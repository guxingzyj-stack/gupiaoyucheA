"""
价值体检评分引擎测试。
"""
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.fundamental import (
    _fetch_financial_indicators,
    _fill_quality_from_abstract,
    _fill_quality_from_indicator,
)
from analysis.value_score import evaluate_value


def base_stock(**overrides):
    data = {
        "industry": "白酒",
    }
    data.update(overrides)
    return data


def base_fundamental(**overrides):
    data = {
        "pe": 10,
        "pb": 2.0,
        "roe": 15,
        "dividend_yield": 2.0,
        "industry": "白酒",
    }
    data.update(overrides)
    return data


def base_metrics(**overrides):
    data = {
        "roe": 15,
        "roe_series": [15, 14, 13],
        "cashflow_to_profit": 1.0,
        "gross_margin": 25,
        "gross_margin_series": [25, 24, 26],
        "revenue_growth": 5,
        "profit_growth": 5,
        "debt_ratio": 45,
        "deducted_profit_ratio": 0.9,
        "missing_fields": [],
        "errors": [],
    }
    data.update(overrides)
    return data


def test_missing_gate():
    result = evaluate_value(
        "000001",
        base_stock(),
        base_fundamental(pe=None, pb=None, roe=None),
        base_metrics(roe=None, cashflow_to_profit=None),
    )
    assert result["gate"]["passed"] is False
    assert result["combo"] == "数据不足，暂不判断"
    assert "ROE" in result["gate"]["missing_fields"]


def test_profit_up_revenue_down_red_flag():
    result = evaluate_value(
        "000002",
        base_stock(),
        base_fundamental(),
        base_metrics(revenue_growth=-3, profit_growth=20),
    )
    assert any(flag["code"] == "R1" for flag in result["red_flags"])
    assert result["quality"]["tier"] == "B"
    assert result["quality"]["score"] <= 7


def test_value_trap_combo():
    result = evaluate_value(
        "000003",
        base_stock(),
        base_fundamental(pe=8, pb=0.8),
        base_metrics(revenue_growth=-5, profit_growth=-1),
    )
    assert any(flag["code"] == "R2" for flag in result["red_flags"])
    assert all(flag["severity"] == "warn" for flag in result["red_flags"])
    assert result["combo"] == "便宜但存疑"
    assert result["valuation"]["dimensions"]["V1_history"]["score"] == 0


def test_bank_branch_uses_pb_roe():
    result = evaluate_value(
        "600000",
        base_stock(industry="银行"),
        base_fundamental(industry="银行", pe=5, pb=0.6, roe=11),
        base_metrics(
            roe=11,
            revenue_growth=3,
            profit_growth=4,
            npl_ratio=1.2,
            provision_coverage=220,
            capital_adequacy=13,
        ),
    )
    assert result["industry_kind"] == "bank"
    assert result["valuation"]["dimensions"]["V1_history"]["score"] == 2
    assert "PB" in result["valuation"]["dimensions"]["V1_history"]["why"]
    assert not any(flag["code"] == "R4" for flag in result["red_flags"])
    assert any(item["code"] == "R4" and item["severity"] == "info" for item in result["cautions"])


def test_low_cashflow_scores_zero():
    result = evaluate_value(
        "000004",
        base_stock(),
        base_fundamental(),
        base_metrics(cashflow_to_profit=0.4),
    )
    assert result["quality"]["dimensions"]["Q2_cashflow"]["score"] == 0
    assert "偏弱" in result["quality"]["dimensions"]["Q2_cashflow"]["why"]


def test_roe_without_history_cannot_score_full():
    result = evaluate_value(
        "000009",
        base_stock(),
        base_fundamental(roe=18),
        base_metrics(roe=18, roe_series=[]),
    )
    q1 = result["quality"]["dimensions"]["Q1_roe"]
    assert q1["score"] == 1
    assert "缺历史序列" in q1["why"]


def test_cashflow_percent_column_always_divides_by_100():
    metrics = {"source_columns": {}}
    df = pd.DataFrame({
        "日期": ["2025-12-31"],
        "经营现金净流量与净利润的比率(%)": [4],
    })
    _fill_quality_from_indicator(metrics, df)

    result = evaluate_value(
        "000007",
        base_stock(),
        base_fundamental(),
        base_metrics(cashflow_to_profit=metrics["cashflow_to_profit"]),
    )
    assert metrics["cashflow_to_profit"] == 0.04
    assert result["quality"]["dimensions"]["Q2_cashflow"]["score"] == 0


def test_fetch_financial_indicators_uses_shared_growth_columns():
    previous = sys.modules.get("akshare")
    fake_df = pd.DataFrame({
        "日期": ["2025-12-31"],
        "净资产收益率(%)": [15],
        "主营业务收入增长率(%)": [12],
        "净利润增长率(%)": [8],
    })
    sys.modules["akshare"] = types.SimpleNamespace(
        stock_financial_analysis_indicator=lambda symbol, start_year="2021": fake_df
    )
    try:
        result = {}
        _fetch_financial_indicators(result, "000010")
    finally:
        if previous is None:
            sys.modules.pop("akshare", None)
        else:
            sys.modules["akshare"] = previous

    assert result["roe"] == 15
    assert result["revenue_growth"] == 12
    assert result["profit_growth"] == 8


def test_deducted_profit_ratio_uses_indicator_and_abstract_fallback():
    metrics = {"source_columns": {}, "errors": []}
    indicator_df = pd.DataFrame({
        "日期": ["2025-12-31"],
        "扣除非经常性损益后的净利润(元)": [60],
    })
    abstract_df = pd.DataFrame({
        "指标": ["归母净利润"],
        "2025": [100],
    })
    _fill_quality_from_indicator(metrics, indicator_df)
    assert metrics.get("deducted_profit_ratio") is None

    _fill_quality_from_abstract(metrics, abstract_df)
    result = evaluate_value(
        "000008",
        base_stock(),
        base_fundamental(),
        base_metrics(deducted_profit_ratio=metrics["deducted_profit_ratio"]),
    )

    assert metrics["deducted_profit_ratio"] == 0.6
    assert metrics["deducted_profit_ratio_source"] == "mixed_indicator_deducted_abstract_parent"
    assert any(flag["code"] == "R1" for flag in result["red_flags"])


def test_tier_boundaries():
    result_a = evaluate_value(
        "000005",
        base_stock(),
        base_fundamental(pe=10, dividend_yield=2),
        base_metrics(debt_ratio=45),
    )
    assert result_a["quality"]["score"] == 8
    assert result_a["quality"]["tier"] == "A"
    assert result_a["valuation"]["score"] == 5
    assert result_a["valuation"]["tier"] == "便宜"

    result_b = evaluate_value(
        "000006",
        base_stock(),
        base_fundamental(pe=10, dividend_yield=0),
        base_metrics(debt_ratio=60),
    )
    assert result_b["quality"]["score"] == 7
    assert result_b["quality"]["tier"] == "B"
    assert result_b["valuation"]["score"] == 4
    assert result_b["valuation"]["tier"] == "合理"


def run_all():
    test_missing_gate()
    test_profit_up_revenue_down_red_flag()
    test_value_trap_combo()
    test_bank_branch_uses_pb_roe()
    test_low_cashflow_scores_zero()
    test_roe_without_history_cannot_score_full()
    test_cashflow_percent_column_always_divides_by_100()
    test_fetch_financial_indicators_uses_shared_growth_columns()
    test_deducted_profit_ratio_uses_indicator_and_abstract_fallback()
    test_tier_boundaries()
    print("test_value_score passed")


if __name__ == "__main__":
    run_all()
