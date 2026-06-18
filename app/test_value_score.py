"""
价值体检评分引擎测试。
"""
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data.fetcher as fetcher
from data.fetcher import _fill_industry_fallback
from analysis.fundamental import (
    _compute_percentile,
    _fetch_financial_indicators,
    _fill_normalized_pe,
    _fill_quality_from_abstract,
    _fill_quality_from_indicator,
)
from analysis.value_pipeline import annotate_signal_conflict
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


def test_value_trap_uses_multi_year_revenue_downtrend():
    result = evaluate_value(
        "000016",
        base_stock(),
        base_fundamental(pe=8, pb=0.8),
        base_metrics(revenue_growth=5, revenue_cagr=-0.02, revenue_years=5),
    )

    r2 = [flag for flag in result["red_flags"] if flag["code"] == "R2"]
    assert r2
    assert "多年营收CAGR -2.00%" in r2[0]["why"]


def test_value_trap_ignores_single_year_drop_when_multi_year_revenue_grows():
    result = evaluate_value(
        "000017",
        base_stock(),
        base_fundamental(pe=8, pb=0.8),
        base_metrics(revenue_growth=-5, revenue_cagr=0.05, revenue_years=5),
    )

    assert not any(flag["code"] == "R2" for flag in result["red_flags"])
    assert result["details"]["metrics"]["revenue_cagr"] == 0.05


def test_value_trap_does_not_flag_slight_positive_revenue_cagr():
    result = evaluate_value(
        "000018",
        base_stock(),
        base_fundamental(pe=8, pb=0.8),
        base_metrics(revenue_growth=-1, revenue_cagr=0.001, revenue_years=5),
    )

    assert not any(flag["code"] == "R2" for flag in result["red_flags"])
    assert result["details"]["metrics"]["revenue_cagr"] == 0.001


def test_signal_conflict_true_for_buy_with_r2_warn_flag():
    result = evaluate_value(
        "000014",
        base_stock(),
        base_fundamental(pe=8, pb=0.8),
        base_metrics(revenue_growth=-5, profit_growth=-1),
    )
    assert any(
        flag.get("code") == "R2" and flag.get("severity") == "warn"
        for flag in result["red_flags"]
    )

    annotated = annotate_signal_conflict(
        result,
        {"rating": "买入", "total_score": 70},
    )

    assert annotated["conflict_with_signal"] is True


def test_signal_conflict_false_without_warn_flag():
    result = evaluate_value(
        "000015",
        base_stock(),
        base_fundamental(),
        base_metrics(),
    )
    assert not any(flag.get("severity") == "warn" for flag in result["red_flags"])

    annotated = annotate_signal_conflict(
        result,
        {"rating": "买入", "total_score": 70},
    )

    assert annotated["conflict_with_signal"] is False


def test_valuation_percentile_used_in_v1():
    result = evaluate_value(
        "000011",
        base_stock(),
        base_fundamental(pe=20),
        base_metrics(pe_percentile=20),
    )
    assert result["valuation"]["dimensions"]["V1_history"]["score"] == 2
    assert "历史分位" in result["valuation"]["dimensions"]["V1_history"]["why"]


def test_missing_valuation_percentile_does_not_block_gate():
    result = evaluate_value(
        "000012",
        base_stock(),
        base_fundamental(),
        base_metrics(missing_fields=["PE历史分位", "PB历史分位"]),
    )
    assert result["gate"]["passed"] is True
    assert any("历史 PE/PB 分位缺失" in item for item in result["open_questions"])


def test_non_core_quality_missing_does_not_block_gate():
    result = evaluate_value(
        "000013",
        base_stock(),
        base_fundamental(),
        base_metrics(missing_fields=["毛利率"]),
    )
    assert result["gate"]["passed"] is True
    assert "毛利率" in result["details"]["missing_fields"]


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


def test_bank_moat_uses_neutral_score():
    result = evaluate_value(
        "600036",
        base_stock(industry="银行"),
        base_fundamental(industry="银行", pb=0.8, roe=10),
        base_metrics(gross_margin=None),
    )

    q3 = result["quality"]["dimensions"]["Q3_moat"]
    assert q3["score"] == 1
    assert q3["why"] == "银行护城河口径不同，按中性处理"


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


def test_stock_info_industry_fallback_from_individual_info():
    previous = sys.modules.get("akshare")
    fake_df = pd.DataFrame({
        "item": ["总市值", "行业"],
        "value": ["100亿", "银行"],
    })
    sys.modules["akshare"] = types.SimpleNamespace(
        stock_individual_info_em=lambda symbol: fake_df
    )
    try:
        info = {}
        _fill_industry_fallback(info, "600036")
    finally:
        if previous is None:
            sys.modules.pop("akshare", None)
        else:
            sys.modules["akshare"] = previous

    assert info["所属行业"] == "银行"
    assert info["industry"] == "银行"
    assert info["industry_source"] == "live:stock_individual_info_em"


def test_industry_fallback_uses_persistent_cache_when_live_empty():
    old_live = fetcher._fetch_live_industry
    old_cache = fetcher._load_industry_cache
    old_map = fetcher._load_industry_fallback_map
    try:
        fetcher._fetch_live_industry = lambda info, symbol: ("", "")
        fetcher._load_industry_cache = lambda: {
            "600036": {"industry": "银行", "source": "live:f10_orgprofile"}
        }
        fetcher._load_industry_fallback_map = lambda: {}

        info = {}
        _fill_industry_fallback(info, "600036")
    finally:
        fetcher._fetch_live_industry = old_live
        fetcher._load_industry_cache = old_cache
        fetcher._load_industry_fallback_map = old_map

    assert info["所属行业"] == "银行"
    assert info["industry"] == "银行"
    assert info["industry_source"] == "persistent_cache"


def test_industry_fallback_uses_map_when_live_and_cache_empty():
    old_live = fetcher._fetch_live_industry
    old_cache = fetcher._load_industry_cache
    old_map = fetcher._load_industry_fallback_map
    try:
        fetcher._fetch_live_industry = lambda info, symbol: ("", "")
        fetcher._load_industry_cache = lambda: {}
        fetcher._load_industry_fallback_map = lambda: {
            "601398": {"industry": "银行", "source": "manual_verified"}
        }

        info = {}
        _fill_industry_fallback(info, "601398")
    finally:
        fetcher._fetch_live_industry = old_live
        fetcher._load_industry_cache = old_cache
        fetcher._load_industry_fallback_map = old_map

    assert info["所属行业"] == "银行"
    assert info["industry"] == "银行"
    assert info["industry_source"] == "fallback_map"


def test_industry_fallback_leaves_empty_without_live_cache_or_map():
    old_live = fetcher._fetch_live_industry
    old_cache = fetcher._load_industry_cache
    old_map = fetcher._load_industry_fallback_map
    try:
        fetcher._fetch_live_industry = lambda info, symbol: ("", "")
        fetcher._load_industry_cache = lambda: {}
        fetcher._load_industry_fallback_map = lambda: {}

        info = {}
        _fill_industry_fallback(info, "999999")
    finally:
        fetcher._fetch_live_industry = old_live
        fetcher._load_industry_cache = old_cache
        fetcher._load_industry_fallback_map = old_map

    assert "所属行业" not in info
    assert "industry_source" not in info


def test_gross_margin_fallback_from_abstract_revenue_cost():
    metrics = {"source_columns": {}, "errors": []}
    abstract_df = pd.DataFrame({
        "指标": ["营业总收入", "营业成本"],
        "2025": [100, 64],
    })

    _fill_quality_from_abstract(metrics, abstract_df)

    assert metrics["gross_margin"] == 36
    assert metrics["gross_margin_series"] == [36]
    assert metrics["source_columns"]["gross_margin"] == "stock_financial_abstract:(营业总收入-营业成本)/营业总收入"


def test_bank_risk_metrics_from_abstract_feed_q5():
    metrics = {"source_columns": {}, "errors": []}
    abstract_df = pd.DataFrame({
        "指标": ["不良贷款率", "拨备覆盖率", "资本充足率"],
        "2025": [1.2, 220, 13],
    })

    _fill_quality_from_abstract(metrics, abstract_df)
    result = evaluate_value(
        "600036",
        base_stock(industry="银行"),
        base_fundamental(industry="银行", pb=0.8, roe=10),
        base_metrics(**metrics),
    )

    assert metrics["npl_ratio"] == 1.2
    assert metrics["provision_coverage"] == 220
    assert metrics["capital_adequacy"] == 13
    assert result["quality"]["dimensions"]["Q5_balance"]["score"] == 2


def test_normalized_pe_from_annual_profit_series():
    metrics = {"source_columns": {}, "errors": [], "pe": 8}
    abstract_df = pd.DataFrame({
        "指标": ["归母净利润"],
        "20251231": [300],
        "20241231": [50],
        "20231231": [50],
        "20221231": [50],
        "20211231": [50],
    })

    _fill_quality_from_abstract(metrics, abstract_df)
    _fill_normalized_pe(metrics)

    assert metrics["normalized_years"] == 5
    assert metrics["normalized_factor"] == 3
    assert metrics["normalized_pe"] == 24


def test_revenue_cagr_from_annual_revenue_series():
    metrics = {"source_columns": {}, "errors": []}
    abstract_df = pd.DataFrame({
        "指标": ["营业总收入"],
        "20251231": [121],
        "20241231": [110],
        "20231231": [100],
        "20221231": [90],
        "20211231": [80],
    })

    _fill_quality_from_abstract(metrics, abstract_df)

    assert metrics["revenue_series"] == [121, 110, 100, 90, 80]
    assert metrics["revenue_years"] == 5
    assert round(metrics["revenue_cagr"], 4) == round((121 / 80) ** (1 / 4) - 1, 4)


def test_cyclical_high_profit_uses_normalized_pe_not_raw_low_pe():
    result = evaluate_value(
        "600019",
        base_stock(industry="钢铁"),
        base_fundamental(industry="钢铁", pe=8, pb=1.2),
        base_metrics(normalized_factor=3, normalized_years=5),
    )

    v1 = result["valuation"]["dimensions"]["V1_history"]
    assert result["industry_kind"] == "heavy_asset"
    assert v1["score"] == 1
    assert "按正常化PE 24.0" in v1["why"]


def test_cyclical_low_profit_uses_lower_normalized_pe():
    result = evaluate_value(
        "600019",
        base_stock(industry="钢铁"),
        base_fundamental(industry="钢铁", pe=20, pb=1.2),
        base_metrics(normalized_factor=0.5, normalized_years=5),
    )

    v1 = result["valuation"]["dimensions"]["V1_history"]
    assert v1["score"] == 2
    assert "按正常化PE 10.0" in v1["why"]


def test_cyclical_missing_normalized_pe_adds_open_question():
    result = evaluate_value(
        "600019",
        base_stock(industry="钢铁"),
        base_fundamental(industry="钢铁", pe=8, pb=1.2),
        base_metrics(),
    )

    assert any("周期股低 PE 慎判" in item for item in result["open_questions"])


def test_compute_percentile_filters_invalid_values():
    percentile = _compute_percentile([-1, 0, 1, 3, 5, None], 3)
    assert round(percentile, 2) == 66.67
    assert _compute_percentile([0, None, -1], 3) is None
    assert _compute_percentile([1, 2, 3], 0) is None


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
    test_value_trap_uses_multi_year_revenue_downtrend()
    test_value_trap_ignores_single_year_drop_when_multi_year_revenue_grows()
    test_value_trap_does_not_flag_slight_positive_revenue_cagr()
    test_signal_conflict_true_for_buy_with_r2_warn_flag()
    test_signal_conflict_false_without_warn_flag()
    test_valuation_percentile_used_in_v1()
    test_missing_valuation_percentile_does_not_block_gate()
    test_non_core_quality_missing_does_not_block_gate()
    test_bank_branch_uses_pb_roe()
    test_low_cashflow_scores_zero()
    test_roe_without_history_cannot_score_full()
    test_cashflow_percent_column_always_divides_by_100()
    test_fetch_financial_indicators_uses_shared_growth_columns()
    test_stock_info_industry_fallback_from_individual_info()
    test_industry_fallback_uses_persistent_cache_when_live_empty()
    test_industry_fallback_uses_map_when_live_and_cache_empty()
    test_industry_fallback_leaves_empty_without_live_cache_or_map()
    test_gross_margin_fallback_from_abstract_revenue_cost()
    test_bank_moat_uses_neutral_score()
    test_bank_risk_metrics_from_abstract_feed_q5()
    test_normalized_pe_from_annual_profit_series()
    test_revenue_cagr_from_annual_revenue_series()
    test_cyclical_high_profit_uses_normalized_pe_not_raw_low_pe()
    test_cyclical_low_profit_uses_lower_normalized_pe()
    test_cyclical_missing_normalized_pe_adds_open_question()
    test_compute_percentile_filters_invalid_values()
    test_deducted_profit_ratio_uses_indicator_and_abstract_fallback()
    test_tier_boundaries()
    print("test_value_score passed")


if __name__ == "__main__":
    run_all()
