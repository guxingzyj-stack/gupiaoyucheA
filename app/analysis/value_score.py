"""
价值体检评分引擎。

本模块只输出独立的价值评估结构，不参与现有综合评分和模型融合。
"""
from __future__ import annotations

from statistics import pstdev
from typing import Any


GENERAL_REQUIRED_FIELDS = {
    "roe": "ROE",
    "pe": "PE",
    "pb": "PB",
    "cashflow_to_profit": "经营现金流/净利润",
}

BANK_REQUIRED_FIELDS = {
    "roe": "ROE",
    "pb": "PB",
}

BANK_KEYWORDS = ("银行",)
HEAVY_ASSET_KEYWORDS = ("钢铁", "水泥", "煤炭", "有色", "化工", "航运", "电力", "地产")
CYCLICAL_KEYWORDS = ("钢铁", "水泥", "煤炭", "有色", "化工", "航运")


def evaluate_value(
    symbol: str,
    stock_info: dict,
    fundamental: dict,
    quality_metrics: dict | None = None,
) -> dict:
    """
    评估股票的长期价值属性。

    返回字段固定为 gate、industry_kind、quality、valuation、red_flags、
    cautions、combo、open_questions、conflict_with_signal、details，
    供后续 UI 旁路展示。
    """
    quality_metrics = quality_metrics or {}
    industry = _pick("industry", stock_info, fundamental, quality_metrics) or "未知"
    industry_kind = _industry_kind(industry)

    data = {
        "symbol": symbol,
        "industry": industry,
        "roe": _pick_num("roe", quality_metrics, fundamental, stock_info),
        "roe_series": _pick_series("roe_series", quality_metrics),
        "cashflow_to_profit": _pick_num("cashflow_to_profit", quality_metrics, fundamental),
        "cashflow_to_profit_series": _pick_series("cashflow_to_profit_series", quality_metrics),
        "deducted_profit_ratio": _pick_num("deducted_profit_ratio", quality_metrics, fundamental),
        "gross_margin": _pick_num("gross_margin", quality_metrics, fundamental),
        "gross_margin_series": _pick_series("gross_margin_series", quality_metrics),
        "revenue_growth": _pick_num("revenue_growth", quality_metrics, fundamental),
        "profit_growth": _pick_num("profit_growth", quality_metrics, fundamental),
        "debt_ratio": _pick_num("debt_ratio", quality_metrics, fundamental),
        "pe": _pick_num("pe", fundamental, stock_info, quality_metrics),
        "pb": _pick_num("pb", fundamental, stock_info, quality_metrics),
        "pe_percentile": _pick_num("pe_percentile", quality_metrics, fundamental),
        "pb_percentile": _pick_num("pb_percentile", quality_metrics, fundamental),
        "dividend_yield": _pick_num("dividend_yield", fundamental, stock_info, quality_metrics),
        "npl_ratio": _pick_num("npl_ratio", quality_metrics, fundamental, stock_info),
        "provision_coverage": _pick_num("provision_coverage", quality_metrics, fundamental, stock_info),
        "capital_adequacy": _pick_num("capital_adequacy", quality_metrics, fundamental, stock_info),
        "top_customer_ratio": _pick_num("top_customer_ratio", quality_metrics, fundamental),
        "top_customer_risk": bool(quality_metrics.get("top_customer_risk") or fundamental.get("top_customer_risk")),
    }

    gate = _build_gate(data, industry_kind, quality_metrics)
    red_flags = _build_red_flags(data, industry_kind)
    cautions = _build_cautions(industry_kind)
    quality = _score_quality(data, industry_kind, red_flags)
    valuation = _score_valuation(data, industry_kind, quality, red_flags)
    combo = _combine_judgement(gate, quality, valuation, red_flags)
    open_questions = _build_open_questions(data, industry_kind, gate, quality_metrics)

    return {
        "gate": gate,
        "industry_kind": industry_kind,
        "quality": quality,
        "valuation": valuation,
        "red_flags": red_flags,
        "cautions": cautions,
        "combo": combo,
        "open_questions": open_questions,
        "conflict_with_signal": None,
        "details": {
            "symbol": symbol,
            "industry": industry,
            "metrics": data,
            "source_columns": quality_metrics.get("source_columns", {}),
            "missing_fields": quality_metrics.get("missing_fields", []),
            "errors": quality_metrics.get("errors", []),
        },
    }


def _build_gate(data: dict, industry_kind: str, quality_metrics: dict) -> dict:
    required = BANK_REQUIRED_FIELDS if industry_kind == "bank" else GENERAL_REQUIRED_FIELDS
    missing = [label for key, label in required.items() if data.get(key) is None]
    missing = sorted(set(missing))
    passed = len(missing) == 0
    return {
        "passed": passed,
        "status": "可评估" if passed else "补数据",
        "missing_fields": missing,
        "reasons": [] if passed else [f"缺少关键字段：{', '.join(missing)}"],
    }


def _score_quality(data: dict, industry_kind: str, red_flags: list[dict]) -> dict:
    dimensions = {
        "Q1_roe": _score_roe(data),
        "Q2_cashflow": _score_cashflow(data, industry_kind),
        "Q3_moat": _score_moat(data, industry_kind),
        "Q4_growth": _score_growth(data),
        "Q5_balance": _score_balance(data, industry_kind),
    }
    score = sum(item["score"] for item in dimensions.values())
    if any(flag["code"] == "R1" for flag in red_flags):
        score = max(0, score - 2)
    score = min(10, score)
    return {
        "score": score,
        "tier": _quality_tier(score),
        "dimensions": dimensions,
    }


def _score_valuation(data: dict, industry_kind: str, quality: dict, red_flags: list[dict]) -> dict:
    dimensions = {
        "V1_history": _score_history_valuation(data, industry_kind),
        "V2_vs_quality": _score_quality_price_match(data, industry_kind, quality),
        "V3_yield": _score_yield(data),
    }
    if any(flag["code"] == "R2" for flag in red_flags):
        dimensions["V1_history"] = {
            "score": 0,
            "why": "便宜来自收入下滑阶段，便宜项暂不计正面",
        }
    score = min(6, sum(item["score"] for item in dimensions.values()))
    return {
        "score": score,
        "tier": _valuation_tier(score),
        "dimensions": dimensions,
    }


def _build_red_flags(data: dict, industry_kind: str) -> list[dict]:
    flags = []
    revenue_growth = data.get("revenue_growth")
    profit_growth = data.get("profit_growth")
    deducted_ratio = data.get("deducted_profit_ratio")
    if (
        profit_growth is not None
        and profit_growth > 0
        and revenue_growth is not None
        and revenue_growth <= 0
    ) or (deducted_ratio is not None and deducted_ratio < 0.7):
        flags.append({
            "code": "R1",
            "name": "增利不增收/扣非不足",
            "severity": "warn",
            "effect": "质量降一档",
        })

    if _looks_cheap(data, industry_kind) and revenue_growth is not None and revenue_growth <= 0:
        flags.append({
            "code": "R2",
            "name": "低估值陷阱",
            "severity": "warn",
            "effect": "便宜不计正面",
        })

    if data.get("top_customer_risk") or (
        data.get("top_customer_ratio") is not None and data["top_customer_ratio"] >= 40
    ):
        flags.append({
            "code": "R3",
            "name": "客户集中",
            "severity": "warn",
            "effect": "列为待核实风险",
        })
    return flags


def _build_cautions(industry_kind: str) -> list[dict]:
    if industry_kind in ("bank", "heavy_asset", "cyclical"):
        return [{
            "code": "R4",
            "name": "行业特殊口径",
            "severity": "info",
            "effect": "需结合行业专属指标",
        }]
    return []


def _build_open_questions(data: dict, industry_kind: str, gate: dict, quality_metrics: dict) -> list[str]:
    questions = []
    if gate["missing_fields"]:
        questions.append("关键财务字段缺失，需补齐后再做高置信价值判断")
    if quality_metrics.get("errors"):
        questions.append("部分 AKShare 数据接口返回异常，已按 None 处理")
    if industry_kind == "bank":
        for key, label in (
            ("npl_ratio", "不良贷款率"),
            ("provision_coverage", "拨备覆盖率"),
            ("capital_adequacy", "资本充足率"),
        ):
            if data.get(key) is None:
                questions.append(f"银行专属指标缺失：{label}")
    if any(v is None for v in (data.get("pe_percentile"), data.get("pb_percentile"))):
        questions.append("历史 PE/PB 分位缺失，估值仅按绝对值粗判")
    return questions


def _score_roe(data: dict) -> dict:
    roe = data.get("roe")
    series = _valid_series(data.get("roe_series"))
    if roe is None:
        return {"score": 0, "why": "ROE缺失"}
    if roe >= 15 and not series:
        return {"score": 1, "why": f"ROE {roe:.1f}%，但缺历史序列，稳定性未确认"}
    if roe >= 15 and min(series[:3]) >= 8 and _series_std(series[:5]) <= 6:
        return {"score": 2, "why": f"ROE {roe:.1f}% 且近年较稳定"}
    if roe >= 8:
        return {"score": 1, "why": f"ROE {roe:.1f}% 达到可接受区间"}
    return {"score": 0, "why": f"ROE {roe:.1f}% 偏低"}


def _score_cashflow(data: dict, industry_kind: str) -> dict:
    if industry_kind == "bank":
        return {"score": 1, "why": "银行现金流口径弱相关，按中性处理"}
    ratio = data.get("cashflow_to_profit")
    if ratio is None:
        return {"score": 0, "why": "经营现金流/净利润缺失"}
    if ratio >= 1.0:
        return {"score": 2, "why": f"现金流覆盖净利润 {ratio:.2f}"}
    if ratio >= 0.7:
        return {"score": 1, "why": f"现金流覆盖净利润 {ratio:.2f}，尚可"}
    return {"score": 0, "why": f"现金流覆盖净利润 {ratio:.2f}，偏弱"}


def _score_moat(data: dict, industry_kind: str) -> dict:
    if industry_kind == "bank":
        return {"score": 1, "why": "银行护城河口径不同，按中性处理"}
    gross_margin = data.get("gross_margin")
    series = _valid_series(data.get("gross_margin_series"))
    if gross_margin is None:
        return {"score": 0, "why": "毛利率缺失"}
    if gross_margin >= 35 and _series_std(series[:5]) <= 8:
        return {"score": 2, "why": f"毛利率 {gross_margin:.1f}% 且波动较小"}
    if gross_margin >= 20:
        return {"score": 1, "why": f"毛利率 {gross_margin:.1f}% 有一定业务质量"}
    return {"score": 0, "why": f"毛利率 {gross_margin:.1f}% 偏低"}


def _score_growth(data: dict) -> dict:
    revenue_growth = data.get("revenue_growth")
    profit_growth = data.get("profit_growth")
    if revenue_growth is None or profit_growth is None:
        return {"score": 0, "why": "营收或利润增长缺失"}
    if revenue_growth >= 10 and profit_growth >= 10:
        return {"score": 2, "why": "营收和利润同步增长"}
    if revenue_growth >= 0 and profit_growth >= 0:
        return {"score": 1, "why": "营收和利润未下滑"}
    return {"score": 0, "why": "营收或利润出现下滑"}


def _score_balance(data: dict, industry_kind: str) -> dict:
    if industry_kind == "bank":
        npl = data.get("npl_ratio")
        provision = data.get("provision_coverage")
        if npl is None or provision is None:
            return {"score": 1, "why": "银行风控指标缺失，按中性处理"}
        if npl <= 1.5 and provision >= 150:
            return {"score": 2, "why": f"不良率 {npl:.2f}% 且拨备 {provision:.0f}%"}
        if npl <= 2.5:
            return {"score": 1, "why": f"不良率 {npl:.2f}% 尚可"}
        return {"score": 0, "why": f"不良率 {npl:.2f}% 偏高"}

    debt_ratio = data.get("debt_ratio")
    if debt_ratio is None:
        return {"score": 0, "why": "资产负债率缺失"}
    if debt_ratio <= 50:
        return {"score": 2, "why": f"资产负债率 {debt_ratio:.1f}% 较稳健"}
    if debt_ratio <= 70:
        return {"score": 1, "why": f"资产负债率 {debt_ratio:.1f}% 尚可"}
    return {"score": 0, "why": f"资产负债率 {debt_ratio:.1f}% 偏高"}


def _score_history_valuation(data: dict, industry_kind: str) -> dict:
    pe = data.get("pe")
    pb = data.get("pb")
    pe_percentile = data.get("pe_percentile")
    pb_percentile = data.get("pb_percentile")

    if industry_kind == "bank":
        if pb_percentile is not None:
            if pb_percentile <= 30:
                return {"score": 2, "why": f"PB历史分位 {pb_percentile:.0f}% 偏低"}
            if pb_percentile <= 70:
                return {"score": 1, "why": f"PB历史分位 {pb_percentile:.0f}% 合理"}
            return {"score": 0, "why": f"PB历史分位 {pb_percentile:.0f}% 偏高"}
        if pb is None:
            return {"score": 0, "why": "PB缺失"}
        if pb < 0.8:
            return {"score": 2, "why": f"银行按PB粗判，PB {pb:.2f} 偏低"}
        if pb <= 1.2:
            return {"score": 1, "why": f"银行按PB粗判，PB {pb:.2f} 合理"}
        return {"score": 0, "why": f"银行按PB粗判，PB {pb:.2f} 偏高"}

    if pe_percentile is not None:
        if pe_percentile <= 30:
            return {"score": 2, "why": f"PE历史分位 {pe_percentile:.0f}% 偏低"}
        if pe_percentile <= 70:
            return {"score": 1, "why": f"PE历史分位 {pe_percentile:.0f}% 合理"}
        return {"score": 0, "why": f"PE历史分位 {pe_percentile:.0f}% 偏高"}
    if pe is None:
        return {"score": 0, "why": "PE缺失"}
    if 0 < pe <= 12:
        return {"score": 2, "why": f"PE {pe:.1f} 偏低"}
    if pe <= 25:
        return {"score": 1, "why": f"PE {pe:.1f} 合理"}
    return {"score": 0, "why": f"PE {pe:.1f} 偏高"}


def _score_quality_price_match(data: dict, industry_kind: str, quality: dict) -> dict:
    pe = data.get("pe")
    pb = data.get("pb")
    roe = data.get("roe")
    tier = quality.get("tier")
    if industry_kind == "bank":
        if roe is None or pb is None:
            return {"score": 0, "why": "银行PB/ROE匹配数据缺失"}
        if roe >= 10 and pb <= 1.2:
            return {"score": 2, "why": "银行ROE与PB匹配度较好"}
        if roe >= 8 and pb <= 1.5:
            return {"score": 1, "why": "银行ROE与PB匹配度一般"}
        return {"score": 0, "why": "银行ROE与PB匹配度偏弱"}

    if pe is None:
        return {"score": 0, "why": "PE缺失，无法评估质价匹配"}
    if tier in ("A", "B") and pe <= 25:
        return {"score": 2, "why": "质量不差且估值不贵"}
    if tier in ("A", "B") and pe <= 40:
        return {"score": 1, "why": "质量不差但估值略高"}
    return {"score": 0, "why": "质量或估值匹配度不足"}


def _score_yield(data: dict) -> dict:
    dy = data.get("dividend_yield")
    if dy is None:
        return {"score": 0, "why": "股息率缺失"}
    if dy >= 4:
        return {"score": 2, "why": f"股息率 {dy:.2f}% 较高"}
    if dy >= 2:
        return {"score": 1, "why": f"股息率 {dy:.2f}% 有一定回报"}
    return {"score": 0, "why": f"股息率 {dy:.2f}% 偏低"}


def _combine_judgement(gate: dict, quality: dict, valuation: dict, red_flags: list[dict]) -> str:
    if not gate.get("passed"):
        return "数据不足，暂不判断"
    if any(flag["code"] == "R2" for flag in red_flags):
        return "便宜但存疑"
    q_tier = quality.get("tier")
    v_tier = valuation.get("tier")
    if q_tier in ("A", "B") and v_tier == "便宜":
        return "又好又便宜"
    if q_tier in ("A", "B") and v_tier == "合理":
        return "好生意，价格公允"
    if q_tier in ("A", "B"):
        return "好生意但偏贵"
    if v_tier == "便宜":
        return "便宜但需验证质量"
    return "平庸且不便宜"


def _industry_kind(industry: str) -> str:
    text = str(industry or "")
    if any(k in text for k in BANK_KEYWORDS):
        return "bank"
    if any(k in text for k in HEAVY_ASSET_KEYWORDS):
        return "heavy_asset"
    if any(k in text for k in CYCLICAL_KEYWORDS):
        return "cyclical"
    return "general"


def _quality_tier(score: float) -> str:
    if score >= 8:
        return "A"
    if score >= 5:
        return "B"
    return "C"


def _valuation_tier(score: float) -> str:
    if score >= 5:
        return "便宜"
    if score >= 3:
        return "合理"
    return "贵"


def _looks_cheap(data: dict, industry_kind: str) -> bool:
    pb = data.get("pb")
    pe = data.get("pe")
    if industry_kind == "bank":
        return pb is not None and pb < 0.8
    return (pe is not None and 0 < pe <= 12) or (pb is not None and pb < 1)


def _pick(field: str, *sources: dict) -> Any:
    for source in sources:
        value = source.get(field)
        if value not in (None, ""):
            return value
    return None


def _pick_num(field: str, *sources: dict) -> float | None:
    value = _pick(field, *sources)
    if value is None:
        return None
    try:
        return float(str(value).replace("%", "").replace(",", ""))
    except Exception:
        return None


def _pick_series(field: str, *sources: dict) -> list[float]:
    value = _pick(field, *sources)
    if not value:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return _valid_series(value)


def _valid_series(values) -> list[float]:
    result = []
    for value in values or []:
        try:
            result.append(float(str(value).replace("%", "").replace(",", "")))
        except Exception:
            continue
    return result


def _series_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))
