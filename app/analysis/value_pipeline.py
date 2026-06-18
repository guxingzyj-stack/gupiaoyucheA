"""
价值体检接线层。

保持主流程只依赖一个轻量 helper，避免在多个入口重复写联网取数逻辑。
"""
from __future__ import annotations

import config
from analysis.fundamental import fetch_quality_metrics
from analysis.value_score import evaluate_value


def build_value_assessment(symbol: str, stock_info: dict, fundamental: dict) -> dict | None:
    """构建独立的价值体检结果；关闭开关时返回 None。"""
    if not getattr(config, "SHOW_VALUE_ASSESSMENT", True):
        return None

    metrics = None
    try:
        metrics = fetch_quality_metrics(symbol)
    except Exception as exc:
        metrics = {
            "errors": [f"价值体检取数失败: {exc}"],
            "missing_fields": [],
        }

    return evaluate_value(symbol, stock_info or {}, fundamental or {}, quality_metrics=metrics)


def annotate_signal_conflict(value_assessment: dict | None, score_result: dict | None) -> dict | None:
    """标注价值红旗是否与买入类综合评分冲突。"""
    if not value_assessment:
        return value_assessment

    score_result = score_result or {}
    rating = str(score_result.get("rating", ""))
    total = score_result.get("total_score", 0) or 0
    try:
        total = float(total)
    except Exception:
        total = 0.0

    is_buy = (
        total >= getattr(config, "ALERT_BUY_THRESHOLD", 65)
        or rating in ("强烈买入", "买入", "Strong Buy", "Buy")
    )
    has_warn = any(
        flag.get("severity") == "warn"
        for flag in value_assessment.get("red_flags", [])
    )
    value_assessment["conflict_with_signal"] = bool(is_buy and has_warn)
    return value_assessment
