"""
综合评分与投资建议模块
技术面(40%) + 基本面(30%) + 情感面(15%) + 预测面(15%)
"""
import numpy as np
import config


def compute_score(
    tech_signals: dict,
    fundamental: dict,
    sentiment: dict,
    short_pred: dict | None,
    long_pred:  dict | None,
) -> dict:
    """
    Returns
    -------
    dict: {
        total_score, rating, color,
        technical_score, fundamental_score,
        sentiment_score, prediction_score,
        details, risk, stop_loss, target_price, position_pct
    }
    """
    w = config.SCORE_WEIGHTS

    tech_s  = _technical_score(tech_signals)
    fund_s  = fundamental.get("score", 50.0)
    sent_s  = _sentiment_score(sentiment)
    pred_s  = _prediction_score(short_pred, long_pred)

    total = (
        tech_s  * w["technical"]   +
        fund_s  * w["fundamental"] +
        sent_s  * w["sentiment"]   +
        pred_s  * w["prediction"]
    )
    total = float(np.clip(total, 0, 100))

    rating, color = _get_rating(total)

    # 风险管理
    last_price = (short_pred or long_pred or {}).get("last_price", 0)
    stop_loss, target_price, position = _risk_management(
        last_price, short_pred, long_pred, total
    )

    return {
        "total_score":       round(total, 1),
        "rating":            rating,
        "color":             color,
        "technical_score":   round(tech_s, 1),
        "fundamental_score": round(fund_s, 1),
        "sentiment_score":   round(sent_s, 1),
        "prediction_score":  round(pred_s, 1),
        "details":           fundamental.get("details", []),
        "stop_loss":         stop_loss,
        "target_price":      target_price,
        "position_pct":      position,
    }


def _technical_score(signals: dict) -> float:
    """根据信号买卖数量打分"""
    if not signals:
        return 50.0
    buy   = signals.get("buy_count", 0)
    sell  = signals.get("sell_count", 0)
    total = buy + sell + signals.get("neutral_count", 0)
    if total == 0:
        return 50.0
    ratio = (buy - sell) / total  # -1 ~ +1
    score = 50.0 + ratio * 40.0   # 10 ~ 90
    return float(np.clip(score, 0, 100))


def _sentiment_score(sentiment: dict) -> float:
    """情感分(0-1) → (0-100)"""
    s = sentiment.get("score", 0.5)
    return float(np.clip(s * 100, 0, 100))


def _prediction_score(short: dict | None, long: dict | None) -> float:
    """预测涨幅与置信度综合打分"""
    scores = []
    for pred in [short, long]:
        if not pred:
            continue
        pct = pred.get("pct_change", 0)
        # 涨10%→满分，跌10%→0分
        s = 50.0 + pct * 4.0
        scores.append(float(np.clip(s, 0, 100)))
    if not scores:
        return 50.0
    return float(np.mean(scores))


def _get_rating(score: float) -> tuple:
    for threshold, zh_label, en_label, color in config.RATING_THRESHOLDS:
        if score >= threshold:
            label = zh_label if config.LANGUAGE == "zh" else en_label
            return label, color
    return "强烈卖出", "#D50000"


def _risk_management(
    last_price: float,
    short_pred: dict | None,
    long_pred:  dict | None,
    score: float,
) -> tuple:
    """计算止损、目标价、建议仓位"""
    if last_price <= 0:
        return None, None, "0%"

    # 止损：ATR动态止损（简化：-5% ~ -8%）
    stop_loss_pct = -0.05 if score >= 65 else -0.08
    stop_loss     = round(last_price * (1 + stop_loss_pct), 2)

    # 目标价：保守估计 = 预测均值 + 0.5×(置信带半宽)
    # （避免直接取 max(upper_bound) 过于乐观误导用户）
    target_candidates = [last_price]
    for pred in [short_pred, long_pred]:
        if not pred:
            continue
        preds = pred.get("predictions") or []
        ub    = pred.get("upper_bound") or []
        lb    = pred.get("lower_bound") or []
        if not preds:
            continue
        try:
            mean_p = float(np.mean(preds))
            if ub and lb and len(ub) == len(lb):
                half_width = float(np.mean(np.asarray(ub) - np.asarray(lb))) / 2.0
            elif ub:
                half_width = max(0.0, float(np.mean(ub)) - mean_p)
            else:
                half_width = 0.0
            target_candidates.append(mean_p + 0.5 * half_width)
        except (TypeError, ValueError):
            continue
    target_price = round(max(target_candidates), 2)

    # Kelly公式简化版仓位
    if score >= 80:
        position = "60%-80%"
    elif score >= 65:
        position = "30%-50%"
    elif score >= 45:
        position = "10%-20%（观望）"
    else:
        position = "0%（建议空仓）"

    return stop_loss, target_price, position
