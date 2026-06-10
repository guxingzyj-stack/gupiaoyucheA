"""
历史评分记录模块
每次分析后保存评分，支持趋势查看和预警
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Optional

import config


def _history_path(symbol: str) -> str:
    return os.path.join(config.HISTORY_DIR, f"{symbol}.json")


def load_history(symbol: str) -> List[Dict]:
    path = _history_path(symbol)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_score(
    symbol:       str,
    stock_name:   str,
    score_result: dict,
    short_pred:   Optional[dict],
    long_pred:    Optional[dict],
):
    """追加一条评分记录"""
    records = load_history(symbol)
    today   = datetime.now().strftime("%Y-%m-%d")

    # 同一天的记录直接覆盖（避免重复运行累积）
    records = [r for r in records if r.get("date") != today]

    entry = {
        "date":              today,
        "stock_name":        stock_name,
        "total_score":       round(score_result.get("total_score", 0), 1),
        "technical_score":   round(score_result.get("technical_score", 0), 1),
        "fundamental_score": round(score_result.get("fundamental_score", 0), 1),
        "sentiment_score":   round(score_result.get("sentiment_score", 0), 1),
        "prediction_score":  round(score_result.get("prediction_score", 0), 1),
        "rating":            score_result.get("rating", ""),
        "short_pct":         round(short_pred["pct_change"], 2) if short_pred else None,
        "long_pct":          round(long_pred["pct_change"], 2) if long_pred else None,
        "stop_loss":         score_result.get("stop_loss"),
        "target_price":      score_result.get("target_price"),
    }
    records.append(entry)

    # 最多保留 365 条（约1年）
    records = records[-365:]

    with open(_history_path(symbol), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return entry


def check_alerts(symbol: str, stock_name: str, score: float) -> List[str]:
    """检查是否触发预警，返回预警消息列表"""
    alerts = []
    records = load_history(symbol)

    if len(records) < 2:
        return alerts

    prev_score = records[-2].get("total_score", score)
    delta = score - prev_score

    if score >= config.ALERT_BUY_THRESHOLD and prev_score < config.ALERT_BUY_THRESHOLD:
        alerts.append(f"🔔 {stock_name}（{symbol}）评分升破 {config.ALERT_BUY_THRESHOLD}，当前 {score:.1f}，评级：买入区间")
    elif score <= config.ALERT_SELL_THRESHOLD and prev_score > config.ALERT_SELL_THRESHOLD:
        alerts.append(f"⚠️ {stock_name}（{symbol}）评分跌破 {config.ALERT_SELL_THRESHOLD}，当前 {score:.1f}，评级：卖出区间")

    if abs(delta) >= 10:
        direction = "急升" if delta > 0 else "急降"
        alerts.append(f"📊 {stock_name}（{symbol}）评分{direction} {delta:+.1f}（昨日 {prev_score:.1f} → 今日 {score:.1f}）")

    return alerts
