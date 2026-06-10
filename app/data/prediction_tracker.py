import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

import config


def _prediction_path(symbol: str) -> str:
    return os.path.join(config.PREDICTION_LOG_DIR, f"{symbol}.json")


def load_prediction_records(symbol: str) -> List[Dict]:
    path = _prediction_path(symbol)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_prediction_records(symbol: str, records: List[Dict]) -> None:
    with open(_prediction_path(symbol), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _normalize_prediction_block(pred: Optional[dict], horizon: str) -> Optional[Dict]:
    if not pred:
        return None

    dates = list(pred.get("dates", []) or [])
    predictions = [float(x) for x in (pred.get("predictions") or [])]
    lower_bound = [float(x) for x in (pred.get("lower_bound") or [])]
    upper_bound = [float(x) for x in (pred.get("upper_bound") or [])]
    if not predictions or not dates:
        return None

    horizon_days = len(dates)
    return {
        "horizon": horizon,
        "horizon_days": horizon_days,
        "generated_last_price": float(pred.get("last_price", 0) or 0),
        "target_price": float(predictions[-1]),
        "pct_change": float(pred.get("pct_change", 0) or 0),
        "target_date": dates[-1],
        "dates": dates,
        "predictions": predictions,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "val_rmse": pred.get("val_rmse"),
        "val_rmse_price": pred.get("val_rmse_price"),
        "weights": pred.get("weights", {}) or {},
        "model_rmse": pred.get("model_rmse", {}) or {},
        "component_predictions": pred.get("component_predictions", {}) or {},
    }


def save_prediction_snapshot(
    symbol: str,
    stock_name: str,
    last_price: float,
    short_pred: Optional[dict],
    long_pred: Optional[dict],
    report_confidence: Optional[Dict] = None,
    score_result: Optional[Dict] = None,
) -> Optional[Dict]:
    short_block = _normalize_prediction_block(short_pred, "short")
    long_block = _normalize_prediction_block(long_pred, "long")
    if not short_block and not long_block:
        return None

    records = load_prediction_records(symbol)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    generated_date = generated_at.split(" ")[0]

    records = [r for r in records if r.get("generated_date") != generated_date]
    snapshot = {
        "snapshot_id": f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "symbol": symbol,
        "stock_name": stock_name,
        "generated_at": generated_at,
        "generated_date": generated_date,
        "last_price": float(last_price),
        "report_confidence": report_confidence or {},
        "score_result": score_result or {},
        "predictions": {
            "short": short_block,
            "long": long_block,
        },
        "evaluations": {},
    }
    records.append(snapshot)
    records = records[-365:]
    _save_prediction_records(symbol, records)
    return snapshot


def _direction_label(value: int) -> str:
    if value > 0:
        return "上涨"
    if value < 0:
        return "下跌"
    return "持平"


def build_prediction_review_rows(
    symbol: str,
    latest_price: Optional[float] = None,
    limit: int = 20,
) -> List[Dict]:
    """Build rows that compare saved predictions with the latest available price.

    Matured predictions use their locked evaluation result. Pending predictions
    use latest_price for a live, interim comparison.
    """
    records = load_prediction_records(symbol)
    rows: List[Dict] = []

    for record in reversed(records):
        generated_at = record.get("generated_at") or record.get("generated_date") or ""
        base_price = float(record.get("last_price", 0) or 0)
        score_result = record.get("score_result") or {}
        for horizon in ("short", "long"):
            pred_block = (record.get("predictions") or {}).get(horizon)
            if not pred_block:
                continue

            evaluation = (record.get("evaluations") or {}).get(horizon) or {}
            predicted_price = float(pred_block.get("target_price", 0) or 0)
            predicted_direction = 0
            if predicted_price > base_price:
                predicted_direction = 1
            elif predicted_price < base_price:
                predicted_direction = -1

            actual_price = None
            actual_date = ""
            status = "进行中"
            error_pct = None
            direction_correct = None

            if evaluation.get("status") == "evaluated":
                status = "已验证"
                actual_price = evaluation.get("actual_price")
                actual_date = evaluation.get("actual_date") or ""
                error_pct = evaluation.get("relative_error_pct")
                direction_correct = evaluation.get("direction_correct")
            elif latest_price is not None:
                actual_price = float(latest_price)
                actual_date = "最新"
                if actual_price:
                    error_pct = abs(actual_price - predicted_price) / actual_price * 100
                actual_direction = 0
                if actual_price > base_price:
                    actual_direction = 1
                elif actual_price < base_price:
                    actual_direction = -1
                direction_correct = predicted_direction == actual_direction

            rows.append({
                "分析时间": generated_at,
                "周期": "10天" if horizon == "short" else "30天",
                "状态": status,
                "当时价格": round(base_price, 2) if base_price else None,
                "预测价": round(predicted_price, 2) if predicted_price else None,
                "目标日期": pred_block.get("target_date") or "",
                "对比日期": actual_date,
                "实际/最新价": round(float(actual_price), 2) if actual_price is not None else None,
                "误差%": round(float(error_pct), 2) if error_pct is not None else None,
                "预测方向": _direction_label(predicted_direction),
                "方向判断": "正确" if direction_correct is True else "错误" if direction_correct is False else "待验证",
                "综合评分": round(float(score_result.get("total_score")), 1) if score_result.get("total_score") is not None else None,
                "评级": score_result.get("rating") or "",
                "_snapshot_id": record.get("snapshot_id") or "",
                "_horizon": horizon,
            })
            if len(rows) >= limit:
                return rows

    return rows


def _find_realized_price(df: pd.DataFrame, target_date: Optional[str]) -> Optional[Dict]:
    if not target_date or df is None or df.empty:
        return None

    target_ts = pd.Timestamp(target_date)
    realized = df[df.index >= target_ts]
    if realized.empty:
        return None

    row = realized.iloc[0]
    return {
        "actual_date": realized.index[0].strftime("%Y-%m-%d"),
        "actual_price": float(row["close"]),
    }


def _evaluate_prediction_block(pred_block: Dict, df: pd.DataFrame) -> Optional[Dict]:
    realized = _find_realized_price(df, pred_block.get("target_date"))
    if not realized:
        return None

    predicted_price = float(pred_block.get("target_price", 0) or 0)
    actual_price = float(realized["actual_price"])
    base_price = float(pred_block.get("generated_last_price", 0) or 0)
    lower_bound = pred_block.get("lower_bound") or []
    upper_bound = pred_block.get("upper_bound") or []

    predicted_direction = 0
    actual_direction = 0
    if predicted_price > base_price:
        predicted_direction = 1
    elif predicted_price < base_price:
        predicted_direction = -1
    if actual_price > base_price:
        actual_direction = 1
    elif actual_price < base_price:
        actual_direction = -1

    band_low = float(lower_bound[-1]) if lower_bound else None
    band_high = float(upper_bound[-1]) if upper_bound else None

    result = {
        "status": "evaluated",
        "actual_date": realized["actual_date"],
        "actual_price": actual_price,
        "predicted_price": predicted_price,
        "absolute_error": abs(actual_price - predicted_price),
        "relative_error_pct": abs(actual_price - predicted_price) / actual_price * 100 if actual_price else None,
        "predicted_direction": predicted_direction,
        "actual_direction": actual_direction,
        "direction_correct": predicted_direction == actual_direction,
        "within_interval": (
            band_low is not None and band_high is not None and band_low <= actual_price <= band_high
        ),
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    components = pred_block.get("component_predictions") or {}
    if components:
        result["components"] = {}
        for model_key, comp in components.items():
            comp_pred = float(comp.get("target_price", 0) or 0)
            comp_low = comp.get("lower_bound")
            comp_high = comp.get("upper_bound")
            comp_dir = 0
            if comp_pred > base_price:
                comp_dir = 1
            elif comp_pred < base_price:
                comp_dir = -1
            result["components"][model_key] = {
                "predicted_price": comp_pred,
                "absolute_error": abs(actual_price - comp_pred),
                "relative_error_pct": abs(actual_price - comp_pred) / actual_price * 100 if actual_price else None,
                "predicted_direction": comp_dir,
                "direction_correct": comp_dir == actual_direction,
                "within_interval": (
                    comp_low is not None and comp_high is not None and float(comp_low) <= actual_price <= float(comp_high)
                ),
            }
    return result


def evaluate_matured_predictions(symbol: str, df: pd.DataFrame) -> Dict:
    records = load_prediction_records(symbol)
    updated = 0
    matured = 0

    for record in records:
        evaluations = record.setdefault("evaluations", {})
        for horizon in ("short", "long"):
            pred_block = (record.get("predictions") or {}).get(horizon)
            if not pred_block:
                continue
            if evaluations.get(horizon, {}).get("status") == "evaluated":
                continue

            result = _evaluate_prediction_block(pred_block, df)
            if result:
                evaluations[horizon] = result
                updated += 1
            else:
                matured += 1

    if updated:
        _save_prediction_records(symbol, records)

    return {
        "symbol": symbol,
        "records": len(records),
        "evaluated_predictions": updated,
        "pending_predictions": matured,
    }


def summarize_posterior_metrics(symbol: str, limit: int = 20) -> Dict:
    records = load_prediction_records(symbol)
    items: List[Dict] = []
    for record in reversed(records):
        for horizon in ("short", "long"):
            result = (record.get("evaluations") or {}).get(horizon)
            if result and result.get("status") == "evaluated":
                items.append(result)
        if len(items) >= limit:
            break

    if not items:
        return {
            "count": 0,
            "avg_relative_error_pct": None,
            "direction_accuracy_pct": None,
            "interval_hit_rate_pct": None,
        }

    rel_errors = [x["relative_error_pct"] for x in items if x.get("relative_error_pct") is not None]
    direction_hits = [1 if x.get("direction_correct") else 0 for x in items]
    interval_hits = [1 if x.get("within_interval") else 0 for x in items if x.get("within_interval") is not None]

    return {
        "count": len(items),
        "avg_relative_error_pct": round(sum(rel_errors) / len(rel_errors), 4) if rel_errors else None,
        "direction_accuracy_pct": round(sum(direction_hits) / len(direction_hits) * 100, 2) if direction_hits else None,
        "interval_hit_rate_pct": round(sum(interval_hits) / len(interval_hits) * 100, 2) if interval_hits else None,
    }


def get_posterior_model_weights(symbol: str, horizon: str, limit: int = 30, min_samples: int = 3) -> Dict[str, float]:
    records = load_prediction_records(symbol)
    model_rows: Dict[str, List[Dict]] = {}

    for record in reversed(records):
        result = (record.get("evaluations") or {}).get(horizon)
        if not result or result.get("status") != "evaluated":
            continue
        for model_key, comp in (result.get("components") or {}).items():
            model_rows.setdefault(model_key, []).append(comp)
        if sum(len(v) for v in model_rows.values()) >= limit * 3:
            break

    scored = {}
    for model_key, rows in model_rows.items():
        if len(rows) < min_samples:
            continue
        rel_errors = [r["relative_error_pct"] for r in rows if r.get("relative_error_pct") is not None]
        dir_hits = [1 if r.get("direction_correct") else 0 for r in rows]
        if not rel_errors or not dir_hits:
            continue
        avg_error = sum(rel_errors) / len(rel_errors)
        dir_acc = sum(dir_hits) / len(dir_hits)
        scored[model_key] = (1.0 / max(avg_error, 0.25)) * max(dir_acc, 0.25)

    if not scored:
        return {}

    total = sum(scored.values())
    return {k: v / total for k, v in scored.items()} if total > 0 else {}


def get_posterior_trend(symbol: str, selected_model: Optional[str] = None, limit: int = 40) -> pd.DataFrame:
    records = load_prediction_records(symbol)
    rows: List[Dict] = []

    for record in reversed(records):
        evaluations = record.get("evaluations") or {}
        for horizon in ("short", "long"):
            result = evaluations.get(horizon)
            if not result or result.get("status") != "evaluated":
                continue

            if selected_model:
                component = (result.get("components") or {}).get(selected_model)
                if component:
                    rows.append({
                        "actual_date": result.get("actual_date"),
                        "evaluated_at": result.get("evaluated_at"),
                        "horizon": horizon,
                        "model": selected_model,
                        "relative_error_pct": component.get("relative_error_pct"),
                        "absolute_error": component.get("absolute_error"),
                        "direction_correct": component.get("direction_correct"),
                        "within_interval": component.get("within_interval"),
                    })
            else:
                rows.append({
                    "actual_date": result.get("actual_date"),
                    "evaluated_at": result.get("evaluated_at"),
                    "horizon": horizon,
                    "model": f"fused_{horizon}",
                    "relative_error_pct": result.get("relative_error_pct"),
                    "absolute_error": result.get("absolute_error"),
                    "direction_correct": result.get("direction_correct"),
                    "within_interval": result.get("within_interval"),
                })

        if len(rows) >= limit:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["actual_date"] = pd.to_datetime(df["actual_date"])
    if "evaluated_at" in df.columns:
        df["evaluated_at"] = pd.to_datetime(df["evaluated_at"])
    return df.sort_values("actual_date")


def summarize_posterior_health(symbol: str, selected_model: Optional[str] = None, limit: int = 20) -> Dict:
    df = get_posterior_trend(symbol, selected_model=selected_model, limit=limit)
    if df.empty or len(df) < 4:
        return {
            "status": "样本不足",
            "color": "#ADBAC7",
            "summary": "后验样本不足，暂无法判断趋势",
            "recent_error_pct": None,
            "previous_error_pct": None,
            "direction_accuracy_pct": None,
            "count": len(df),
        }

    df = df.dropna(subset=["relative_error_pct"]).copy()
    if df.empty or len(df) < 4:
        return {
            "status": "样本不足",
            "color": "#ADBAC7",
            "summary": "后验误差样本不足，暂无法判断趋势",
            "recent_error_pct": None,
            "previous_error_pct": None,
            "direction_accuracy_pct": None,
            "count": len(df),
        }

    half = max(2, len(df) // 2)
    previous = df.iloc[:half]
    recent = df.iloc[-half:]
    previous_error = float(previous["relative_error_pct"].mean())
    recent_error = float(recent["relative_error_pct"].mean())
    direction_accuracy = float(df["direction_correct"].fillna(False).astype(int).mean() * 100)
    delta = recent_error - previous_error

    if delta <= -1.0 and direction_accuracy >= 60:
        status, color = "稳定提升", "#22c55e"
        summary = f"最近误差下降 {abs(delta):.2f}% ，方向判断较稳"
    elif delta >= 1.0 and direction_accuracy < 55:
        status, color = "明显退化", "#f85149"
        summary = f"最近误差上升 {delta:.2f}% ，方向判断走弱"
    elif abs(delta) < 1.0 and direction_accuracy >= 60:
        status, color = "基本稳定", "#4CAF50"
        summary = "最近误差波动不大，方向准确率尚可"
    else:
        status, color = "波动加大", "#d29922"
        summary = "最近误差和方向表现出现摇摆"

    return {
        "status": status,
        "color": color,
        "summary": summary,
        "recent_error_pct": round(recent_error, 4),
        "previous_error_pct": round(previous_error, 4),
        "direction_accuracy_pct": round(direction_accuracy, 2),
        "count": len(df),
    }
