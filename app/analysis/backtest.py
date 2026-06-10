"""
滚动前向回测（walk-forward backtest）模块  v1.0
====================================================

目的：诚实评估预测模型对"未来"是否真的有效，输出方向准确率 / 误差，
并与朴素基准对照。绝不复用已训练模型（避免未来函数 lookahead）。

口径（默认值，与产品对齐）：
  - 预测周期 horizon: 10 日（主）/ 30 日（辅），= config.SHORT/LONG_TERM_DAYS
  - 训练窗口: 滚动 2 年（约 504 个交易日）
  - 回测覆盖期: 最近 ~6 个月（120 个交易日）
  - 步长 step: 5 个交易日重训一次（周频）→ 约 24 个评估点
  - 模式:
        fast     仅用 XGBoost 重训（~5s/次），先验证逻辑/批量扫
        accurate 全集成（LSTM+XGB+Prophet），夜里跑核心自选股

核心正确性保证：
  1. 全量历史先算好指标（指标均为后向滚动，不泄露未来），再按 ≤T 切片训练。
  2. 真实值取自全量序列 close[i+h]，预测只见到 ≤i 的数据。
  3. 输出朴素基准（trailing momentum 方向）+ 实际上涨率，供对照判断模型是否有效。
"""

import io
import os
import json
import argparse
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd

import config


# ─── 模块导入兜底（支持 `data.x` 与 `app.data.x` 两种运行方式）──────────
def _imp(module_paths, name):
    last_err = None
    for p in module_paths:
        try:
            return getattr(__import__(p, fromlist=[name]), name)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise ImportError(f"无法导入 {name}: {last_err}")


def _load_pipeline():
    fetch = _imp(["data.fetcher", "app.data.fetcher"], "fetch_stock_data")
    enrich = _imp(["data.fetcher", "app.data.fetcher"], "enrich_with_market_data")
    compute = _imp(["analysis.technical", "app.analysis.technical"], "compute_indicators")
    return fetch, enrich, compute


def _get_xgb():
    return _imp(["models.xgboost_model", "app.models.xgboost_model"], "XGBoostPredictor")


def _get_ensemble():
    return _imp(["models.ensemble", "app.models.ensemble"], "EnsemblePredictor")


# ─── 数据准备：全量历史 + 指标（一次性，后续切片复用）────────────────────
def _prepare_full_df(symbol: str, years: int) -> pd.DataFrame:
    fetch, enrich, compute = _load_pipeline()
    df = fetch(symbol, years=years)
    if df is None or len(df) == 0:
        raise ValueError(f"未取到 {symbol} 的历史数据")
    try:
        df = enrich(df, symbol)
    except Exception:
        pass  # 市场特征拿不到不致命，XGB 会用 0 兜底
    df = compute(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ─── 单点预测（返回 {horizon: (pct_change, pred_end_price)}）────────────
def _predict_fast(df_t: pd.DataFrame, horizons) -> dict:
    XGBoostPredictor = _get_xgb()
    out = {}
    last_price = float(df_t["close"].iloc[-1])
    for h in horizons:
        try:
            with redirect_stdout(io.StringIO()):
                m = XGBoostPredictor(prediction_days=h, use_bayes=False)
                m.train(df_t)
                res = m.predict(df_t)
            pct = float(res["pct_change"])
            out[h] = (pct, last_price * (1 + pct / 100.0))
        except Exception:
            continue
    return out


def _predict_accurate(df_t: pd.DataFrame, horizons) -> dict:
    EnsemblePredictor = _get_ensemble()
    out = {}
    with redirect_stdout(io.StringIO()):
        p = EnsemblePredictor()
        p.train(df_t)
        short = p.predict_short(df_t) if config.SHORT_TERM_DAYS in horizons else None
        long = p.predict_long(df_t) if config.LONG_TERM_DAYS in horizons else None
    if short:
        pct = float(short["pct_change"])
        out[config.SHORT_TERM_DAYS] = (pct, short["last_price"] * (1 + pct / 100.0))
    if long:
        pct = float(long["pct_change"])
        out[config.LONG_TERM_DAYS] = (pct, long["last_price"] * (1 + pct / 100.0))
    return out


# ─── 主回测循环 ──────────────────────────────────────────────────────────
def run_backtest(
    symbol: str,
    mode: str = "fast",
    horizons=None,
    test_days: int = 120,
    step: int = 5,
    train_window_years: float = 2.0,
    years_fetch: int = None,
    min_train: int = None,
    progress=None,
    save: bool = True,
) -> dict:
    """对单只股票做滚动前向回测。

    progress: 可选回调 progress(done:int, total:int, msg:str)，用于 UI / CLI 进度。
    返回 summary dict（含 by_horizon 汇总与逐点 records）。
    """
    horizons = horizons or [config.SHORT_TERM_DAYS, config.LONG_TERM_DAYS]
    horizons = sorted(set(int(h) for h in horizons))
    years_fetch = years_fetch or getattr(config, "DEFAULT_PERIOD_YEARS", 3)
    min_train = min_train or getattr(config, "MIN_TRADING_DAYS", 120)

    df = _prepare_full_df(symbol, years_fetch)
    closes = df["close"].to_numpy(dtype=float)
    dates = df.index
    n = len(df)
    max_h = max(horizons)
    train_bars = int(round(train_window_years * 252))

    # 评估点：需要左侧有训练历史、右侧有 max_h 个真实值
    last_eval = n - 1 - max_h
    first_eval = max(min_train, max_h)  # 至少有 min_train 历史、且能算 momentum 基准
    if last_eval < first_eval:
        raise ValueError(
            f"{symbol} 数据不足以回测：可用 {n} 条，"
            f"需要 ≥ {first_eval + max_h + 1} 条（增大 years_fetch 或减小 horizon）"
        )
    start = max(first_eval, last_eval - test_days + 1)
    eval_idx = list(range(start, last_eval + 1, step))
    total = len(eval_idx)

    records = []
    for cnt, i in enumerate(eval_idx, 1):
        train_start = max(0, i - train_bars + 1)
        df_t = df.iloc[train_start : i + 1]
        np.random.seed(42)  # 固定 XGB 路径噪声，保证可复现
        try:
            if mode == "accurate":
                preds = _predict_accurate(df_t, horizons)
            else:
                preds = _predict_fast(df_t, horizons)
        except Exception as exc:  # noqa: BLE001
            if progress:
                progress(cnt, total, f"跳过 {dates[i].date()}: {exc}")
            continue

        last_price = float(closes[i])
        for h in horizons:
            if h not in preds:
                continue
            pred_pct, pred_price = preds[h]
            actual_price = float(closes[i + h])
            actual_pct = (actual_price - last_price) / last_price * 100.0
            # 朴素基准：过去 h 日趋势延续（动量方向）
            if i - h >= 0:
                prev = float(closes[i - h])
                base_pct = (last_price - prev) / prev * 100.0 if prev else 0.0
            else:
                base_pct = 0.0
            records.append({
                "date": str(dates[i].date()),
                "horizon": h,
                "last_price": round(last_price, 3),
                "pred_pct": round(pred_pct, 3),
                "pred_price": round(pred_price, 3),
                "actual_pct": round(actual_pct, 3),
                "actual_price": round(actual_price, 3),
                "dir_correct": int(np.sign(pred_pct) == np.sign(actual_pct)),
                "abs_err_pct": round(abs(pred_price - actual_price) / actual_price * 100.0, 3),
                "base_pct": round(base_pct, 3),
                "base_correct": int(np.sign(base_pct) == np.sign(actual_pct)),
                "actual_up": int(actual_pct > 0),
            })
        if progress:
            progress(cnt, total, str(dates[i].date()))

    summary = _summarize(symbol, mode, records, horizons, {
        "test_days": test_days,
        "step": step,
        "train_window_years": train_window_years,
        "years_fetch": years_fetch,
        "eval_points": total,
        "data_bars": n,
        "data_start": str(dates[0].date()),
        "data_end": str(dates[-1].date()),
    })
    if save:
        summary["saved_path"] = _save(symbol, summary)
    return summary


def _summarize(symbol, mode, records, horizons, params) -> dict:
    by_h = {}
    for h in horizons:
        rs = [r for r in records if r["horizon"] == h]
        if not rs:
            continue
        n = len(rs)
        by_h[str(h)] = {
            "samples": n,
            "direction_accuracy": round(sum(r["dir_correct"] for r in rs) / n, 4),
            "baseline_accuracy": round(sum(r["base_correct"] for r in rs) / n, 4),
            "actual_up_rate": round(sum(r["actual_up"] for r in rs) / n, 4),
            "mae_pct": round(sum(r["abs_err_pct"] for r in rs) / n, 3),
            "edge_vs_baseline": round(
                (sum(r["dir_correct"] for r in rs) - sum(r["base_correct"] for r in rs)) / n, 4
            ),
        }
    return {
        "symbol": symbol,
        "mode": mode,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "by_horizon": by_h,
        "records": records,
    }


# ─── 落盘 / 读取 ─────────────────────────────────────────────────────────
def _save(symbol: str, summary: dict) -> str:
    os.makedirs(config.BACKTEST_LOG_DIR, exist_ok=True)
    fname = f"{symbol}_{datetime.now().strftime('%Y%m%d')}.json"
    path = os.path.join(config.BACKTEST_LOG_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def get_latest_backtest(symbol: str) -> dict:
    """读取某股票最近一次回测结果（供 UI 展示）。无则返回 {}。"""
    d = config.BACKTEST_LOG_DIR
    if not os.path.isdir(d):
        return {}
    files = [f for f in os.listdir(d) if f.startswith(f"{symbol}_") and f.endswith(".json")]
    if not files:
        return {}
    files.sort(reverse=True)  # 文件名带日期，倒序即最新
    try:
        with open(os.path.join(d, files[0]), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def format_summary_line(summary: dict, horizon: int = None) -> str:
    """把回测结果压成一行中文摘要，供报告页顶部展示。"""
    bh = summary.get("by_horizon", {})
    if not bh:
        return "暂无回测数据"
    h = str(horizon or min(int(k) for k in bh))
    s = bh.get(h)
    if not s:
        return "暂无回测数据"
    return (
        f"近 {s['samples']} 次 {h} 日预测：方向准确率 "
        f"{s['direction_accuracy']:.0%}（基准 {s['baseline_accuracy']:.0%}，"
        f"平均误差 {s['mae_pct']:.1f}%）"
    )


# ─── CLI 入口 ────────────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(description="滚动前向回测")
    parser.add_argument("symbol", help="股票代码，如 600519")
    parser.add_argument("--mode", choices=["fast", "accurate"], default="fast")
    parser.add_argument("--test-days", type=int, default=120)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--train-years", type=float, default=2.0)
    parser.add_argument("--horizons", default=None, help="逗号分隔，如 10,30")
    args = parser.parse_args()

    horizons = None
    if args.horizons:
        horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    def _prog(done, total, msg):
        print(f"  [{done}/{total}] {msg}", flush=True)

    print(f"开始回测 {args.symbol}  mode={args.mode} …")
    summary = run_backtest(
        args.symbol, mode=args.mode, horizons=horizons,
        test_days=args.test_days, step=args.step,
        train_window_years=args.train_years, progress=_prog,
    )
    print("\n==== 回测结果 ====")
    p = summary["params"]
    print(f"数据区间 {p['data_start']} ~ {p['data_end']}（{p['data_bars']} 条），评估点 {p['eval_points']}")
    for h, s in summary["by_horizon"].items():
        print(
            f"  [{h}日] 方向准确率 {s['direction_accuracy']:.1%} | "
            f"基准 {s['baseline_accuracy']:.1%} | "
            f"领先基准 {s['edge_vs_baseline']:+.1%} | "
            f"实际上涨率 {s['actual_up_rate']:.1%} | "
            f"平均误差 {s['mae_pct']:.2f}% | n={s['samples']}"
        )
    print(f"\n已保存：{summary.get('saved_path')}")


if __name__ == "__main__":
    _cli()
