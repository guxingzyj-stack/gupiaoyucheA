"""
回测启动器（命令行 / 计划任务入口）
======================================
用法示例（用项目自带 python）：
    python\\python.exe scripts\\run_backtest.py 600519
    python\\python.exe scripts\\run_backtest.py 600519 --mode accurate
    python\\python.exe scripts\\run_backtest.py --watchlist            # 回测全部自选股
    python\\python.exe scripts\\run_backtest.py 600519 000001 --step 5 --test-days 120

把 app 目录加入 sys.path，解决嵌入式 Python (._pth) 不含 cwd 的问题。
"""

import os
import sys
import argparse
from pathlib import Path


def _bootstrap():
    root = Path(__file__).resolve().parents[1]
    app_dir = root / "app"
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    return app_dir


def _load_watchlist():
    try:
        from data.watchlist import load_watchlist
        items = load_watchlist()
    except Exception as exc:  # noqa: BLE001
        print(f"读取自选股失败：{exc}")
        return []
    out = []
    for it in items:
        if isinstance(it, dict):
            out.append(it.get("symbol") or it.get("code") or "")
        else:
            out.append(str(it))
    return [s for s in out if s]


def main():
    _bootstrap()
    from analysis import backtest

    parser = argparse.ArgumentParser(description="滚动前向回测启动器")
    parser.add_argument("symbols", nargs="*", help="股票代码，可多个")
    parser.add_argument("--watchlist", action="store_true", help="回测全部自选股")
    parser.add_argument("--mode", choices=["fast", "accurate"], default="fast")
    parser.add_argument("--test-days", type=int, default=120)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--train-years", type=float, default=2.0)
    parser.add_argument("--horizons", default=None, help="逗号分隔，如 10,30")
    args = parser.parse_args()

    symbols = list(args.symbols)
    if args.watchlist:
        symbols = _load_watchlist() or symbols
    if not symbols:
        parser.error("请提供股票代码，或用 --watchlist 回测自选股")

    horizons = None
    if args.horizons:
        horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    for sym in symbols:
        print(f"\n===== 回测 {sym}  mode={args.mode} =====")

        def _prog(done, total, msg, _sym=sym):
            print(f"  [{done}/{total}] {msg}", flush=True)

        try:
            summary = backtest.run_backtest(
                sym, mode=args.mode, horizons=horizons,
                test_days=args.test_days, step=args.step,
                train_window_years=args.train_years, progress=_prog,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {sym} 回测失败：{exc}")
            continue

        p = summary["params"]
        print(f"  数据 {p['data_start']}~{p['data_end']}（{p['data_bars']}条），评估点 {p['eval_points']}")
        for h, s in summary["by_horizon"].items():
            print(
                f"    [{h}日] 方向准确率 {s['direction_accuracy']:.1%} | "
                f"基准 {s['baseline_accuracy']:.1%} | "
                f"领先 {s['edge_vs_baseline']:+.1%} | "
                f"实际涨率 {s['actual_up_rate']:.1%} | "
                f"误差 {s['mae_pct']:.2f}% | n={s['samples']}"
            )
        print(f"  已保存：{summary.get('saved_path')}")


if __name__ == "__main__":
    main()
