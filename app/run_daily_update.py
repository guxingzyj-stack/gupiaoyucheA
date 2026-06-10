import argparse
import json
import os
import sys
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


def _load_watchlist_symbols():
    try:
        from data.watchlist import load_watchlist
    except Exception:
        return []
    return [item.get("symbol") for item in load_watchlist() if item.get("symbol")]


def main():
    parser = argparse.ArgumentParser(description="Run self-learning daily update outside Streamlit.")
    parser.add_argument("--symbols", default="", help="Comma-separated stock symbols. Empty means current watchlist.")
    parser.add_argument("--output", default="", help="Optional path to save the summary JSON.")
    parser.add_argument("--workers", type=int, default=1, help="Worker count for stock-level updates. Default 1 for quiet background runs.")
    args = parser.parse_args()

    from models.online_learner import run_daily_update_all

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = _load_watchlist_symbols()
    if not symbols:
        print("No symbols found for daily update.", file=sys.stderr)
        return 1

    reports = run_daily_update_all(symbols, workers=args.workers)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": symbols,
        "reports": reports,
    }
    output_path = args.output
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
