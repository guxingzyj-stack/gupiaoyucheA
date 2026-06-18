"""
A股智能分析预测系统 — 主入口
用法:
  python main.py analyze --ticker 600519
  python main.py analyze --ticker 000858 --no-backtest
  python main.py board 600519 000858 300750 002594
  python main.py history 600519
  python main.py schedule 600519 000858   # 每日定时分析
"""
import os
import sys
import webbrowser
import warnings
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

console = Console()

BANNER = r"""
 ____  _             _      _                _           _
/ ___|| |_ ___   ___| | __ / \   _ __   __ _| |_   _ ___| |_
\___ \| __/ _ \ / __| |/ // _ \ | '_ \ / _` | | | | / __| __|
 ___) | || (_) | (__|   </ ___ \| | | | (_| | | |_| \__ \ |_
|____/ \__\___/ \___|_|\_/_/   \_\_| |_|\__,_|_|\__, |___/\__|
                                                 |___/
"""


def _print_banner():
    console.print(Text(BANNER, style="bold cyan"), justify="center")
    console.print(
        Panel("A股智能分析 · 三模型融合预测 · 情感分析 · 回测验证",
              style="bold blue", border_style="cyan"),
        justify="center",
    )


def _step(c, msg):  c.print(f"  [bold cyan]▶[/bold cyan] {msg}")
def _ok(c, msg):    c.print(f"  [bold green]✓[/bold green] {msg}")
def _warn(c, msg):  c.print(f"  [bold yellow]⚠[/bold yellow] {msg}")
def _section(c, t): c.rule(f"[bold blue]{t}[/bold blue]")


def _safe(d, key, fmt=".2f", default="—"):
    v = d.get(key)
    if v is None:
        return default
    try:
        return f"{float(v):{fmt}}"
    except Exception:
        return str(v)


# ─────────────────────────────────────────────────────────────
#  核心分析流程（单股）
# ─────────────────────────────────────────────────────────────

def run_analysis(symbol: str, lang: str, do_backtest: bool, open_browser: bool,
                 save_history: bool = True) -> dict:
    """完整分析单只股票，返回结果字典（供 board 调用）"""
    import config as cfg
    cfg.LANGUAGE = lang

    from i18n import set_lang
    set_lang(lang)

    _print_banner()
    console.print(f"\n[bold]正在分析：[/bold] [yellow]{symbol}[/yellow]\n")

    result = {}

    # ── 1. 数据获取 ──────────────────────────────────────
    _section(console, "数据获取")
    _step(console, "获取A股历史行情（AKShare）...")
    try:
        from data.fetcher import (fetch_stock_data, fetch_stock_info,
                                  get_stock_name, enrich_with_market_data)
        df         = fetch_stock_data(symbol, years=cfg.DEFAULT_PERIOD_YEARS)
        stock_info = fetch_stock_info(symbol)
        stock_name = get_stock_name(symbol)
        _ok(console, f"{stock_name}（{symbol}）历史数据：{len(df)}个交易日")
        result["stock_name"] = stock_name
    except Exception as e:
        console.print(f"\n[bold red]✗ 数据获取失败：{e}[/bold red]")
        return {}

    _step(console, "获取新闻数据...")
    try:
        from data.news import fetch_news, get_news_texts
        news_list  = fetch_news(symbol)
        news_texts = get_news_texts(news_list)
        _ok(console, f"获取新闻 {len(news_list)} 条")
    except Exception as e:
        _warn(console, f"新闻获取失败: {e}")
        news_list, news_texts = [], []

    _step(console, "加载市场基准数据（沪深300 + 资金流向）...")
    try:
        df = enrich_with_market_data(df, symbol)
        added = [c for c in ["hs300_ret1", "main_flow_z"] if c in df.columns]
        _ok(console, f"市场特征已加入：{', '.join(added) if added else '(网络限制，跳过)'}")
    except Exception as e:
        _warn(console, f"市场数据加载失败（继续）: {e}")

    # ── 2. 技术分析 ──────────────────────────────────────
    _section(console, "技术分析")
    _step(console, "计算技术指标（MA/EMA/MACD/RSI/布林带/KDJ/ATR...）")
    try:
        from analysis.technical import compute_indicators, generate_signals, compute_support_resistance
        df        = compute_indicators(df)
        signals   = generate_signals(df)
        sr_levels = compute_support_resistance(df)
        _ok(console, f"技术信号：{signals['buy_count']}买入 / {signals['sell_count']}卖出 / {signals['neutral_count']}中性")
    except Exception as e:
        console.print(f"[red]✗ 技术分析失败：{e}[/red]")
        signals   = {"signals": [], "buy_count": 0, "sell_count": 0, "neutral_count": 0}
        sr_levels = {}

    # ── 3. 基本面分析 ────────────────────────────────────
    _section(console, "基本面分析")
    _step(console, "获取财务指标...")
    try:
        from analysis.fundamental import fetch_and_analyze
        fundamental = fetch_and_analyze(symbol, stock_info)
        _ok(console, f"PE={_safe(fundamental,'pe','.1f')}  PB={_safe(fundamental,'pb','.2f')}  ROE={_safe(fundamental,'roe','.1f')}%")
    except Exception as e:
        _warn(console, f"基本面分析失败: {e}")
        fundamental = {"score": 50, "details": []}

    # ── 4. 情感分析 ──────────────────────────────────────
    _section(console, "新闻情感分析")
    _step(console, "分析新闻情感（SnowNLP中文NLP）...")
    try:
        from analysis.sentiment import analyze_sentiment
        sentiment = analyze_sentiment(news_texts)
        lc = {"正面": "green", "负面": "red", "中性": "yellow"}.get(sentiment["label"], "yellow")
        _ok(console,
            f"情感：[{lc}]{sentiment['label']}[/{lc}]  "
            f"得分：{sentiment['score']:.2%}  "
            f"(正面{sentiment['positive_count']} / 负面{sentiment['negative_count']} / 中性{sentiment['neutral_count']})")
    except Exception as e:
        _warn(console, f"情感分析失败: {e}")
        sentiment = {"score": 0.5, "label": "中性",
                     "positive_count": 0, "negative_count": 0, "neutral_count": 0, "news_count": 0}

    # ── 5. 模型训练 & 预测 ───────────────────────────────
    _section(console, "AI模型训练与预测（LSTM + XGBoost + Prophet）")
    from models.ensemble import EnsemblePredictor
    predictor = EnsemblePredictor()
    predictor.symbol_context = symbol
    predictor.posterior_weights = predictor._load_posterior_weights(symbol)
    predictor.model_control = predictor._load_model_control(symbol)
    short_pred = long_pred = None

    try:
        rmse_info = {}
        if getattr(cfg, "REPORT_USE_PRETRAINED_MODELS", True):
            rmse_info = predictor.load_pretrained(symbol, console=console)
        if not rmse_info and getattr(cfg, "REPORT_RETRAIN_IF_MISSING", True):
            rmse_info = predictor.train(df, console=console)
        _ok(console, f"模型训练完成：{', '.join(f'{k}={v:.4f}' for k,v in rmse_info.items() if v < 1e9)}")
    except Exception as e:
        _warn(console, f"模型训练失败：{e}")

    _step(console, "生成短期（10天）预测...")
    try:
        short_pred = predictor.predict_short(df)
        if short_pred:
            pct = short_pred["pct_change"]
            c   = "green" if pct >= 0 else "red"
            _ok(console, f"短期预测：[{c}]{'▲' if pct>=0 else '▼'}{abs(pct):.2f}%[/{c}]  "
                         f"目标价：{short_pred['predictions'][-1]:.2f}")
    except Exception as e:
        _warn(console, f"短期预测失败: {e}")

    _step(console, "生成中期（30天）预测...")
    try:
        long_pred = predictor.predict_long(df)
        if long_pred:
            pct = long_pred["pct_change"]
            c   = "green" if pct >= 0 else "red"
            _ok(console, f"中期预测：[{c}]{'▲' if pct>=0 else '▼'}{abs(pct):.2f}%[/{c}]  "
                         f"目标价：{long_pred['predictions'][-1]:.2f}")
    except Exception as e:
        _warn(console, f"中期预测失败: {e}")

    # ── 6. 综合评分 ──────────────────────────────────────
    _section(console, "综合评分")
    from models.scorer import compute_score
    score_result = compute_score(signals, fundamental, sentiment, short_pred, long_pred)

    try:
        from analysis.value_pipeline import build_value_assessment, annotate_signal_conflict
        value_assessment = build_value_assessment(symbol, stock_info, fundamental)
        annotate_signal_conflict(value_assessment, score_result)
        if value_assessment:
            result["value_assessment"] = value_assessment
            _ok(console, f"价值体检：{value_assessment.get('combo', '已生成')}")
    except Exception as e:
        _warn(console, f"价值体检失败: {e}")

    score  = score_result["total_score"]
    rating = score_result["rating"]
    color  = score_result["color"]
    _print_score_table(score_result, score, rating, color)

    result.update({"score_result": score_result, "short_pred": short_pred, "long_pred": long_pred})

    # ── 7. 历史评分保存 & 预警 ───────────────────────────
    history_records = []
    if save_history:
        try:
            from data.score_history import save_score, check_alerts, load_history
            from data.prediction_tracker import save_prediction_snapshot
            save_score(symbol, stock_name, score_result, short_pred, long_pred)
            save_prediction_snapshot(
                symbol=symbol,
                stock_name=stock_name,
                last_price=float(df["close"].iloc[-1]),
                short_pred=short_pred,
                long_pred=long_pred,
                score_result=score_result,
            )
            history_records = load_history(symbol)
            alerts = check_alerts(symbol, stock_name, score)
            for alert in alerts:
                console.print(f"\n[bold yellow]{alert}[/bold yellow]")
            _ok(console, f"评分已记录（共 {len(history_records)} 条历史）")
        except Exception as e:
            _warn(console, f"历史记录失败: {e}")

    # ── 8. 回测 ──────────────────────────────────────────
    backtest_result = None
    if do_backtest:
        _section(console, "策略回测")
        _step(console, "执行历史回测（MA金叉/死叉 + RSI策略）...")
        try:
            from backtest.engine import run_backtest
            backtest_result = run_backtest(df)
            if backtest_result:
                _print_backtest(backtest_result)
        except Exception as e:
            _warn(console, f"回测失败: {e}")

    # ── 9. 支撑/阻力 ─────────────────────────────────────
    if sr_levels:
        _section(console, "支撑位 & 阻力位")
        _print_sr(sr_levels)

    # ── 10. 图表 & 报告 ──────────────────────────────────
    _section(console, "生成报告")
    _step(console, "生成交互式图表...")

    figures = {}
    try:
        from visualization.charts import (
            create_candlestick_chart, create_prediction_chart,
            create_score_gauge, create_fundamental_radar, create_signal_bar,
            create_score_history_chart,
        )
        figures["candlestick"] = create_candlestick_chart(df, stock_name, symbol)
        figures["prediction"]  = create_prediction_chart(df, short_pred, long_pred, stock_name)
        figures["gauge"]       = create_score_gauge(score, rating, color)
        figures["radar"]       = create_fundamental_radar(fundamental, stock_name)
        if signals.get("signals"):
            figures["signals"] = create_signal_bar(signals["signals"])
        if len(history_records) > 1:
            figures["history"] = create_score_history_chart(history_records, stock_name)
        _ok(console, f"已生成 {len(figures)} 个图表")
    except Exception as e:
        _warn(console, f"图表生成失败: {e}")

    _step(console, "生成HTML报告...")
    try:
        from visualization.report import generate_html_report
        report_path = generate_html_report(
            symbol          = symbol,
            stock_name      = stock_name,
            df              = df,
            tech_signals    = signals,
            fundamental     = fundamental,
            sentiment       = sentiment,
            short_pred      = short_pred,
            long_pred       = long_pred,
            score_result    = score_result,
            backtest        = backtest_result,
            figures         = figures,
            news_list       = news_list,
            history_records = history_records,
        )
        result["report_path"] = report_path
        _ok(console, f"报告已保存：[link={report_path}]{report_path}[/link]")

        if open_browser:
            webbrowser.open(f"file:///{report_path.replace(os.sep, '/')}")
            console.print("  [dim]已在浏览器中打开报告[/dim]")
    except Exception as e:
        _warn(console, f"报告生成失败: {e}")

    # ── 11. 终端摘要 ─────────────────────────────────────
    _section(console, "分析完成")
    _print_final_summary(stock_name, symbol, score, rating, color, short_pred, long_pred, score_result)
    console.print()
    console.print("[bold yellow]⚠  风险提示：本报告仅供参考，不构成投资建议。股市有风险，投资须谨慎。[/bold yellow]")

    return result


# ─────────────────────────────────────────────────────────────
#  快速分析（用于 board / compare，不生成报告）
# ─────────────────────────────────────────────────────────────

def _compare_one(tk: str) -> dict:
    """compare 单股 worker（模块顶层，ProcessPool 可 pickle）"""
    try:
        from concurrent.futures import ThreadPoolExecutor
        from data.fetcher import fetch_stock_data, fetch_stock_info
        from analysis.technical import compute_indicators, generate_signals
        from analysis.fundamental import fetch_and_analyze
        from analysis.sentiment import analyze_sentiment
        from data.news import fetch_news, get_news_texts
        from models.scorer import compute_score

        # 网络 IO 并行
        with ThreadPoolExecutor(max_workers=3) as io_pool:
            f_df   = io_pool.submit(fetch_stock_data, tk, 1)
            f_info = io_pool.submit(fetch_stock_info, tk)
            f_news = io_pool.submit(fetch_news, tk, 15)
            df   = f_df.result()
            info = f_info.result()
            news = f_news.result()
        name = info.get("股票简称") or info.get("名称") or tk
        df   = compute_indicators(df)
        sig  = generate_signals(df)
        fund = fetch_and_analyze(tk, info)
        sent = analyze_sentiment(get_news_texts(news))
        res  = compute_score(sig, fund, sent, None, None)
        return {"symbol": tk, "name": name, "score_result": res}
    except Exception as e:
        return {"symbol": tk, "name": "—", "error": str(e)}


def _subprocess_init():
    """ProcessPool 子进程初始化：
    1. 限制 TF 线程数，防止 N 个子进程各开 N 线程造成 oversubscription
    2. 屏蔽过多日志噪音
    """
    import os
    try:
        import config as _cfg
        intra = int(getattr(_cfg, "TF_INTRAOP_THREADS", 1))
        inter = int(getattr(_cfg, "TF_INTEROP_THREADS", 2))
    except Exception:
        intra, inter = 1, 2
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OMP_NUM_THREADS", str(intra))
    os.environ.setdefault("MKL_NUM_THREADS", str(intra))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(intra))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(inter))
    # TF 在 import 后才能设置 thread 数，且必须在创建任何 op 之前，故此处用环境变量
    # 子进程内 import tensorflow 时会自动读取


def run_quick(symbol: str) -> dict:
    """快速分析（1年数据，含预测）"""
    try:
        from concurrent.futures import ThreadPoolExecutor
        from data.fetcher import fetch_stock_data, fetch_stock_info
        from analysis.technical import compute_indicators, generate_signals
        from analysis.fundamental import fetch_and_analyze
        from analysis.sentiment import analyze_sentiment
        from data.news import fetch_news, get_news_texts
        from models.scorer import compute_score
        from models.ensemble import EnsemblePredictor

        # 4 个独立网络 IO 并行：行情、个股信息、新闻；name 从 info 内派生
        with ThreadPoolExecutor(max_workers=3) as io_pool:
            f_df   = io_pool.submit(fetch_stock_data, symbol, 2)
            f_info = io_pool.submit(fetch_stock_info, symbol)
            f_news = io_pool.submit(fetch_news, symbol, 15)
            df   = f_df.result()
            info = f_info.result()
            news = f_news.result()
        name = info.get("股票简称") or info.get("名称") or symbol
        df   = compute_indicators(df)
        sig  = generate_signals(df)
        fund = fetch_and_analyze(symbol, info)
        sent = analyze_sentiment(get_news_texts(news))

        pred = EnsemblePredictor()
        pred.symbol_context = symbol
        pred.posterior_weights = pred._load_posterior_weights(symbol)
        pred.model_control = pred._load_model_control(symbol)
        short_pred = long_pred = None
        try:
            pred.train(df)
            short_pred = pred.predict_short(df)
            long_pred  = pred.predict_long(df)
        except Exception:
            pass

        res = compute_score(sig, fund, sent, short_pred, long_pred)

        return {
            "symbol": symbol, "name": name,
            "score_result": res, "short_pred": short_pred, "long_pred": long_pred,
            "fundamental": fund, "sentiment": sent, "signals": sig,
            "last_price": float(df["close"].iloc[-1]),
        }
    except Exception as e:
        return {"symbol": symbol, "name": "—", "error": str(e)}


# ─────────────────────────────────────────────────────────────
#  辅助打印
# ─────────────────────────────────────────────────────────────

def _print_score_table(result, score, rating, color):
    table = Table(box=box.ROUNDED, border_style="blue", show_header=True)
    table.add_column("维度",     style="cyan",  width=14)
    table.add_column("得分",     style="white", width=8,  justify="right")
    table.add_column("权重",     style="dim",   width=8,  justify="right")
    table.add_column("加权贡献", style="white", width=10, justify="right")

    import config as cfg
    w = cfg.SCORE_WEIGHTS
    rows = [
        ("技术面",  result["technical_score"],   w["technical"],   "blue"),
        ("基本面",  result["fundamental_score"],  w["fundamental"],  "green"),
        ("情感面",  result["sentiment_score"],    w["sentiment"],    "magenta"),
        ("预测面",  result["prediction_score"],   w["prediction"],   "yellow"),
    ]
    for name, s, weight, clr in rows:
        table.add_row(name, f"[{clr}]{s:.1f}[/{clr}]", f"{weight:.0%}", f"{s*weight:.1f}")
    table.add_section()
    cn = {"#00C853":"green","#69F0AE":"green","#FFD600":"yellow","#FF6D00":"yellow","#D50000":"red"}.get(color,"white")
    table.add_row("[bold]综合评分[/bold]", f"[bold {cn}]{score:.1f}[/bold {cn}]",
                  "100%", f"[bold {cn}]{rating}[/bold {cn}]")
    console.print(table)

    rt = Table(box=box.SIMPLE, show_header=False, border_style="dim")
    rt.add_column("", width=14); rt.add_column("", width=16)
    rt.add_row("[dim]建议止损[/dim]", f"[red]{result.get('stop_loss','—')}[/red]")
    rt.add_row("[dim]目标价位[/dim]", f"[green]{result.get('target_price','—')}[/green]")
    rt.add_row("[dim]建议仓位[/dim]", f"[yellow]{result.get('position_pct','—')}[/yellow]")
    console.print(rt)


def _print_backtest(bt):
    rc = "green" if bt.get("annual_return", 0) > 0 else "red"
    ec = "green" if bt.get("excess_return", 0) > 0 else "red"
    t = Table(box=box.SIMPLE_HEAD, border_style="dim")
    t.add_column("指标", width=14); t.add_column("数值", width=12, justify="right")
    t.add_row("胜率",       f"{bt.get('win_rate',0):.1%}")
    t.add_row("年化收益",   f"[{rc}]{bt.get('annual_return',0):.2%}[/{rc}]")
    t.add_row("最大回撤",   f"[red]{bt.get('max_drawdown',0):.2%}[/red]")
    t.add_row("夏普比率",   f"{bt.get('sharpe_ratio',0):.2f}")
    t.add_row("总交易次数", f"{bt.get('total_trades',0)}")
    t.add_row("超额收益",   f"[{ec}]{bt.get('excess_return',0):+.2%}[/{ec}]")
    t.add_row("回测区间",   bt.get("period", ""))
    console.print(t)


def _print_sr(sr):
    console.print(
        f"  当前价格：[yellow]{sr.get('close',0):.2f}[/yellow]  "
        f"阻力位：[red]{' / '.join(str(x) for x in sr.get('resistance',[]))}[/red]  "
        f"支撑位：[green]{' / '.join(str(x) for x in sr.get('support',[]))}[/green]"
    )


def _print_final_summary(name, sym, score, rating, color, short, long_p, result):
    cn = {"#00C853":"green","#69F0AE":"green","#FFD600":"yellow","#FF6D00":"yellow","#D50000":"red"}.get(color,"white")
    lines = [
        f"[bold]{name}（{sym}）[/bold]",
        f"综合评分：[bold {cn}]{score:.1f}分  {rating}[/bold {cn}]",
    ]
    if short:
        pct = short["pct_change"]; c = "green" if pct >= 0 else "red"
        lines.append(f"短期预测（10天）：[{c}]{'▲' if pct>=0 else '▼'}{abs(pct):.2f}%[/{c}]  预测价 {short['predictions'][-1]:.2f}")
    if long_p:
        pct = long_p["pct_change"]; c = "green" if pct >= 0 else "red"
        lines.append(f"中期预测（30天）：[{c}]{'▲' if pct>=0 else '▼'}{abs(pct):.2f}%[/{c}]  预测价 {long_p['predictions'][-1]:.2f}")
    lines.append(f"止损价：[red]{result.get('stop_loss','—')}[/red]  "
                 f"目标价：[green]{result.get('target_price','—')}[/green]  "
                 f"建议仓位：[yellow]{result.get('position_pct','—')}[/yellow]")
    console.print(Panel("\n".join(lines), title="[bold cyan]分析摘要[/bold cyan]",
                        border_style="cyan", padding=(1, 2)))


# ─────────────────────────────────────────────────────────────
#  多股看板 HTML
# ─────────────────────────────────────────────────────────────

def _generate_board_html(results: list) -> str:
    """生成多股对比看板 HTML"""
    from datetime import datetime
    import plotly.graph_objects as go
    import plotly.io as pio

    rows = ""
    for r in results:
        if "error" in r:
            rows += f"<tr><td>{r['symbol']}</td><td>—</td><td colspan='7' style='color:#D50000'>{r['error'][:50]}</td></tr>\n"
            continue

        sr    = r.get("score_result", {})
        score = sr.get("total_score", 0)
        color = sr.get("color", "#78909C")
        rating = sr.get("rating", "—")
        sp    = r.get("short_pred")
        lp    = r.get("long_pred")
        price = r.get("last_price", 0)

        sp_txt = (f"{'▲' if sp['pct_change']>=0 else '▼'}{abs(sp['pct_change']):.1f}%"
                  if sp else "—")
        lp_txt = (f"{'▲' if lp['pct_change']>=0 else '▼'}{abs(lp['pct_change']):.1f}%"
                  if lp else "—")
        sp_color = "#00C853" if (sp and sp["pct_change"] >= 0) else "#EF5350"
        lp_color = "#00C853" if (lp and lp["pct_change"] >= 0) else "#EF5350"

        rows += f"""<tr>
  <td><b>{r['symbol']}</b></td>
  <td>{r['name']}</td>
  <td style='font-size:1.1em;font-weight:700;color:{color}'>{score:.1f}</td>
  <td style='color:{color}'>{rating}</td>
  <td style='color:#2196F3'>{sr.get('technical_score','—')}</td>
  <td style='color:#4CAF50'>{sr.get('fundamental_score','—')}</td>
  <td style='color:#CE93D8'>{sr.get('sentiment_score','—')}</td>
  <td>{price:.2f}</td>
  <td style='color:{sp_color}'>{sp_txt}</td>
  <td style='color:{lp_color}'>{lp_txt}</td>
</tr>\n"""

    # 雷达对比图
    radar_html = ""
    try:
        cats = ["技术面", "基本面", "情感面", "预测面", "技术面"]
        fig = go.Figure()
        for r in results:
            if "error" in r:
                continue
            sr = r.get("score_result", {})
            vals = [
                sr.get("technical_score", 50),
                sr.get("fundamental_score", 50),
                sr.get("sentiment_score", 50),
                sr.get("prediction_score", 50),
                sr.get("technical_score", 50),
            ]
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                                          name=f"{r['symbol']} {r['name']}"))
        fig.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            font=dict(color="#C9D1D9"),
            polar=dict(bgcolor="#161B22",
                       radialaxis=dict(range=[0,100], showticklabels=False),
                       angularaxis=dict(gridcolor="#21262D")),
            title="各股维度对比雷达图", height=450,
        )
        radar_html = pio.to_html(fig, full_html=False, include_plotlyjs=True)
    except Exception:
        pass

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>A股多股对比看板</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0D1117; color:#C9D1D9; font-family:'Microsoft YaHei',Arial,sans-serif; font-size:14px; }}
  .container {{ max-width:1400px; margin:0 auto; padding:24px; }}
  h1 {{ color:#58A6FF; margin-bottom:6px; }}
  .meta {{ color:#8B949E; font-size:0.9em; margin-bottom:20px; }}
  table {{ width:100%; border-collapse:collapse; background:#161B22;
           border:1px solid #30363D; border-radius:8px; overflow:hidden; }}
  th {{ background:#21262D; color:#8B949E; padding:10px 14px; text-align:left; font-weight:normal; }}
  td {{ padding:10px 14px; border-bottom:1px solid #21262D; }}
  tr:hover td {{ background:rgba(255,255,255,0.03); }}
  tr:last-child td {{ border-bottom:none; }}
  .card {{ background:#161B22; border:1px solid #30363D; border-radius:8px;
           padding:16px; margin-top:20px; }}
  .warning {{ background:rgba(255,152,0,0.1); border:1px solid rgba(255,152,0,0.3);
              border-radius:6px; padding:12px; color:#FFB74D; margin-top:20px; text-align:center; }}
</style>
</head>
<body>
<div class="container">
<h1>📊 A股多股对比看板</h1>
<div class="meta">生成时间：{datetime.now().strftime("%Y年%m月%d日 %H:%M")} &nbsp;|&nbsp; 共分析 {len(results)} 只股票（按评分排序）</div>

<table>
  <tr>
    <th>代码</th><th>名称</th><th>综合评分</th><th>评级</th>
    <th>技术面</th><th>基本面</th><th>情感面</th>
    <th>当前价</th><th>10天预测</th><th>30天预测</th>
  </tr>
  {rows}
</table>

<div class="card">
{radar_html}
</div>

<div class="warning">⚠️ 仅供参考，不构成投资建议。股市有风险，投资须谨慎。</div>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
#  CLI 命令
# ─────────────────────────────────────────────────────────────

@click.group()
def cli():
    """A股智能分析预测系统"""
    pass


@cli.command()
@click.option("--ticker",       "-t", required=True,  help="股票代码，如 600519")
@click.option("--lang",         "-l", default="zh",   help="语言：zh / en", show_default=True)
@click.option("--years",        "-y", default=3,      help="历史数据年数", show_default=True)
@click.option("--no-backtest",        is_flag=True,   help="跳过回测")
@click.option("--no-browser",         is_flag=True,   help="不自动打开浏览器")
@click.option("--no-history",         is_flag=True,   help="不保存历史评分")
def analyze(ticker, lang, years, no_backtest, no_browser, no_history):
    """深度分析单只股票（技术+基本面+情感+预测+回测）"""
    import config as cfg
    cfg.DEFAULT_PERIOD_YEARS = years
    run_analysis(
        symbol       = ticker,
        lang         = lang,
        do_backtest  = not no_backtest,
        open_browser = not no_browser,
        save_history = not no_history,
    )


@cli.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--lang",       "-l", default="zh",  help="语言", show_default=True)
@click.option("--no-browser",       is_flag=True,  help="不自动打开浏览器")
def board(tickers, lang, no_browser):
    """多股对比看板：分析多只股票并生成排行榜（含预测）"""
    import config as cfg
    cfg.LANGUAGE = lang

    # 多股并行分析，worker 数由 config.PARALLEL_STOCK_WORKERS 控制
    workers = min(len(tickers), max(1, getattr(cfg, "PARALLEL_STOCK_WORKERS", 2)))
    console.print(Panel(
        f"[bold cyan]多股对比看板[/bold cyan]  并行分析 {len(tickers)} 只股票，workers={workers}"
    ))

    results = []
    if workers <= 1 or len(tickers) <= 1:
        # 单 worker 或单股，走原顺序逻辑（避免 ProcessPool 启动开销）
        for tk in tickers:
            console.print(f"\n[cyan]→ 分析 {tk}...[/cyan]")
            r = run_quick(tk)
            results.append(r)
            if "score_result" in r:
                s = r["score_result"]["total_score"]
                console.print(f"  [green]✓ {r['name']}  评分：{s:.1f}[/green]")
            else:
                console.print(f"  [red]✗ 失败：{r.get('error','')}[/red]")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # 用 ProcessPool 避开 GIL；提交后按完成顺序收集
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_subprocess_init,
        ) as pool:
            futures = {pool.submit(run_quick, tk): tk for tk in tickers}
            for fut in as_completed(futures):
                tk = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"symbol": tk, "name": "—", "error": str(e)}
                results.append(r)
                if "score_result" in r:
                    s = r["score_result"]["total_score"]
                    console.print(f"  [green]✓ {r['name']:<12}({tk})  评分：{s:.1f}[/green]")
                else:
                    console.print(f"  [red]✗ {tk} 失败：{r.get('error','')}[/red]")

    # 按评分排序
    results.sort(key=lambda x: x.get("score_result", {}).get("total_score", 0), reverse=True)

    # 终端表格
    table = Table(box=box.ROUNDED, border_style="blue", title="多股评分排行榜")
    table.add_column("排名", width=4,  justify="right")
    table.add_column("代码", width=8)
    table.add_column("名称", width=12)
    table.add_column("综合评分", width=8, justify="right")
    table.add_column("技术", width=7,  justify="right")
    table.add_column("基本面", width=7, justify="right")
    table.add_column("情感", width=7,  justify="right")
    table.add_column("10天预测", width=9, justify="right")
    table.add_column("评级", width=8)

    for i, r in enumerate(results, 1):
        if "error" in r:
            table.add_row(str(i), r["symbol"], "—", "—", "—", "—", "—", "—", f"[red]失败[/red]")
            continue
        sr    = r.get("score_result", {})
        score = sr.get("total_score", 0)
        clr   = {"强烈买入":"green","买入":"green","观望":"yellow","卖出":"red","强烈卖出":"red"}.get(sr.get("rating",""),"white")
        sp    = r.get("short_pred")
        sp_txt = (f"{'▲' if sp['pct_change']>=0 else '▼'}{abs(sp['pct_change']):.1f}%" if sp else "—")
        sp_clr = "green" if (sp and sp["pct_change"] >= 0) else "red"
        table.add_row(
            str(i), r["symbol"], r["name"],
            f"[bold]{score:.1f}[/bold]",
            f"{sr.get('technical_score','—')}",
            f"{sr.get('fundamental_score','—')}",
            f"{sr.get('sentiment_score','—')}",
            f"[{sp_clr}]{sp_txt}[/{sp_clr}]",
            f"[{clr}]{sr.get('rating','—')}[/{clr}]",
        )

    console.print(table)

    # 生成HTML看板
    try:
        html = _generate_board_html(results)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(cfg.OUTPUT_DIR, f"board_{ts}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        console.print(f"\n[green]✓ 看板已保存：{path}[/green]")
        if not no_browser:
            webbrowser.open(f"file:///{path.replace(os.sep, '/')}")
    except Exception as e:
        _warn(console, f"HTML看板生成失败: {e}")


@cli.command()
@click.argument("ticker")
@click.option("--days", "-d", default=90, help="显示最近N天", show_default=True)
def history(ticker, days):
    """查看某只股票的历史评分走势"""
    from data.score_history import load_history
    records = load_history(ticker)

    if not records:
        console.print(f"[yellow]暂无 {ticker} 的历史记录。运行 analyze 命令后会自动保存。[/yellow]")
        return

    recent = records[-days:]
    name   = recent[-1].get("stock_name", ticker)
    console.print(Panel(f"[bold cyan]{name}（{ticker}）历史评分[/bold cyan]  共 {len(records)} 条记录"))

    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("日期",     width=12)
    table.add_column("综合评分", width=8,  justify="right")
    table.add_column("技术",     width=7,  justify="right")
    table.add_column("基本面",   width=7,  justify="right")
    table.add_column("情感",     width=7,  justify="right")
    table.add_column("10天预测", width=9,  justify="right")
    table.add_column("评级",     width=8)

    for r in recent:
        score = r.get("total_score", 0)
        clr   = {"强烈买入":"green","买入":"green","观望":"yellow","卖出":"red","强烈卖出":"red"}.get(r.get("rating",""),"white")
        sp    = r.get("short_pct")
        sp_txt = f"{'▲' if sp and sp>=0 else '▼'}{abs(sp):.1f}%" if sp is not None else "—"
        sp_clr = "green" if sp and sp >= 0 else "red"
        table.add_row(
            r["date"],
            f"[bold]{score:.1f}[/bold]",
            str(r.get("technical_score", "—")),
            str(r.get("fundamental_score", "—")),
            str(r.get("sentiment_score", "—")),
            f"[{sp_clr}]{sp_txt}[/{sp_clr}]",
            f"[{clr}]{r.get('rating','—')}[/{clr}]",
        )

    console.print(table)

    # 趋势简图
    scores = [r.get("total_score", 0) for r in recent]
    if len(scores) > 1:
        delta = scores[-1] - scores[-2]
        trend = "↗" if delta > 0 else ("↘" if delta < 0 else "→")
        dc    = "green" if delta > 0 else ("red" if delta < 0 else "yellow")
        console.print(f"\n  最新评分：[bold]{scores[-1]:.1f}[/bold]  "
                      f"较前次：[{dc}]{trend} {delta:+.1f}[/{dc}]")


@cli.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--lang", "-l", default="zh")
def compare(tickers, lang):
    """快速对比多只股票（不做深度预测，速度更快）"""
    import config as cfg
    cfg.LANGUAGE = lang

    console.print(Panel("[bold cyan]多股快速对比[/bold cyan]"))
    table = Table(box=box.ROUNDED, border_style="blue")
    table.add_column("代码",   width=10)
    table.add_column("名称",   width=12)
    table.add_column("综合评分",width=10, justify="right")
    table.add_column("技术面", width=9,  justify="right")
    table.add_column("基本面", width=9,  justify="right")
    table.add_column("情感",   width=9,  justify="right")
    table.add_column("评级",   width=10)

    # 并行执行（compare 不训练模型，几乎纯 IO + 轻量计算，ThreadPool 即可）
    workers = min(len(tickers), max(1, getattr(cfg, "PARALLEL_STOCK_WORKERS", 2)))
    if workers <= 1 or len(tickers) <= 1:
        results = [_compare_one(tk) for tk in tickers]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_compare_one, tickers))

    # 按输入顺序渲染表格
    for r in results:
        tk = r.get("symbol", "—")
        if "error" in r:
            table.add_row(tk, "—", "—", "—", "—", "—", f"[red]{str(r['error'])[:30]}[/red]")
            continue
        name = r["name"]
        res = r["score_result"]
        clr = {"强烈买入":"green","买入":"green","观望":"yellow",
               "卖出":"red","强烈卖出":"red"}.get(res["rating"],"white")
        table.add_row(
            tk, name,
            f"[bold]{res['total_score']:.1f}[/bold]",
            f"{res['technical_score']:.1f}",
            f"{res['fundamental_score']:.1f}",
            f"{res['sentiment_score']:.1f}",
            f"[{clr}]{res['rating']}[/{clr}]",
        )

    console.print(table)


@cli.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--lang", "-l", default="zh")
def schedule(tickers, lang):
    """对一组股票执行定时分析（每日收盘后运行，无需浏览器）"""
    import config as cfg
    cfg.LANGUAGE = lang

    console.print(Panel(f"[bold cyan]定时批量分析[/bold cyan]  {len(tickers)} 只股票"))

    all_alerts = []
    for tk in tickers:
        console.print(f"\n[cyan]── 分析 {tk} ──[/cyan]")
        try:
            from data.fetcher import fetch_stock_data, fetch_stock_info, get_stock_name
            from analysis.technical import compute_indicators, generate_signals
            from analysis.fundamental import fetch_and_analyze
            from analysis.sentiment import analyze_sentiment
            from data.news import fetch_news, get_news_texts
            from models.scorer import compute_score
            from data.score_history import save_score, check_alerts

            df   = fetch_stock_data(tk, years=2)
            info = fetch_stock_info(tk)
            name = get_stock_name(tk)
            df   = compute_indicators(df)
            sig  = generate_signals(df)
            fund = fetch_and_analyze(tk, info)
            news = fetch_news(tk, limit=15)
            sent = analyze_sentiment(get_news_texts(news))
            res  = compute_score(sig, fund, sent, None, None)

            save_score(tk, name, res, None, None)
            alerts = check_alerts(tk, name, res["total_score"])
            all_alerts.extend(alerts)

            clr = {"强烈买入":"green","买入":"green","观望":"yellow","卖出":"red","强烈卖出":"red"}.get(res["rating"],"white")
            console.print(f"  [green]✓[/green] {name}  评分：[{clr}]{res['total_score']:.1f}  {res['rating']}[/{clr}]")
        except Exception as e:
            console.print(f"  [red]✗ {tk} 失败: {e}[/red]")

    if all_alerts:
        console.print("\n" + "="*50)
        console.print("[bold yellow]📢 预警通知：[/bold yellow]")
        for a in all_alerts:
            console.print(f"  {a}")
    else:
        console.print("\n[dim]无预警触发[/dim]")

    console.print(f"\n[green]✓ 批量分析完成[/green]  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    cli()
