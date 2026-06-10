"""
HTML 报告生成模块
将所有图表和分析结果打包成单个自包含HTML文件
"""
import os
from datetime import datetime
import plotly.io as pio

import config


def generate_html_report(
    symbol:        str,
    stock_name:    str,
    df,
    tech_signals:  dict,
    fundamental:   dict,
    sentiment:     dict,
    short_pred:    dict | None,
    long_pred:     dict | None,
    score_result:  dict,
    backtest:      dict | None,
    figures:       dict,
    news_list:     list  = None,
    history_records: list = None,
) -> str:
    from i18n import t

    date_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{symbol}_{date_str}.html"
    filepath  = os.path.join(config.OUTPUT_DIR, filename)

    # ── 图表转HTML ────────────────────────────────────────
    charts_html = {}
    first = True
    for name, fig in figures.items():
        charts_html[name] = pio.to_html(
            fig, full_html=False, include_plotlyjs=first
        )
        first = False

    color = score_result.get("color", "#FFD600")
    score = score_result.get("total_score", 0)

    # ── 基本面表格行 ──────────────────────────────────────
    fund_rows = ""
    fund_map = [
        ("行业",      fundamental.get("industry",       "—")),
        ("市盈率PE",  _fmt(fundamental.get("pe"),       ".1f")),
        ("市净率PB",  _fmt(fundamental.get("pb"),       ".2f")),
        ("ROE",       _fmt(fundamental.get("roe"),      ".1f", "%")),
        ("净利润增长",_fmt(fundamental.get("profit_growth"), ".1f", "%")),
        ("营收增长",  _fmt(fundamental.get("revenue_growth"),".1f", "%")),
        ("股息率",    _fmt(fundamental.get("dividend_yield"),".2f", "%")),
        ("总市值",    fundamental.get("market_cap", "—")),
    ]
    for label, val in fund_map:
        fund_rows += f"<tr><td>{label}</td><td><b>{val}</b></td></tr>\n"

    # ── 信号行 ────────────────────────────────────────────
    signal_rows = ""
    for s in tech_signals.get("signals", []):
        sig_color = {"buy": "#00C853", "sell": "#D50000", "neutral": "#78909C"}[s["signal"]]
        sig_text  = {"buy": "买入", "sell": "卖出", "neutral": "中性"}[s["signal"]]
        signal_rows += (
            f"<tr>"
            f"<td>{s['name']}</td>"
            f"<td>{s['value']}</td>"
            f"<td style='color:{sig_color}'><b>{sig_text}</b></td>"
            f"<td style='color:#9E9E9E;font-size:0.9em'>{s['desc']}</td>"
            f"</tr>\n"
        )

    # ── 预测摘要 ─────────────────────────────────────────
    short_summary = _pred_summary(short_pred, "短期(10天)")
    long_summary  = _pred_summary(long_pred,  "中期(30天)")

    # ── 情感摘要 ─────────────────────────────────────────
    sent_color = ("#00C853" if sentiment.get("score", 0.5) > 0.65
                  else ("#D50000" if sentiment.get("score", 0.5) < 0.35 else "#FFD600"))

    # ── 回测 ─────────────────────────────────────────────
    bt_html = _backtest_html(backtest) if backtest else "<p style='color:#78909C'>未执行回测</p>"

    # ── 新闻列表 ─────────────────────────────────────────
    news_html = _news_html(news_list or [])

    # ── 历史评分 ─────────────────────────────────────────
    history_section = ""
    if "history" in charts_html and history_records and len(history_records) > 1:
        history_section = f"""
<div class="chart-wrap card">
  <h2>历史评分趋势（共 {len(history_records)} 次记录）</h2>
  {charts_html.get('history', '')}
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{stock_name}（{symbol}）分析报告</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0D1117; color:#C9D1D9; font-family:'Microsoft YaHei',SimHei,Arial,sans-serif;
         font-size:14px; line-height:1.6; }}
  .container {{ max-width:1400px; margin:0 auto; padding:20px; }}
  h1 {{ color:#58A6FF; font-size:1.8em; margin-bottom:4px; }}
  h2 {{ color:#79C0FF; font-size:1.2em; margin:20px 0 10px; border-left:4px solid #1F6FEB;
         padding-left:10px; }}
  .meta {{ color:#8B949E; font-size:0.9em; margin-bottom:20px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }}
  .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:16px; }}
  .card {{ background:#161B22; border:1px solid #30363D; border-radius:8px; padding:16px;
           margin-bottom:16px; }}
  .score-big {{ font-size:4em; font-weight:700; color:{color}; text-align:center;
                line-height:1; margin:10px 0; }}
  .rating {{ font-size:1.6em; font-weight:600; color:{color}; text-align:center; }}
  .score-sub {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }}
  .score-item {{ background:#0D1117; border-radius:6px; padding:10px; text-align:center; }}
  .score-item .val {{ font-size:1.4em; font-weight:600; }}
  .score-item .lbl {{ color:#8B949E; font-size:0.85em; margin-top:2px; }}
  table {{ width:100%; border-collapse:collapse; }}
  th,td {{ padding:8px 12px; border-bottom:1px solid #21262D; text-align:left; }}
  th {{ color:#8B949E; font-weight:normal; font-size:0.9em; }}
  tr:hover {{ background:rgba(255,255,255,0.03); }}
  .pred-box {{ background:#0D1117; border-radius:6px; padding:14px; margin-bottom:10px; }}
  .pred-price {{ font-size:1.8em; font-weight:700; }}
  .pred-pct {{ font-size:1.1em; margin-left:10px; }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.85em; }}
  .tag-pos {{ background:rgba(0,200,83,0.15); color:#00C853; }}
  .tag-neg {{ background:rgba(213,0,0,0.15); color:#D50000; }}
  .tag-neu {{ background:rgba(120,144,156,0.15); color:#78909C; }}
  .risk-box {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:12px; }}
  .risk-item {{ background:#0D1117; border-radius:6px; padding:12px; text-align:center; }}
  .risk-item .val {{ font-size:1.3em; font-weight:600; }}
  .risk-item .lbl {{ color:#8B949E; font-size:0.8em; }}
  .warning {{ background:rgba(255,152,0,0.1); border:1px solid rgba(255,152,0,0.3);
              border-radius:6px; padding:12px; color:#FFB74D; margin-top:20px; text-align:center; }}
  .chart-wrap {{ margin-bottom:16px; }}
  .bt-stat {{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; }}
  .bt-stat-item {{ background:#0D1117; border-radius:6px; padding:10px; text-align:center; }}
  .news-item {{ padding:10px 0; border-bottom:1px solid #21262D; }}
  .news-item:last-child {{ border-bottom:none; }}
  .news-title {{ color:#C9D1D9; font-size:0.95em; line-height:1.5; }}
  .news-title a {{ color:#58A6FF; text-decoration:none; }}
  .news-title a:hover {{ text-decoration:underline; }}
  .news-meta {{ color:#8B949E; font-size:0.82em; margin-top:3px; }}
  .news-content {{ color:#8B949E; font-size:0.85em; margin-top:4px; line-height:1.4; }}
  .sent-pos {{ color:#00C853; }}
  .sent-neg {{ color:#D50000; }}
  .sent-neu {{ color:#78909C; }}
</style>
</head>
<body>
<div class="container">

<!-- 标题 -->
<h1>{stock_name}（{symbol}）</h1>
<div class="meta">
  分析日期：{datetime.now().strftime("%Y年%m月%d日 %H:%M")} &nbsp;|&nbsp;
  数据来源：AKShare · 东方财富 &nbsp;|&nbsp;
  历史数据：{len(df)}个交易日
</div>

<!-- 综合评分 + 风险 -->
<div class="grid2">
  <div class="card">
    <h2>综合评分</h2>
    <div class="score-big">{score}</div>
    <div class="rating">{score_result.get('rating','—')}</div>
    <div class="score-sub">
      <div class="score-item">
        <div class="val" style="color:#2196F3">{score_result.get('technical_score','—')}</div>
        <div class="lbl">技术面 (40%)</div>
      </div>
      <div class="score-item">
        <div class="val" style="color:#4CAF50">{score_result.get('fundamental_score','—')}</div>
        <div class="lbl">基本面 (30%)</div>
      </div>
      <div class="score-item">
        <div class="val" style="color:{sent_color}">{score_result.get('sentiment_score','—')}</div>
        <div class="lbl">情感面 (15%)</div>
      </div>
      <div class="score-item">
        <div class="val" style="color:#FF9800">{score_result.get('prediction_score','—')}</div>
        <div class="lbl">预测面 (15%)</div>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>风险管理建议</h2>
    <div class="risk-box">
      <div class="risk-item">
        <div class="val" style="color:#EF5350">{score_result.get('stop_loss','—')}</div>
        <div class="lbl">建议止损价</div>
      </div>
      <div class="risk-item">
        <div class="val" style="color:#00C853">{score_result.get('target_price','—')}</div>
        <div class="lbl">目标价位</div>
      </div>
      <div class="risk-item">
        <div class="val" style="color:#FF9800">{score_result.get('position_pct','—')}</div>
        <div class="lbl">建议仓位</div>
      </div>
    </div>
    <div style="margin-top:14px; color:#8B949E; font-size:0.9em;">
      <b>新闻情感：</b>
      <span class="tag {'tag-pos' if sentiment.get('score',0.5)>0.65 else ('tag-neg' if sentiment.get('score',0.5)<0.35 else 'tag-neu')}">
        {sentiment.get('label','中性')} ({sentiment.get('score',0.5):.2%})
      </span>
      &nbsp; 分析新闻 <b>{sentiment.get('news_count',0)}</b> 条
      &nbsp;·&nbsp;
      正面 <span class="sent-pos">{sentiment.get('positive_count',0)}</span> /
      负面 <span class="sent-neg">{sentiment.get('negative_count',0)}</span> /
      中性 <span class="sent-neu">{sentiment.get('neutral_count',0)}</span>
    </div>
  </div>
</div>

<!-- 历史评分趋势（如有）-->
{history_section}

<!-- K线图 -->
<div class="chart-wrap card">
  <h2>行情走势 & 技术指标</h2>
  {charts_html.get('candlestick', '')}
</div>

<!-- 预测图 -->
<div class="chart-wrap card">
  <h2>价格预测走势</h2>
  {charts_html.get('prediction', '')}
  <div class="grid2" style="margin-top:12px">
    {short_summary}
    {long_summary}
  </div>
</div>

<!-- 信号 + 评分仪表 -->
<div class="grid2">
  <div class="card">
    <h2>技术信号汇总（{tech_signals.get('buy_count',0)}买 / {tech_signals.get('sell_count',0)}卖 / {tech_signals.get('neutral_count',0)}中性）</h2>
    {charts_html.get('signals', '')}
    <table style="margin-top:10px">
      <tr><th>指标</th><th>数值</th><th>信号</th><th>说明</th></tr>
      {signal_rows}
    </table>
  </div>
  <div class="card">
    <h2>综合评分仪表</h2>
    {charts_html.get('gauge', '')}
  </div>
</div>

<!-- 基本面 -->
<div class="grid2">
  <div class="card">
    <h2>基本面数据</h2>
    <table>
      <tr><th>指标</th><th>数值</th></tr>
      {fund_rows}
    </table>
    {''.join(f"<p style='color:#8B949E;font-size:0.85em;margin-top:4px'>• {d}</p>" for d in score_result.get('details',[]))}
  </div>
  <div class="card">
    <h2>基本面雷达图</h2>
    {charts_html.get('radar', '')}
  </div>
</div>

<!-- 新闻列表 -->
{news_html}

<!-- 回测 -->
<div class="card">
  <h2>策略回测结果</h2>
  {bt_html}
</div>

<!-- 风险警示 -->
<div class="warning">
  ⚠️ 风险提示：本报告由AI模型自动生成，仅供参考，不构成任何投资建议。
  股市有风险，投资须谨慎。历史数据不代表未来表现，预测存在不确定性。
</div>

</div>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return filepath


def _fmt(val, fmt=".2f", suffix=""):
    if val is None:
        return "—"
    try:
        return f"{float(val):{fmt}}{suffix}"
    except Exception:
        return str(val)


def _pred_summary(pred: dict | None, label: str) -> str:
    if not pred:
        return f"<div class='pred-box'><div style='color:#8B949E'>{label}：预测不可用</div></div>"
    pct   = pred.get("pct_change", 0)
    price = pred.get("predictions", [0])[-1] if pred.get("predictions") else 0
    color = "#00C853" if pct >= 0 else "#EF5350"
    sign  = "▲" if pct >= 0 else "▼"
    return (
        f"<div class='pred-box'>"
        f"<div style='color:#8B949E;font-size:0.9em'>{label}</div>"
        f"<div class='pred-price' style='color:{color}'>{price:.2f} 元"
        f"<span class='pred-pct'>{sign}{abs(pct):.2f}%</span></div>"
        f"</div>"
    )


def _news_html(news_list: list) -> str:
    if not news_list:
        return ""

    rows = ""
    for item in news_list:
        title   = item.get("title", "").strip()
        content = item.get("content", "").strip()
        date    = str(item.get("date", ""))[:10]
        source  = item.get("source", "")
        url     = item.get("url", "")

        if not title:
            continue

        title_html = (f"<a href='{url}' target='_blank'>{title}</a>"
                      if url else title)
        content_html = (f"<div class='news-content'>{content[:200]}...</div>"
                        if content and content != title and len(content) > 10 else "")

        rows += f"""
<div class="news-item">
  <div class="news-title">{title_html}</div>
  <div class="news-meta">{date} &nbsp;·&nbsp; {source}</div>
  {content_html}
</div>"""

    if not rows:
        return ""

    return f"""
<div class="card">
  <h2>相关新闻（{len(news_list)} 条）</h2>
  {rows}
</div>"""


def _backtest_html(bt: dict) -> str:
    color_return = "#00C853" if bt.get("annual_return", 0) > 0 else "#EF5350"
    color_vs     = "#00C853" if bt.get("excess_return", 0) > 0 else "#EF5350"
    return f"""
<div class="bt-stat">
  <div class="bt-stat-item">
    <div class="val">{bt.get('win_rate',0):.1%}</div>
    <div class="lbl">胜率</div>
  </div>
  <div class="bt-stat-item">
    <div class="val" style="color:{color_return}">{bt.get('annual_return',0):.1%}</div>
    <div class="lbl">年化收益</div>
  </div>
  <div class="bt-stat-item">
    <div class="val" style="color:#EF5350">{bt.get('max_drawdown',0):.1%}</div>
    <div class="lbl">最大回撤</div>
  </div>
  <div class="bt-stat-item">
    <div class="val">{bt.get('sharpe_ratio',0):.2f}</div>
    <div class="lbl">夏普比率</div>
  </div>
  <div class="bt-stat-item">
    <div class="val" style="color:{color_vs}">{bt.get('excess_return',0):+.1%}</div>
    <div class="lbl">超额收益(vs持有)</div>
  </div>
</div>
<p style="color:#8B949E;font-size:0.85em;margin-top:8px">
  回测周期：{bt.get('period','')} &nbsp;|&nbsp; 总交易 {bt.get('total_trades',0)} 次
</p>"""
