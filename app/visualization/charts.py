"""
Plotly 交互式图表模块
K线图、技术指标、预测曲线、基本面雷达图、综合评分仪表盘
"""
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# 配色方案
COLORS = {
    "up":        "#EF5350",   # 涨（红）
    "down":      "#26A69A",   # 跌（绿）
    "ma5":       "#FF9800",
    "ma10":      "#2196F3",
    "ma20":      "#9C27B0",
    "ma60":      "#F44336",
    "ema12":     "#00BCD4",
    "ema26":     "#FF5722",
    "macd":      "#1565C0",
    "signal":    "#E53935",
    "hist_pos":  "#EF5350",
    "hist_neg":  "#26A69A",
    "bb":        "#78909C",
    "predict":   "#FF6F00",
    "conf_band": "rgba(255,111,0,0.15)",
    "bg":        "#0D1117",
    "grid":      "#21262D",
    "text":      "#E6EDF3",
}

def _base_layout(**extra) -> dict:
    """返回基础布局字典，调用方可覆盖任意键"""
    base = dict(
        paper_bgcolor = COLORS["bg"],
        plot_bgcolor  = COLORS["bg"],
        font          = dict(color=COLORS["text"], family="Microsoft YaHei, SimHei, Arial"),
        xaxis         = dict(gridcolor=COLORS["grid"], showgrid=True),
        yaxis         = dict(gridcolor=COLORS["grid"], showgrid=True),
        legend        = dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#444C56",
                             font=dict(color="#E6EDF3", size=12)),
        hovermode     = "x unified",
        margin        = dict(l=50, r=20, t=60, b=40),
    )
    base.update(extra)
    return base

# 向后兼容别名
LAYOUT_DEFAULTS = _base_layout()


def create_candlestick_chart(
    df: pd.DataFrame,
    stock_name: str,
    symbol: str,
    max_bars: int = 500,
) -> go.Figure:
    """K线图 + 均线 + 布林带 + 成交量 + MACD + RSI

    max_bars: 最多显示的 K 线数（默认 500 ≈ 2 年），超出的截尾
              避免渲染 725+ 根 K 线导致 plotly 卡顿
    """

    # 限制 K 线数量加速渲染
    if max_bars and len(df) > max_bars:
        df = df.iloc[-max_bars:]

    # 降低 JSON 体积以加速浏览器传输/解析（数据一根不减，仅减小数）：
    #   价格类保留 2 位小数、指标类 3 位、成交量取整
    df = df.copy()
    for _c in ("open", "high", "low", "close",
               "sma_5", "sma_10", "sma_20", "sma_60", "bb_upper", "bb_lower"):
        if _c in df.columns:
            df[_c] = df[_c].round(2)
    for _c in ("macd", "macd_signal", "macd_hist", "rsi_14"):
        if _c in df.columns:
            df[_c] = df[_c].round(3)
    if "volume" in df.columns:
        df["volume"] = df["volume"].round(0)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=["", "MACD", "RSI(14)"],
    )

    # ── K线 ─────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x     = df.index,
            open  = df["open"],
            high  = df["high"],
            low   = df["low"],
            close = df["close"],
            name  = "K线",
            increasing_fillcolor = COLORS["up"],
            decreasing_fillcolor = COLORS["down"],
            increasing_line_color= COLORS["up"],
            decreasing_line_color= COLORS["down"],
        ),
        row=1, col=1,
    )

    # ── 均线 ─────────────────────────────────────────────────
    for period, color, name in [
        (5,  COLORS["ma5"],  "MA5"),
        (10, COLORS["ma10"], "MA10"),
        (20, COLORS["ma20"], "MA20"),
        (60, COLORS["ma60"], "MA60"),
    ]:
        col = f"sma_{period}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=name,
                           line=dict(width=1, color=color), opacity=0.9),
                row=1, col=1,
            )

    # ── 布林带 ─────────────────────────────────────────────
    if "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_upper"], name="BB上轨",
                       line=dict(width=1, color=COLORS["bb"], dash="dash"), opacity=0.6),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["bb_lower"], name="BB下轨",
                       fill="tonexty", fillcolor="rgba(120,144,156,0.08)",
                       line=dict(width=1, color=COLORS["bb"], dash="dash"), opacity=0.6),
            row=1, col=1,
        )

    # ── 成交量（用颜色区分涨跌） ───────────────────────────
    if "volume" in df.columns:
        colors_vol = [
            COLORS["up"] if c >= o else COLORS["down"]
            for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["volume"], name="成交量",
                   marker_color=colors_vol, opacity=0.7,
                   showlegend=False),
            row=1, col=1,
        )
        # 成交量轴叠加在K线右侧，用secondary_y效果不佳，改为缩小比例叠加
        # 实际上放到第2行更清晰，这里volume叠加于K线上方轻微显示

    # ── MACD ────────────────────────────────────────────────
    if "macd" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd"], name="MACD",
                       line=dict(width=1.5, color=COLORS["macd"])),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                       line=dict(width=1.5, color=COLORS["signal"])),
            row=2, col=1,
        )
        if "macd_hist" in df.columns:
            hist_colors = [
                COLORS["hist_pos"] if v >= 0 else COLORS["hist_neg"]
                for v in df["macd_hist"]
            ]
            fig.add_trace(
                go.Bar(x=df.index, y=df["macd_hist"], name="柱状",
                       marker_color=hist_colors, opacity=0.7),
                row=2, col=1,
            )

    # ── RSI ─────────────────────────────────────────────────
    if "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["rsi_14"], name="RSI(14)",
                       line=dict(width=1.5, color="#CE93D8")),
            row=3, col=1,
        )
        for level, color in [(70, "rgba(239,83,80,0.3)"), (30, "rgba(38,166,154,0.3)")]:
            fig.add_hline(y=level, line_dash="dash",
                          line_color=color, opacity=0.8, row=3, col=1)

    # ── 布局 ────────────────────────────────────────────────
    last_price  = df["close"].iloc[-1]
    pct_today   = df.get("pct_change", pd.Series([0])).iloc[-1] if "pct_change" in df.columns else 0
    title_color = COLORS["up"] if pct_today >= 0 else COLORS["down"]
    arrow       = "▲" if pct_today >= 0 else "▼"

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=(f"<span style='color:#E6EDF3;font-size:16px'><b>{stock_name}（{symbol}）</b></span>"
                  f"<span style='color:#ADBAC7;font-size:13px'>  最新: </span>"
                  f"<span style='color:{title_color};font-size:15px;font-weight:bold'>{last_price:.2f}</span>"
                  f"<span style='color:{title_color};font-size:13px'>  {arrow}{abs(pct_today):.2f}%</span>"),
            font=dict(size=16, color="#E6EDF3"),
        ),
        xaxis3=dict(
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1月",  step="month", stepmode="backward"),
                    dict(count=3,  label="3月",  step="month", stepmode="backward"),
                    dict(count=6,  label="6月",  step="month", stepmode="backward"),
                    dict(count=1,  label="1年",  step="year",  stepmode="backward"),
                    dict(step="all", label="全部"),
                ],
                bgcolor=COLORS["grid"],
                activecolor="#1F6FEB",
            ),
        ),
        height=800,
    )
    fig.update_xaxes(showspikes=True, spikecolor=COLORS["text"], spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor=COLORS["text"], spikethickness=1)

    return fig


def create_prediction_chart(
    df: pd.DataFrame,
    short_pred: dict | None,
    long_pred:  dict | None,
    stock_name: str,
) -> go.Figure:
    """预测走势图（历史价格 + 短期/中期预测 + 置信区间）"""
    fig = go.Figure()

    # ── 历史收盘价（最近90天） ────────────────────────────
    hist = df.tail(90)
    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist["close"],
            name="历史收盘价",
            line=dict(color="#64B5F6", width=2),
        )
    )

    # ── 短期预测 ─────────────────────────────────────────
    if short_pred:
        dates = pd.to_datetime(short_pred["dates"])
        preds = short_pred["predictions"]
        lb    = short_pred["lower_bound"]
        ub    = short_pred["upper_bound"]

        fig.add_trace(go.Scatter(
            x=dates, y=ub, fill=None,
            line=dict(color="rgba(255,111,0,0)", width=0),
            showlegend=False, name="短期上界",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=lb, fill="tonexty",
            fillcolor=COLORS["conf_band"],
            line=dict(color="rgba(255,111,0,0)", width=0),
            name="短期置信区间",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=preds, name="短期预测(10天)",
            line=dict(color=COLORS["predict"], width=2.5, dash="dot"),
            mode="lines+markers",
            marker=dict(size=5),
        ))

    # ── 中期预测 ─────────────────────────────────────────
    if long_pred:
        dates = pd.to_datetime(long_pred["dates"])
        preds = long_pred["predictions"]
        lb    = long_pred["lower_bound"]
        ub    = long_pred["upper_bound"]

        fig.add_trace(go.Scatter(
            x=dates, y=ub, fill=None,
            line=dict(color="rgba(102,187,106,0)", width=0),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=lb, fill="tonexty",
            fillcolor="rgba(102,187,106,0.12)",
            line=dict(color="rgba(102,187,106,0)", width=0),
            name="中期置信区间",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=preds, name="中期预测(30天)",
            line=dict(color="#66BB6A", width=2.5, dash="dash"),
        ))

    fig.update_layout(**_base_layout(
        title=f"<b>{stock_name} — 价格预测走势</b>",
        yaxis_title="价格 (元)",
        height=500,
    ))
    return fig


def create_score_gauge(score: float, rating: str, color: str) -> go.Figure:
    """综合评分仪表盘"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title=dict(text=f"综合评分<br><span style='font-size:1.2em;color:{color}'>{rating}</span>",
                   font=dict(size=16)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1,
                      tickcolor=COLORS["text"],
                      tickvals=[0, 30, 45, 65, 80, 100]),
            bar=dict(color=color),
            bgcolor=COLORS["grid"],
            borderwidth=2,
            bordercolor=COLORS["text"],
            steps=[
                dict(range=[0,  30], color="#B71C1C"),
                dict(range=[30, 45], color="#E53935"),
                dict(range=[45, 65], color="#F9A825"),
                dict(range=[65, 80], color="#43A047"),
                dict(range=[80,100], color="#1B5E20"),
            ],
            threshold=dict(
                line=dict(color="white", width=4),
                thickness=0.75,
                value=score,
            ),
        ),
        number=dict(font=dict(size=36, color=color), suffix="分"),
    ))
    layout = {**LAYOUT_DEFAULTS, "height": 300, "margin": dict(l=30, r=30, t=80, b=20)}
    fig.update_layout(**layout)
    return fig


def create_fundamental_radar(fundamental: dict, stock_name: str) -> go.Figure:
    """基本面雷达图"""
    def normalize(val, low, high, reverse=False):
        if val is None:
            return 50
        v = float(np.clip((val - low) / (high - low) * 100, 0, 100))
        return 100 - v if reverse else v

    categories = ["估值(PE)", "市净率(PB)", "盈利能力(ROE)",
                  "利润增长", "营收增长", "股息率"]
    values = [
        normalize(fundamental.get("pe"),             0,  60, reverse=True),
        normalize(fundamental.get("pb"),             0,   8, reverse=True),
        normalize(fundamental.get("roe"),            0,  30),
        normalize(fundamental.get("profit_growth"),  -20, 50),
        normalize(fundamental.get("revenue_growth"), -10, 40),
        normalize(fundamental.get("dividend_yield"),  0,   6),
    ]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(33,150,243,0.15)",
        line=dict(color="#2196F3", width=2),
        name="基本面",
    ))
    fig.update_layout(**_base_layout(
        polar=dict(
            bgcolor=COLORS["bg"],
            radialaxis=dict(range=[0, 100], showticklabels=False,
                            gridcolor=COLORS["grid"]),
            angularaxis=dict(gridcolor=COLORS["grid"]),
        ),
        title=f"<b>{stock_name} — 基本面雷达图</b>",
        height=400,
        showlegend=False,
    ))
    return fig


def create_score_history_chart(records: list, stock_name: str) -> go.Figure:
    """历史评分走势图（折线 + 区间背景色）"""
    if not records:
        return go.Figure()

    dates  = [r["date"] for r in records]
    scores = [r["total_score"] for r in records]

    fig = go.Figure()

    # 背景色带（买入/观望/卖出区间）
    for y0, y1, clr, label in [
        (80, 100, "rgba(0,200,83,0.06)",  "强烈买入"),
        (65,  80, "rgba(0,200,83,0.04)",  "买入"),
        (45,  65, "rgba(255,214,0,0.04)", "观望"),
        (30,  45, "rgba(255,109,0,0.04)", "卖出"),
        (0,   30, "rgba(213,0,0,0.06)",   "强烈卖出"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=clr, line_width=0, annotation_text=label,
                      annotation_position="right", annotation_font_size=10,
                      annotation_font_color="#555")

    # 评分走势线
    fig.add_trace(go.Scatter(
        x=dates, y=scores,
        mode="lines+markers",
        name="综合评分",
        line=dict(color="#58A6FF", width=2.5),
        marker=dict(size=6, color=scores,
                    colorscale=[[0,"#D50000"],[0.5,"#FFD600"],[1,"#00C853"]],
                    showscale=False),
        hovertemplate="%{x}<br>评分: %{y:.1f}<extra></extra>",
    ))

    # 各维度折线（可选显示）
    for key, color, name in [
        ("technical_score",   "#2196F3", "技术面"),
        ("fundamental_score", "#4CAF50", "基本面"),
        ("sentiment_score",   "#CE93D8", "情感面"),
    ]:
        vals = [r.get(key) for r in records]
        if any(v is not None for v in vals):
            fig.add_trace(go.Scatter(
                x=dates, y=vals, name=name,
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.7,
                visible="legendonly",
            ))

    fig.update_layout(**_base_layout(
        title=f"<b>{stock_name} — 历史评分趋势</b>",
        yaxis=dict(range=[0, 100], gridcolor=COLORS["grid"]),
        height=350,
    ))
    return fig


def create_signal_bar(signals: list) -> go.Figure:
    """信号条形图"""
    names  = [s["name"]  for s in signals]
    values = []
    colors = []
    for s in signals:
        sig = s["signal"]
        if sig == "buy":
            values.append(1);  colors.append(COLORS["up"])
        elif sig == "sell":
            values.append(-1); colors.append(COLORS["down"])
        else:
            values.append(0);  colors.append("#607D8B")

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[s["desc"] for s in signals],
        textposition="outside",
        hovertext=[f"{s['name']}: {s['value']}<br>{s['desc']}" for s in signals],
    ))
    fig.update_layout(**_base_layout(
        title="<b>技术信号汇总</b>",
        xaxis=dict(range=[-1.5, 1.5], tickvals=[-1, 0, 1],
                   ticktext=["卖出", "中性", "买入"],
                   gridcolor=COLORS["grid"]),
        height=max(250, len(signals) * 40),
        margin=dict(l=80, r=120, t=60, b=40),
    ))
    return fig
