"""
A股智能分析系统 — 手机版
启动: streamlit run mobile.py --server.port 8502
"""
import sys, os, json, warnings
from datetime import datetime
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="A股分析·手机版",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

import config

# 注意：主 streamlit 进程不限制 numpy/MKL 线程数，否则会拖慢 plotly 渲染

# ── 启动时孤儿/0字节模型文件自动清理（每天最多一次）────────────
try:
    from models.online_learner import maybe_startup_cleanup
    _cleanup_result = maybe_startup_cleanup()
    if _cleanup_result and _cleanup_result.get("removed_count", 0) > 0:
        print(f"[startup cleanup] 已清理 {_cleanup_result['removed_count']} 个旧模型，"
              f"释放 {_cleanup_result.get('freed_mb', 0)} MB")
except Exception as _e:
    print(f"[startup cleanup] 跳过: {_e}")

WATCHLIST_FILE = config.WATCHLIST_FILE

# ══════════════════════════════════════════════════════
#  CSS — 手机优先
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
* { box-sizing: border-box; }
.stApp { background:#0D1117; color:#E6EDF3;
         font-family:'Microsoft YaHei',Arial,sans-serif; }
.block-container { padding: 0.4rem 0.6rem 2rem !important; max-width:100% !important; }
/* 压缩所有元素间距 */
.element-container { margin-bottom: 0 !important; }
[data-testid="stMarkdownContainer"] { margin: 0 !important; }

/* 隐藏杂项 */
#MainMenu, footer, header,
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display:none !important; }

/* 全局文字 —— 不包含 span，避免覆盖彩色数值 */
p, label, li { color:#E6EDF3 !important; }
h1,h2,h3,h4 { color:#E6EDF3 !important; }
.stCaption, [data-testid="stCaptionContainer"] p { color:#ADBAC7 !important; }
[data-testid="stWidgetLabel"] { color:#E6EDF3 !important; font-weight:500 !important; }
.stCheckbox label, .stCheckbox span { color:#E6EDF3 !important; }
.streamlit-expanderHeader { color:#E6EDF3 !important; font-weight:600 !important; }
/* Streamlit 内部 span 文字（不影响彩色 mval） */
.stMarkdown p, [data-testid="stText"] { color:#E6EDF3 !important; }

/* 彩色数值 class —— 优先级高于全局继承 */
.mval { font-size:1.05em; font-weight:700; }
.mlbl { color:#ADBAC7 !important; font-size:0.88em; }
.c-red    { color:#EF5350 !important; }
.c-green  { color:#22c55e !important; }
.c-yellow { color:#FFA726 !important; }
.c-white  { color:#E6EDF3 !important; }
.c-gray   { color:#ADBAC7 !important; }
.c-blue   { color:#58A6FF !important; }

/* 自选股紧凑行 */
.wl-row   { padding: 2px 0; line-height: 1.5; }
.wl-name  { font-size: 0.98em; font-weight: 700; color: #E6EDF3; }
.wl-code  { font-size: 0.75em; color: #8B949E; }
.wl-score { font-size: 0.88em; font-weight: 700; }
.wl-tag   { font-size: 0.72em; color: #8B949E; }
/* 全局压缩 block 间距 */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlock"] > div { margin-bottom: 0 !important; }
[data-testid="stHorizontalBlock"] { gap: 6px !important; margin-bottom: 4px !important; align-items: center !important; }
/* 旧卡片兼容 */
.stock-card {
    background:#161B22; border:1px solid #30363D; border-radius:12px;
    padding:14px 16px; margin-bottom:10px;
    display:flex; justify-content:space-between; align-items:center;
}
.stock-name { font-size:1.05em; font-weight:700; color:#E6EDF3; }
.stock-code { font-size:0.78em; color:#ADBAC7; margin-top:2px; }
.stock-score { font-size:1.6em; font-weight:700; text-align:right; }
.stock-rating { font-size:0.75em; color:#ADBAC7; text-align:right; }

/* 评分大卡 */
.big-score {
    background:#161B22; border:1px solid #30363D; border-radius:12px;
    padding:20px; text-align:center; margin-bottom:12px;
}
.big-score .num { font-size:3.5em; font-weight:700; line-height:1; }
.big-score .label { font-size:1em; color:#ADBAC7; margin-top:6px; }

/* 小评分卡（2列） */
.score-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px; }
.mini-card {
    background:#161B22; border:1px solid #30363D; border-radius:10px;
    padding:12px; text-align:center;
}
.mini-card .val { font-size:1.6em; font-weight:700; }
.mini-card .lbl { font-size:0.72em; color:#ADBAC7; margin-top:3px; }

/* 指标行 */
.metric-row {
    background:#161B22; border:1px solid #30363D; border-radius:10px;
    padding:12px 14px; margin-bottom:8px;
    display:flex; justify-content:space-between; align-items:center;
}
.metric-row .mlbl { color:#ADBAC7; font-size:0.88em; }
.metric-row .mval { font-size:1.05em; font-weight:700; }

/* 普通按钮（股票列表分析按钮等） */
.stButton > button {
    background:#1F6FEB !important; color:#fff !important;
    border:none !important; border-radius:6px !important;
    min-height:34px !important;
    font-size:0.88em !important; font-weight:600 !important;
    width:100% !important; padding:4px 6px !important;
    white-space:nowrap !important; overflow:visible !important;
}
.stButton > button[kind="secondary"] {
    background:#21262D !important; color:#E6EDF3 !important;
    border:1px solid #444C56 !important;
}

/* 输入框 */
[data-testid="stTextInput"] input {
    background:#21262D !important; color:#E6EDF3 !important;
    border:1px solid #444C56 !important; border-radius:10px !important;
    font-size:16px !important; min-height:48px !important; padding:0 12px !important;
}
[data-testid="stTextInput"] input::placeholder { color:#768390 !important; }
[data-testid="stFormSubmitButton"] > button {
    background:#1F6FEB !important; color:#fff !important;
    border:none !important; border-radius:10px !important;
    min-height:44px !important; font-size:1em !important; width:100% !important;
}

/* 表单 */
[data-testid="stForm"] { border:none !important; padding:0 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background:#161B22 !important; border-radius:10px; padding:4px;
    gap:4px;
}
.stTabs [data-baseweb="tab"] {
    color:#ADBAC7 !important; font-size:0.95em !important;
    border-radius:8px !important; padding:10px 8px !important;
    min-height:44px !important;
}
.stTabs [aria-selected="true"] {
    background:#21262D !important; color:#58A6FF !important; font-weight:600 !important;
}

/* slider */
[data-testid="stSlider"] div { color:#E6EDF3 !important; }

/* selectbox 下拉框 */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background:#21262D !important; color:#E6EDF3 !important;
    border:1px solid #444C56 !important; border-radius:8px !important;
}
[data-baseweb="popover"] [role="listbox"] {
    background:#21262D !important;
}
[data-baseweb="popover"] [role="option"] {
    background:#21262D !important; color:#E6EDF3 !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background:#30363D !important; color:#fff !important;
}

/* checkbox */
[data-testid="stCheckbox"] { transform:scale(1.2); transform-origin:left; }

/* expander */
[data-testid="stExpander"] { border:1px solid #30363D !important; border-radius:10px !important; }

/* 分隔线 */
hr { border:none; border-top:1px solid #21262D; margin:10px 0; }

/* ── 分析页顶部 Hero ── */
.hero-card-top {
    background: linear-gradient(135deg, #161B22 0%, #1C2128 100%);
    border: 1px solid #30363D; border-radius: 14px;
    padding: 16px 18px 12px; margin-bottom: 8px;
}
.hero-name {
    font-size: 1.6em; font-weight: 700; color: #E6EDF3;
    line-height: 1.2; margin-bottom: 4px;
}
.hero-code {
    font-size: 0.78em; color: #8B949E;
}
/* 价格行容器：flex 并排 */
.hero-prices {
    display: flex; gap: 10px; margin-bottom: 10px;
}
/* 价格块 */
.hero-cur {
    flex: 1; background: #161B22; border: 1px solid #30363D;
    border-radius: 12px; padding: 12px 14px;
}
.hero-lbl {
    font-size: 0.75em; color: #8B949E !important; margin-bottom: 5px;
}
.hero-price {
    font-size: 1.9em; font-weight: 700; color: #E6EDF3; line-height: 1.1;
}
/* 价格颜色优先于基础色（A股：涨红跌绿） */
.hero-price.c-red    { color: #EF5350 !important; }
.hero-price.c-green  { color: #22c55e !important; }
.hero-price.c-white  { color: #E6EDF3 !important; }
.hero-pred-val {
    font-size: 1.9em; font-weight: 700; line-height: 1.1;
}
.hero-pred-pct {
    font-size: 0.9em; font-weight: 600; margin-top: 3px;
}

/* 新闻条目 */
.news-item {
    background:#161B22; border:1px solid #30363D; border-radius:10px;
    padding:12px 14px; margin-bottom:8px;
}
.news-title { font-size:0.92em; font-weight:600; color:#E6EDF3; line-height:1.4; }
.news-meta  { font-size:0.75em; color:#ADBAC7; margin-top:4px; }

/* 应用标题 */
.app-title {
    font-size: 1.25em; font-weight: 700; color: #E6EDF3;
    padding: 4px 0 8px; margin-bottom: 2px;
}

/* Radio 导航 Tab 样式 */
[data-testid="stRadio"] {
    background: #161B22 !important;
    border-radius: 10px !important;
    padding: 3px 4px !important;
    border-bottom: 1px solid #30363D;
    margin-bottom: 10px !important;
}
[data-testid="stRadio"] > div { gap: 0 !important; }
[data-testid="stRadio"] label {
    flex: 1 !important; text-align: center !important;
    padding: 6px 2px !important; border-radius: 6px !important;
    color: #ADBAC7 !important; font-size: 0.88em !important;
    font-weight: 500 !important; cursor: pointer !important;
    white-space: nowrap !important;
}
[data-testid="stRadio"] label > div:first-child { display: none !important; }
[data-testid="stRadio"] [aria-checked="true"] ~ * { color: #E6EDF3 !important; }
[data-testid="stRadio"] label:has(input:checked) {
    background: #21262D !important; color: #E6EDF3 !important;
    font-weight: 700 !important; border-bottom: 2px solid #EF5350 !important;
    border-radius: 6px 6px 0 0 !important;
}

/* 自定义底部导航 */
.nav-bar {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 999;
    background: #161B22; border-top: 1px solid #30363D;
    display: flex; height: 56px;
}
.nav-btn {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: #8B949E; font-size: 0.7em; cursor: pointer;
    background: none; border: none; gap: 2px;
}
.nav-btn.active { color: #58A6FF; }
.nav-icon { font-size: 1.4em; line-height: 1; }
</style>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════
def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return []

def save_watchlist(wl):
    os.makedirs(os.path.dirname(WATCHLIST_FILE), exist_ok=True)
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)

def normalize_symbol(s):
    s = s.upper().strip()
    for suffix in (".SS",".SZ",".SH",".BJ","SH","SZ"):
        s = s.replace(suffix,"")
    return s.lstrip("0").zfill(6) if s.isdigit() else s

def score_color(v):
    if v >= 80: return "c-green"
    if v >= 65: return "c-green"
    if v >= 45: return "c-yellow"
    if v >= 30: return "c-red"
    return "c-red"

def score_hex(v):
    if v >= 80: return "#22c55e"
    if v >= 65: return "#4CAF50"
    if v >= 45: return "#FFA726"
    if v >= 30: return "#EF5350"
    return "#f44336"

def pct_cls(v):
    # A股：涨红跌绿
    return "c-red" if v > 0 else ("c-green" if v < 0 else "c-gray")

def pct_color(v):
    return "#EF5350" if v > 0 else ("#22c55e" if v < 0 else "#ADBAC7")

def hex_to_cls(color):
    m = {"#EF5350":"c-red","#f44336":"c-red","#22c55e":"c-green",
         "#4CAF50":"c-green","#FFA726":"c-yellow","#ADBAC7":"c-gray",
         "#E6EDF3":"c-white","#58A6FF":"c-blue"}
    return m.get(color, "c-white")

def fmt(val, spec=".1f", suffix=""):
    if val is None: return "—"
    try:    return f"{float(val):{spec}}{suffix}"
    except: return str(val)

def metric_row(label, value, color="#E6EDF3", sub=""):
    cls = hex_to_cls(color) if color.startswith("#") else color
    sub_html = f'<em class="c-gray" style="font-size:0.8em;margin-left:6px">{sub}</em>' if sub else ""
    return (f'<div class="metric-row">'
            f'<div class="mlbl">{label}</div>'
            f'<div class="mval {cls}">{value}{sub_html}</div>'
            f'</div>')


# ══════════════════════════════════════════════════════
#  快速分析（XGBoost only）
# ══════════════════════════════════════════════════════
def fast_xgb(df, days):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor
    from models.xgboost_model import _build_features
    feat_df = _build_features(df)
    future_ret = (df["close"].shift(-days) / df["close"] - 1).reindex(feat_df.index)
    mask = future_ret.notna()
    feat_df = feat_df[mask]; target = future_ret[mask]
    if len(feat_df) < 60: return None
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values); y = target.values
    split = int(len(X)*0.85)
    # n_jobs 跟随 config，避免 ProcessPool 嵌套时过度订阅
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=config.XGB_N_JOBS, early_stopping_rounds=20, eval_metric="rmse")
    model.fit(X[:split], y[:split], eval_set=[(X[split:], y[split:])], verbose=False)
    pred_ret = model.predict(X[split:])
    close_val = df["close"].reindex(feat_df.index)[mask][split:]
    rmse = float(np.sqrt(mean_squared_error(
        close_val.values*(1+y[split:]), close_val.values*(1+pred_ret))))
    last_feat = feat_df.iloc[[-1]]
    ret_end = float(model.predict(scaler.transform(last_feat.values))[0])
    last_price = float(df["close"].iloc[-1])
    end_price = last_price*(1+ret_end)
    predictions = np.linspace(last_price, end_price, days+1)[1:]
    noise = np.random.normal(0, rmse*0.2, len(predictions))
    predictions = np.clip(predictions+noise, last_price*0.6, last_price*1.6)
    import pandas as pd
    dates = pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=days, freq="B")
    # 用 RMSE 生成置信区间（图表需要）
    sigma = np.linspace(rmse * 0.5, rmse * 1.2, len(predictions))
    lower = (predictions - sigma).tolist()
    upper = (predictions + sigma).tolist()
    return {"predictions": predictions.tolist(),
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "last_price": last_price,
            "pct_change": float((end_price-last_price)/last_price*100),
            "val_rmse": rmse,
            "lower_bound": lower,
            "upper_bound": upper}

def run_analysis(symbol, years=2, do_backtest=False, fast_mode=True):
    r = {"symbol": symbol, "ok": False}
    prog = st.progress(0, "获取行情...")
    try:
        from data.fetcher import fetch_stock_data, fetch_stock_info, get_stock_name, enrich_with_market_data
        df = fetch_stock_data(symbol, years=years)
        info = fetch_stock_info(symbol)
        name = get_stock_name(symbol)
        r.update(stock_name=name, last_price=float(df["close"].iloc[-1]), trading_days=len(df))

        prog.progress(15, "市场数据...")
        try: df = enrich_with_market_data(df, symbol)
        except: pass

        prog.progress(25, "获取新闻...")
        try:
            from data.news import fetch_news, get_news_texts
            news_list  = fetch_news(symbol)
            news_texts = get_news_texts(news_list)
        except: news_list = news_texts = []
        r["news_list"] = news_list

        prog.progress(38, "技术指标...")
        from analysis.technical import compute_indicators, generate_signals, compute_support_resistance
        df = compute_indicators(df)
        r.update(df=df, signals=generate_signals(df), sr=compute_support_resistance(df))

        prog.progress(50, "基本面...")
        try:
            from analysis.fundamental import fetch_and_analyze
            r["fundamental"] = fetch_and_analyze(symbol, info)
        except: r["fundamental"] = {"score":50,"details":[]}

        prog.progress(60, "情感分析...")
        try:
            from analysis.sentiment import analyze_sentiment
            r["sentiment"] = analyze_sentiment(news_texts)
        except: r["sentiment"] = {"score":0.5,"label":"中性","news_count":0}

        if fast_mode:
            prog.progress(70, "预测模型（快速，约30秒）...")
            try:
                short_pred = fast_xgb(df, config.SHORT_TERM_DAYS)
                long_pred  = fast_xgb(df, config.LONG_TERM_DAYS)
            except: short_pred = long_pred = None
        else:
            prog.progress(70, "预测模型（完整，约2分钟）...")
            try:
                from models.ensemble import EnsemblePredictor
                pred = EnsemblePredictor()
                pred.train(df)
                short_pred = pred.predict_short(df)
                long_pred  = pred.predict_long(df)
            except: short_pred = long_pred = None
        r.update(short_pred=short_pred, long_pred=long_pred)

        prog.progress(88, "综合评分...")
        from models.scorer import compute_score
        r["score_result"] = compute_score(r["signals"], r["fundamental"], r["sentiment"], short_pred, long_pred)

        prog.progress(92, "保存历史...")
        try:
            from data.score_history import save_score, load_history
            save_score(symbol, name, r["score_result"], short_pred, long_pred)
            r["history"] = load_history(symbol)
        except: r["history"] = []

        if do_backtest:
            prog.progress(95, "回测...")
            try:
                from backtest.engine import run_backtest
                r["backtest"] = run_backtest(df)
            except: r["backtest"] = None

        r["ok"] = True
        r["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        r["error"] = str(e)
    prog.progress(100, "完成")
    prog.empty()
    return r


# ══════════════════════════════════════════════════════
#  Session State
# ══════════════════════════════════════════════════════
if "watchlist"  not in st.session_state: st.session_state.watchlist  = load_watchlist()
if "selected"   not in st.session_state: st.session_state.selected   = None
if "results"    not in st.session_state: st.session_state.results    = {}
if "page"       not in st.session_state: st.session_state.page       = "watchlist"
if "years"      not in st.session_state: st.session_state.years      = 2
if "do_bt"      not in st.session_state: st.session_state.do_bt      = False
if "fast_mode"  not in st.session_state: st.session_state.fast_mode  = True

wl = st.session_state.watchlist
page = st.session_state.page

# ══════════════════════════════════════════════════════
#  顶部标题 + Tab 导航
# ══════════════════════════════════════════════════════
st.markdown('<div class="app-title">📈 A股智能分析预测</div>', unsafe_allow_html=True)

_NAV_LABELS = {"📋 自选股": "watchlist", "📊 分析": "analyze", "➕ 添加股票": "add"}
_nav_sel = st.radio(
    "nav", list(_NAV_LABELS.keys()),
    index=list(_NAV_LABELS.values()).index(page),
    horizontal=True, label_visibility="collapsed",
)
if _NAV_LABELS[_nav_sel] != page:
    st.session_state.page = _NAV_LABELS[_nav_sel]
    st.rerun()


# ──────────────────────────────────────────────────────
#  Page 1: 自选股列表
# ──────────────────────────────────────────────────────
if page == "watchlist":
    if not wl:
        st.info("还没有自选股，点「＋添加」添加")
    else:
        st.caption(f"共 {len(wl)} 只")
        for item in wl:
            sym  = item["symbol"]
            name = item.get("name", sym)
            cached = st.session_state.results.get(sym)
            if cached and cached.get("ok"):
                sr     = cached["score_result"]
                score  = sr["total_score"]
                rating = sr["rating"]
                clr    = score_color(score)
                sp     = cached.get("short_pred")
                pct    = sp["pct_change"] if sp else None
            else:
                score = None; rating = "未分析"; clr = "c-gray"; pct = None

            score_str = f"{score:.0f}" if score is not None else "—"
            pct_str = ""
            if pct is not None:
                arrow = "▲" if pct >= 0 else "▼"
                pct_str = f'<span class="{pct_cls(pct)}"> {arrow}{abs(pct):.1f}%</span>'

            cl, cr = st.columns([7, 2])
            cl.markdown(
                f'<div class="wl-row">'
                f'<span class="wl-name">{name}</span>'
                f'<span class="wl-code"> {sym}</span>'
                f'<span class="wl-score {clr}"> · {score_str}</span>'
                f'<span class="wl-tag"> {rating}</span>'
                f'{pct_str}'
                f'</div>',
                unsafe_allow_html=True)
            if cr.button("分析", key=f"anl_{sym}", use_container_width=True, type="primary"):
                st.session_state.selected = sym
                st.session_state._do_analyze = True
                st.session_state.page = "analyze"
                st.rerun()


# ──────────────────────────────────────────────────────
#  Page 2: 分析报告
# ──────────────────────────────────────────────────────
elif page == "analyze":
    sym = st.session_state.selected

    # 触发分析
    if st.session_state.pop("_do_analyze", False) and sym:
        fm = st.session_state.fast_mode
        eta = "约30秒" if fm else "约2分钟"
        with st.spinner(f"分析 {sym} 中，{eta}..."):
            st.session_state.results[sym] = run_analysis(
                sym,
                years      = st.session_state.years,
                do_backtest= st.session_state.do_bt,
                fast_mode  = fm,
            )
        st.rerun()

    if not sym:
        st.info("👆 在「自选股」页点击「分析」按钮")
    else:
        r = st.session_state.results.get(sym)
        if not r:
            st.info(f"已选 **{sym}**，点击上方「分析」按钮开始")
        elif not r.get("ok"):
            st.error(f"分析失败：{r.get('error','未知')}")
        else:
            sr   = r["score_result"]
            sp   = r.get("short_pred")
            lp   = r.get("long_pred")
            fund = r["fundamental"]
            sent = r["sentiment"]

            # ── 顶部 Hero 卡片：股票名 + 副标题 ──
            st.markdown(
                f'<div class="hero-card-top">'
                f'<div class="hero-name">{r["stock_name"]}</div>'
                f'<div class="hero-code">{sym} &nbsp;·&nbsp; {r.get("ts","")[:16]} &nbsp;·&nbsp; {r["trading_days"]}个交易日</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── 今日涨跌（用于当前价颜色）──
            try:
                _df2 = r["df"]
                _today_pct = float((_df2["close"].iloc[-1] / _df2["close"].iloc[-2] - 1) * 100)
            except:
                _today_pct = 0
            _cur_cls = pct_cls(_today_pct) if _today_pct != 0 else "c-white"

            # ── 价格行：当前价 + 预测价（单个 HTML 块，flex 并排）──
            if sp:
                _pct   = sp["pct_change"]
                _arrow = "▲" if _pct >= 0 else "▼"
                _pc    = pct_cls(_pct)
                pred_block = (
                    f'<div class="hero-cur">'
                    f'<div class="hero-lbl">10天预测</div>'
                    f'<div class="hero-price {_pc}">¥{sp["predictions"][-1]:.2f}</div>'
                    f'<div class="hero-pred-pct {_pc}">{_arrow}{abs(_pct):.2f}%</div>'
                    f'</div>'
                )
            else:
                pred_block = ""
            st.markdown(
                f'<div class="hero-prices">'
                f'<div class="hero-cur">'
                f'<div class="hero-lbl">当前价格</div>'
                f'<div class="hero-price {_cur_cls}">¥{r["last_price"]:.2f}</div>'
                f'<div class="hero-pred-pct {_cur_cls}">{"▲" if _today_pct>0 else "▼" if _today_pct<0 else ""}{abs(_today_pct):.2f}% 今日</div>'
                f'</div>'
                f'{pred_block}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── 1. 综合评分大卡 ──
            total = sr["total_score"]
            clr   = score_color(total)
            st.markdown(
                f'<div class="big-score">'
                f'<div class="num {clr}">{total:.1f}</div>'
                f'<div class="label">{sr["rating"]}</div>'
                f'</div>', unsafe_allow_html=True)

            # ── 2. 4个分项评分（2×2网格）──
            st.markdown('<div class="score-grid">' +
                "".join([
                    f'<div class="mini-card">'
                    f'<div class="val {score_color(v)}">{v:.1f}</div>'
                    f'<div class="lbl">{lbl}</div></div>'
                    for v, lbl in [
                        (sr["technical_score"],   "技术面 40%"),
                        (sr["fundamental_score"], "基本面 30%"),
                        (sr["sentiment_score"],   "情感面 15%"),
                        (sr["prediction_score"],  "预测面 15%"),
                    ]
                ]) + '</div>', unsafe_allow_html=True)

            # ── 3. 风险管理（默认展开）──
            with st.expander("📌 风险管理", expanded=True):
                pos_cls  = score_color(total)
                sent_cls = {"正面":"c-red","负面":"c-green"}.get(sent["label"],"c-gray")
                st.markdown(
                    metric_row("建议止损", f'¥{sr.get("stop_loss","—")}', "c-red", "跌破须止损") +
                    metric_row("目标价位", f'¥{sr.get("target_price","—")}', "c-green") +
                    metric_row("建议仓位", sr.get("position_pct","—"), pos_cls) +
                    metric_row("新闻情感", sent["label"], sent_cls,
                               f'{sent["score"]:.0%} · {sent.get("news_count",0)}条'),
                    unsafe_allow_html=True)

            # ── 图表通用配置 ──
            try:
                from visualization.charts import (
                    create_candlestick_chart, create_prediction_chart,
                    create_score_gauge, create_fundamental_radar,
                )
                _df   = r["df"]
                _name = r["stock_name"]
                _sp   = r.get("short_pred")
                _lp   = r.get("long_pred")
                _mob_legend = dict(
                    orientation="v", x=1.0, y=1.0,
                    xanchor="right", yanchor="top",
                    bgcolor="rgba(13,17,23,0.85)",
                    bordercolor="rgba(68,76,86,0.6)", borderwidth=1,
                    font=dict(size=9, color="#E6EDF3"),
                    itemwidth=30, tracegroupgap=2,
                )
            except:
                _df = _name = _sp = _lp = _mob_legend = None

            # ── 4. 综合评分仪表 ──
            if _mob_legend is not None:
                with st.expander("🎯 综合评分仪表"):
                    try:
                        fig_g = create_score_gauge(sr["total_score"], sr["rating"], sr["color"])
                        fig_g.update_layout(height=280, margin=dict(t=30,b=10,l=10,r=10))
                        st.plotly_chart(fig_g, use_container_width=True, key=f"m_g_{sym}")
                    except: st.caption("图表加载失败")

            # ── 5. 价格预测（数值）──
            if sp or lp:
                with st.expander("🔮 价格预测"):
                    if sp:
                        pct = sp["pct_change"]
                        arrow = "▲" if pct >= 0 else "▼"
                        st.markdown(metric_row("短期10天", f'¥{sp["predictions"][-1]:.2f}',
                                               pct_cls(pct), f'{arrow}{abs(pct):.2f}%'),
                                    unsafe_allow_html=True)
                    if lp:
                        pct = lp["pct_change"]
                        arrow = "▲" if pct >= 0 else "▼"
                        st.markdown(metric_row("中期30天", f'¥{lp["predictions"][-1]:.2f}',
                                               pct_cls(pct), f'{arrow}{abs(pct):.2f}%'),
                                    unsafe_allow_html=True)

            # ── 6. 价格预测走势图 ──
            if _df is not None and (_sp or _lp):
                with st.expander("📈 预测走势图"):
                    try:
                        fig_p = create_prediction_chart(_df, _sp, _lp, _name)
                        fig_p.update_layout(
                            height=300, margin=dict(t=30,b=30,l=45,r=8),
                            legend=_mob_legend, title=None,
                        )
                        st.plotly_chart(fig_p, use_container_width=True, key=f"m_p_{sym}")
                    except Exception as _pe:
                        st.caption(f"图表加载失败：{_pe}")

            # ── 7. 技术信号 ──
            sigs = r["signals"].get("signals", [])
            if sigs:
                with st.expander("📡 技术信号"):
                    for s in sigs:
                        icon    = {"buy":"🔴","sell":"🟢","neutral":"⚪"}[s["signal"]]
                        sig_txt = {"buy":"买入","sell":"卖出","neutral":"中性"}[s["signal"]]
                        sig_cls = {"buy":"c-red","sell":"c-green","neutral":"c-gray"}[s["signal"]]
                        st.markdown(
                            f'<div class="metric-row">'
                            f'<div class="mlbl">{icon} {s["name"]}</div>'
                            f'<div class="mval {sig_cls}">{s["value"]} '
                            f'<em style="font-size:0.8em">{sig_txt}</em>'
                            f'</div></div>', unsafe_allow_html=True)

            # ── 8. K线图 ──
            if _df is not None:
                with st.expander("📉 K线图"):
                    try:
                        fig_c = create_candlestick_chart(_df, _name, sym)
                        fig_c.update_layout(
                            height=420, margin=dict(t=10,b=30,l=45,r=8),
                            legend=_mob_legend, title=None,
                        )
                        st.plotly_chart(fig_c, use_container_width=True, key=f"m_c_{sym}")
                    except: st.caption("图表加载失败")

            # ── 9. 基本面 ──
            with st.expander("📋 基本面"):
                pe  = fund.get("pe")
                pb  = fund.get("pb")
                roe = fund.get("roe")
                pg  = fund.get("profit_growth") or 0
                rg  = fund.get("revenue_growth") or 0
                dy  = fund.get("dividend_yield") or 0
                pe_c  = ("c-green" if pe and pe<15 else
                         "c-yellow" if pe and pe<30 else "c-red") if pe else "c-gray"
                pb_c  = ("c-green" if pb and pb<1 else
                         "c-yellow" if pb and pb<3 else "c-red") if pb else "c-gray"
                roe_c = ("c-green" if roe and roe>15 else
                         "c-yellow" if roe and roe>8 else "c-red") if roe else "c-gray"
                pg_c  = "c-red" if pg>0 else ("c-green" if pg<0 else "c-gray")
                rg_c  = "c-red" if rg>0 else ("c-green" if rg<0 else "c-gray")
                dy_c  = "c-green" if dy>3 else ("c-yellow" if dy>1 else "c-red")
                rows = [
                    ("行业",     fund.get("industry","—"),   "c-white"),
                    ("市盈率PE", fmt(pe,".1f"),              pe_c),
                    ("市净率PB", fmt(pb,".2f"),              pb_c),
                    ("ROE",      fmt(roe,".1f","%"),         roe_c),
                    ("净利增长", fmt(pg,".1f","%"),          pg_c),
                    ("营收增长", fmt(rg,".1f","%"),          rg_c),
                    ("股息率",   fmt(dy,".2f","%"),          dy_c),
                    ("总市值",   fund.get("market_cap","—"), "c-white"),
                ]
                st.markdown("".join(metric_row(l,v,c) for l,v,c in rows), unsafe_allow_html=True)

            # ── 10. 基本面雷达 ──
            if _df is not None:
                with st.expander("🕸️ 基本面雷达"):
                    try:
                        fig_r = create_fundamental_radar(r["fundamental"], _name)
                        fig_r.update_layout(height=300, margin=dict(t=30,b=10,l=10,r=10), title=None)
                        st.plotly_chart(fig_r, use_container_width=True, key=f"m_r_{sym}")
                    except: st.caption("图表加载失败")

            # ── 11. 支撑阻力 ──
            sr_lv = r.get("sr", {})
            if sr_lv:
                with st.expander("📍 支撑 & 阻力"):
                    res = " / ".join(f"¥{x}" for x in sr_lv.get("resistance",[])[:3])
                    sup = " / ".join(f"¥{x}" for x in sr_lv.get("support",[])[:3])
                    st.markdown(
                        metric_row("当前价", f'¥{sr_lv.get("close",0):.2f}', "c-white") +
                        metric_row("阻力位", res, "c-red") +
                        metric_row("支撑位", sup, "c-green"),
                        unsafe_allow_html=True)

            # ── 12. 新闻 ──
            news = r.get("news_list", [])
            if news:
                with st.expander(f"📰 新闻 ({len(news)}条)"):
                    for item in news[:8]:
                        title   = item.get("title","").strip()
                        date_   = str(item.get("date",""))[:10]
                        source  = item.get("source","")
                        url_    = item.get("url","")
                        if not title: continue
                        link = f'<a href="{url_}" style="color:#58A6FF;font-size:0.8em">阅读原文</a>' if url_ else ""
                        st.markdown(
                            f'<div class="news-item">'
                            f'<div class="news-title">{title[:60]}</div>'
                            f'<div class="news-meta">{date_} · {source} {link}</div>'
                            f'</div>', unsafe_allow_html=True)

            st.divider()
            st.caption("⚠️ 仅供参考，不构成投资建议")


# ──────────────────────────────────────────────────────
#  Page 3: 添加股票
# ──────────────────────────────────────────────────────
elif page == "add":
    st.markdown('<div style="font-size:1em;font-weight:700;margin-bottom:4px">➕ 添加股票 <span style="font-size:0.72em;color:#8B949E;font-weight:400">· 空格/逗号/换行分隔可批量</span></div>', unsafe_allow_html=True)
    with st.form("add_form", clear_on_submit=True):
        codes_input = st.text_area("股票代码", placeholder="600519 000858 601318",
                                   height=68, label_visibility="collapsed")
        if st.form_submit_button("添加", use_container_width=True):
            import re
            raw_codes = re.split(r"[\s,，、;；\n]+", codes_input.strip())
            raw_codes = [c.strip() for c in raw_codes if c.strip()]
            if not raw_codes:
                st.error("请输入股票代码")
            else:
                from data.fetcher import fetch_stock_data, get_stock_name
                added, skipped, failed = [], [], []
                prog = st.progress(0, f"共 {len(raw_codes)} 只，验证中...")
                for i, raw in enumerate(raw_codes):
                    sym = normalize_symbol(raw)
                    prog.progress((i+1)/len(raw_codes), f"验证 {sym}...")
                    if any(w["symbol"]==sym for w in wl):
                        skipped.append(sym)
                        continue
                    try:
                        df_test = fetch_stock_data(sym, years=1)
                        if df_test is None or df_test.empty: raise ValueError()
                        name = get_stock_name(sym)
                        wl.append({"symbol":sym,"name":name})
                        added.append(f"{name}({sym})")
                    except:
                        failed.append(sym)
                prog.empty()
                if added:
                    save_watchlist(wl)
                    st.session_state.watchlist = wl
                    st.success(f"✅ 已添加 {len(added)} 只：{'、'.join(added)}")
                if skipped:
                    st.warning(f"⚠️ 已存在 {len(skipped)} 只：{'、'.join(skipped)}")
                if failed:
                    st.error(f"❌ 无效代码 {len(failed)} 只：{'、'.join(failed)}")
                if added:
                    import time; time.sleep(1.5)
                    st.session_state.page = "watchlist"
                    st.rerun()

    st.markdown('<div style="font-size:0.85em;font-weight:600;color:#ADBAC7;margin:10px 0 4px">📋 自选股列表</div>', unsafe_allow_html=True)
    if not wl:
        st.caption("暂无自选股")
    # 列表一次性渲染为 HTML
    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:7px 0;border-bottom:1px solid #21262D">'
        f'<span style="font-size:0.95em;color:#E6EDF3;font-weight:600">{item.get("name",item["symbol"])}'
        f'<span style="color:#8B949E;font-size:0.78em;font-weight:400;margin-left:6px">{item["symbol"]}</span>'
        f'</span></div>'
        for item in wl
    )
    st.markdown(f'<div style="margin-bottom:10px">{rows_html}</div>', unsafe_allow_html=True)

    # 删除：下拉选择 + 按钮
    if wl:
        del_labels = [f'{w.get("name",w["symbol"])}（{w["symbol"]}）' for w in wl]
        del_sel = st.selectbox("选择要删除的股票", del_labels, label_visibility="collapsed")
        if st.button("删除选中", type="secondary", use_container_width=True):
            idx = del_labels.index(del_sel)
            sym_del = wl[idx]["symbol"]
            wl.pop(idx)
            save_watchlist(wl)
            st.session_state.watchlist = wl
            if st.session_state.selected == sym_del:
                st.session_state.selected = None
            st.rerun()

    st.markdown('<div style="font-size:0.85em;font-weight:600;color:#ADBAC7;margin:10px 0 4px">⚙️ 分析参数</div>', unsafe_allow_html=True)
    st.session_state.years = st.slider("历史年数", 1, 5, st.session_state.years)
    c1, c2 = st.columns(2)
    st.session_state.fast_mode = c1.checkbox("⚡ 快速模式", value=st.session_state.fast_mode,
        help="仅XGBoost约30秒；关闭后全模型约2分钟")
    st.session_state.do_bt = c2.checkbox("📊 执行回测", value=st.session_state.do_bt)
    mode_txt = "⚡ 快速" if st.session_state.fast_mode else "🎯 完整"
    st.caption(f"{mode_txt} · {st.session_state.years}年 · {'含回测' if st.session_state.do_bt else '不回测'}")
