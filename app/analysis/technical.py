"""
技术指标计算模块（纯 numpy/pandas 实现，无需 pandas-ta）
支持 Python 3.10+，兼容 Python 3.14
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  基础指标函数
# ─────────────────────────────────────────────────────────────

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=n - 1, adjust=False, min_periods=n).mean()
    avg_l  = loss.ewm(com=n - 1, adjust=False, min_periods=n).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = _ema(series, fast)
    ema_slow   = _ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist       = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger(series: pd.Series, n=20, k=2):
    mid   = _sma(series, n)
    std   = series.rolling(n, min_periods=n).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def _kdj(high: pd.Series, low: pd.Series, close: pd.Series,
         n=9, m1=3, m2=3):
    lowest_low   = low.rolling(n, min_periods=n).min()
    highest_high = high.rolling(n, min_periods=n).max()
    denom  = (highest_high - lowest_low).replace(0, np.nan)
    rsv    = (close - lowest_low) / denom * 100
    K      = rsv.ewm(com=m1 - 1, adjust=False, min_periods=1).mean()
    D      = K.ewm(com=m2 - 1, adjust=False, min_periods=1).mean()
    J      = 3 * K - 2 * D
    return K, D, J


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False, min_periods=n).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ─────────────────────────────────────────────────────────────
#  主计算函数
# ─────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算全套技术指标，返回带指标列的 DataFrame
    纯 numpy/pandas 实现，不依赖 pandas-ta
    """
    d = df.copy()
    c = d["close"]
    h = d.get("high",  c)
    l = d.get("low",   c)
    v = d.get("volume", pd.Series(np.ones(len(c)), index=c.index))

    # ── 均线 ──────────────────────────────────────────────
    for n in [5, 10, 20, 60]:
        d[f"sma_{n}"] = _sma(c, n)
    d["ema_12"] = _ema(c, 12)
    d["ema_26"] = _ema(c, 26)

    # ── MACD ──────────────────────────────────────────────
    d["macd"], d["macd_signal"], d["macd_hist"] = _macd(c)

    # ── RSI ───────────────────────────────────────────────
    d["rsi_14"] = _rsi(c, 14)
    d["rsi_6"]  = _rsi(c, 6)

    # ── 布林带 ────────────────────────────────────────────
    d["bb_upper"], d["bb_mid"], d["bb_lower"] = _bollinger(c, 20, 2)
    bw          = d["bb_upper"] - d["bb_lower"]
    d["bb_pct"] = np.where(bw > 0, (c - d["bb_lower"]) / bw, 0.5)

    # ── KDJ ───────────────────────────────────────────────
    d["kdj_k"], d["kdj_d"], d["kdj_j"] = _kdj(h, l, c)

    # ── ATR ───────────────────────────────────────────────
    d["atr_14"] = _atr(h, l, c, 14)

    # ── OBV ───────────────────────────────────────────────
    d["obv"] = _obv(c, v)

    # ── 成交量均线 & 量比 ─────────────────────────────────
    d["vol_ma_5"]  = _sma(v, 5)
    d["vol_ma_20"] = _sma(v, 20)
    d["vol_ratio"] = np.where(d["vol_ma_5"] > 0, v / d["vol_ma_5"], 1.0)

    # ── 价格动量 ──────────────────────────────────────────
    for n in [1, 3, 5, 10, 20]:
        d[f"momentum_{n}"] = c.pct_change(n) * 100

    # ── 波动率 ────────────────────────────────────────────
    d["volatility_10"] = c.pct_change().rolling(10).std() * 100
    d["volatility_20"] = c.pct_change().rolling(20).std() * 100

    return d


# ─────────────────────────────────────────────────────────────
#  信号生成
# ─────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 2:
        return {"signals": [], "buy_count": 0, "sell_count": 0, "neutral_count": 0}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    def add(name, value, sig, desc):
        signals.append({"name": name, "value": value, "signal": sig, "desc": desc})

    # RSI
    rsi = last.get("rsi_14", np.nan)
    if not _isnan(rsi):
        if rsi < 30:
            add("RSI(14)", f"{rsi:.1f}", "buy",  "超卖区域，可能反弹")
        elif rsi > 70:
            add("RSI(14)", f"{rsi:.1f}", "sell", "超买区域，注意回调")
        else:
            add("RSI(14)", f"{rsi:.1f}", "neutral", f"中性区间")

    # MACD 金叉/死叉
    mc, ms = last.get("macd", np.nan), last.get("macd_signal", np.nan)
    mp, sp = prev.get("macd", np.nan), prev.get("macd_signal", np.nan)
    if not any(_isnan(x) for x in [mc, ms, mp, sp]):
        if mc > ms and mp <= sp:
            add("MACD", f"{mc:.4f}", "buy",  "MACD金叉，看涨信号")
        elif mc < ms and mp >= sp:
            add("MACD", f"{mc:.4f}", "sell", "MACD死叉，看跌信号")
        elif mc > ms:
            add("MACD", f"{mc:.4f}", "buy",  "MACD在信号线上方")
        else:
            add("MACD", f"{mc:.4f}", "sell", "MACD在信号线下方")

    # MA20 / MA60
    close = last["close"]
    for period, label in [(20, "MA20"), (60, "MA60")]:
        ma = last.get(f"sma_{period}", np.nan)
        if not _isnan(ma):
            if close > ma:
                add(label, f"{ma:.2f}", "buy",  f"价格在{label}上方，趋势向上")
            else:
                add(label, f"{ma:.2f}", "sell", f"价格在{label}下方，趋势向下")

    # 布林带
    bb_pct = last.get("bb_pct", np.nan)
    if not _isnan(bb_pct):
        if bb_pct > 0.95:
            add("布林带", f"{bb_pct:.2%}", "sell", "价格触及上轨，压力较大")
        elif bb_pct < 0.05:
            add("布林带", f"{bb_pct:.2%}", "buy",  "价格触及下轨，支撑较强")
        else:
            add("布林带", f"{bb_pct:.2%}", "neutral", "价格在布林带中部")

    # KDJ
    kk, kd = last.get("kdj_k", np.nan), last.get("kdj_d", np.nan)
    pk, pd_ = prev.get("kdj_k", np.nan), prev.get("kdj_d", np.nan)
    if not any(_isnan(x) for x in [kk, kd, pk, pd_]):
        if kk > kd and pk <= pd_ and kk < 80:
            add("KDJ", f"K:{kk:.1f} D:{kd:.1f}", "buy",  "KDJ金叉，短期看涨")
        elif kk < kd and pk >= pd_ and kk > 20:
            add("KDJ", f"K:{kk:.1f} D:{kd:.1f}", "sell", "KDJ死叉，短期看跌")
        elif kk < 20:
            add("KDJ", f"K:{kk:.1f}", "buy",  "KDJ超卖")
        elif kk > 80:
            add("KDJ", f"K:{kk:.1f}", "sell", "KDJ超买")
        else:
            add("KDJ", f"K:{kk:.1f} D:{kd:.1f}", "neutral", "中性")

    # 成交量
    vr  = last.get("vol_ratio", np.nan)
    pct = last.get("pct_change", 0) if "pct_change" in last.index else 0
    if not _isnan(vr):
        if vr > 2.0 and pct > 0:
            add("成交量", f"量比{vr:.1f}x", "buy",  "放量上涨，资金介入")
        elif vr > 2.0 and pct < 0:
            add("成交量", f"量比{vr:.1f}x", "sell", "放量下跌，警惕出货")
        elif vr < 0.5:
            add("成交量", f"量比{vr:.1f}x", "neutral", "缩量，方向待确认")
        else:
            add("成交量", f"量比{vr:.1f}x", "neutral", "成交量正常")

    buy_cnt     = sum(1 for s in signals if s["signal"] == "buy")
    sell_cnt    = sum(1 for s in signals if s["signal"] == "sell")
    neutral_cnt = sum(1 for s in signals if s["signal"] == "neutral")

    return {
        "signals":       signals,
        "buy_count":     buy_cnt,
        "sell_count":    sell_cnt,
        "neutral_count": neutral_cnt,
    }


def compute_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    if len(df) < window:
        return {}
    recent = df.tail(window * 2)
    highs  = recent["high"].nlargest(3).values   if "high" in recent.columns else []
    lows   = recent["low"].nsmallest(3).values   if "low"  in recent.columns else []
    return {
        "resistance": [round(float(h), 2) for h in highs],
        "support":    [round(float(l), 2) for l in lows],
        "close":      float(df["close"].iloc[-1]),
    }


def _isnan(v) -> bool:
    try:
        return np.isnan(float(v))
    except Exception:
        return True
