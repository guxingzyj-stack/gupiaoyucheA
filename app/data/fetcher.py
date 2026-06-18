"""
数据获取模块
使用 curl_cffi 直接访问东方财富API（兼容各类代理环境）
"""
import os
import json
import logging
import hashlib
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config

_logger = logging.getLogger(__name__)

# curl_cffi 模拟浏览器指纹，绕过 requests/urllib3 的代理限制
from curl_cffi import requests as cfreq

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept":  "application/json, text/plain, */*",
}

# 市场ID：沪市1，深市0，北交所0
_MARKET_MAP = {
    "6": "1",   # 60xxxx 上交所
    "0": "0",   # 00xxxx / 30xxxx 深交所
    "3": "0",
    "4": "0",
    "8": "0",
    "9": "1",
}


def _cache_path(key: str) -> str:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(config.CACHE_DIR, f"{h}.pkl")


def _cache_valid(path: str, hours: int = 4) -> bool:
    if not os.path.exists(path):
        return False
    try:
        if os.path.getsize(path) == 0:
            return False
    except OSError:
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - mtime < timedelta(hours=hours)


def _safe_read_pickle(path: str) -> Optional[pd.DataFrame]:
    """安全读取 pickle 缓存，失败时返回 None（视为缓存失效）"""
    try:
        return pd.read_pickle(path)
    except Exception as exc:
        _logger.warning("缓存文件读取失败 %s: %s", path, exc)
        # 删除损坏的缓存文件，避免下次再被读到
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _normalize_symbol(symbol: str) -> str:
    s = symbol.upper().strip()
    for suffix in (".SS", ".SZ", ".SH", ".BJ", "SH", "SZ"):
        s = s.replace(suffix, "")
    if not s.isdigit():
        return s
    # 直接 zfill(6)，不再 lstrip("0")
    # 旧逻辑会把 "01000001" 这种异常输入静默接受，掩盖代码错误
    if len(s) > 6:
        raise ValueError(f"股票代码长度异常 ({len(s)}位): {symbol}")
    return s.zfill(6)


def _get_secid(symbol: str) -> str:
    """生成东方财富 secid（市场ID.股票代码）"""
    mkt = _MARKET_MAP.get(symbol[0], "1")
    return f"{mkt}.{symbol}"


def _non_empty(value) -> str:
    text = str(value or "").strip()
    if text in ("", "-", "None", "nan"):
        return ""
    return text


def _fill_industry_fallback(info: dict, symbol: str) -> None:
    """为行业字段做保守兜底：取不到就保持空值，不编造。"""
    if _non_empty(info.get("所属行业")):
        return

    try:
        import akshare as ak

        df = ak.stock_individual_info_em(symbol=symbol)
        if df is not None and not df.empty:
            key_col = _find_info_column(df, ("item", "指标", "项目", "key"))
            value_col = _find_info_column(df, ("value", "值", "内容", "数据"))
            if key_col and value_col:
                for _, row in df.iterrows():
                    name = str(row.get(key_col, ""))
                    if "行业" in name:
                        industry = _non_empty(row.get(value_col))
                        if industry:
                            info["所属行业"] = industry
                            info["industry_source"] = "stock_individual_info_em"
                            return
    except Exception as exc:
        _logger.warning("AKShare行业兜底失败 %s: %s", symbol, exc)

    industry = _fetch_industry_from_quote_board(symbol)
    if industry:
        info["所属行业"] = industry
        info["industry_source"] = "eastmoney_quote_f127"


def _find_info_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_map = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found:
            return found
    return None


def _fetch_industry_from_quote_board(symbol: str) -> str:
    try:
        secid = _get_secid(symbol)
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid": secid,
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "fields": "f57,f58,f127",
        }
        r = cfreq.get(url, params=params, headers=_HEADERS, timeout=15)
        d = r.json()
        if d.get("data"):
            return _non_empty(d["data"].get("f127"))
    except Exception as exc:
        _logger.warning("东方财富行业板块兜底失败 %s: %s", symbol, exc)
    return ""


def _fetch_kline_direct(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """
    直接调用东方财富K线API（curl_cffi，模拟浏览器请求头）
    adjust: qfq=前复权(1), hfq=后复权(2), 不复权=0
    """
    fqt_map = {"qfq": "1", "hfq": "2", "": "0", None: "0"}
    fqt     = fqt_map.get(adjust, "1")
    secid   = _get_secid(symbol)

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid":   secid,
        "klt":     "101",       # 日K
        "fqt":     fqt,
        "beg":     start,
        "end":     end,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut":      "7eea3edcaed734bea9cbfc24409ed989",
    }

    r = cfreq.get(url, params=params, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()

    if not data.get("data") or not data["data"].get("klines"):
        raise ValueError(f"无K线数据返回，请确认股票代码正确: {symbol}")

    rows = []
    for line in data["data"]["klines"]:
        parts = line.split(",")
        rows.append({
            "date":       parts[0],
            "open":       float(parts[1]),
            "close":      float(parts[2]),
            "high":       float(parts[3]),
            "low":        float(parts[4]),
            "volume":     float(parts[5]),
            "amount":     float(parts[6]),
            "amplitude":  float(parts[7]),
            "pct_change": float(parts[8]),
            "change":     float(parts[9]),
            "turnover":   float(parts[10]),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def fetch_stock_data(
    symbol: str,
    years: int      = config.DEFAULT_PERIOD_YEARS,
    adjust: str     = "qfq",
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取A股日线行情（前复权）"""
    symbol = _normalize_symbol(symbol)
    end    = datetime.now()
    start  = end - timedelta(days=365 * years)

    cache_key  = f"hist_{symbol}_{years}_{adjust}"
    cache_file = _cache_path(cache_key)

    if use_cache and _cache_valid(cache_file):
        cached = _safe_read_pickle(cache_file)
        if cached is not None:
            return cached

    try:
        df = _fetch_kline_direct(
            symbol,
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            adjust,
        )
    except Exception as e:
        raise RuntimeError(f"获取行情数据失败 [{symbol}]: {e}")

    if df is None or df.empty:
        raise ValueError(f"无数据返回，请确认股票代码: {symbol}")

    df = df.dropna(subset=["close"])

    if len(df) < config.MIN_TRADING_DAYS:
        raise ValueError(
            f"历史数据不足（{len(df)}天），至少需要{config.MIN_TRADING_DAYS}天"
        )

    df.to_pickle(cache_file)
    return df


def fetch_stock_info(symbol: str) -> dict:
    """获取个股基本信息（名称/行业/市值/PE/PB）"""
    symbol     = _normalize_symbol(symbol)
    cache_key  = f"info_{symbol}"
    json_path  = os.path.join(config.CACHE_DIR, f"info_{symbol}.json")

    if _cache_valid(json_path, hours=12) and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    info = {}

    # ── 方式1：东方财富个股信息接口 ──────────────────────
    try:
        url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
        params = {
            "reportName": "RPT_F10_INFO_ORGPROFILE",
            "columns":    "ALL",
            "filter":     f'(SECURITY_CODE="{symbol}")',
            "pageNumber": "1", "pageSize": "1",
            "source": "WEB", "client": "WEB",
        }
        r = cfreq.get(url, params=params, headers=_HEADERS, timeout=15)
        d = r.json()
        if d.get("result") and d["result"].get("data"):
            row = d["result"]["data"][0]
            info["股票简称"] = row.get("SECURITY_NAME_ABBR", symbol)
            info["所属行业"] = row.get("INDUSTY_CSRC2", "")
    except Exception as exc:
        _logger.warning("获取 %s 股票基础信息失败: %s", symbol, exc)

    # ── 方式2：东方财富实时行情接口（含PE/PB/市值） ──────
    try:
        secid = _get_secid(symbol)
        url   = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid":  secid,
            "ut":     "7eea3edcaed734bea9cbfc24409ed989",
            "fields": "f57,f58,f162,f167,f168,f116,f117",
        }
        r = cfreq.get(url, params=params, headers=_HEADERS, timeout=15)
        d = r.json()
        if d.get("data"):
            raw = d["data"]
            info.setdefault("股票简称", str(raw.get("f58", symbol)))
            # f162=PE(TTM) 单位：0.01，需除以100
            pe_raw = raw.get("f162")
            if pe_raw and pe_raw != "-":
                try:
                    info["pe_ttm"] = round(float(pe_raw) / 100, 2)
                except Exception:
                    pass
            # f167=PB 单位：0.01，需除以100
            pb_raw = raw.get("f167")
            if pb_raw and pb_raw != "-":
                try:
                    info["pb"] = round(float(pb_raw) / 100, 2)
                except Exception:
                    pass
            # f116=总市值（元），转亿
            mc = raw.get("f116")
            if mc:
                try:
                    info["总市值"] = f"{float(mc)/1e8:.2f}亿"
                except Exception:
                    pass
            # f168=股息率，单位0.01%，需除以10000
            dy = raw.get("f168")
            if dy and dy != "-":
                try:
                    info["dividend_yield"] = round(float(dy) / 10000 * 100, 4)
                except Exception:
                    pass
    except Exception as exc:
        _logger.warning("获取 %s 行情/PE/PB 信息失败: %s", symbol, exc)

    # 确保有名字
    if "股票简称" not in info or not info["股票简称"]:
        info["股票简称"] = symbol

    _fill_industry_fallback(info, symbol)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)

    return info


def get_stock_name(symbol: str) -> str:
    """获取股票中文名称"""
    try:
        info = fetch_stock_info(symbol)
        for key in ["股票简称", "名称", "股票名称"]:
            if key in info and info[key]:
                return str(info[key])
    except Exception:
        pass
    return symbol


# ══════════════════════════════════════════════════════════════
#  市场数据增强特征
# ══════════════════════════════════════════════════════════════

# 进程内内存缓存：批量分析时避免每只股票都读 pickle / 走网络
# key=(years, mtime); value=DataFrame
_HS300_MEMORY_CACHE: dict = {}


def fetch_hs300_data(years: int = 3) -> pd.DataFrame:
    """获取沪深300指数日线数据（用作市场基准协变量）"""
    cache_key  = f"hs300_{years}"
    cache_file = _cache_path(cache_key)
    # 1. 进程内内存缓存（最快路径）
    if os.path.exists(cache_file):
        try:
            mtime = os.path.getmtime(cache_file)
            mem_key = (years, mtime)
            if mem_key in _HS300_MEMORY_CACHE:
                return _HS300_MEMORY_CACHE[mem_key]
        except OSError:
            pass
    if _cache_valid(cache_file, hours=6):
        cached = _safe_read_pickle(cache_file)
        if cached is not None:
            try:
                _HS300_MEMORY_CACHE[(years, os.path.getmtime(cache_file))] = cached
            except OSError:
                pass
            return cached

    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df = df.rename(columns={"date": "date", "close": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # 只保留收盘价
        df = df[["close"]].rename(columns={"close": "hs300_close"})
        cutoff = datetime.now() - timedelta(days=365 * years)
        df = df[df.index >= cutoff]
        df.to_pickle(cache_file)
        try:
            _HS300_MEMORY_CACHE[(years, os.path.getmtime(cache_file))] = df
        except OSError:
            pass
        return df
    except Exception as exc:
        _logger.warning("获取沪深300数据失败: %s", exc)
        return pd.DataFrame()


def fetch_money_flow(symbol: str) -> pd.DataFrame:
    """获取个股主力资金流向（最近60天）"""
    cache_key  = f"flow_{symbol}"
    cache_file = _cache_path(cache_key)
    if _cache_valid(cache_file, hours=4):
        cached = _safe_read_pickle(cache_file)
        if cached is not None:
            return cached

    try:
        import akshare as ak
        mkt = "sh" if symbol[0] == "6" else "sz"
        df  = ak.stock_individual_fund_flow(stock=symbol, market=mkt)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"日期": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # 主力净流入列可能叫"主力净流入"或"main_net"
        for col_name in ["主力净流入", "主力净流入-净额"]:
            if col_name in df.columns:
                df = df[[col_name]].rename(columns={col_name: "main_flow"})
                break
        if "main_flow" not in df.columns and not df.empty:
            # 取第2列作为主力流入
            df = df.iloc[:, [1]].rename(columns={df.columns[1]: "main_flow"})
        df["main_flow"] = pd.to_numeric(df["main_flow"], errors="coerce")
        df.to_pickle(cache_file)
        return df
    except Exception as exc:
        _logger.warning("获取 %s 资金流向失败: %s", symbol, exc)
        return pd.DataFrame()


def enrich_with_market_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """将沪深300和资金流向数据合并到股票DataFrame"""
    result = df.copy()

    # ── 沪深300协变量 ─────────────────────────────────────
    try:
        hs300 = fetch_hs300_data()
        if not hs300.empty:
            result = result.join(hs300, how="left")
            result["hs300_close"] = result["hs300_close"].ffill()
            result["hs300_ret1"]  = result["hs300_close"].pct_change(1)
            result["hs300_ret5"]  = result["hs300_close"].pct_change(5)
            result["hs300_ret20"] = result["hs300_close"].pct_change(20)
            # 股票与指数的相对强弱（RS）
            result["rel_strength"] = result["close"].pct_change(20) - result["hs300_ret20"]
    except Exception as exc:
        _logger.warning("合并沪深300数据失败 %s: %s", symbol, exc)

    # ── 资金流向 ──────────────────────────────────────────
    try:
        flow = fetch_money_flow(symbol)
        if not flow.empty:
            result = result.join(flow[["main_flow"]], how="left")
            result["main_flow"] = result["main_flow"].ffill().fillna(0)
            # 归一化：取20日滚动z-score
            mu  = result["main_flow"].rolling(20).mean()
            std = result["main_flow"].rolling(20).std().replace(0, 1)
            result["main_flow_z"] = (result["main_flow"] - mu) / std
    except Exception as exc:
        _logger.warning("合并 %s 资金流向数据失败: %s", symbol, exc)

    return result
