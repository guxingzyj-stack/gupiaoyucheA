"""
新闻数据获取模块
优先使用 AKShare stock_news_em，备用东方财富公告 API
"""
import re
import warnings
from typing import List, Dict

warnings.filterwarnings("ignore")

import config
from data.fetcher import _normalize_symbol, _HEADERS, cfreq


def _norm_title_key(title: str) -> str:
    """归一化标题用于去重：去除空白/标点/全半角差异"""
    if not title:
        return ""
    # 去除所有空白字符和常见标点
    s = re.sub(r"[\s　，。、；：！？\.\,\;\:\!\?\-\(\)\[\]【】《》\"'`~]+", "", title)
    return s.lower()


def fetch_news(symbol: str, limit: int = config.NEWS_LIMIT) -> List[Dict]:
    """获取个股相关新闻"""
    symbol    = _normalize_symbol(symbol)
    news_list = []

    # ── 优先：AKShare 个股新闻（东方财富数据源）────────────
    try:
        import akshare as ak
        df = ak.stock_news_em(symbol=symbol)
        if df is not None and not df.empty:
            for _, row in df.head(limit).iterrows():
                title   = str(row.get("新闻标题", "")).strip()
                content = str(row.get("新闻内容", "")).strip()[:400]
                date    = str(row.get("发布时间", "")).strip()
                source  = str(row.get("文章来源", "东方财富")).strip()
                url_str = str(row.get("新闻链接", "")).strip()
                if title:
                    news_list.append({
                        "title":   title,
                        "content": content,
                        "date":    date,
                        "source":  source,
                        "url":     url_str,
                    })
    except Exception:
        pass

    # ── 备用1：东方财富公司公告 API ──────────────────────────
    if len(news_list) < 5:
        try:
            url    = "https://np-anotice-stock.eastmoney.com/api/security/ann"
            params = {
                "sr":            "-1",
                "page_size":     "20",
                "page_index":    "1",
                "ann_type":      "A",
                "client_source": "web",
                "stock_list":    symbol,
            }
            r = cfreq.get(url, params=params, headers=_HEADERS, timeout=15)
            d = r.json()
            for item in (d.get("data") or {}).get("list") or []:
                title = item.get("title") or item.get("title_ch", "")
                date  = item.get("notice_date") or item.get("display_time", "")
                if title:
                    news_list.append({
                        "title":   title,
                        "content": title,
                        "date":    str(date)[:10],
                        "source":  "公司公告",
                        "url":     "",
                    })
        except Exception:
            pass

    # ── 备用2：东方财富快讯 API ──────────────────────────────
    if len(news_list) < 5:
        try:
            url    = "https://np-listapi.eastmoney.com/comm/web/getListInfo"
            params = {
                "client":       "web",
                "type":         "2",
                "mTypeAndCode": f"1,{symbol}",
                "pageSize":     str(limit),
                "pageIndex":    "1",
            }
            r    = cfreq.get(url, params=params, headers=_HEADERS, timeout=15)
            d    = r.json()
            items = (d.get("data") or {}).get("list") or []
            for item in items:
                title = item.get("title", "")
                if title:
                    news_list.append({
                        "title":   title,
                        "content": item.get("digest", item.get("summary", ""))[:400],
                        "date":    item.get("publishTime", item.get("ctime", "")),
                        "source":  item.get("mediaName", "东方财富"),
                        "url":     item.get("url", ""),
                    })
        except Exception:
            pass

    # 去重（使用归一化全标题，避免"X公司发布公告..." 不同新闻被误判重复）
    seen, unique = set(), []
    for item in news_list:
        key = _norm_title_key(item.get("title", ""))
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    return unique[:limit]


def get_news_texts(news_list: List[Dict]) -> List[str]:
    texts = []
    for item in news_list:
        text = (item.get("title", "") + " " + item.get("content", "")).strip()
        if len(text) > 5:
            texts.append(text)
    return texts
