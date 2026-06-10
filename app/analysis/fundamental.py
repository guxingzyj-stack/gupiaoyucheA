"""
基本面分析模块
通过 AKShare 获取财务指标并打分
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def fetch_and_analyze(symbol: str, stock_info: dict) -> dict:
    """
    获取并分析基本面数据

    Returns
    -------
    dict: {pe, pb, roe, revenue_growth, profit_growth,
           dividend_yield, market_cap, industry,
           score(0-100), details}
    """
    result = {
        "pe":             None,
        "pb":             None,
        "roe":            None,
        "revenue_growth": None,
        "profit_growth":  None,
        "dividend_yield": None,
        "market_cap":     None,
        "industry":       "未知",
        "score":          50,
        "details":        [],
    }

    # ── 从 stock_info 提取已有字段 ─────────────────────────
    _extract_from_info(result, stock_info)

    # ── 从 AKShare 获取财务指标 ────────────────────────────
    _fetch_financial_indicators(result, symbol)

    # ── 打分 ───────────────────────────────────────────────
    result["score"] = _score(result, result["details"])

    return result


def _extract_from_info(result: dict, info: dict):
    """从个股信息字典提取基本面字段"""
    mapping = {
        "总市值":    "market_cap",
        "市盈率(TTM)": "pe",
        "pe_ttm":   "pe",
        "市净率":   "pb",
        "pb":        "pb",
        "所属行业": "industry",
        "行业":      "industry",
        "dividend_yield": "dividend_yield",
    }
    for raw_key, field in mapping.items():
        if raw_key in info and info[raw_key]:
            val = info[raw_key]
            if field in ("pe", "pb", "dividend_yield"):
                try:
                    result[field] = float(str(val).replace(",", "").replace("%", ""))
                except Exception:
                    pass
            elif field == "market_cap":
                try:
                    result[field] = _parse_market_cap(str(val))
                except Exception:
                    pass
            else:
                result[field] = str(val)


def _parse_market_cap(s: str) -> float:
    """将 '1234.56亿' 或 '12345.6万' 解析为数值（亿元）"""
    s = s.strip()
    if "亿" in s:
        return float(s.replace("亿", "").replace(",", ""))
    if "万" in s:
        return float(s.replace("万", "").replace(",", "")) / 10000
    return float(s.replace(",", ""))


def _fetch_financial_indicators(result: dict, symbol: str):
    """从 AKShare 获取 ROE、营收增长等"""
    try:
        import akshare as ak

        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2021")
        if df is None or df.empty:
            return

        df = df.sort_values(by=df.columns[0], ascending=False)

        # ROE
        for col in df.columns:
            if "净资产收益率" in col or "ROE" in col:
                try:
                    val = float(str(df[col].iloc[0]).replace("%", ""))
                    result["roe"] = val
                except Exception:
                    pass
                break

        # 营收增长率（同比）
        for col in df.columns:
            if "营业收入" in col and "增长" in col:
                try:
                    val = float(str(df[col].iloc[0]).replace("%", ""))
                    result["revenue_growth"] = val
                except Exception:
                    pass
                break

        # 净利润增长率
        for col in df.columns:
            if "净利润" in col and "增长" in col:
                try:
                    val = float(str(df[col].iloc[0]).replace("%", ""))
                    result["profit_growth"] = val
                except Exception:
                    pass
                break

    except Exception:
        pass


def _score(result: dict, details: list) -> float:
    """综合评分(0-100)"""
    score = 50.0

    # PE 评分（行业中位数约15-25）
    pe = result.get("pe")
    if pe is not None:
        if pe <= 0:
            details.append("PE为负（亏损），扣分")
            score -= 10
        elif pe < 10:
            details.append(f"PE={pe:.1f}，估值极低，加分")
            score += 10
        elif pe < 20:
            details.append(f"PE={pe:.1f}，估值合理，加分")
            score += 5
        elif pe < 40:
            details.append(f"PE={pe:.1f}，估值偏高，轻微扣分")
            score -= 5
        else:
            details.append(f"PE={pe:.1f}，估值过高，扣分")
            score -= 10

    # PB 评分
    pb = result.get("pb")
    if pb is not None:
        if pb < 1:
            details.append(f"PB={pb:.2f}，低于净资产，加分")
            score += 8
        elif pb < 3:
            details.append(f"PB={pb:.2f}，估值合理")
            score += 3
        elif pb > 5:
            details.append(f"PB={pb:.2f}，估值较贵，扣分")
            score -= 5

    # ROE 评分
    roe = result.get("roe")
    if roe is not None:
        if roe > 20:
            details.append(f"ROE={roe:.1f}%，盈利能力强，加分")
            score += 12
        elif roe > 10:
            details.append(f"ROE={roe:.1f}%，盈利能力一般，轻微加分")
            score += 5
        elif roe < 0:
            details.append(f"ROE={roe:.1f}%，净资产亏损，扣分")
            score -= 10

    # 利润增长率
    pg = result.get("profit_growth")
    if pg is not None:
        if pg > 30:
            details.append(f"净利润增长{pg:.1f}%，高增长，加分")
            score += 10
        elif pg > 10:
            details.append(f"净利润增长{pg:.1f}%，稳定增长，加分")
            score += 5
        elif pg < 0:
            details.append(f"净利润下滑{abs(pg):.1f}%，扣分")
            score -= 8

    # 股息率
    dy = result.get("dividend_yield")
    if dy is not None and dy > 0:
        details.append(f"股息率{dy:.2f}%，有分红")
        score += min(dy * 2, 6)

    return float(np.clip(score, 0, 100))
