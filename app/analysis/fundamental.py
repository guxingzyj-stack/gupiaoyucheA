"""
基本面分析模块
通过 AKShare 获取财务指标并打分
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

VALUATION_PERIOD = "近十年"


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

        fields = {
            "roe": (("净资产收益率(%)", "加权净资产收益率(%)"), ("净资产收益率",)),
            "revenue_growth": (("主营业务收入增长率(%)",), ("收入", "增长")),
            "profit_growth": (("净利润增长率(%)",), ("净利润", "增长")),
        }
        for field, (candidates, keywords) in fields.items():
            col = _find_column(df, candidates=candidates, keywords=keywords)
            if not col:
                continue
            val = _latest_notna(_numeric_series(df[col]))
            if val is not None:
                result[field] = val

    except Exception:
        pass


def fetch_quality_metrics(symbol: str) -> dict:
    """
    获取价值体检需要的财务质量数据。

    AKShare 的接口和中文列名可能随版本变化，本函数只使用当前返回中
    实际存在的列；取不到的数据返回 None，并写入 missing_fields/errors。
    """
    metrics = {
        "symbol": symbol,
        "roe": None,
        "roe_series": [],
        "cashflow_to_profit": None,
        "cashflow_to_profit_series": [],
        "deducted_net_profit": None,
        "deducted_net_profit_indicator": None,
        "deducted_net_profit_abstract": None,
        "parent_net_profit": None,
        "parent_net_profit_indicator": None,
        "parent_net_profit_abstract": None,
        "deducted_profit_ratio": None,
        "deducted_profit_ratio_source": None,
        "gross_margin": None,
        "gross_margin_series": [],
        "revenue_growth": None,
        "profit_growth": None,
        "debt_ratio": None,
        "pe_percentile": None,
        "pb_percentile": None,
        "npl_ratio": None,
        "provision_coverage": None,
        "capital_adequacy": None,
        "net_interest_margin": None,
        "source_columns": {},
        "missing_fields": [],
        "errors": [],
    }

    try:
        import akshare as ak
    except Exception as exc:
        metrics["errors"].append(f"AKShare不可用: {exc}")
        return metrics

    indicator_df = None
    try:
        indicator_df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2021")
        if indicator_df is not None and not indicator_df.empty:
            indicator_df = indicator_df.sort_values(by=indicator_df.columns[0], ascending=False)
            _fill_quality_from_indicator(metrics, indicator_df)
    except Exception as exc:
        metrics["errors"].append(f"财务指标获取失败: {exc}")

    try:
        abstract_df = ak.stock_financial_abstract(symbol=symbol)
        if abstract_df is not None and not abstract_df.empty:
            _fill_quality_from_abstract(metrics, abstract_df)
    except Exception as exc:
        metrics["errors"].append(f"财务摘要获取失败: {exc}")

    _fetch_valuation_percentiles(metrics, symbol, ak)
    _mark_missing_quality_fields(metrics)
    return metrics


def _fill_quality_from_indicator(metrics: dict, df: pd.DataFrame) -> None:
    fields = {
        "roe": (("净资产收益率(%)", "加权净资产收益率(%)"), ("净资产收益率",)),
        "cashflow_to_profit": (("经营现金净流量与净利润的比率(%)",), ("经营现金", "净利润", "比率")),
        "gross_margin": (("销售毛利率(%)",), ("毛利率",)),
        "revenue_growth": (("主营业务收入增长率(%)",), ("收入", "增长")),
        "profit_growth": (("净利润增长率(%)",), ("净利润", "增长")),
        "debt_ratio": (("资产负债率(%)",), ("资产负债率",)),
    }
    for field, (candidates, keywords) in fields.items():
        col = _find_column(df, candidates=candidates, keywords=keywords)
        metrics["source_columns"][field] = col
        if not col:
            continue
        series = _numeric_series(df[col])
        latest = _latest_notna(series)
        if field in ("cashflow_to_profit",):
            metrics[field] = _percent_to_ratio(latest, is_percent=True)
            metrics[f"{field}_series"] = [
                _percent_to_ratio(x, is_percent=True)
                for x in series.dropna().head(5).tolist()
            ]
        elif field in ("roe", "gross_margin"):
            metrics[field] = latest
            metrics[f"{field}_series"] = series.dropna().head(5).tolist()
        else:
            metrics[field] = latest

    deducted_col = _find_column(
        df,
        candidates=("扣除非经常性损益后的净利润(元)",),
        keywords=("扣除非经常性", "净利润"),
    )
    metrics["source_columns"]["deducted_net_profit_indicator"] = deducted_col
    if deducted_col:
        metrics["deducted_net_profit_indicator"] = _latest_notna(_numeric_series(df[deducted_col]))

    parent_col = _find_column(
        df,
        candidates=(
            "归属于母公司所有者的净利润(元)",
            "归属于母公司股东的净利润(元)",
            "归属母公司股东的净利润(元)",
            "净利润(元)",
        ),
    )
    metrics["source_columns"]["parent_net_profit_indicator"] = parent_col
    if parent_col:
        metrics["parent_net_profit_indicator"] = _latest_notna(_numeric_series(df[parent_col]))

    _refresh_deducted_profit_ratio(metrics)


def _fill_quality_from_abstract(metrics: dict, df: pd.DataFrame) -> None:
    if "指标" not in df.columns:
        metrics["errors"].append("财务摘要缺少指标列")
        return
    parent = _latest_abstract_value(df, ("归母净利润", "归属于母公司股东的净利润"))
    deducted = _latest_abstract_value(df, ("扣非", "扣除非经常性损益"))
    revenue = _latest_abstract_value(df, ("营业总收入", "营业收入"))

    if parent is not None:
        metrics["parent_net_profit_abstract"] = parent
    if deducted is not None:
        metrics["deducted_net_profit_abstract"] = deducted
    _refresh_deducted_profit_ratio(metrics)
    if revenue is not None:
        metrics["latest_revenue"] = revenue


def _refresh_deducted_profit_ratio(metrics: dict) -> None:
    pairs = (
        ("abstract", "deducted_net_profit_abstract", "parent_net_profit_abstract"),
        ("indicator", "deducted_net_profit_indicator", "parent_net_profit_indicator"),
        ("mixed_indicator_deducted_abstract_parent", "deducted_net_profit_indicator", "parent_net_profit_abstract"),
        ("mixed_abstract_deducted_indicator_parent", "deducted_net_profit_abstract", "parent_net_profit_indicator"),
    )

    for source, deducted_key, parent_key in pairs:
        deducted = metrics.get(deducted_key)
        parent = metrics.get(parent_key)
        if deducted is None or parent in (None, 0):
            continue
        ratio = deducted / parent
        if abs(ratio) > 5:
            metrics["errors"].append(f"扣非比率口径疑似不一致，已跳过: {source}")
            continue
        metrics["deducted_net_profit"] = deducted
        metrics["parent_net_profit"] = parent
        metrics["deducted_profit_ratio"] = ratio
        metrics["deducted_profit_ratio_source"] = source
        return

    metrics["deducted_net_profit"] = (
        metrics.get("deducted_net_profit_abstract")
        if metrics.get("deducted_net_profit_abstract") is not None
        else metrics.get("deducted_net_profit_indicator")
    )
    metrics["parent_net_profit"] = (
        metrics.get("parent_net_profit_abstract")
        if metrics.get("parent_net_profit_abstract") is not None
        else metrics.get("parent_net_profit_indicator")
    )


def _fetch_valuation_percentiles(metrics: dict, symbol: str, ak) -> None:
    try:
        if _try_lg_valuation_percentiles(metrics, symbol, ak):
            _mark_missing_valuation_fields(metrics)
            return
    except Exception as exc:
        metrics["errors"].append(f"乐咕历史估值获取失败: {exc}")

    try:
        if _try_baidu_valuation_percentiles(metrics, symbol, ak):
            _mark_missing_valuation_fields(metrics)
            return
    except Exception as exc:
        metrics["errors"].append(f"百度历史估值获取失败: {exc}")

    _mark_missing_valuation_fields(metrics)


def _try_lg_valuation_percentiles(metrics: dict, symbol: str, ak) -> bool:
    fn = getattr(ak, "stock_a_indicator_lg", None)
    if fn is None:
        return False

    for candidate in _symbol_candidates(symbol):
        try:
            df = fn(symbol=candidate)
        except Exception as exc:
            metrics["errors"].append(f"stock_a_indicator_lg({candidate})失败: {exc}")
            continue
        if df is None or df.empty:
            continue
        df = _recent_valuation_window(df)
        pe_col = _find_column(df, candidates=("pe_ttm", "PE_TTM", "pe", "PE"))
        pb_col = _find_column(df, candidates=("pb", "PB"))
        updated = False
        if pe_col:
            _fill_percentile_from_series(metrics, "pe_percentile", df[pe_col], f"stock_a_indicator_lg:{pe_col}")
            updated = metrics.get("pe_percentile") is not None or updated
        if pb_col:
            _fill_percentile_from_series(metrics, "pb_percentile", df[pb_col], f"stock_a_indicator_lg:{pb_col}")
            updated = metrics.get("pb_percentile") is not None or updated
        if updated:
            return True
    return False


def _try_baidu_valuation_percentiles(metrics: dict, symbol: str, ak) -> bool:
    fn = getattr(ak, "stock_zh_valuation_baidu", None)
    if fn is None:
        metrics["errors"].append("当前AKShare无stock_zh_valuation_baidu接口，历史估值缺失")
        return False

    specs = (
        ("pe_percentile", "市盈率(TTM)"),
        ("pb_percentile", "市净率"),
    )
    updated = False
    for field, indicator in specs:
        if metrics.get(field) is not None:
            updated = True
            continue
        try:
            df = fn(symbol=symbol, indicator=indicator, period=VALUATION_PERIOD)
        except Exception as exc:
            metrics["errors"].append(f"百度{indicator}估值获取失败: {exc}")
            continue
        if df is None or df.empty:
            metrics["errors"].append(f"百度{indicator}估值返回为空")
            continue
        value_col = _find_column(df, candidates=("value", "Value", "估值"))
        if not value_col:
            metrics["errors"].append(f"百度{indicator}估值缺少value列")
            continue
        _fill_percentile_from_series(metrics, field, df[value_col], f"stock_zh_valuation_baidu:{indicator}:{value_col}")
        updated = metrics.get(field) is not None or updated
    return updated


def _fill_percentile_from_series(metrics: dict, field: str, series, source: str) -> None:
    values = _positive_numeric_series(series)
    current = _last_notna(values)
    percentile = _compute_percentile(values, current)
    metrics["source_columns"][field] = source
    if percentile is None:
        return
    metrics[field] = percentile
    if len(values) < 250:
        label = "PE" if field == "pe_percentile" else "PB"
        metrics["errors"].append(f"{label}历史估值样本不足: {len(values)}")


def _compute_percentile(series, current) -> float | None:
    values = _positive_numeric_series(series)
    if values.empty:
        return None
    try:
        current_value = float(current)
    except Exception:
        return None
    if current_value <= 0 or np.isnan(current_value):
        return None
    return float((values <= current_value).mean() * 100.0)


def _positive_numeric_series(series) -> pd.Series:
    values = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return values[values > 0]


def _recent_valuation_window(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _find_column(df, candidates=("trade_date", "date", "日期"))
    if not date_col:
        return df
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.dropna().empty:
        return df
    end = dates.max()
    start = end - pd.DateOffset(years=10)
    result = df.loc[dates >= start].copy()
    result["_valuation_date"] = dates.loc[result.index]
    result = result.sort_values("_valuation_date")
    return result.drop(columns=["_valuation_date"])


def _symbol_candidates(symbol: str) -> tuple[str, ...]:
    symbol = str(symbol)
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    upper_prefix = prefix.upper()
    return (
        symbol,
        f"{prefix}{symbol}",
        f"{upper_prefix}{symbol}",
    )


def _find_column(df: pd.DataFrame, candidates=(), keywords=()) -> str | None:
    columns = [str(c) for c in df.columns]
    for col in candidates:
        if col in columns:
            return col
    if keywords:
        for col in columns:
            if all(k in col for k in keywords):
                return col
    return None


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def _latest_notna(series: pd.Series) -> float | None:
    series = series.dropna()
    if series.empty:
        return None
    return float(series.iloc[0])


def _last_notna(series: pd.Series) -> float | None:
    series = series.dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def _percent_to_ratio(value, is_percent: bool = False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = float(value)
    if is_percent:
        return value / 100.0
    if abs(value) > 5:
        return value / 100.0
    return value


def _latest_abstract_value(df: pd.DataFrame, keywords: tuple[str, ...]) -> float | None:
    value_cols = [c for c in df.columns if str(c).isdigit()]
    if not value_cols:
        return None
    value_cols = sorted(value_cols, reverse=True)
    for _, row in df.iterrows():
        name = str(row.get("指标", ""))
        if any(k in name for k in keywords):
            vals = pd.to_numeric(row[value_cols], errors="coerce").dropna()
            if not vals.empty:
                return float(vals.iloc[0])
    return None


def _mark_missing_quality_fields(metrics: dict) -> None:
    required = {
        "roe": "ROE",
        "cashflow_to_profit": "经营现金流/净利润",
        "gross_margin": "毛利率",
        "revenue_growth": "营收增长率",
        "profit_growth": "净利润增长率",
        "debt_ratio": "资产负债率",
    }
    for field, label in required.items():
        if metrics.get(field) is None:
            metrics["missing_fields"].append(label)


def _mark_missing_valuation_fields(metrics: dict) -> None:
    optional = {
        "pe_percentile": "PE历史分位",
        "pb_percentile": "PB历史分位",
    }
    for field, label in optional.items():
        if metrics.get(field) is None and label not in metrics["missing_fields"]:
            metrics["missing_fields"].append(label)


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
