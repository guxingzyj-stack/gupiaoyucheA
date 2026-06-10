"""
中文新闻情感分析模块
使用 SnowNLP 对财经新闻进行情感评分
"""
import warnings
from typing import List

import numpy as np

warnings.filterwarnings("ignore")

# 负面关键词权重加成
NEGATIVE_KEYWORDS = [
    "暴跌", "崩盘", "亏损", "违规", "处罚", "调查", "退市", "停牌",
    "债务", "违约", "诉讼", "造假", "欺诈", "减持", "套现", "解禁",
    "下调", "警示", "风险", "危机", "问题", "质疑", "下跌", "跌停",
]

POSITIVE_KEYWORDS = [
    "涨停", "利好", "增长", "突破", "创新高", "业绩预增", "分红",
    "回购", "增持", "战略合作", "中标", "获批", "上市", "扩产",
    "新产品", "盈利", "超预期", "强势", "龙头", "行业第一",
]


def analyze_sentiment(news_texts: List[str]) -> dict:
    """
    对新闻文本列表进行情感分析

    Returns
    -------
    dict: {score(0-1), label, positive_count, negative_count,
           neutral_count, keyword_boost, details}
    """
    if not news_texts:
        return {
            "score":          0.5,
            "label":          "中性",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count":  0,
            "keyword_boost":  0.0,
            "news_count":     0,
        }

    scores = []
    pos_count = neg_count = neu_count = 0
    keyword_boost = 0.0

    try:
        from snownlp import SnowNLP
    except ImportError:
        return _fallback_sentiment(news_texts)

    article_boosts = []  # 单条新闻的关键词贡献，最终求均值
    for text in news_texts:
        if not text or len(text.strip()) < 4:
            continue
        try:
            s = SnowNLP(text[:300]).sentiments  # 截断，避免过长
            scores.append(s)

            # 单条新闻关键词贡献：±0.03/命中，但单条最多累计 ±0.10
            per_article = 0.0
            for kw in NEGATIVE_KEYWORDS:
                if kw in text:
                    per_article -= 0.03
            for kw in POSITIVE_KEYWORDS:
                if kw in text:
                    per_article += 0.03
            per_article = float(np.clip(per_article, -0.10, 0.10))
            article_boosts.append(per_article)

            if s > 0.65:
                pos_count += 1
            elif s < 0.35:
                neg_count += 1
            else:
                neu_count += 1
        except Exception:
            scores.append(0.5)
            article_boosts.append(0.0)
            neu_count += 1
    # 全局 keyword_boost 改为单条新闻贡献的均值（而非累加），避免长新闻把分数推到边界
    if article_boosts:
        keyword_boost = float(np.mean(article_boosts))

    if not scores:
        avg = 0.5
    else:
        # 加权均值：近期新闻权重更高
        weights = np.linspace(1.5, 0.5, len(scores))
        avg = float(np.average(scores, weights=weights))

    # 关键词修正（限幅）
    keyword_boost = float(np.clip(keyword_boost, -0.15, 0.15))
    final_score   = float(np.clip(avg + keyword_boost, 0.0, 1.0))

    if final_score > 0.65:
        label = "正面"
    elif final_score < 0.35:
        label = "负面"
    else:
        label = "中性"

    return {
        "score":          final_score,
        "label":          label,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count":  neu_count,
        "keyword_boost":  keyword_boost,
        "news_count":     len(scores),
    }


def _fallback_sentiment(texts: List[str]) -> dict:
    """SnowNLP不可用时的关键词回退方案"""
    pos = neg = 0
    for text in texts:
        for kw in POSITIVE_KEYWORDS:
            if kw in text:
                pos += 1
        for kw in NEGATIVE_KEYWORDS:
            if kw in text:
                neg += 1

    total = pos + neg or 1
    score = 0.5 + (pos - neg) / (total * 4)
    score = float(np.clip(score, 0, 1))

    return {
        "score":          score,
        "label":          "正面" if score > 0.65 else ("负面" if score < 0.35 else "中性"),
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count":  len(texts) - pos - neg,
        "keyword_boost":  0.0,
        "news_count":     len(texts),
    }
