"""
XGBoost 预测模型（完整修复版 v3.0）
- 目标变量改为收益率（更平稳）
- 加入更多滞后天数 + 市场特征
- 超参数优化（贝叶斯优化）
- 完整实现：69个特征（行业、资金流、市场情绪、量价、月份效应）
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

import config

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# 尝试导入贝叶斯优化库
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    from sklearn.model_selection import TimeSeriesSplit
    BAYES_AVAILABLE = True
except ImportError:
    BAYES_AVAILABLE = False


# ── 基础价格特征 ─────────────────────────────────────
BASE_FEATURES = [
    "close", "open", "high", "low", "volume",
]

# ── 移动平均线特征 ──────────────────────────────────
MA_FEATURES = [
    "sma_5", "sma_10", "sma_20", "sma_60",
    "ema_12", "ema_26",
]

# ── 技术指标特征 ────────────────────────────────────
TECH_FEATURES = [
    "rsi_14", "rsi_6",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_pct",
    "kdj_k", "kdj_d", "kdj_j",
    "atr_14",
    "vol_ratio",
    "momentum_1", "momentum_3", "momentum_5", 
    "momentum_10", "momentum_20",
    "volatility_10", "volatility_20",
    "obv",
]

# ── 市场宏观特征 ───────────────────────────────────────
MARKET_FEATURES = [
    "hs300_ret1", "hs300_ret5", "hs300_ret20",
    "rel_strength",
    "main_flow_z",
]

# ── 行业指数特征 ──────────────────────────────────
INDUSTRY_FEATURES = [
    "industry_ret_1",    # 行业指数1日收益率
    "industry_ret_5",    # 行业指数5日收益率
]

# ── 资金流向特征 ──────────────────────────────────
MONEY_FLOW_FEATURES = [
    "margin_change",      # 融资余额变化
    "main_money_inflow",  # 主力资金净流入
]

# ── 市场情绪特征 ──────────────────────────────────
SENTIMENT_FEATURES = [
    "market_vix",         # 市场波动率（用沪深300波动率替代）
]

# 收益率滞后天数（8个）
RETURN_LAGS = [1, 2, 3, 5, 7, 10, 15, 20]

# 价格滞后（相对当前价格，避免量纲）（5个）
PRICE_LAGS  = [1, 2, 3, 5, 10]

# 滚动窗口（4个）
ROLL_WINDOWS = [5, 10, 20, 60]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建特征工程矩阵（完整69个特征）
    
    特征列表（共69个）：
    1. 基础价格特征（5个）：close, open, high, low, volume
    2. 移动平均线特征（6个）：sma_5/10/20/60, ema_12/26
    3. 技术指标特征（18个）：rsi_14/6, macd系列, bb系列, kdj系列, atr_14, vol_ratio, 
                               momentum_1/3/5/10/20, volatility_10/20, obv
    4. 市场宏观特征（5个）：hs300_ret1/5/20, rel_strength, main_flow_z
    5. 行业指数特征（2个）：industry_ret_1, industry_ret_5
    6. 资金流向特征（2个）：margin_change, main_money_inflow
    7. 市场情绪特征（1个）：market_vix
    8. 收益率滞后特征（8个）：ret_lag_1/2/3/5/7/10/15/20
    9. 价格滞后特征（5个）：close_lag_r_1/2/3/5/10
    10. 滚动统计特征（16个）：roll_mean_r/std/max/min 对于 5/10/20/60 日
    11. 量价关系特征（3个）：vol_ret_corr_10, vol_roll_r_5, vol_roll_r_20
    12. 高级量价特征（3个）：vol_rank_10, vol_rank_20, pv_divergence
    13. 日期特征（3个）：day_of_week, month, week_of_year
    14. 月份效应特征（2个）：is_month_end, is_month_start
    
    总计：5+6+18+5+2+2+1+8+5+16+3+3+3+2 = 69个特征
    """
    feat = pd.DataFrame(index=df.index)
    
    # ── 1. 基础价格特征（5个）────────────────────────────
    for col in BASE_FEATURES:
        if col in df.columns:
            feat[col] = df[col]
        else:
            feat[col] = 0.0
    
    # ── 2. 移动平均线特征（6个）─────────────────────────
    for col in MA_FEATURES:
        if col in df.columns:
            feat[col] = df[col]
        else:
            feat[col] = 0.0
    
    # ── 3. 技术指标特征（18个）──────────────────────────
    for col in TECH_FEATURES:
        if col in df.columns:
            feat[col] = df[col]
        else:
            feat[col] = 0.0
    
    # ── 4. 市场宏观特征（5个）────────────────────────────
    for col in MARKET_FEATURES:
        if col in df.columns:
            feat[col] = df[col]
        else:
            feat[col] = 0.0
    
    # ── 5. 行业指数特征（2个）────────────────────────────
    for col in INDUSTRY_FEATURES:
        if col in df.columns:
            feat[col] = df[col]
        else:
            # 如果数据中没有，用0填充（保持向后兼容）
            feat[col] = 0.0
    
    # ── 6. 资金流向特征（2个）────────────────────────────
    for col in MONEY_FLOW_FEATURES:
        if col in df.columns:
            feat[col] = df[col]
        else:
            # 如果数据中没有，用0填充（保持向后兼容）
            feat[col] = 0.0
    
    # ── 7. 市场情绪特征（1个）────────────────────────────
    if "market_vix" in df.columns:
        feat["market_vix"] = df["market_vix"]
    else:
        # 计算沪深300的20日波动率作为市场情绪指标
        if "hs300_close" in df.columns:
            hs300_ret = df["hs300_close"].pct_change()
            market_vix = hs300_ret.rolling(20).std() * np.sqrt(252)  # 年化波动率
            feat["market_vix"] = market_vix.fillna(0.2)  # 用0.2填充NaN（约20%年化波动）
        else:
            # 如果没有沪深300数据，用个股波动率替代
            ret = df["close"].pct_change()
            market_vix = ret.rolling(20).std() * np.sqrt(252)
            feat["market_vix"] = market_vix.fillna(0.25)
    
    # ── 8. 收益率滞后（8个）─────────────────────────────
    ret = df["close"].pct_change()
    for lag in RETURN_LAGS:
        feat[f"ret_lag_{lag}"] = ret.shift(lag)
    
    # ── 9. 价格滞后（5个）───────────────────────────────
    last_close = df["close"]
    for lag in PRICE_LAGS:
        feat[f"close_lag_r_{lag}"] = df["close"].shift(lag) / last_close - 1
    
    # ── 10. 滚动统计（16个）─────────────────────────────
    for win in ROLL_WINDOWS:
        roll = df["close"].rolling(win)
        feat[f"roll_mean_r_{win}"] = roll.mean() / last_close - 1
        feat[f"roll_std_{win}"]    = roll.std() / last_close
        feat[f"roll_max_r_{win}"]  = roll.max() / last_close - 1
        feat[f"roll_min_r_{win}"]  = roll.min() / last_close - 1
    
    # ── 11. 量价关系（3个）──────────────────────────────
    if "volume" in df.columns:
        vol = df["volume"]
        feat["vol_ret_corr_10"]  = ret.rolling(10).corr(vol.pct_change())
        feat["vol_roll_r_5"]     = vol.rolling(5).mean() / (vol.rolling(20).mean().replace(0, np.nan))
        feat["vol_roll_r_20"]    = vol.rolling(20).mean() / (vol.rolling(60).mean().replace(0, np.nan))
    
    # ── 12. 高级量价特征（3个）─────────────────────────
    if "volume" in df.columns:
        vol = df["volume"]
        # 成交量相对历史分位数
        for win in [10, 20]:
            vol_rank = vol.rolling(win).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            feat[f"vol_rank_{win}"] = vol_rank
        # 价量背离指标
        price_20d_ret = df["close"].pct_change(20)
        vol_20d_ret = vol.pct_change(20)
        feat["pv_divergence"] = price_20d_ret - vol_20d_ret
    
    # ── 13. 日期特征（3个）──────────────────────────────
    feat["day_of_week"]  = df.index.dayofweek
    feat["month"]        = df.index.month
    feat["week_of_year"] = df.index.isocalendar().week.astype(int)
    
    # ── 14. 月份效应（2个）──────────────────────────────
    feat["is_month_end"] = (df.index.day >= 25).astype(int)
    feat["is_month_start"] = (df.index.day <= 5).astype(int)
    
    # 打印特征数量（用于验证）
    print(f"✓ 特征工程完成：共生成 {len(feat.columns)} 个特征")
    
    return feat.dropna()


def optimize_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    使用贝叶斯优化搜索XGBoost最优超参数
    
    参数：
        X_train: 训练特征矩阵
        y_train: 训练目标变量
    
    返回：
        最优参数字典
    """
    if not BAYES_AVAILABLE:
        print("警告：skopt未安装，使用默认参数。请安装：pip install scikit-optimize")
        return {
            "n_estimators": 800,
            "learning_rate": 0.02,
            "max_depth": 4,
            "subsample": 0.75,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
        }
    
    # 定义参数搜索空间
    search_space = {
        "n_estimators": Integer(300, 1500),
        "learning_rate": Real(0.005, 0.1, prior="log-uniform"),
        "max_depth": Integer(3, 8),
        "subsample": Real(0.6, 0.95, prior="uniform"),
        "colsample_bytree": Real(0.5, 0.95, prior="uniform"),
        "reg_alpha": Real(0.01, 5.0, prior="log-uniform"),
        "reg_lambda": Real(0.01, 10.0, prior="log-uniform"),
        "min_child_weight": Integer(1, 10),
        "gamma": Real(0, 0.5, prior="uniform"),
    }
    
    # 时间序列交叉验证（保持时间顺序）
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 创建XGBoost模型
    xgb = XGBRegressor(
        random_state=42,
        n_jobs=config.XGB_N_JOBS,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    
    # 贝叶斯优化搜索
    opt = BayesSearchCV(
        estimator=xgb,
        search_spaces=search_space,
        n_iter=50,               # 优化迭代次数
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=1,                # 避免嵌套并行
        random_state=42,
        verbose=0,
    )
    
    # 需要验证集用于early_stopping
    split = int(len(X_train) * 0.85)
    X_opt_train, X_opt_val = X_train[:split], X_train[split:]
    y_opt_train, y_opt_val = y_train[:split], y_train[split:]
    
    # 修复：正确的语法
    opt.fit(X_opt_train, y_opt_train, eval_set=[(X_opt_val, y_opt_val)], verbose=False)
    
    print(f"贝叶斯优化最佳参数：{opt.best_params_}")
    print(f"最佳CV分数（负MSE）：{opt.best_score_:.6f}")
    
    return opt.best_params_


class XGBoostPredictor:
    def __init__(self, prediction_days: int = config.SHORT_TERM_DAYS, use_bayes: bool | None = None):
        """
        初始化XGBoost预测器
        
        参数：
            prediction_days: 预测天数
            use_bayes: 是否使用贝叶斯优化（默认True）
        """
        self.prediction_days = prediction_days
        self.model           = None
        self.scaler          = StandardScaler()
        self.is_trained      = False
        self.val_rmse        = float("inf")
        self.feature_cols    = []
        self._last_close     = None
        self.use_bayes       = config.XGB_USE_BAYES_IN_REPORT if use_bayes is None else use_bayes
        self.best_params     = None  # 保存优化后的参数

        if not XGB_AVAILABLE:
            raise ImportError("请安装 xgboost: pip install xgboost")

    def train(self, df: pd.DataFrame) -> float:
        """
        训练XGBoost模型
        
        参数：
            df: 包含价格和特征的DataFrame
        
        返回：
            验证集RMSE
        """
        feat_df = _build_features(df)
        
        # 目标：N天后的累计收益率（更平稳，更易学习）
        future_ret = (
            df["close"].shift(-self.prediction_days) / df["close"] - 1
        ).reindex(feat_df.index)
        
        mask    = future_ret.notna()
        feat_df = feat_df[mask]
        target  = future_ret[mask]
        
        if len(feat_df) < 80:
            raise ValueError("特征数据不足，无法训练XGBoost模型")
        
        self.feature_cols = feat_df.columns.tolist()
        X_raw = feat_df.values
        y = target.values
        
        train_boundary = int(len(X_raw) * 0.70)
        holdout_start  = int(len(X_raw) * 0.85)
        train_end      = train_boundary - self.prediction_days
        early_end      = holdout_start - self.prediction_days
        if train_end < 60:
            raise ValueError("purge 后训练集少于 60 条，无法训练 XGBoost 模型")
        if early_end <= train_boundary:
            raise ValueError("早停集为空，无法训练 XGBoost 模型")
        if holdout_start >= len(X_raw):
            raise ValueError("holdout 评估集为空，无法训练 XGBoost 模型")

        X_train = self.scaler.fit_transform(X_raw[:train_end])
        X_early = self.scaler.transform(X_raw[train_boundary:early_end])
        X_holdout = self.scaler.transform(X_raw[holdout_start:])
        y_train = y[:train_end]
        y_early = y[train_boundary:early_end]
        y_holdout = y[holdout_start:]
        
        # ── 使用贝叶斯优化或默认参数 ─────────────────────
        if self.use_bayes and BAYES_AVAILABLE:
            print("正在使用贝叶斯优化搜索最优超参数...")
            X_bayes = self.scaler.transform(X_raw[:early_end])
            y_bayes = y[:early_end]
            self.best_params = optimize_xgboost(X_bayes, y_bayes)
            
            # 使用优化后的参数训练最终模型
            self.model = XGBRegressor(
                **self.best_params,
                random_state=42,
                n_jobs=config.XGB_N_JOBS,
                early_stopping_rounds=50,
                eval_metric="rmse",
            )
        else:
            # 使用默认参数
            print("使用默认超参数训练...")
            self.model = XGBRegressor(
                n_estimators         = 800,
                learning_rate        = 0.02,
                max_depth            = 4,
                min_child_weight     = 5,
                subsample            = 0.75,
                colsample_bytree     = 0.7,
                colsample_bylevel    = 0.8,
                reg_alpha            = 0.5,
                reg_lambda           = 2.0,
                gamma                = 0.1,
                random_state         = 42,
                n_jobs               = config.XGB_N_JOBS,
                early_stopping_rounds= 50,
                eval_metric          = "rmse",
            )
        
        # 修复：正确的语法
        self.model.fit(X_train, y_train, eval_set=[(X_early, y_early)], verbose=False)
        
        pred_ret_val = self.model.predict(X_holdout)
        # 将收益率预测还原为价格，计算RMSE
        close_val   = df["close"].reindex(feat_df.index).iloc[holdout_start:]
        pred_price  = close_val.values * (1 + pred_ret_val)
        true_price  = close_val.values * (1 + y_holdout)
        self.val_rmse   = float(np.sqrt(mean_squared_error(true_price, pred_price)))
        self.is_trained = True
        
        print(f"训练完成！验证集RMSE: {self.val_rmse:.4f}")
        print(f"特征数量：{len(self.feature_cols)} 个")
        return self.val_rmse
        
    def predict(self, df: pd.DataFrame) -> dict:
        """
        使用训练好的模型进行预测
        
        参数：
            df: 包含价格和特征的DataFrame
        
        返回：
            预测结果字典（包含预测价格、置信区间、日期等）
        """
        if not self.is_trained:
            raise RuntimeError("XGBoost模型尚未训练")
        
        feat_df   = _build_features(df)
        if feat_df.empty:
            raise ValueError("特征构建失败")
        
        last_feat  = feat_df.iloc[[-1]]
        # 只取训练时的特征列（保证一致性）
        for c in self.feature_cols:
            if c not in last_feat.columns:
                last_feat[c] = 0.0
        last_feat = last_feat[self.feature_cols]
        X_last    = self.scaler.transform(last_feat.values)
        
        # 预测累计收益率
        ret_at_end = float(self.model.predict(X_last)[0])
        last_price = float(df["close"].iloc[-1])
        end_price  = last_price * (1 + ret_at_end)
        
        # 线性插值路径（加噪声模拟真实波动）
        predictions = np.linspace(last_price, end_price, self.prediction_days + 1)[1:]
        noise       = np.random.normal(0, self.val_rmse * 0.25, len(predictions))
        predictions = np.clip(predictions + noise, last_price * 0.6, last_price * 1.6)
        
        last_date = df.index[-1]
        dates     = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.prediction_days,
            freq="B",
        )
        
        margin = self.val_rmse * 1.5
        return {
            "predictions":   predictions.tolist(),
            "lower_bound":   (predictions - margin).tolist(),
            "upper_bound":   (predictions + margin).tolist(),
            "dates":         [d.strftime("%Y-%m-%d") for d in dates],
            "last_price":    last_price,
            "pct_change":    float((end_price - last_price) / last_price * 100),
            "val_rmse":      self.val_rmse,
        }
    
    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性排序"""
        if not self.is_trained:
            return pd.Series()
        scores = self.model.feature_importances_
        return pd.Series(scores, index=self.feature_cols).sort_values(ascending=False)
