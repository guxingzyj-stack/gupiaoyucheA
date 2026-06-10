"""
趋势预测模型
优先使用 Facebook Prophet（Python<=3.12）
Python 3.13+ 自动降级为多项式回归趋势外推
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

import config

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    pass


class ProphetPredictor:
    def __init__(self, prediction_days: int = config.LONG_TERM_DAYS):
        self.prediction_days = prediction_days
        self.model           = None
        self.is_trained      = False
        self.val_rmse        = float("inf")
        self.backend         = "prophet" if PROPHET_AVAILABLE else "regression"
        self._df             = None

    def train(self, df: pd.DataFrame) -> float:
        self._df = df
        if self.backend == "prophet":
            return self._train_prophet(df)
        else:
            return self._train_regression(df)

    def _train_prophet(self, df: pd.DataFrame) -> float:
        prop_df = pd.DataFrame({"ds": df.index, "y": df["close"].values}).reset_index(drop=True)

        if "volume" in df.columns:
            vol_norm = (df["volume"] - df["volume"].mean()) / (df["volume"].std() + 1e-9)
            prop_df["volume_norm"] = vol_norm.values

        split    = int(len(prop_df) * 0.85)
        train_df = prop_df.iloc[:split]
        val_df   = prop_df.iloc[split:]

        self.model = Prophet(
            changepoint_prior_scale = 0.1,
            seasonality_prior_scale = 5.0,
            yearly_seasonality      = True,
            weekly_seasonality      = True,
            daily_seasonality       = False,
            uncertainty_samples     = 200,
        )
        if "volume_norm" in prop_df.columns:
            self.model.add_regressor("volume_norm")

        self.model.fit(train_df, iter=200)

        future_val = val_df[["ds"]].copy()
        if "volume_norm" in prop_df.columns:
            future_val["volume_norm"] = val_df["volume_norm"].values

        fc       = self.model.predict(future_val)
        pred_val = fc["yhat"].values
        true_val = val_df["y"].values

        self.val_rmse   = float(np.sqrt(mean_squared_error(true_val, pred_val)))
        self.is_trained = True
        return self.val_rmse

    def _train_regression(self, df: pd.DataFrame) -> float:
        """多项式回归 + 季节性分解作为 Prophet 降级替代"""
        close  = df["close"].values
        n      = len(close)

        # 时间特征
        t      = np.arange(n).reshape(-1, 1)
        # 周期性特征（月份、星期）
        month  = np.array([d.month for d in df.index]).reshape(-1, 1)
        dow    = np.array([d.dayofweek for d in df.index]).reshape(-1, 1)
        sin_m  = np.sin(2 * np.pi * month / 12)
        cos_m  = np.cos(2 * np.pi * month / 12)

        X = np.hstack([t, t**2, t**3, sin_m, cos_m, dow])

        split   = int(n * 0.85)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = close[:split], close[split:]

        self.model = Ridge(alpha=1.0)
        self.model.fit(X_tr, y_tr)

        pred_val      = self.model.predict(X_val)
        self.val_rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))

        # 保存参数供预测使用
        self._n_train   = n
        self._last_date = df.index[-1]

        self.is_trained = True
        return self.val_rmse

    def predict(self, df: pd.DataFrame) -> dict:
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")

        if self.backend == "prophet":
            return self._predict_prophet(df)
        else:
            return self._predict_regression(df)

    def _predict_prophet(self, df: pd.DataFrame) -> dict:
        last_date = df.index[-1]
        future    = self.model.make_future_dataframe(periods=self.prediction_days, freq="B")
        if "volume_norm" in self.model.extra_regressors:
            future["volume_norm"] = 0.0

        fc        = self.model.predict(future)
        future_fc = fc[fc["ds"] > last_date].head(self.prediction_days)
        predictions = future_fc["yhat"].values
        lower_bound = future_fc["yhat_lower"].values
        upper_bound = future_fc["yhat_upper"].values
        dates       = future_fc["ds"].dt.strftime("%Y-%m-%d").tolist()

        last_price  = float(df["close"].iloc[-1])
        if len(predictions) > 0:
            shift       = last_price - predictions[0]
            decay       = np.exp(-np.linspace(0, 2, len(predictions)))
            predictions = predictions + shift * decay
            lower_bound = lower_bound + shift * decay
            upper_bound = upper_bound + shift * decay

        return self._build_result(predictions, lower_bound, upper_bound, dates, last_price)

    def _predict_regression(self, df: pd.DataFrame) -> dict:
        n         = self._n_train
        last_date = df.index[-1]
        dates_fut = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.prediction_days, freq="B"
        )
        t_fut = np.arange(n, n + self.prediction_days).reshape(-1, 1)
        month_fut = np.array([d.month for d in dates_fut]).reshape(-1, 1)
        dow_fut   = np.array([d.dayofweek for d in dates_fut]).reshape(-1, 1)
        sin_m = np.sin(2 * np.pi * month_fut / 12)
        cos_m = np.cos(2 * np.pi * month_fut / 12)

        X_fut       = np.hstack([t_fut, t_fut**2, t_fut**3, sin_m, cos_m, dow_fut])
        predictions = self.model.predict(X_fut)

        # 使预测从当前价格平滑过渡
        last_price = float(df["close"].iloc[-1])
        shift      = last_price - predictions[0]
        decay      = np.exp(-np.linspace(0, 3, self.prediction_days))
        predictions = predictions + shift * decay

        margin      = self.val_rmse * 1.8
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        dates       = [d.strftime("%Y-%m-%d") for d in dates_fut]

        return self._build_result(predictions, lower_bound, upper_bound, dates, last_price)

    def _build_result(self, predictions, lower_bound, upper_bound, dates, last_price) -> dict:
        return {
            "predictions": predictions.tolist(),
            "lower_bound": lower_bound.tolist(),
            "upper_bound": upper_bound.tolist(),
            "dates":       dates,
            "last_price":  last_price,
            "pct_change":  float((predictions[-1] - last_price) / last_price * 100),
            "val_rmse":    self.val_rmse,
            "backend":     self.backend,
        }
