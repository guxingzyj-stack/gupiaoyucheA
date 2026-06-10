"""
三模型融合模块（完整修复版 v3.0）
按验证RMSE动态分配权重：RMSE越低权重越高

高级功能：
1. 动态加权集成 - 根据最近预测误差动态调整权重
2. Stacking集成 - 使用Ridge回归作为元模型学习最优组合
3. 修复：使用TimeSeriesSplit替代KFold（避免时间序列数据泄露）
"""

import warnings
import traceback
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any
from collections import deque

warnings.filterwarnings("ignore")

import config


def _rmse_weights(rmse_dict: dict) -> dict:
    """RMSE越小权重越高（反比权重，归一化）"""
    vals  = {k: v for k, v in rmse_dict.items() if v < float("inf") and v > 0}
    if not vals:
        return {k: 1.0 / len(rmse_dict) for k in rmse_dict}
    median_rmse = float(np.median(list(vals.values())))
    outlier_mult = getattr(config, "RMSE_OUTLIER_MULTIPLIER", 2.5)
    outlier_pen = getattr(config, "RMSE_OUTLIER_PENALTY", 0.25)
    inv = {}
    for k, v in vals.items():
        penalty = 1.0
        if median_rmse > 0 and v > median_rmse * outlier_mult:
            penalty = outlier_pen
        inv[k] = (1.0 / v) * penalty
    total = sum(inv.values())
    weights = {k: inv[k] / total for k in vals}
    for k in rmse_dict:
        if k not in weights:
            weights[k] = 0.0
    return weights


def _align_predictions(preds_list: list, target_days: int) -> list:
    """将多个预测序列对齐到同一长度"""
    aligned = []
    for p in preds_list:
        arr = np.array(p)
        if len(arr) >= target_days:
            aligned.append(arr[:target_days])
        else:
            # 用最后一个值填充
            pad   = np.full(target_days - len(arr), arr[-1] if len(arr) > 0 else 0)
            aligned.append(np.concatenate([arr, pad]))
    return aligned


class DynamicEnsemble:
    """
    动态加权集成类
    
    功能：
    1. 记录每个模型最近N次预测误差
    2. 根据平均误差动态计算权重（误差越小权重越大）
    3. 支持在线更新权重
    """
    
    def __init__(self, window_size: int = 20):
        """
        初始化动态集成器
        
        Parameters
        ----------
        window_size : int, 默认 20
            记录最近多少次预测误差用于计算动态权重
        """
        self.window_size = window_size
        # 使用deque存储每个模型的最近预测误差（RMSE）
        self.error_history: Dict[str, deque] = {}
        # 动态权重缓存
        self.dynamic_weights: Dict[str, float] = {}
        
    def update_weights(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        根据真实值和各模型预测值更新动态权重
        
        原理：计算每个模型的最近预测误差，误差越小权重越大
        
        Parameters
        ----------
        y_true : np.ndarray
            真实值数组
        predictions_dict : Dict[str, np.ndarray]
            各模型的预测值，格式为 {模型名: 预测值数组}
            
        Returns
        -------
        Dict[str, float] : 更新后的动态权重字典
        """
        from sklearn.metrics import mean_squared_error
        
        # 计算每个模型的预测误差（RMSE）
        for model_name, y_pred in predictions_dict.items():
            if model_name not in self.error_history:
                self.error_history[model_name] = deque(maxlen=self.window_size)
            
            # 确保y_true和y_pred长度一致
            min_len = min(len(y_true), len(y_pred))
            y_true_trimmed = y_true[:min_len]
            y_pred_trimmed = y_pred[:min_len]
            
            # 计算RMSE
            rmse = np.sqrt(mean_squared_error(y_true_trimmed, y_pred_trimmed))
            self.error_history[model_name].append(rmse)
        
        # 计算每个模型的平均误差
        avg_errors = {}
        for model_name, errors in self.error_history.items():
            if len(errors) > 0:
                avg_errors[model_name] = np.mean(errors)
        
        # 根据平均误差计算动态权重（误差越小权重越大）
        if avg_errors:
            # 使用反比权重
            inv_errors = {}
            for model_name, error in avg_errors.items():
                if error > 0:
                    inv_errors[model_name] = 1.0 / error
                else:
                    # 如果误差为0，给予最大权重
                    inv_errors[model_name] = float('inf')
            
            # 归一化权重
            total_inv = sum(inv_errors.values())
            if total_inv > 0 and not np.isinf(total_inv):
                self.dynamic_weights = {k: v / total_inv for k, v in inv_errors.items()}
            else:
                # 处理无穷大值（某个模型误差为0）
                self.dynamic_weights = {}
                for model_name, inv_val in inv_errors.items():
                    if np.isinf(inv_val):
                        self.dynamic_weights[model_name] = 1.0
                    else:
                        self.dynamic_weights[model_name] = 0.0
                # 归一化
                weight_sum = sum(self.dynamic_weights.values())
                if weight_sum > 0:
                    self.dynamic_weights = {k: v / weight_sum for k, v in self.dynamic_weights.items()}
        
        return self.dynamic_weights
    
    def get_weights(self) -> Dict[str, float]:
        """
        获取当前动态权重
        
        Returns
        -------
        Dict[str, float] : 当前动态权重字典，如果尚未计算则返回空字典
        """
        return self.dynamic_weights
    
    def get_error_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取各模型误差统计摘要
        
        Returns
        -------
        Dict[str, Dict[str, float]] : 各模型的误差统计（均值、标准差、最近误差）
        """
        summary = {}
        for model_name, errors in self.error_history.items():
            if len(errors) > 0:
                summary[model_name] = {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'last_error': errors[-1] if errors else None,
                    'sample_count': len(errors)
                }
        return summary


def stacking_ensemble(models: List[Any], X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, meta_model_type: str = 'ridge') -> np.ndarray:
    """
    Stacking第二层集成学习（修复版：使用TimeSeriesSplit）
    
    原理：
    1. 第一层：各模型分别进行预测，预测结果作为新特征
    2. 第二层：使用元模型（默认Ridge回归）学习最优组合权重
    
    Parameters
    ----------
    models : List[Any]
        第一层模型列表，每个模型需有predict方法
    X_train : np.ndarray
        训练特征
    y_train : np.ndarray
        训练目标
    X_test : np.ndarray
        测试特征
    meta_model_type : str, 默认 'ridge'
        元模型类型，目前支持 'ridge'
        
    Returns
    -------
    np.ndarray : Stacking集成模型的预测结果
    """
    from sklearn.linear_model import Ridge
    
    if len(models) < 2:
        raise ValueError("Stacking需要至少2个模型")
    
    # 第一层：获取每个模型的预测（作为新特征）
    # 使用TimeSeriesSplit生成训练集的预测（避免过拟合和数据泄露）
    from sklearn.model_selection import TimeSeriesSplit
    
    n_models = len(models)
    n_samples = len(X_train)
    
    # 存储第一层预测（训练集）
    layer1_preds_train = np.zeros((n_samples, n_models))
    
    # 使用TimeSeriesSplit（不是KFold！保持时间顺序）
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in tscv.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train = y_train[train_idx]
        
        for i, model in enumerate(models):
            # 训练模型
            model.fit(X_fold_train, y_fold_train)
            # 预测验证集（这部分预测用于训练元模型）
            pred_val = model.predict(X_fold_val)
            layer1_preds_train[val_idx, i] = pred_val
    
    # 第一层：获取测试集预测（使用全部训练数据训练每个模型）
    layer1_preds_test = np.zeros((len(X_test), n_models))
    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        layer1_preds_test[:, i] = model.predict(X_test)
    
    # 第二层：训练元模型
    if meta_model_type == 'ridge':
        meta_model = Ridge(alpha=1.0, random_state=42)
    else:
        raise ValueError(f"不支持的元模型类型: {meta_model_type}")
    
    # 使用第一层预测作为特征，训练元模型
    meta_model.fit(layer1_preds_train, y_train)
    
    # 使用元模型进行最终预测
    final_predictions = meta_model.predict(layer1_preds_test)
    
    return final_predictions


class EnsemblePredictor:
    """
    训练并融合 LSTM / XGBoost / Prophet 三个模型
    自动处理任意模型训练失败的情况（至少1个成功即可运行）
    
    增强功能：
    1. 支持动态加权集成
    2. 支持Stacking集成（使用TimeSeriesSplit）
    """

    def __init__(self, use_dynamic_weights: bool = False, use_stacking: bool = False):
        """
        初始化集成预测器
        
        Parameters
        ----------
        use_dynamic_weights : bool, 默认 False
            是否使用动态加权（根据最近预测误差调整权重）
        use_stacking : bool, 默认 False
            是否使用Stacking集成（需要更多计算资源）
        """
        self.models            = {}
        self.weights           = {}
        self.rmse_dict         = {}
        self.is_trained        = False
        self.use_dynamic_weights = use_dynamic_weights
        self.use_stacking        = use_stacking
        self.posterior_weights   = {"short": {}, "long": {}}
        self.model_control       = {}
        self.symbol_context      = None
        
        # 动态集成器（如果启用）
        if use_dynamic_weights:
            self.dynamic_ensemble = DynamicEnsemble(window_size=20)
        else:
            self.dynamic_ensemble = None
        
        # Stacking模型列表（如果启用）
        self.stacking_models = []
        self.stacking_meta_model = None

    def _load_posterior_weights(self, symbol: str):
        # 解决双层 try 嵌套和模块导入兜底的可读性问题
        get_posterior_model_weights = None
        for import_path in ("data.prediction_tracker", "app.data.prediction_tracker"):
            try:
                module = __import__(import_path, fromlist=["get_posterior_model_weights"])
                get_posterior_model_weights = module.get_posterior_model_weights
                break
            except ImportError:
                continue
        if get_posterior_model_weights is None:
            return {"short": {}, "long": {}}
        try:
            return {
                "short": get_posterior_model_weights(symbol, "short"),
                "long": get_posterior_model_weights(symbol, "long"),
            }
        except Exception as exc:
            print(f"加载后验权重失败 [{symbol}]: {exc}")
            return {"short": {}, "long": {}}

    def _load_model_control(self, symbol: str):
        try:
            from models.online_learner import OnlineLearner
        except Exception:
            try:
                from app.models.online_learner import OnlineLearner
            except Exception:
                return {}
        try:
            learner = OnlineLearner()
            return learner.get_model_control(symbol)
        except Exception:
            return {}

    @staticmethod
    def _blend_posterior_weights(base_weights: Dict[str, float], posterior_weights: Dict[str, float]) -> Dict[str, float]:
        if not posterior_weights:
            return base_weights
        combined = {}
        for key, base in base_weights.items():
            posterior = posterior_weights.get(key)
            factor = 1.0 if posterior is None else (0.60 + 0.80 * posterior)
            combined[key] = max(base * factor, 0.0)
        total = sum(combined.values())
        if total <= 0:
            return base_weights
        return {k: v / total for k, v in combined.items()}

    def load_pretrained(self, symbol: str, console=None) -> dict:
        """Load best saved online models for report-time inference."""

        def log(msg):
            if console:
                console.print(msg)

        model_map = [
            ("lstm_short", "lstm_short"),
            ("lstm_long", "lstm_long"),
            ("xgb_short", "xgb_short"),
            ("xgb_long", "xgb_long"),
            ("prophet", "prophet"),
        ]

        try:
            from models.online_learner import OnlineLearner
        except Exception:
            from app.models.online_learner import OnlineLearner

        learner = OnlineLearner()
        self.models = {}
        self.rmse_dict = {}
        self.symbol_context = symbol
        self.posterior_weights = self._load_posterior_weights(symbol)
        self.model_control = self._load_model_control(symbol)

        for model_type, key in model_map:
            model, rmse = learner._load_best_model(symbol, model_type)
            if model is not None and rmse < float("inf"):
                self.models[key] = model
                self.rmse_dict[key] = rmse
                log(f"  [green]loaded {model_type} RMSE={rmse:.4f}[/green]")
            else:
                self.rmse_dict[key] = float("inf")

        self.is_trained = any(v < float("inf") for v in self.rmse_dict.values())
        return self.rmse_dict if self.is_trained else {}

    def train(self, df: pd.DataFrame, console=None) -> dict:
        """
        训练三个模型（5 个子模型并发训练）

        Parameters
        ----------
        df      : 带技术指标的 DataFrame
        console : rich Console 对象（可选，用于打印进度）

        Returns
        -------
        dict: {model_name: rmse}
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        log_lock = threading.Lock()

        def log(msg):
            if console:
                with log_lock:
                    console.print(msg)

        # ── 训练任务定义 ─────────────────────────────────
        def _train_lstm(horizon, days, lb):
            key = f"lstm_{horizon}"
            try:
                from models.lstm_model import LSTMPredictor
                log(f"  [cyan]训练 LSTM-{horizon}[/cyan] (预测{days}天)...")
                m = LSTMPredictor(lookback=lb, prediction_days=days, name=key)
                rmse = m.train(df)
                log(f"  [green]✓ LSTM-{horizon} 验证RMSE={rmse:.4f}[/green]")
                return key, m, rmse
            except Exception as e:
                log(f"  [yellow]⚠ LSTM-{horizon} 训练失败: {e}[/yellow]")
                return key, None, float("inf")

        def _train_xgb(horizon, days):
            key = f"xgb_{horizon}"
            try:
                from models.xgboost_model import XGBoostPredictor
                log(f"  [cyan]训练 XGBoost-{horizon}[/cyan] (预测{days}天)...")
                m = XGBoostPredictor(prediction_days=days)
                rmse = m.train(df)
                log(f"  [green]✓ XGBoost-{horizon} 验证RMSE={rmse:.4f}[/green]")
                return key, m, rmse
            except Exception as e:
                log(f"  [yellow]⚠ XGBoost-{horizon} 训练失败: {e}[/yellow]")
                return key, None, float("inf")

        def _train_prophet():
            key = "prophet"
            try:
                from models.prophet_model import ProphetPredictor
                log(f"  [cyan]训练 Prophet[/cyan] (预测{config.LONG_TERM_DAYS}天)...")
                m = ProphetPredictor(prediction_days=config.LONG_TERM_DAYS)
                rmse = m.train(df)
                log(f"  [green]✓ Prophet 验证RMSE={rmse:.4f}[/green]")
                return key, m, rmse
            except Exception as e:
                log(f"  [yellow]⚠ Prophet 训练失败: {e}[/yellow]")
                return key, None, float("inf")

        # ── 并发执行（TF/XGB/Prophet 训练时释放 GIL，ThreadPool 即可）──
        tasks = [
            (_train_lstm, "short", config.SHORT_TERM_DAYS, config.LSTM_LOOKBACK_SHORT),
            (_train_lstm, "long",  config.LONG_TERM_DAYS,  config.LSTM_LOOKBACK_LONG),
            (_train_xgb,  "short", config.SHORT_TERM_DAYS),
            (_train_xgb,  "long",  config.LONG_TERM_DAYS),
            (_train_prophet,),
        ]
        workers = max(1, min(len(tasks), getattr(config, "ENSEMBLE_TRAIN_WORKERS", 2)))
        log(f"  [dim]并行训练 {len(tasks)} 个模型，workers={workers}[/dim]")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(task[0], *task[1:]) for task in tasks]
            for fut in as_completed(futures):
                try:
                    key, model, rmse = fut.result()
                    if model is not None:
                        self.models[key] = model
                    self.rmse_dict[key] = rmse
                except Exception as e:
                    log(f"  [yellow]⚠ 训练任务异常: {e}[/yellow]")

        if not any(v < float("inf") for v in self.rmse_dict.values()):
            raise RuntimeError("所有预测模型均训练失败，请检查依赖安装")

        self.is_trained = True
        return self.rmse_dict

    def predict_short(self, df: pd.DataFrame) -> Optional[dict]:
        """融合短期预测（10天）"""
        return self._fuse(df, "short", config.SHORT_TERM_DAYS)

    def predict_long(self, df: pd.DataFrame) -> Optional[dict]:
        """融合中期预测（30天）"""
        return self._fuse(df, "long", config.LONG_TERM_DAYS)

    def _fuse(self, df: pd.DataFrame, horizon: str, days: int) -> Optional[dict]:
        if not self.is_trained:
            return None

        if self.symbol_context and not self.model_control:
            self.model_control = self._load_model_control(self.symbol_context)

        # 3 个模型推理并行（ThreadPool；TF/XGB predict 释放 GIL）
        from concurrent.futures import ThreadPoolExecutor, as_completed
        keys_to_run = [
            k for k in [f"lstm_{horizon}", f"xgb_{horizon}", "prophet"]
            if k in self.models and not self.model_control.get(k, {}).get("disabled")
        ]
        candidates = {}
        if keys_to_run:
            def _run_one(k):
                try:
                    return k, self.models[k].predict(df)
                except Exception:
                    return k, None
            with ThreadPoolExecutor(max_workers=min(3, len(keys_to_run))) as pool:
                for fut in as_completed([pool.submit(_run_one, k) for k in keys_to_run]):
                    k, result = fut.result()
                    if result is not None:
                        candidates[k] = result

        if not candidates:
            return None

        # 如果使用Stacking集成
        if self.use_stacking and len(candidates) >= 2:
            # TODO: 实现Stacking集成的预测逻辑
            # 当前版本暂时使用静态权重
            print("⚠️ Stacking集成预测功能待实现，暂时使用静态权重")
            rmse_sub = {k: self.rmse_dict.get(k, float("inf")) for k in candidates}
            weights  = self._blend_posterior_weights(_rmse_weights(rmse_sub), self.posterior_weights.get(horizon, {}))
        else:
            # 计算权重（根据配置选择静态或动态权重）
            if self.use_dynamic_weights and self.dynamic_ensemble is not None:
                # 使用动态权重（如果有历史误差记录）
                dynamic_weights = self.dynamic_ensemble.get_weights()
                if dynamic_weights and len(dynamic_weights) > 0:
                    weights = dynamic_weights
                else:
                    # 如果没有动态权重，使用静态权重作为初始值
                    rmse_sub = {k: self.rmse_dict.get(k, float("inf")) for k in candidates}
                    weights  = self._blend_posterior_weights(_rmse_weights(rmse_sub), self.posterior_weights.get(horizon, {}))
            else:
                # 使用静态权重（基于验证集RMSE）
                rmse_sub = {k: self.rmse_dict.get(k, float("inf")) for k in candidates}
                weights  = self._blend_posterior_weights(_rmse_weights(rmse_sub), self.posterior_weights.get(horizon, {}))

        adjusted_weights = {}
        for key, weight in weights.items():
            factor = float(self.model_control.get(key, {}).get("downweight_factor", 1.0) or 1.0)
            adjusted_weights[key] = max(weight * factor, 0.0)
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        preds_list  = [np.array(candidates[k]["predictions"]) for k in candidates]
        lower_list  = [np.array(candidates[k]["lower_bound"]) for k in candidates]
        upper_list  = [np.array(candidates[k]["upper_bound"]) for k in candidates]

        preds_al = _align_predictions(preds_list, days)
        lower_al = _align_predictions(lower_list, days)
        upper_al = _align_predictions(upper_list, days)

        w_arr   = np.array([weights[k] for k in candidates])
        fused   = np.average(np.stack(preds_al), axis=0, weights=w_arr)
        f_lower = np.average(np.stack(lower_al), axis=0, weights=w_arr)
        f_upper = np.average(np.stack(upper_al), axis=0, weights=w_arr)

        # 日期用第一个成功模型的日期
        dates      = list(candidates.values())[0]["dates"][:days]
        last_price = float(df["close"].iloc[-1])
        component_predictions = {}
        candidate_keys = list(candidates.keys())
        for idx, key in enumerate(candidate_keys):
            component_predictions[key] = {
                "target_date": dates[-1],
                "target_price": float(preds_al[idx][-1]),
                "lower_bound": float(lower_al[idx][-1]),
                "upper_bound": float(upper_al[idx][-1]),
                "pct_change": float((preds_al[idx][-1] - last_price) / last_price * 100),
            }

        return {
            "predictions":   fused.tolist(),
            "lower_bound":   f_lower.tolist(),
            "upper_bound":   f_upper.tolist(),
            "dates":         dates,
            "last_price":    last_price,
            "pct_change":    float((fused[-1] - last_price) / last_price * 100),
            "weights":       weights,
            "model_rmse":    {k: self.rmse_dict[k] for k in candidates},
            "dynamic_weights": self.dynamic_ensemble.get_weights() if self.dynamic_ensemble else {},
            "component_predictions": component_predictions,
        }

    def update_dynamic_weights(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        更新动态权重（在获得真实值后调用）
        
        Parameters
        ----------
        y_true : np.ndarray
            真实价格数组
        predictions_dict : Dict[str, np.ndarray]
            各模型的预测结果
            
        Returns
        -------
        Dict[str, float] : 更新后的动态权重
        """
        if self.dynamic_ensemble is not None:
            return self.dynamic_ensemble.update_weights(y_true, predictions_dict)
        return {}

    def get_dynamic_weight_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取动态权重误差统计摘要
        
        Returns
        -------
        Dict[str, Dict[str, float]] : 各模型误差统计
        """
        if self.dynamic_ensemble is not None:
            return self.dynamic_ensemble.get_error_summary()
        return {}
