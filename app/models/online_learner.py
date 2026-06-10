"""
在线学习框架（重构版 v2.0）

功能：
1. 每日增量训练（Daily Update）
2. 模型性能评估与对比（RMSE对比）
3. 模型版本管理（保存多个版本，自动加载RMSE最优版本）
4. 性能日志记录与可视化

使用方式：
- 可选启用，不影响现有流程
- 可通过定时任务（每天15:30收盘后）自动触发
"""

import os
import json
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config


class OnlineLearner:
    """
    在线学习器
    
    功能：
    1. 每日增量训练更新模型
    2. 评估新模型性能，保留更优者
    3. 模型版本管理（保存带时间戳的多个版本）
    4. 自动加载RMSE最优的模型
    5. 记录性能日志
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        初始化在线学习器
        
        Parameters
        ----------
        model_dir : str, 可选
            模型保存目录，默认为 config.ONLINE_MODEL_DIR
        """
        if model_dir is None:
            self.model_dir = config.ONLINE_MODEL_DIR
        else:
            self.model_dir = model_dir
        
        # 确保模型目录存在
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 性能日志目录
        self.log_dir = config.PERFORMANCE_LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 模型版本记录文件
        self.version_file = os.path.join(self.model_dir, "model_versions.json")
        self.control_file = os.path.join(self.model_dir, "model_controls.json")
        
    def daily_update(self, symbol: str, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        每日增量训练更新
        
        流程：
        1. 加载现有模型（RMSE最优版本）
        2. 使用新数据重新训练模型
        3. 评估新模型性能（RMSE）
        4. 与旧模型对比，保留更优者
        5. 保存新版本模型
        
        Parameters
        ----------
        symbol : str
            股票代码
        new_data : pd.DataFrame
            [DEPRECATED] 此参数已弃用，函数内部会主动 fetch_stock_data 拉取完整数据。
            传入空 DataFrame 即可，保留参数仅为向后兼容。
            
        Returns
        -------
        Dict[str, Any] : 更新结果报告
        """
        from sklearn.metrics import mean_squared_error
        
        report = {
            "symbol": symbol,
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models_updated": [],
            "models_skipped": [],
            "performance_log": {}
        }
        
        # 需要更新的模型类型
        try:
            from app.data.fetcher import fetch_stock_data
            from app.analysis.technical import compute_indicators
            from app.data.prediction_tracker import (
                evaluate_matured_predictions,
                summarize_posterior_metrics,
                summarize_posterior_health,
            )
        except Exception:
            from data.fetcher import fetch_stock_data
            from analysis.technical import compute_indicators
            from data.prediction_tracker import (
                evaluate_matured_predictions,
                summarize_posterior_metrics,
                summarize_posterior_health,
            )

        full_df = fetch_stock_data(symbol, years=config.DEFAULT_PERIOD_YEARS)
        if full_df is None or len(full_df) < config.MIN_TRADING_DAYS:
            report["models_skipped"].append({
                "model": "all",
                "reason": "data_not_enough",
            })
            self._save_report(symbol, report)
            return report

        full_df = compute_indicators(full_df)
        report["posterior_eval"] = evaluate_matured_predictions(symbol, full_df)
        report["posterior_summary"] = summarize_posterior_metrics(symbol)
        report["model_controls"] = {}
        report["alerts"] = []

        model_types = ["lstm_short", "lstm_long", "xgb_short", "xgb_long", "prophet"]
        for model_type in model_types:
            control_state, alerts = self._update_model_control(symbol, model_type, summarize_posterior_health(symbol, model_type))
            report["model_controls"][model_type] = control_state
            report["alerts"].extend(alerts)
        
        # 一次性加载所有 model_type 的最优旧模型，避免循环中重复反序列化
        # 同时只保留 rmse（不持有模型引用，节省内存）
        old_rmse_cache: Dict[str, float] = {}
        old_has_cache: Dict[str, bool] = {}
        for mt in model_types:
            _old_model, _old_rmse = self._load_best_model(symbol, mt)
            old_rmse_cache[mt] = _old_rmse
            old_has_cache[mt] = _old_model is not None
            # 显式释放大模型引用
            del _old_model
        # 注意：上面 full_df 已校验过长度，此处不再重复检查
        for model_type in model_types:
            try:
                control_state = report["model_controls"].get(model_type, {})
                # 1. 使用预加载的旧模型 RMSE（不再持有旧模型引用，节省内存）
                old_rmse = old_rmse_cache.get(model_type, float('inf'))
                has_old_model = old_has_cache.get(model_type, False)

                # 2. 训练新模型
                new_model, new_rmse = self._train_model(full_df, model_type)
                

                if new_model is None:
                    report["models_skipped"].append({
                        "model": model_type,
                        "reason": "训练失败"
                    })
                    continue
                
                # 4. 对比性能，决定保留哪个模型
                if has_old_model and old_rmse < float('inf'):
                    # 与旧模型对比
                    decision = self._should_accept_new_model(old_rmse, new_rmse, control_state)
                    if decision["accept"]:
                        # 新模型更优或相当，保存新模型
                        self._save_model(symbol, model_type, new_model, new_rmse)
                        improvement_pct = (old_rmse - new_rmse) / old_rmse * 100
                        report["models_updated"].append({
                            "model": model_type,
                            "old_rmse": round(old_rmse, 4),
                            "new_rmse": round(new_rmse, 4),
                            "improvement_pct": round(improvement_pct, 2),
                            "accept_rule": decision["rule"],
                        })
                    else:
                        # 旧模型更优，保留旧模型
                        report["models_skipped"].append({
                            "model": model_type,
                            "reason": f"旧模型RMSE({old_rmse:.4f})优于新模型({new_rmse:.4f})",
                            "accept_rule": decision["rule"],
                        })
                else:
                    # 没有旧模型，直接保存新模型
                    self._save_model(symbol, model_type, new_model, new_rmse)
                    report["models_updated"].append({
                        "model": model_type,
                        "old_rmse": None,
                        "new_rmse": round(new_rmse, 4),
                        "improvement_pct": None
                    })
                
                # 5. 记录性能日志
                self._log_performance(symbol, model_type, new_rmse, old_rmse if has_old_model else None)
                report["performance_log"][model_type] = {
                    "new_rmse": round(new_rmse, 4),
                    "old_rmse": round(old_rmse, 4) if has_old_model else None
                }
                
            except Exception as e:
                report["models_skipped"].append({
                    "model": model_type,
                    "reason": f"异常: {str(e)}"
                })
                print(f"更新模型 {model_type} 时出错: {e}")
        
        # 保存更新报告
        self._save_report(symbol, report)
        
        return report

    def _control_key(self, symbol: str, model_type: str) -> str:
        return f"{symbol}_{model_type}"

    def _load_control_store(self) -> Dict[str, Dict]:
        if not os.path.exists(self.control_file):
            return {}
        try:
            with open(self.control_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_control_store(self, store: Dict[str, Dict]) -> None:
        with open(self.control_file, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

    def get_model_control(self, symbol: str) -> Dict[str, Dict]:
        store = self._load_control_store()
        result = {}
        for key, value in store.items():
            if key.startswith(f"{symbol}_"):
                result[key[len(symbol) + 1:]] = value
        return result

    def _update_model_control(self, symbol: str, model_type: str, health: Dict) -> tuple[Dict, List[str]]:
        store = self._load_control_store()
        key = self._control_key(symbol, model_type)
        current = store.get(key, {})
        streak = int(current.get("degrade_streak", 0))
        alerts: List[str] = []

        state = {
            "status": health.get("status", "样本不足"),
            "summary": health.get("summary", ""),
            "count": health.get("count", 0),
            "downweight_factor": 1.0,
            "disabled": False,
            "degrade_streak": streak,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if health.get("count", 0) < config.MODEL_CONTROL_MIN_SAMPLES:
            store[key] = {**current, **state}
            self._save_control_store(store)
            return store[key], alerts

        status = health.get("status")
        if status == "稳定提升":
            state["degrade_streak"] = 0
        elif status == "基本稳定":
            state["degrade_streak"] = 0
        elif status == "波动加大":
            state["downweight_factor"] = config.MODEL_DOWNWEIGHT_VOLATILE
            state["degrade_streak"] = streak + 1
            alerts.append(f"{symbol} {model_type} 后验波动加大，已自动轻度降权")
        elif status == "明显退化":
            state["downweight_factor"] = config.MODEL_DOWNWEIGHT_DEGRADED
            state["degrade_streak"] = streak + 1
            alerts.append(f"{symbol} {model_type} 后验明显退化，已自动降权")

        if (
            status == "明显退化"
            and state["degrade_streak"] >= config.MODEL_DISABLE_DEGRADE_STREAK
            and health.get("count", 0) >= config.MODEL_DISABLE_MIN_SAMPLES
        ):
            state["disabled"] = True
            alerts.append(f"{symbol} {model_type} 连续退化，已自动停用")
        elif status in ("稳定提升", "基本稳定"):
            if current.get("disabled"):
                alerts.append(f"{symbol} {model_type} 表现恢复，已重新启用")
            state["disabled"] = False

        store[key] = state
        self._save_control_store(store)
        return state, alerts

    def _should_accept_new_model(self, old_rmse: float, new_rmse: float, control_state: Dict) -> Dict[str, Any]:
        status = control_state.get("status", "样本不足")
        if not np.isfinite(old_rmse):
            return {"accept": True, "rule": "no_baseline"}
        if not np.isfinite(new_rmse):
            return {"accept": False, "rule": "invalid_new_rmse"}

        threshold = 1.0
        rule = "match_or_better"
        if status == "明显退化":
            threshold = 0.98
            rule = "degraded_require_2pct_better"
        elif status == "波动加大":
            threshold = 0.995
            rule = "volatile_require_0.5pct_better"
        elif status == "稳定提升":
            # 稳定提升中也要求新模型不劣于旧模型，避免逐步劣化
            # （原值 1.01 会允许每次最多 1% 的退化，长期积累严重）
            threshold = 1.0
            rule = "improving_require_not_worse"

        return {
            "accept": new_rmse <= old_rmse * threshold,
            "rule": rule,
        }
    
    def _train_model(self, df: pd.DataFrame, model_type: str):
        """
        训练指定类型的模型
        
        Parameters
        ----------
        df : pd.DataFrame
            完整训练数据
        model_type : str
            模型类型（lstm_short, lstm_long, xgb_short, xgb_long, prophet）
            
        Returns
        -------
        tuple : (训练好的模型, RMSE)
        """
        try:
            if "lstm" in model_type:
                horizon = "short" if "short" in model_type else "long"
                days = config.SHORT_TERM_DAYS if horizon == "short" else config.LONG_TERM_DAYS
                lb = config.LSTM_LOOKBACK_SHORT if horizon == "short" else config.LSTM_LOOKBACK_LONG
                try:
                    from app.models.lstm_model import LSTMPredictor
                except Exception:
                    from models.lstm_model import LSTMPredictor
                model = LSTMPredictor(lookback=lb, prediction_days=days, name=f"lstm_{horizon}")
                rmse = model.train(df)
                return model, rmse
                
            elif "xgb" in model_type:
                days = config.SHORT_TERM_DAYS if "short" in model_type else config.LONG_TERM_DAYS
                try:
                    from app.models.xgboost_model import XGBoostPredictor
                except Exception:
                    from models.xgboost_model import XGBoostPredictor
                model = XGBoostPredictor(
                    prediction_days=days,
                    use_bayes=getattr(config, "ONLINE_XGB_USE_BAYES", False),
                )
                rmse = model.train(df)
                return model, rmse
                
            elif model_type == "prophet":
                try:
                    from app.models.prophet_model import ProphetPredictor
                except Exception:
                    from models.prophet_model import ProphetPredictor
                model = ProphetPredictor(prediction_days=config.LONG_TERM_DAYS)
                rmse = model.train(df)
                return model, rmse
                
        except Exception as e:
            print(f"训练模型 {model_type} 失败: {e}")
            return None, float('inf')
        
        return None, float('inf')
    
    def _evaluate_model(self, model, df: pd.DataFrame) -> float:
        """
        评估模型性能（使用验证集RMSE）
        
        Parameters
        ----------
        model : 训练好的模型对象
        df : pd.DataFrame
            评估数据
            
        Returns
        -------
        float : RMSE值
        """
        try:
            # 直接使用模型训练时计算的验证集RMSE
            if hasattr(model, 'val_rmse'):
                return model.val_rmse
            
            # 如果没有val_rmse属性，尝试使用模型进行预测评估
            # 这里使用简单的回测方式
            from sklearn.metrics import mean_squared_error
            
            # 获取模型预测
            result = model.predict(df)
            
            if result is None or "predictions" not in result:
                return float('inf')
            
            # 对于在线学习框架，我们主要依赖模型训练时的验证RMSE
            # 如果需要更复杂的评估，可以在这里实现
            if hasattr(model, 'val_rmse'):
                return model.val_rmse
            
            return float('inf')
                    
        except Exception as e:
            print(f"评估模型失败: {e}")
            return float('inf')
    
    def _save_model(self, symbol: str, model_type: str, model, rmse: float):
        """
        保存模型，带版本管理
        
        Parameters
        ----------
        symbol : str
            股票代码
        model_type : str
            模型类型
        model : 训练好的模型对象
        rmse : float
            模型RMSE
        """
        # 生成带时间戳的版本号
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}"
        
        # 保存模型文件
        model_filename = f"{symbol}_{model_type}_{version}.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'rmse': rmse,
                'timestamp': timestamp,
                'version': version
            }, f)
        
        # 更新版本记录
        self._update_version_record(symbol, model_type, version, model_filename, rmse)
        
        print(f"model saved: {model_filename} (RMSE={rmse:.4f})")
    
    def _load_best_model(self, symbol: str, model_type: str):
        """
        加载RMSE最优的模型（而不仅仅是最新版本）
        
        Parameters
        ----------
        symbol : str
            股票代码
        model_type : str
            模型类型
            
        Returns
        -------
        tuple : (模型对象, RMSE) 或 (None, inf)
        """
        # 读取版本记录
        versions = self._get_version_record(symbol, model_type)
        
        if not versions:
            return None, float('inf')

        control = self.get_model_control(symbol).get(model_type, {})
        if control.get("disabled"):
            return None, float('inf')
        
        # 按 RMSE 升序排序，跳过损坏/空文件，依次尝试加载
        sorted_versions = sorted(versions, key=lambda x: x.get('rmse', float('inf')))
        for candidate in sorted_versions:
            filename = candidate.get("filename")
            if not filename:
                continue
            model_path = os.path.join(self.model_dir, filename)
            if not os.path.exists(model_path):
                continue
            # 跳过 0 字节文件（训练中断导致的损坏快照）
            try:
                if os.path.getsize(model_path) == 0:
                    print(f"跳过 0 字节模型文件: {filename}")
                    continue
            except OSError:
                continue
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                # 校验数据结构
                if not isinstance(data, dict) or 'model' not in data or 'rmse' not in data:
                    print(f"模型文件结构异常，跳过: {filename}")
                    continue
                print(f"loaded best model: {filename} (RMSE={data['rmse']:.4f})")
                return data['model'], data['rmse']
            except (EOFError, pickle.UnpicklingError, ValueError) as e:
                print(f"模型文件损坏，跳过 {filename}: {e}")
                continue
            except Exception as e:
                print(f"加载模型 {filename} 失败: {e}")
                continue
        return None, float('inf')
    
    def _load_latest_model(self, symbol: str, model_type: str):
        """
        加载最新的模型（按时间戳）
        
        Parameters
        ----------
        symbol : str
            股票代码
        model_type : str
            模型类型
            
        Returns
        -------
        模型对象 或 None
        """
        # 读取版本记录
        versions = self._get_version_record(symbol, model_type)
        
        if not versions:
            return None, float('inf')
        
        # 按 timestamp 字段降序排序，依次尝试加载（跳过 0 字节/损坏）
        def _ts_key(v):
            return v.get("timestamp", "")
        sorted_versions = sorted(versions, key=_ts_key, reverse=True)
        for candidate in sorted_versions:
            filename = candidate.get("filename")
            if not filename:
                continue
            model_path = os.path.join(self.model_dir, filename)
            if not os.path.exists(model_path):
                continue
            try:
                if os.path.getsize(model_path) == 0:
                    continue
            except OSError:
                continue
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                if not isinstance(data, dict) or 'model' not in data or 'rmse' not in data:
                    continue
                return data['model'], data['rmse']
            except (EOFError, pickle.UnpicklingError, ValueError) as e:
                print(f"模型文件损坏，跳过 {filename}: {e}")
                continue
            except Exception as e:
                print(f"加载模型 {filename} 失败: {e}")
                continue
        return None, float('inf')
    
    def _update_version_record(self, symbol: str, model_type: str, version: str, 
                             filename: str, rmse: float):
        """更新版本记录JSON文件"""
        # 读取现有记录
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r', encoding='utf-8') as f:
                all_versions = json.load(f)
        else:
            all_versions = {}
        
        # 更新记录
        key = f"{symbol}_{model_type}"
        if key not in all_versions:
            all_versions[key] = []
        
        all_versions[key].append({
            "version": version,
            "filename": filename,
            "rmse": rmse,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_active": True
        })

        versions = all_versions[key]
        recent_keep = versions[-config.MODEL_VERSION_KEEP_RECENT:]
        best_keep = sorted(versions, key=lambda x: x.get("rmse", float("inf")))[:config.MODEL_VERSION_KEEP_BEST]
        keep_names = {item["filename"] for item in recent_keep + best_keep}
        removed = [item for item in versions if item["filename"] not in keep_names]
        all_versions[key] = [item for item in versions if item["filename"] in keep_names]
        for item in removed:
            stale_path = os.path.join(self.model_dir, item["filename"])
            if os.path.exists(stale_path):
                try:
                    os.remove(stale_path)
                except Exception:
                    pass
        
        # 保存记录
        with open(self.version_file, 'w', encoding='utf-8') as f:
            json.dump(all_versions, f, ensure_ascii=False, indent=2)

    def cleanup_model_versions(
        self,
        symbols: Optional[List[str]] = None,
        keep_recent: Optional[int] = None,
        keep_best: Optional[int] = None,
    ) -> Dict[str, Any]:
        """清理旧模型版本，按每只股票/每种模型保留最近版本和RMSE最优版本。"""
        keep_recent = config.MODEL_VERSION_KEEP_RECENT if keep_recent is None else max(0, int(keep_recent))
        keep_best = config.MODEL_VERSION_KEEP_BEST if keep_best is None else max(0, int(keep_best))
        symbol_set = set(symbols or [])

        if os.path.exists(self.version_file):
            with open(self.version_file, 'r', encoding='utf-8') as f:
                all_versions = json.load(f)
        else:
            all_versions = {}

        removed_files: List[str] = []
        freed_bytes = 0

        for key, versions in list(all_versions.items()):
            symbol = key.split("_", 1)[0]
            if symbol_set and symbol not in symbol_set:
                continue
            if not isinstance(versions, list):
                all_versions[key] = []
                continue

            recent_keep = versions[-keep_recent:] if keep_recent else []
            best_keep = (
                sorted(versions, key=lambda x: x.get("rmse", float("inf")))[:keep_best]
                if keep_best
                else []
            )
            keep_names = {
                item.get("filename")
                for item in recent_keep + best_keep
                if item.get("filename")
            }

            pruned_versions = []
            for item in versions:
                filename = item.get("filename")
                if filename in keep_names:
                    pruned_versions.append(item)
                    continue
                if filename:
                    model_path = os.path.join(self.model_dir, filename)
                    if os.path.exists(model_path):
                        try:
                            size = os.path.getsize(model_path)
                            os.remove(model_path)
                            freed_bytes += size
                            removed_files.append(filename)
                        except Exception:
                            pass
            all_versions[key] = pruned_versions

        with open(self.version_file, 'w', encoding='utf-8') as f:
            json.dump(all_versions, f, ensure_ascii=False, indent=2)

        # 额外扫描：删除 0 字节文件 + 孤儿文件（未在 version 记录中的 .pkl）
        orphan_removed: List[str] = []
        zero_byte_removed: List[str] = []
        try:
            all_known_files = {
                item.get("filename")
                for versions in all_versions.values()
                if isinstance(versions, list)
                for item in versions
                if item.get("filename")
            }
            for entry in os.listdir(self.model_dir):
                if not entry.endswith(".pkl"):
                    continue
                # 如果指定了 symbol 过滤，只清理目标 symbol 的孤儿文件
                if symbol_set:
                    file_symbol = entry.split("_", 1)[0]
                    if file_symbol not in symbol_set:
                        continue
                file_path = os.path.join(self.model_dir, entry)
                try:
                    size = os.path.getsize(file_path)
                except OSError:
                    continue
                # 0 字节文件直接删除
                if size == 0:
                    try:
                        os.remove(file_path)
                        zero_byte_removed.append(entry)
                    except OSError:
                        pass
                    continue
                # 孤儿文件（不在版本记录中）
                if entry not in all_known_files:
                    try:
                        os.remove(file_path)
                        freed_bytes += size
                        orphan_removed.append(entry)
                    except OSError:
                        pass
        except OSError:
            pass

        return {
            "removed_count": len(removed_files) + len(orphan_removed) + len(zero_byte_removed),
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / 1024 / 1024, 2),
            "keep_recent": keep_recent,
            "keep_best": keep_best,
            "removed_files": removed_files,
            "orphan_removed": orphan_removed,
            "zero_byte_removed": zero_byte_removed,
        }
    
    def _get_version_record(self, symbol: str, model_type: str) -> List[Dict]:
        """获取版本记录"""
        if not os.path.exists(self.version_file):
            return []
        
        with open(self.version_file, 'r', encoding='utf-8') as f:
            all_versions = json.load(f)
        
        key = f"{symbol}_{model_type}"
        return all_versions.get(key, [])
    
    def _log_performance(self, symbol: str, model_type: str, new_rmse: float, old_rmse: Optional[float]):
        """
        记录性能日志
        
        Parameters
        ----------
        symbol : str
            股票代码
        model_type : str
            模型类型
        new_rmse : float
            新模型RMSE
        old_rmse : float, 可选
            旧模型RMSE
        """
        log_file = os.path.join(self.log_dir, f"{symbol}_{model_type}_performance.csv")
        
        # 创建或追加日志
        import csv
        
        file_exists = os.path.exists(log_file)
        
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头（如果文件不存在）
            if not file_exists:
                writer.writerow(['timestamp', 'new_rmse', 'old_rmse', 'improvement_pct'])
            
            # 计算改进百分比
            if old_rmse and old_rmse > 0:
                improvement = (old_rmse - new_rmse) / old_rmse * 100
            else:
                improvement = None
            
            # 写入记录（improvement=0 也是有效值，需用 is not None 判断）
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(new_rmse, 4),
                round(old_rmse, 4) if (old_rmse is not None and old_rmse > 0) else '',
                round(improvement, 2) if improvement is not None else ''
            ])
    
    def _save_report(self, symbol: str, report: Dict):
        """保存更新报告"""
        report_file = os.path.join(
            self.log_dir, 
            f"{symbol}_update_report_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def get_latest_report(self, symbol: str) -> Dict[str, Any]:
        report_files = [
            os.path.join(self.log_dir, name)
            for name in os.listdir(self.log_dir)
            if name.startswith(f"{symbol}_update_report_") and name.endswith(".json")
        ]
        if not report_files:
            return {}
        report_file = max(report_files, key=os.path.getmtime)
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def collect_model_alerts(self, symbol: str) -> List[str]:
        alerts: List[str] = []
        controls = self.get_model_control(symbol)
        for model_type, state in controls.items():
            status = state.get("status")
            if state.get("disabled"):
                alerts.append(f"{symbol} {model_type} 已自动停用")
            elif status == "明显退化":
                alerts.append(f"{symbol} {model_type} 明显退化，当前已强降权")
            elif status == "波动加大":
                alerts.append(f"{symbol} {model_type} 波动加大，当前已轻度降权")
        return alerts
    
    def get_performance_trend(self, symbol: str, model_type: str) -> pd.DataFrame:
        """
        获取性能趋势数据（用于可视化）
        
        Parameters
        ----------
        symbol : str
            股票代码
        model_type : str
            模型类型
            
        Returns
        -------
        pd.DataFrame : 包含时间戳和RMSE的DataFrame
        """
        log_file = os.path.join(self.log_dir, f"{symbol}_{model_type}_performance.csv")
        
        if not os.path.exists(log_file):
            return pd.DataFrame()
        
        df = pd.read_csv(log_file)
        return df
    
    def visualize_performance(self, symbol: str, model_type: str):
        """
        可视化RMSE趋势图
        
        Parameters
        ----------
        symbol : str
            股票代码
        model_type : str
            模型类型
        """
        try:
            import plotly.express as px
            
            df = self.get_performance_trend(symbol, model_type)
            
            if df.empty:
                print(f"没有找到 {symbol} {model_type} 的性能日志")
                return None
            
            # 创建趋势图
            fig = px.line(
                df, 
                x='timestamp', 
                y='new_rmse',
                title=f'{symbol} - {model_type} - RMSE趋势',
                labels={'timestamp': '时间', 'new_rmse': 'RMSE'}
            )
            
            fig.update_layout(
                template='plotly_dark',
                xaxis_title='时间',
                yaxis_title='RMSE'
            )
            
            return fig
            
        except Exception as e:
            print(f"可视化失败: {e}")
            return None
    

def maybe_startup_cleanup() -> Optional[Dict[str, Any]]:
    """启动时尝试清理孤儿/0字节模型文件。
    使用日期 lockfile 保证每天最多清理一次，避免每次启动都扫描。
    返回清理报告字典（None 表示当天已清理过/已禁用）。
    """
    if not getattr(config, "AUTO_CLEANUP_ON_STARTUP", True):
        return None
    today = datetime.now().strftime("%Y%m%d")
    lock_path = os.path.join(config.ONLINE_MODEL_DIR, f".cleanup_{today}.lock")
    try:
        if os.path.exists(lock_path):
            return None
        # 触发清理
        learner = OnlineLearner()
        result = learner.cleanup_model_versions()
        # 写 lockfile
        try:
            os.makedirs(config.ONLINE_MODEL_DIR, exist_ok=True)
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            # 顺便清理超过 7 天的旧 lockfile
            for entry in os.listdir(config.ONLINE_MODEL_DIR):
                if entry.startswith(".cleanup_") and entry.endswith(".lock"):
                    full = os.path.join(config.ONLINE_MODEL_DIR, entry)
                    try:
                        if (datetime.now().timestamp() - os.path.getmtime(full)) > 7 * 86400:
                            os.remove(full)
                    except OSError:
                        pass
        except OSError:
            pass
        return result
    except Exception as exc:
        print(f"启动时模型清理失败: {exc}")
        return None


def _daily_update_one(symbol: str):
    # daily_update 内部会自己 fetch 完整数据，此处不再传入 new_data
    # 第二个参数保留是为了向后兼容签名，传入空 DataFrame 占位
    learner = OnlineLearner()
    try:
        return symbol, learner.daily_update(symbol, pd.DataFrame())
    except Exception as e:
        return symbol, {"status": "failed", "reason": str(e)}


def _worker_threading_init():
    """ProcessPool worker 启动时限制 TF/MKL/OMP 线程数，避免过度订阅。"""
    try:
        import config as _cfg
        intra = int(getattr(_cfg, "TF_INTRAOP_THREADS", 1))
        inter = int(getattr(_cfg, "TF_INTEROP_THREADS", 2))
    except Exception:
        intra, inter = 1, 2
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OMP_NUM_THREADS", str(intra))
    os.environ.setdefault("MKL_NUM_THREADS", str(intra))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(intra))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(intra))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(inter))


def run_daily_update_all(symbols: List[str], workers: int | None = None) -> Dict[str, Any]:
    """Run daily model updates for multiple stocks, using process-level parallelism."""
    all_reports = {}
    symbols = list(symbols)
    if not symbols:
        return all_reports

    if workers is None:
        workers = min(len(symbols), getattr(config, "PARALLEL_STOCK_WORKERS", 4))
    else:
        workers = max(1, min(len(symbols), int(workers)))

    if workers <= 1:
        for symbol in symbols:
            key, report = _daily_update_one(symbol)
            all_reports[key] = report
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_threading_init,
        ) as pool:
            futures = [pool.submit(_daily_update_one, symbol) for symbol in symbols]
            for fut in as_completed(futures):
                key, report = fut.result()
                all_reports[key] = report

    return all_reports


def write_scheduler_helper(symbols: List[str], schedule_time: str = "15:30") -> Dict[str, str]:
    os.makedirs(config.SCHEDULER_DIR, exist_ok=True)
    runner_path = os.path.join(config.SCHEDULER_DIR, "daily_update_runner.ps1")
    register_path = os.path.join(config.SCHEDULER_DIR, "register_daily_update_task.ps1")
    project_root = config.PROJECT_ROOT
    python_exe = os.path.join(project_root, "python", "python.exe")
    update_script = os.path.join(config.BASE_DIR, "run_daily_update.py")
    data_dir = config.APP_DATA_DIR
    symbols_arg = ",".join(symbols)

    runner_content = f"""$ErrorActionPreference = 'Stop'
$projectRoot = "{project_root}"
$env:GUANGHUAN_STOCK_DATA_DIR = "{data_dir}"
Set-Location $projectRoot
& "{python_exe}" "{update_script}" --symbols "{symbols_arg}"
"""
    register_content = f"""$taskName = "GuangHuanStockDailyUpdate"
$scriptPath = "{runner_path}"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "{schedule_time}"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force
"""
    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(runner_content)
    with open(register_path, "w", encoding="utf-8") as f:
        f.write(register_content)
    return {"runner": runner_path, "register": register_path}


def setup_scheduler(symbols: List[str]):
    """
    设置定时任务调度器（使用schedule库）
    
    功能：
    - 每个交易日15:30收盘后自动执行模型更新
    
    Parameters
    ----------
    symbols : List[str]
        要更新的股票代码列表
    """
    helper_paths: Dict[str, str] = {}
    try:
        helper_paths = write_scheduler_helper(symbols)
        import schedule
        import time
        from threading import Thread
        
        def daily_update_job():
            """每日更新任务"""
            print(f"[{datetime.now()}] 开始执行每日模型更新...")
            reports = run_daily_update_all(symbols)
            print(f"[{datetime.now()}] 每日模型更新完成")
            return reports
        
        # 设置定时任务：每个交易日15:30执行
        schedule.every().monday.at("15:30").do(daily_update_job)
        schedule.every().tuesday.at("15:30").do(daily_update_job)
        schedule.every().wednesday.at("15:30").do(daily_update_job)
        schedule.every().thursday.at("15:30").do(daily_update_job)
        schedule.every().friday.at("15:30").do(daily_update_job)
        
        def run_scheduler():
            """运行调度器（在后台线程中）"""
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        
        # 启动后台线程运行调度器
        scheduler_thread = Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        print("✓ 定时任务调度器已启动（每个交易日15:30执行模型更新）")
        
        return {
            "mode": "in_process",
            "thread": scheduler_thread,
            "helper": helper_paths,
        }

    except ImportError:
        if not helper_paths:
            helper_paths = write_scheduler_helper(symbols)
        return {
            "mode": "external_only",
            "helper": helper_paths,
            "thread": None,
            "reason": "schedule_not_installed",
        }
    except Exception as e:
        if not helper_paths:
            helper_paths = write_scheduler_helper(symbols)
        return {
            "mode": "external_only",
            "helper": helper_paths,
            "thread": None,
            "reason": str(e),
        }
    

# ══════════════════════════════════════════════════════════════════
# 主函数（供测试使用）
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 测试在线学习框架
    test_symbol = "600519"  # 贵州茅台
    
    print("=" * 60)
    print("在线学习框架测试")
    print("=" * 60)
    
    # 创建在线学习器
    learner = OnlineLearner()
    
    # 获取测试数据
    from app.data.fetcher import fetch_stock_data
    test_data = fetch_stock_data(test_symbol, years=3)
    
    if test_data is not None and not test_data.empty:
        print(f"\n开始为 {test_symbol} 执行在线更新...")
        
        # 执行更新
        report = learner.daily_update(test_symbol, test_data)
        
        # 打印报告
        print("\n更新报告:")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        
        # 生成可视化
        print("\n生成性能趋势图...")
        for model_type in ["lstm_short", "xgb_short"]:
            fig = learner.visualize_performance(test_symbol, model_type)
            if fig:
                fig.show()
    else:
        print("无法获取测试数据")
