# AI Agent Instructions

本文件是给 ChatGPT、Claude、Codex 等 AI 编程助手的接手规则。修改代码前请先阅读：

- [README.md](README.md)
- [docs/AI_CONTEXT.md](docs/AI_CONTEXT.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [CHANGELOG.md](CHANGELOG.md)

## 工作原则

- 不要无关重构。
- 不要随意改公共函数签名。
- 不要为了让指标变好看而回退防泄漏修复。
- 修改模型逻辑后必须考虑历史 pkl 兼容性。
- 每次重要修改完成后单独 git commit，并推送到 GitHub。
- 用户要求“完成后 GitHub”时，默认表示提交并推送到 `origin main`。

## 高风险文件

以下文件改动需谨慎：

- [app/app.py](app/app.py)：主界面，体量较大，易引入 Streamlit 前端状态问题。
- [app/models/ensemble.py](app/models/ensemble.py)：集成预测字段依赖强。
- [app/models/scorer.py](app/models/scorer.py)：报告评分字段依赖强。
- [app/models/lstm_model.py](app/models/lstm_model.py)：LSTM/MLP 训练、预测和 pickle 兼容。
- [app/models/xgboost_model.py](app/models/xgboost_model.py)：XGBoost 训练目标和时间序列切分。
- [app/models/online_learner.py](app/models/online_learner.py)：后台在线学习、模型版本和性能日志。

如果必须修改 `ensemble.py`、`scorer.py` 或 `app.py` 的公共接口，请先说明原因再动手。

## 模型规则

- 训练和验证必须按时间顺序切分。
- scaler、标准化器、特征选择器只能在训练段 fit。
- 目标为未来 N 日收益时，切分点前训练样本要做 purge gap，避免标签窗口伸进验证期。
- XGBoost 的 early stopping 集不能同时作为最终评估集。
- `self.val_rmse` 应按还原价格后的误差计算。
- LSTM 当前目标口径是累计收益率，`target_kind` 必须保持 `"return"`。
- 没有 `target_kind="return"` 的旧 LSTM pkl 应加载失效并触发重训。

## 验证命令

优先运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
python app/test_models.py
```

辅助 grep：

```powershell
rg -n "fit_transform|target_kind|purge|holdout" app/models
rg -n "torch|transformers" .
```

## 不要引入的东西

- 不要重新加入 `torch` 或 `transformers`，当前情感分析使用轻量依赖。
- 不要新增大型依赖，除非用户明确同意。
- 不要把用户数据、模型 pkl、缓存、日志加入 git。

## 常见问题

- Streamlit 页面必须通过 `streamlit run app/app.py` 或启动脚本运行。
- 浏览器关闭不代表后台服务停止。
- Windows 控制台可能需要 `PYTHONIOENCODING=utf-8`，否则中文或符号打印可能报编码错误。
- 第一次生成报告可能较慢，因为需要下载行情、构建特征和训练初始模型。
