# AI Context

这是光环智能股票预测系统的 AI 接手背景说明。

## 项目目标

本项目提供 A 股本地预测和分析：

- 读取自选股。
- 拉取行情和新闻。
- 生成技术、基本面、情绪和预测评分。
- 训练或加载多个模型。
- 输出单股报告、多股对比、历史预测回顾和在线学习结果。

## 当前关键状态

- 当前版本：`v2.3.1`，定义在 [app/config.py](../app/config.py)。
- 主界面：Streamlit，入口为 [app/app.py](../app/app.py)。
- 推荐启动：项目根目录运行 `启动光环智能.bat` 或 `launch.bat`。
- 推荐测试：`python app/test_models.py`。
- 远端仓库：`https://github.com/guxingzyj-stack/gupiaoyucheA.git`。

## 已完成的重要修复

### 时间序列数据泄漏修复

LSTM 和 XGBoost 已修复训练/验证泄漏问题：

- LSTM scaler 不再对全量数据 fit，只在训练段 fit。
- LSTM 目标由缩放价格改为未来累计收益率。
- LSTM 和 XGBoost 均在切分点做 purge gap。
- XGBoost 已拆分为训练集、早停集和 holdout 评估集。
- XGBoost 的 `self.val_rmse` 来自 holdout，而不是 early stopping 集。

### LSTM 旧模型兼容防护

旧版 LSTM pkl 是按“缩放价格目标”训练的。当前版本按“收益率目标”预测。

因此 [app/models/lstm_model.py](../app/models/lstm_model.py) 增加：

```python
self.target_kind = "return"
```

反序列化时，如果旧 state 没有 `target_kind="return"`，会：

- `self.model = None`
- `self.is_trained = False`
- `self.target_kind = "return"`
- 打印旧模型失效提示

这样上游会按缺失模型处理并触发重训。

## 重要约束

- 不要为了降低 RMSE 回退防泄漏修复。
- 验证 RMSE 变差是正常的，代表更真实的泛化评估。
- 不要改动模型返回 dict 的字段结构，`ensemble.py` 和 `scorer.py` 依赖这些字段。
- 不要随意清理用户的 `online_models`、`performance_logs`、`prediction_logs`。
- 不要提交模型 pkl、缓存、日志或用户自选股数据。

## 常用路径

配置集中在 [app/config.py](../app/config.py)：

- `APP_DATA_DIR`
- `MODEL_DIR`
- `ONLINE_MODEL_DIR`
- `PREDICTION_LOG_DIR`
- `PERFORMANCE_LOG_DIR`
- `REPORT_RETRAIN_IF_MISSING`
- `MODEL_VERSION_KEEP_RECENT`
- `MODEL_VERSION_KEEP_BEST`

安装版用户数据默认位于 `%LOCALAPPDATA%\GuangHuanStock`，也可通过 `GUANGHUAN_STOCK_DATA_DIR` 指定。

## 推荐修改流程

1. 先读本文件、[AGENTS.md](../AGENTS.md) 和 [docs/ARCHITECTURE.md](ARCHITECTURE.md)。
2. 用 `rg` 定位代码，不要全局盲改。
3. 小范围修改。
4. 运行 `python app/test_models.py`。
5. 检查 `git diff`。
6. 单独 commit。
7. 推送 GitHub。

## 给外部 AI 的推荐提示

```text
请先阅读 README.md、AGENTS.md、docs/AI_CONTEXT.md、docs/ARCHITECTURE.md 和 CHANGELOG.md。
这是一个 A 股本地预测系统，重点关注时间序列防泄漏、模型输出兼容和 Streamlit 稳定性。
修改时不要做无关重构，完成后运行 python app/test_models.py。
```
