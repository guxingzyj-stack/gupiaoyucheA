# Architecture

本文档说明光环智能股票预测系统的主要结构，方便后续维护和 AI 助手接手。

## 启动流程

```text
启动光环智能.bat
  -> launch.bat
    -> scripts/launch_app.py
      -> python -m streamlit run app/app.py
      -> 打开 http://localhost:8501
```

[scripts/launch_app.py](../scripts/launch_app.py) 会：

- 检查 `8501` 端口。
- 设置 `GUANGHUAN_STOCK_DATA_DIR`。
- 后台启动 Streamlit。
- 打开浏览器。

## 主界面

[app/app.py](../app/app.py) 是 Streamlit 主界面，包含：

- 左侧品牌、自选股和分析设置。
- 单股分析报告。
- 多股对比看板。
- 历史评分。
- 在线学习和模型管理。
- 历史预测回顾。

该文件较大，修改时应尽量局部处理，避免破坏组件 key 和 Streamlit 前端状态。

## 配置层

[app/config.py](../app/config.py) 定义：

- 应用版本。
- 基础目录和用户数据目录。
- 模型、缓存、日志、历史记录路径。
- 短期和长期预测天数。
- LSTM、XGBoost、Prophet 参数。
- 在线学习和模型保留策略。

## 数据与分析

主要目录：

- `app/data`：数据获取和自选股相关代码。
- `app/analysis`：行情、技术指标、基本面、新闻情绪等分析模块。
- `app/visualization`：图表构建。
- `app/history`：历史记录。

实际数据可能来自 AkShare、缓存文件和用户数据目录。

## 模型层

### [app/models/xgboost_model.py](../app/models/xgboost_model.py)

职责：

- 构建 XGBoost 特征。
- 使用未来 N 日累计收益率作为目标。
- 按时间顺序拆分训练、早停和 holdout。
- 在训练边界做 purge gap。
- 用 holdout 价格 RMSE 作为 `val_rmse`。

注意：

- scaler 只能在训练段 fit。
- 贝叶斯优化不能接触最后 15% holdout 数据。

### [app/models/lstm_model.py](../app/models/lstm_model.py)

职责：

- TensorFlow 可用时使用 Keras LSTM/BiLSTM。
- TensorFlow 不可用时降级为 scikit-learn MLP。
- 特征缩放使用训练段 fit。
- 目标为未来各日累计收益率。
- 预测输出仍保持价格序列、上下界、日期、涨跌幅等字段。
- 通过 `target_kind="return"` 防止旧价格目标 pkl 静默误用。

### [app/models/prophet_model.py](../app/models/prophet_model.py)

职责：

- Prophet 时间序列预测。
- 通常作为集成模型中的一个子模型。

### [app/models/ensemble.py](../app/models/ensemble.py)

职责：

- 训练或加载多个子模型。
- 根据模型 RMSE 等信息组合预测结果。
- 尝试加载在线学习保存的最佳模型。
- 对外提供报告生成所需的预测 dict。

不要轻易改变预测 dict 字段结构。

### [app/models/scorer.py](../app/models/scorer.py)

职责：

- 汇总技术面、基本面、情绪面和预测面。
- 输出综合评分、评级、颜色和风险建议相关字段。

## 在线学习

[app/models/online_learner.py](../app/models/online_learner.py) 负责：

- 每只股票的每日模型更新。
- 新旧模型 RMSE 对比。
- 模型版本保存和清理。
- 性能日志记录。
- 后台批量更新。
- Windows 计划任务脚本生成。

关键输出：

- `online_models`
- `performance_logs`
- 每只股票的 update report json

## 历史预测与报告

历史预测用于回看过去报告的预测和后续实际价格：

- 10 天预测。
- 30 天预测。
- 阶段性对比。
- 目标日期到达后的正式验证。

这些逻辑与报告生成、prediction logs 和历史回顾 UI 相关，改动时要防止破坏旧报告读取。

## 安装包

[installer](../installer) 存放 Windows 安装包脚本和说明。

安装版需要特别注意：

- 后台 Streamlit 进程退出。
- 卸载时用户数据是否清理。
- `%LOCALAPPDATA%\GuangHuanStock` 下的模型、缓存、日志体积。
- 首次运行模型训练较慢，需要用户提示。

## 测试建议

基础验证：

```powershell
$env:PYTHONIOENCODING='utf-8'
python app/test_models.py
```

代码检查：

```powershell
rg -n "fit_transform|target_kind|purge|holdout" app/models
rg -n "torch|transformers" .
```

前端验证：

- 启动应用。
- 生成单股报告。
- 切换自选股。
- 展开相关消息。
- 打开历史预测回顾。
- 打开在线学习页。
- 检查浏览器控制台是否出现 Streamlit 前端红错。
