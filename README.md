# 光环智能股票预测系统

光环智能股票预测系统是一个面向 A 股的本地化分析工具，使用 Streamlit 提供桌面浏览器界面，结合行情数据、技术指标、新闻情绪、机器学习模型和在线学习机制生成股票预测报告。

当前应用版本号位于 [app/config.py](app/config.py)：`APP_VERSION = "v2.3.1"`。

## 主要功能

- 自选股管理和单股分析报告生成。
- 多股对比看板。
- 历史预测回顾，与最新价格或目标日期价格对比。
- XGBoost、LSTM/MLP、Prophet 等模型集成预测。
- 在线学习和后台每日自学习。
- 模型版本管理、性能日志、预测日志和报告回看。
- Windows 本地启动脚本和安装包构建脚本。

## 快速启动

在项目根目录运行：

```bat
启动光环智能.bat
```

或：

```bat
launch.bat
```

启动脚本会调用 [scripts/launch_app.py](scripts/launch_app.py)，自动检查端口并打开：

```text
http://localhost:8501
```

源码方式也可以直接运行：

```powershell
python -m streamlit run app/app.py --server.port 8501
```

## 重要目录

- [app/app.py](app/app.py)：Streamlit 主界面。
- [app/config.py](app/config.py)：路径、版本、模型和运行配置。
- [app/models](app/models)：模型、集成预测、在线学习和评分逻辑。
- `app/online_models`：在线学习模型目录。
- `app/performance_logs`：在线学习性能报告。
- `app/prediction_logs`：历史预测记录。
- `app/history`：历史结果数据。
- [installer](installer)：Windows 安装包相关脚本。
- [scripts](scripts)：启动器和辅助脚本。

用户安装版可通过环境变量 `GUANGHUAN_STOCK_DATA_DIR` 把模型、缓存、日志和自选股数据放到独立用户数据目录。

## 模型口径

模型训练必须按时间顺序处理，避免未来数据泄漏。

- XGBoost 目标为未来 N 日累计收益率。
- LSTM 当前目标也已改为未来各日累计收益率。
- 验证 RMSE 必须在还原为价格后计算，便于不同模型权重可比。
- 旧版按缩放价格训练的 LSTM pkl 已通过 `target_kind = "return"` 做兼容防护，加载旧模型时会失效并触发重训。

## 验证

首选测试：

```powershell
$env:PYTHONIOENCODING='utf-8'
python app/test_models.py
```

如果运行 [run_training_test.py](run_training_test.py)，注意该脚本历史上可能包含旧路径，需要先确认路径是否匹配当前工作目录。

## 给 AI 助手的入口

让 ChatGPT、Claude 或 Codex 接手本项目时，请先阅读：

- [AGENTS.md](AGENTS.md)
- [docs/AI_CONTEXT.md](docs/AI_CONTEXT.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [CHANGELOG.md](CHANGELOG.md)

推荐提示词：

```text
请先阅读 README.md、AGENTS.md、docs/AI_CONTEXT.md、docs/ARCHITECTURE.md 和 CHANGELOG.md，
再分析或修改这个仓库。不要做无关重构，修改后运行 app/test_models.py。
```
