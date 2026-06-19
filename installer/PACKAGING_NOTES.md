# 安装包打包说明

## 当前版本

`v2.3.2 价值体检完善版`

## 打包工具

使用 Inno Setup 6 编译安装包。

安装好 Inno Setup 6 后，在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\installer\build_installer.ps1
```

生成文件位置：

```text
installer\光环智能股票预测系统_v2.3.2_Setup.exe
```

## 安装包包含

- app 程序代码
- python 内置运行环境
- scripts 启动器
- CHANGELOG.md
- 使用说明.txt

## 安装包排除

- app\online_models
- app\performance_logs
- app\prediction_logs
- app\history
- app\.cache
- app\output
- app\scheduler
- app\watchlist.json
- __pycache__
- *.pyc
- 第三方库测试目录 tests / test / testing
- Python 开发/调试文件 *.pdb / *.lib / *.a / *.h
- *.bak / *.bak2

## 用户数据

安装版启动器会把用户数据放在：

```text
%LOCALAPPDATA%\GuangHuanStock
```

这里保存：

- 自选股
- 自学习模型
- 预测历史
- 在线学习报告
- 缓存
- 启动日志

升级安装程序时，不会覆盖这些用户数据。

## 注意事项

当前机器若未安装 Inno Setup，`build_installer.ps1` 会提示未找到 `ISCC.exe`，不会生成安装包。
