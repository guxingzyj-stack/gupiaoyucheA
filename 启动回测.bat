@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 光环智能 - 模型回测（滚动前向验证）
echo ============================================================
echo   模型回测 / 滚动前向验证（walk-forward）
echo   用历史数据检验预测模型的真实方向准确率，并对照朴素基准
echo ============================================================
echo.
set /p CODE=请输入股票代码（多个用空格隔开；直接回车=回测全部自选股）:
echo.
echo   模式 1) 快档：仅 XGBoost，约 2 分钟/股，先看方向是否超基准
echo   模式 2) 精档：全模型(LSTM+XGB+Prophet)，约 25 分钟/股，夜里跑
set /p MODE=请选择模式 [1/2，默认 1]:
if "%MODE%"=="2" (set MODEARG=accurate) else (set MODEARG=fast)
echo.
echo 开始回测（mode=%MODEARG%），请耐心等待……
echo.
if "%CODE%"=="" (
  "python\python.exe" "scripts\run_backtest.py" --watchlist --mode %MODEARG%
) else (
  "python\python.exe" "scripts\run_backtest.py" %CODE% --mode %MODEARG%
)
echo.
echo ============================================================
echo 回测完成，结果已保存到 backtest_logs。
echo 重新打开桌面版后，报告页「价格预测走势」下方会显示历史准确率。
echo ============================================================
pause
