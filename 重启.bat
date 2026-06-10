@echo off
chcp 65001 >nul
title 重启光环智能
cd /d "%~dp0"

echo ============================================
echo   关闭旧的 streamlit 进程...
echo ============================================

REM 找到所有跑 streamlit 的 python 进程并杀掉
for /f "tokens=2 delims=," %%a in ('wmic process where "name='python.exe' and commandline like '%%streamlit%%'" get processid /format:csv ^| findstr [0-9]') do (
    echo   杀掉 PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)

REM 等 1 秒让端口释放
timeout /t 2 /nobreak >nul

echo.
echo ============================================
echo   重新启动...
echo ============================================
call "launch.bat"

echo.
echo ✓ 已启动，浏览器请刷新 http://localhost:8501
echo.
timeout /t 3 /nobreak >nul
