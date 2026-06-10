@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 光环智能 - 停止服务
echo ============================================================
echo   正在停止光环智能后台服务（端口 8501 / 8502）……
echo ============================================================
echo.
set "FOUND=0"
for %%P in (8501 8502) do (
  for /f "tokens=5" %%i in ('netstat -ano ^| findstr ":%%P " ^| findstr LISTENING') do (
    taskkill /F /PID %%i >nul 2>&1
    if not errorlevel 1 (
      echo   已结束端口 %%P 上的进程  PID=%%i
      set "FOUND=1"
    )
  )
)
echo.
if "%FOUND%"=="0" (
  echo   未发现正在运行的光环智能服务（可能已退出）。
) else (
  echo   ✓ 光环智能已全部停止。
)
echo.
timeout /t 3 >nul
