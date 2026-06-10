@echo off
cd /d "%~dp0"
if exist "python\pythonw.exe" (
  start "" "python\pythonw.exe" "scripts\launch_app.py"
) else (
  start "" "python\python.exe" "scripts\launch_app.py"
)
