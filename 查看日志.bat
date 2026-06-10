@echo off
chcp 65001 >nul
title 查看光环智能日志
echo 打开日志文件...
start notepad "%LOCALAPPDATA%\GuangHuanStock\logs\streamlit_startup.log"
