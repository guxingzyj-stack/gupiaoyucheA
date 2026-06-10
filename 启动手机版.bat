@echo off
cd /d "%~dp0"
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "127.0.0.1"') do (set "IP=%%a" & goto :go)
:go
set IP=%IP: =%
echo http://%IP%:8502
python\python.exe -m streamlit run app\mobile.py --server.port 8502 --server.address 0.0.0.0 --browser.gatherUsageStats false --server.headless true
pause
