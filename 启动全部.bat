@echo off
cd /d "%~dp0"
start "desktop" cmd /k "cd /d "%~dp0" && python\python.exe -m streamlit run app\app.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false --server.headless true"
timeout /t 3 /nobreak >nul
start "mobile" cmd /k "cd /d "%~dp0" && python\python.exe -m streamlit run app\mobile.py --server.port 8502 --server.address 0.0.0.0 --browser.gatherUsageStats false --server.headless true"
pause
