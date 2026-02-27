@echo off
title MuleNet Launcher
echo ==============================================
echo        Starting MuleNet Services...
echo ==============================================

echo [1/2] Starting FastAPI Backend...
start "MuleNet Backend" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && cd backend && python main.py"

echo [2/2] Starting React Frontend...
start "MuleNet Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo ==============================================
echo Both services have been launched in separate windows!
echo  - Backend running at: http://localhost:8000
echo  - Frontend running at: http://localhost:5173 / 5174
echo.
echo To stop the servers, close the newly opened command windows.
echo ==============================================
pause
