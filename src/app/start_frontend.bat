@echo off
REM Medical RAG Frontend Server - Windows Startup Script

title Medical RAG Frontend Server

echo.
echo 🌐 Medical RAG Frontend Server
echo ==============================
echo Starting frontend web interface...

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ and add it to PATH.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "start_frontend.py" (
    echo ❌ Please run this script from the src/app directory
    echo    Current directory: %CD%
    pause
    exit /b 1
)

echo ✅ Starting frontend server...
echo    Host: 0.0.0.0 (accessible from network)
echo    Port: 3000
echo    Can connect to backend on any machine
echo.
echo 🔗 Frontend will be available at: http://localhost:3000
echo 🌐 Network access: http://your-ip:3000
echo.
echo 📋 Setup Steps:
echo    1. Start backend server on GPU machine (start_backend.bat)  
echo    2. Configure backend URL in the web interface
echo    3. Click Connect and start chatting!
echo.

REM Start the frontend server
python start_frontend.py --host 0.0.0.0 --port 3000

echo.
echo Frontend server stopped.
pause
