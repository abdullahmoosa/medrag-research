@echo off
REM Medical RAG Backend Server - Windows Startup Script

title Medical RAG Backend Server

echo.
echo 🏥 Medical RAG Backend Server
echo =============================
echo Starting backend API server on GPU machine...

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ and add it to PATH.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "backend.py" (
    echo ❌ Please run this script from the src/app directory
    echo    Current directory: %CD%
    pause
    exit /b 1
)

echo ✅ Starting backend server...
echo    Host: 0.0.0.0 (accessible from network)
echo    Port: 8000
echo    Frontend clients can connect from other machines
echo.
echo 🔗 Backend will be available at: http://localhost:8000
echo 📖 API Documentation: http://localhost:8000/docs
echo.
echo ⏳ Initializing (this may take a moment to load indexes)...
echo.

REM Start the backend server
python start_backend.py --host 0.0.0.0 --port 8000

echo.
echo Backend server stopped.
pause
