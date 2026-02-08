@echo off
REM Medical RAG Chat Application - Windows Startup Script

title Medical RAG Chat Application

echo.
echo 🏥 Medical RAG Chat Application
echo ===============================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ and add it to PATH.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "main.py" (
    echo ❌ Please run this script from the src/app directory
    echo    Current directory: %CD%
    pause
    exit /b 1
)

echo ✅ Starting Medical RAG Chat Application...
echo.
echo 📊 Features:
echo    - Medical question answering with RAG
echo    - Evidence-based responses with sources  
echo    - Interactive web interface
echo    - Customizable retrieval parameters
echo.
echo 🔗 Application will be available at: http://localhost:8000
echo 📖 Documentation: README.md
echo.
echo ⏳ Initializing (this may take a moment)...
echo.

REM Start the application
python run.py --host 127.0.0.1 --port 8000

echo.
echo Application stopped.
pause
