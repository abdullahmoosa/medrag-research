@echo off
echo 🌐 Medical RAG Frontend - Virtual Environment  
echo ==========================================

cd /d "E:\nusrat\medrag\src\app"
"E:\nusrat\medrag\src\app\venv\Scripts\python.exe" start_frontend.py --host 0.0.0.0 --port 3829
pause
