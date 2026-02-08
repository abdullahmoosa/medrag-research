@echo off
echo 🏥 Medical RAG Backend - Virtual Environment
echo ==========================================

cd /d "E:\nusrat\medrag\src\app"
"E:\nusrat\medrag\src\app\venv\Scripts\python.exe" start_backend.py --host 0.0.0.0 --port 8547
pause
