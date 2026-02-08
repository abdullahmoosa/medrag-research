#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Frontend Server Startup Script  
Runs the frontend web interface that connects to the backend API
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Start Medical RAG Frontend Server")
    parser.add_argument("--port", type=int, default=3829, help="Port to serve frontend on")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host to bind to (0.0.0.0 for all interfaces)")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8547",
                       help="Default backend server URL")
    parser.add_argument("--no-browser", action="store_true", 
                       help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    print(f"""
🌐 Medical RAG Frontend Server
==============================

🎨 Starting frontend web interface...

Configuration:
- Frontend Host: {args.host}
- Frontend Port: {args.port}
- Backend URL: {args.backend_url}

Access URLs:
- Local: http://localhost:{args.port}
- Network: http://<your-ip>:{args.port}

Setup Instructions:
1. 🖥️  Start backend server on GPU machine: 
   python start_backend.py --host 0.0.0.0 --port 8000
   
2. 🌐 Configure backend URL in frontend interface
   Default: {args.backend_url}
   Network: http://<backend-machine-ip>:8000
   
3. 🔗 Click 'Connect' to connect to backend
   
4. 💬 Start asking medical questions!

Features:
✓ Cross-machine connectivity
✓ Real-time chat interface  
✓ Source citations display
✓ Live parameter configuration
✓ Connection status monitoring
✓ Mobile-friendly design

Note: Backend and frontend can run on different machines
      Backend should have GPU + indexes, frontend is lightweight
""")
    
    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return 1
    
    print("=" * 60)
    
    try:
        # Start the frontend server
        subprocess.run([
            sys.executable,
            str(frontend_dir / "serve.py"),
            "--host", args.host,
            "--port", str(args.port),
            "--backend-url", args.backend_url,
        ] + (["--no-browser"] if args.no_browser else []))
        
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped.")
    except Exception as e:
        print(f"❌ Failed to start frontend server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
