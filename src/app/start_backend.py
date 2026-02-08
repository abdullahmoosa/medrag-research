#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backend Server Startup Script
Runs the Medical RAG API server on the GPU machine
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Start Medical RAG Backend Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host to bind to (0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=8547, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", type=str, default="info", 
                       choices=["critical", "error", "warning", "info", "debug"],
                       help="Log level")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print(f"""
🏥 Medical RAG Backend Server
============================

🚀 Starting backend API server...

Configuration:
- Host: {args.host} (accessible from other machines)
- Port: {args.port}
- Workers: {args.workers}
- Log Level: {args.log_level}

Features:
✓ Medical RAG with hybrid retrieval
✓ MedEmbed + GPU acceleration  
✓ Evidence-based responses
✓ Source citations
✓ Cross-origin requests (CORS enabled)

API Endpoints:
- Health Check: http://{args.host}:{args.port}/api/health
- Chat: http://{args.host}:{args.port}/api/chat
- Config: http://{args.host}:{args.port}/api/config
- Docs: http://{args.host}:{args.port}/docs

🌐 Frontend clients can connect from other machines
   Configure frontend to use: http://<your-ip>:{args.port}

Prerequisites:
✓ Ollama running on localhost:11434
✓ Required models downloaded
✓ Index files in indexes/ directory
✓ GPU available (optional, will fallback to CPU)

""")
    
    # Check if required directories exist
    indexes_dir = os.path.join(PROJECT_ROOT, "indexes")
    if not os.path.exists(indexes_dir):
        print(f"⚠️  Warning: Indexes directory not found: {indexes_dir}")
        print("   Make sure to build indexes before starting the backend.")
    else:
        # List available indexes
        indexes = [d for d in os.listdir(indexes_dir) if os.path.isdir(os.path.join(indexes_dir, d))]
        if indexes:
            print(f"📊 Available indexes: {', '.join(indexes)}")
        else:
            print("⚠️  Warning: No indexes found in indexes directory.")
    
    print("=" * 60)
    
    try:
        # Start the server
        uvicorn.run(
            "src.app.backend:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            workers=args.workers if not args.reload else 1,  # Reload doesn't work with multiple workers
            access_log=True
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
