#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Startup script for Medical RAG Chat Application
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
    parser = argparse.ArgumentParser(description="Start Medical RAG Chat Application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", type=str, default="info", 
                       choices=["critical", "error", "warning", "info", "debug"],
                       help="Log level")
    
    args = parser.parse_args()
    
    print(f"""
    🏥 Medical RAG Chat Application
    ==============================
    
    Starting server on: http://{args.host}:{args.port}
    
    Features:
    - Medical question answering with RAG
    - Evidence-based responses with sources
    - Customizable retrieval parameters
    - Multiple embedding models support
    - Interactive web interface
    
    Make sure you have:
    ✓ Ollama running on localhost:11434
    ✓ Required models downloaded
    ✓ Index files in the indexes/ directory
    
    """)
    
    # Check if required directories exist
    indexes_dir = os.path.join(PROJECT_ROOT, "indexes")
    if not os.path.exists(indexes_dir):
        print(f"⚠️  Warning: Indexes directory not found: {indexes_dir}")
        print("   Make sure to build indexes before starting the application.")
    else:
        # List available indexes
        indexes = [d for d in os.listdir(indexes_dir) if os.path.isdir(os.path.join(indexes_dir, d))]
        if indexes:
            print(f"📊 Available indexes: {', '.join(indexes)}")
        else:
            print("⚠️  Warning: No indexes found in indexes directory.")
    
    print("\n" + "="*50 + "\n")
    
    # Start the server
    uvicorn.run(
        "src.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()
