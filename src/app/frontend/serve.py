#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple HTTP Server for Medical RAG Frontend
This serves the static frontend files that connect to the backend API
"""

import os
import http.server
import socketserver
import webbrowser
import argparse
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler with CORS support"""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        print(f"[{self.address_string()}] {format % args}")

def main():
    parser = argparse.ArgumentParser(description="Serve Medical RAG Frontend")
    parser.add_argument("--port", type=int, default=3000, help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8000", 
                       help="Default backend URL to suggest")
    
    args = parser.parse_args()
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    print(f"""
🌐 Medical RAG Frontend Server
==============================

Starting frontend server...
Host: {args.host}
Port: {args.port}
Directory: {frontend_dir}

Frontend URL: http://localhost:{args.port}
Backend URL: {args.backend_url}

Instructions:
1. Open the frontend URL in your browser
2. Configure the backend server URL 
3. Click 'Connect' to connect to your RAG backend
4. Start asking medical questions!

Note: The backend server should be running separately
      on your GPU-enabled machine at {args.backend_url}
""")
    
    # Create server
    with socketserver.TCPServer((args.host, args.port), CustomHTTPRequestHandler) as httpd:
        print(f"✅ Frontend server running at http://{args.host}:{args.port}")
        
        # Open browser if not disabled
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.port}")
        
        print("\n🔗 Access from other machines:")
        print(f"   Local network: http://<your-ip>:{args.port}")
        print(f"   Localhost: http://localhost:{args.port}")
        
        print("\n📱 Mobile access:")
        print(f"   Use your computer's IP address on the same network")
        
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down frontend server...")
            print("👋 Frontend server stopped.")

if __name__ == "__main__":
    main()
