#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick start demo for Medical RAG Chat Application
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path

def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama():
    """Try to start Ollama if not running"""
    if not check_ollama():
        print("Starting Ollama server...")
        try:
            # Try to start Ollama
            subprocess.Popen(["ollama", "serve"], shell=True)
            time.sleep(5)  # Give it time to start
            if check_ollama():
                print("✅ Ollama server started")
                return True
            else:
                print("❌ Failed to start Ollama")
                return False
        except:
            print("❌ Could not start Ollama. Please start it manually: ollama serve")
            return False
    else:
        print("✅ Ollama server is running")
        return True

def demo_chat_api():
    """Demonstrate the chat API"""
    test_queries = [
        "What are the symptoms of diabetes?",
        "How is hypertension diagnosed?",
        "What are the treatment options for asthma?"
    ]
    
    print("\n🤖 Testing Chat API...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"message": query, "conversation_id": "demo"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Response: {data['response'][:200]}...")
                print(f"📊 Sources: {data['num_sources']} found")
                print(f"⏱️  Time: {data['retrieval_time']:.2f}s retrieval + {data['generation_time']:.2f}s generation")
            else:
                print(f"❌ API Error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("⏱️  Request timed out (this is normal for the first request)")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("🏥 Medical RAG Chat Application - Quick Demo")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Please run this from the src/app directory")
        return 1
    
    # Check Ollama
    if not start_ollama():
        print("\n💡 Please ensure Ollama is running:")
        print("   1. Install Ollama: https://ollama.ai")
        print("   2. Start server: ollama serve")
        print("   3. Pull models: ollama pull thewindmom/llama3-med42-8b")
        return 1
    
    # Start the FastAPI app
    print("\n🚀 Starting Medical RAG Chat Application...")
    print("   This will take a moment to initialize...")
    
    try:
        # Start the server in the background
        server_process = subprocess.Popen([
            sys.executable, "run.py", "--host", "127.0.0.1", "--port", "8000"
        ])
        
        # Wait for server to start
        print("⏳ Waiting for server to initialize...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Server is ready!")
                    break
            except:
                pass
            time.sleep(1)
            if i % 5 == 0:
                print(f"   Still initializing... ({i+1}/30)")
        else:
            print("❌ Server failed to start within 30 seconds")
            server_process.terminate()
            return 1
        
        # Show available endpoints
        print(f"\n🌐 Application is running at:")
        print(f"   Web Interface: http://localhost:8000")
        print(f"   API Health:    http://localhost:8000/health")
        print(f"   Available Indexes: http://localhost:8000/available-indexes")
        
        # Demo the API
        demo_chat_api()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"   Visit http://localhost:8000 to use the web interface")
        print(f"   Press Ctrl+C to stop the server")
        
        # Keep running until user interrupts
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    print("👋 Demo finished. Thank you!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
