#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for Medical RAG Chat Application
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command"""
    print(f"Running: {cmd}")
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    result = subprocess.run(
        cmd, 
        check=check, 
        capture_output=capture_output,
        text=True
    )
    
    if capture_output:
        return result.stdout.strip()
    return result

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            models = response.json().get("models", [])
            print(f"   Available models: {len(models)}")
            for model in models[:3]:  # Show first 3
                print(f"   - {model['name']}")
            return True
        else:
            print("❌ Ollama server not responding correctly")
            return False
    except Exception as e:
        print(f"❌ Ollama server not running: {str(e)}")
        print("   Start with: ollama serve")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    req_file = Path(__file__).parent / "requirements.txt"
    
    try:
        run_command([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_indexes():
    """Check for available indexes"""
    project_root = Path(__file__).parent.parent.parent
    indexes_dir = project_root / "indexes"
    
    if not indexes_dir.exists():
        print(f"❌ Indexes directory not found: {indexes_dir}")
        print("   You need to build indexes first using the build scripts")
        return False
    
    indexes = [d.name for d in indexes_dir.iterdir() if d.is_dir()]
    if not indexes:
        print(f"❌ No indexes found in {indexes_dir}")
        return False
    
    print(f"✅ Found {len(indexes)} indexes:")
    for idx in indexes:
        print(f"   - {idx}")
    return True

def pull_ollama_models():
    """Pull required Ollama models"""
    models = [
        "thewindmom/llama3-med42-8b",
        "deepseek-ai/deepseek-r1:8b",
    ]
    
    print("\nPulling Ollama models (this may take a while)...")
    
    for model in models:
        try:
            print(f"Pulling {model}...")
            run_command(["ollama", "pull", model])
            print(f"✅ {model} downloaded")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to pull {model}")
            print(f"   Try manually: ollama pull {model}")

def test_setup():
    """Test the complete setup"""
    print("\nTesting setup...")
    
    # Test imports
    try:
        from src.app.config import RAGConfig
        from src.app.rag_service import RAGService
        print("✅ Module imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test configuration
    try:
        config = RAGConfig()
        print(f"✅ Configuration created (using index: {config.index_name})")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    return True

def main():
    print("🏥 Medical RAG Chat Application Setup")
    print("=" * 40)
    
    all_checks_passed = True
    
    # Check Python version
    if not check_python_version():
        all_checks_passed = False
    
    # Install dependencies
    if not install_dependencies():
        all_checks_passed = False
    
    # Check Ollama
    if not check_ollama():
        all_checks_passed = False
        print("\n💡 To start Ollama:")
        print("   ollama serve")
    
    # Check indexes
    if not check_indexes():
        all_checks_passed = False
        print("\n💡 To build indexes, see the main project documentation")
    
    # Pull models if Ollama is running
    if check_ollama():
        pull_ollama_models()
    
    # Test setup
    if not test_setup():
        all_checks_passed = False
    
    print("\n" + "=" * 40)
    
    if all_checks_passed:
        print("🎉 Setup completed successfully!")
        print("\n🚀 To start the application:")
        print("   python run.py")
        print("\n   Then visit: http://localhost:8000")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
