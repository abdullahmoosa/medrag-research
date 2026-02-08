"""
Quick setup test script to diagnose the installation issue
"""
import subprocess
import sys
from pathlib import Path

def test_venv_creation():
    """Test creating a minimal virtual environment"""
    venv_path = Path("test_venv")
    
    # Remove if exists
    if venv_path.exists():
        print("Removing existing test venv...")
        import shutil
        shutil.rmtree(venv_path)
    
    # Create venv
    print("Creating test virtual environment...")
    try:
        result = subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                               check=True, capture_output=True, text=True)
        print("✅ Virtual environment created successfully")
        
        # Test pip without upgrade
        pip_exe = venv_path / "Scripts" / "pip.exe"
        print(f"Testing pip at: {pip_exe}")
        
        # Just check pip version
        result = subprocess.run([str(pip_exe), "--version"], 
                               capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"✅ Pip working: {result.stdout.strip()}")
            
            # Try a simple install
            print("Testing simple package install...")
            result = subprocess.run([str(pip_exe), "install", "requests"], 
                                   capture_output=True, text=True, check=False)
            if result.returncode == 0:
                print("✅ Package installation works")
            else:
                print(f"❌ Package install failed: {result.stderr}")
        else:
            print(f"❌ Pip not working: {result.stderr}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create venv: {e}")
        print(f"Error: {e.stderr}")
    
    # Cleanup
    if venv_path.exists():
        import shutil
        shutil.rmtree(venv_path)
        print("Cleaned up test environment")

if __name__ == "__main__":
    test_venv_creation()
