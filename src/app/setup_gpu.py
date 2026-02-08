#!/usr/bin/env python3
"""GPU Setup Script for Medical RAG Application."""

import subprocess
import sys
import platform
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        print(f"   Error: {e.stderr}")
        return None

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    print("🔍 Checking for NVIDIA GPU...")
    result = run_command("nvidia-smi", check=False)
    
    if result and result.returncode == 0:
        print("✅ NVIDIA GPU detected")
        # Extract CUDA version from nvidia-smi output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'CUDA Version:' in line:
                cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                print(f"   CUDA Version: {cuda_version}")
                return True, cuda_version
        return True, "Unknown"
    else:
        print("❌ No NVIDIA GPU detected or drivers not installed")
        return False, None

def install_pytorch_cuda(venv_path):
    """Install PyTorch with CUDA support."""
    print("🚀 Installing PyTorch with CUDA support...")
    
    if platform.system() == "Windows":
        pip_exe = venv_path / "Scripts" / "pip.exe"
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        pip_exe = venv_path / "bin" / "pip"
        python_exe = venv_path / "bin" / "python"
    
    # Uninstall CPU version
    print("   Removing CPU-only PyTorch...")
    run_command(f'"{pip_exe}" uninstall torch torchvision torchaudio -y', check=False)
    
    # Install CUDA version
    print("   Installing PyTorch with CUDA 12.1 support...")
    cuda_cmd = f'"{pip_exe}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
    result = run_command(cuda_cmd)
    
    if result and result.returncode == 0:
        print("✅ PyTorch CUDA installation successful")
        return True
    else:
        print("❌ PyTorch CUDA installation failed")
        return False

def test_pytorch_gpu():
    """Test PyTorch GPU support."""
    print("🧪 Testing PyTorch GPU support...")
    
    test_script = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    # Test tensor creation
    try:
        x = torch.randn(10, 10).cuda()
        print("✅ GPU tensor creation successful")
        print(f"Tensor device: {x.device}")
    except Exception as e:
        print(f"❌ GPU tensor creation failed: {e}")
else:
    print("❌ CUDA not available in PyTorch")
'''
    
    if platform.system() == "Windows":
        python_exe = "venv\\Scripts\\python.exe"
    else:
        python_exe = "venv/bin/python"
    
    result = run_command(f'"{python_exe}" -c "{test_script}"')
    return result and "CUDA available: True" in result.stdout

def main():
    """Main setup function."""
    print("🏥 Medical RAG - GPU Setup")
    print("=" * 50)
    
    # Check current directory
    app_dir = Path(__file__).parent
    venv_path = app_dir / "venv"
    
    if not venv_path.exists():
        print(f"❌ Virtual environment not found at: {venv_path}")
        print("   Please run setup_environment.py first")
        return 1
    
    # Check for NVIDIA GPU
    has_gpu, cuda_version = check_nvidia_gpu()
    if not has_gpu:
        print("\\n⚠️  No NVIDIA GPU detected.")
        print("   The application will run on CPU only.")
        return 1
    
    # Install PyTorch with CUDA
    if install_pytorch_cuda(venv_path):
        print("\\n🧪 Testing GPU support...")
        if test_pytorch_gpu():
            print("\\n🎉 GPU setup completed successfully!")
            print("\\nNext steps:")
            print("1. Restart the backend server")
            print("2. Check the logs for 'CUDA available' message")
            print("3. GPU acceleration should now be active")
        else:
            print("\\n❌ GPU test failed. Check the installation.")
            return 1
    else:
        print("\\n❌ GPU setup failed.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
