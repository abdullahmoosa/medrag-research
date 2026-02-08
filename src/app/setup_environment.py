#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment Setup Script for Medical RAG Application
Creates virtual environment and installs dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, check=True, cwd=None):
    """Run a command and handle errors"""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            capture_output=True, 
            text=True,
            cwd=cwd,
            shell=platform.system() == "Windows"
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        raise

def detect_available_pythons():
    """Detect available Python installations on Windows"""
    available_pythons = []
    
    if platform.system() == "Windows":
        try:
            # Use py launcher to list available versions
            result = subprocess.run(
                ["py", "--list"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith('-3.'):
                    # Extract version like "-3.11-64"
                    version_part = line.split()[0][1:]  # Remove leading "-"
                    if '-64' in version_part:
                        version_part = version_part.replace('-64', '')
                    
                    # Test if this version works
                    try:
                        test_result = subprocess.run(
                            [f"py", f"-{version_part}", "--version"], 
                            capture_output=True, 
                            text=True, 
                            check=True
                        )
                        available_pythons.append({
                            'version': version_part,
                            'command': f"py -{version_part}",
                            'full_version': test_result.stdout.strip()
                        })
                    except:
                        continue
            
        except subprocess.CalledProcessError:
            pass
    
    # Always include current python as fallback
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    available_pythons.append({
        'version': current_version,
        'command': sys.executable,
        'full_version': f"Python {sys.version.split()[0]}"
    })
    
    return available_pythons

def select_python_version():
    """Allow user to select Python version if multiple are available"""
    available = detect_available_pythons()
    
    if len(available) <= 1:
        return sys.executable
    
    print(f"\n🐍 Multiple Python versions detected:")
    for i, python_info in enumerate(available):
        marker = " (current)" if python_info['command'] == sys.executable else ""
        marker += " (recommended)" if python_info['version'].startswith('3.11') else ""
        print(f"  {i+1}. {python_info['full_version']}{marker}")
    
    while True:
        try:
            choice = input(f"\nSelect Python version (1-{len(available)}, Enter for current): ").strip()
            
            if not choice:  # Use current Python
                return sys.executable
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available):
                selected = available[choice_idx]
                print(f"✅ Selected: {selected['full_version']}")
                return selected['command']
            else:
                print(f"❌ Invalid choice. Please select 1-{len(available)}")
                
        except ValueError:
            print("❌ Please enter a number or press Enter for default")
        except KeyboardInterrupt:
            print("\n🛑 Setup cancelled")
            return None

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    recommended_version = (3, 11)
    min_version = (3, 8)
    
    if version.major < 3 or (version.major == 3 and version.minor < min_version[1]):
        print(f"❌ Python {min_version[0]}.{min_version[1]}+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print(f"   Recommended version: Python {recommended_version[0]}.{recommended_version[1]} (matches Docker)")
        return False
    
    if (version.major, version.minor) == recommended_version:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Optimal (matches Docker)")
    elif (version.major, version.minor) >= recommended_version:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible (newer than Docker)")
    else:
        print(f"⚠️  Python {version.major}.{version.minor}.{version.micro} - Compatible but older than Docker")
        print(f"   Consider using Python {recommended_version[0]}.{recommended_version[1]} for consistency")
    
    return True

def create_virtual_environment(python_executable=None):
    """Create virtual environment for the project"""
    if python_executable is None:
        python_executable = sys.executable
    
    app_dir = Path(__file__).parent
    venv_path = app_dir / "venv"
    
    if venv_path.exists():
        print(f"✅ Virtual environment already exists at: {venv_path}")
        
        # Check which Python version the existing venv uses
        if platform.system() == "Windows":
            existing_python = venv_path / "Scripts" / "python.exe"
        else:
            existing_python = venv_path / "bin" / "python"
        
        if existing_python.exists():
            try:
                result = subprocess.run([str(existing_python), "--version"], capture_output=True, text=True)
                print(f"   Existing venv uses: {result.stdout.strip()}")
            except:
                pass
        
        return venv_path
    
    print(f"📦 Creating virtual environment at: {venv_path}")
    print(f"   Using Python: {python_executable}")
    
    # Create venv with selected Python
    if isinstance(python_executable, str) and python_executable.startswith("py -"):
        # Handle Windows py launcher format
        launcher_args = python_executable.split()
        cmd = launcher_args + ["-m", "venv", str(venv_path)]
    else:
        cmd = [python_executable, "-m", "venv", str(venv_path)]
    
    run_command(cmd)
    
    print(f"✅ Virtual environment created successfully")
    return venv_path

def get_activation_command(venv_path):
    """Get the command to activate virtual environment"""
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        return str(activate_script)
    else:
        activate_script = venv_path / "bin" / "activate"
        return f"source {activate_script}"

def install_dependencies(venv_path):
    """Install required dependencies in virtual environment"""
    app_dir = Path(__file__).parent
    requirements_file = app_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    # Get python executable from venv
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    print(f"📦 Installing dependencies from: {requirements_file}")
    
    # Upgrade pip first (but don't fail if it doesn't work)
    print("Upgrading pip...")
    try:
        run_command([str(pip_exe), "install", "--upgrade", "pip"])
        print("✅ Pip upgraded successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Pip upgrade failed (code {e.returncode}), continuing with current version...")
    except Exception as e:
        print(f"⚠️  Warning: Pip upgrade failed ({e}), continuing...")
    
    # Install requirements
    print("Installing project dependencies...")
    try:
        run_command([str(pip_exe), "install", "-r", str(requirements_file)])
        print("✅ Project dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install project dependencies (code {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ Failed to install project dependencies: {e}")
        return False
    
    # Install additional packages that might be needed
    additional_packages = [
        "torch>=2.1.0",  # Ensure PyTorch is available
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.2"
    ]
    
    print("Installing additional ML packages...")
    for package in additional_packages:
        try:
            print(f"   Installing {package}...")
            run_command([str(pip_exe), "install", package])
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Warning: Failed to install {package} (code {e.returncode})")
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to install {package}: {e}")
    
    print(f"✅ Dependencies installation completed")
    return True

def check_gpu_availability(venv_path):
    """Check if GPU is available for PyTorch"""
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    gpu_check_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
else:
    print("No CUDA devices found - will use CPU")
"""
    
    try:
        result = run_command([str(python_exe), "-c", gpu_check_script])
        print("✅ GPU check completed")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Could not check GPU availability")
        return False

def create_run_scripts(venv_path):
    """Create convenient run scripts that use the virtual environment"""
    app_dir = Path(__file__).parent
    
    if platform.system() == "Windows":
        # Windows batch files
        backend_script = app_dir / "run_backend_venv.bat"
        frontend_script = app_dir / "run_frontend_venv.bat"
        python_exe = venv_path / "Scripts" / "python.exe"
        
        backend_content = f"""@echo off
echo 🏥 Medical RAG Backend - Virtual Environment
echo ==========================================

cd /d "{app_dir}"
"{python_exe}" start_backend.py --host 0.0.0.0 --port 8000
pause
"""
        
        frontend_content = f"""@echo off
echo 🌐 Medical RAG Frontend - Virtual Environment  
echo ==========================================

cd /d "{app_dir}"
"{python_exe}" start_frontend.py --host 0.0.0.0 --port 3000
pause
"""
        
    else:
        # Unix shell scripts
        backend_script = app_dir / "run_backend_venv.sh"
        frontend_script = app_dir / "run_frontend_venv.sh"
        activate_script = venv_path / "bin" / "activate"
        
        backend_content = f"""#!/bin/bash
echo "🏥 Medical RAG Backend - Virtual Environment"
echo "=========================================="

cd "{app_dir}"
source "{activate_script}"
python start_backend.py --host 0.0.0.0 --port 8000
"""
        
        frontend_content = f"""#!/bin/bash
echo "🌐 Medical RAG Frontend - Virtual Environment"
echo "=========================================="

cd "{app_dir}"
source "{activate_script}"
python start_frontend.py --host 0.0.0.0 --port 3000
"""
    
    # Write scripts
    with open(backend_script, 'w') as f:
        f.write(backend_content)
    
    with open(frontend_script, 'w') as f:
        f.write(frontend_content)
    
    # Make executable on Unix
    if platform.system() != "Windows":
        os.chmod(backend_script, 0o755)
        os.chmod(frontend_script, 0o755)
    
    print(f"✅ Created run scripts:")
    print(f"   Backend: {backend_script}")
    print(f"   Frontend: {frontend_script}")

def main():
    print("🏥 Medical RAG Application - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Select Python version (if multiple available)
    selected_python = select_python_version()
    if selected_python is None:
        return 1
    
    try:
        # Create virtual environment with selected Python
        venv_path = create_virtual_environment(selected_python)
        
        # Install dependencies
        if not install_dependencies(venv_path):
            return 1
        
        # Check GPU
        check_gpu_availability(venv_path)
        
        # Create run scripts
        create_run_scripts(venv_path)
        
        # Get activation command
        activation_cmd = get_activation_command(venv_path)
        
        print("\n" + "=" * 50)
        print("🎉 Environment setup completed successfully!")
        
        # Show Python version info
        if platform.system() == "Windows":
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        
        try:
            result = run_command([str(python_exe), "--version"], check=False)
            if result and result.stdout:
                venv_python_version = result.stdout.strip()
                print(f"📋 Virtual environment Python: {venv_python_version}")
        except:
            pass
        
        print(f"🐳 Docker Python version: Python 3.11 (for consistency)")
        print("\n📋 Next steps:")
        
        if platform.system() == "Windows":
            print(f"   1. Use the convenient scripts:")
            print(f"      Backend:  run_backend_venv.bat")
            print(f"      Frontend: run_frontend_venv.bat")
            print(f"\n   2. Or manually activate environment:")
            print(f"      {activation_cmd}")
            print(f"      python start_backend.py")
        else:
            print(f"   1. Use the convenient scripts:")
            print(f"      Backend:  ./run_backend_venv.sh")
            print(f"      Frontend: ./run_frontend_venv.sh")
            print(f"\n   2. Or manually activate environment:")
            print(f"      {activation_cmd}")
            print(f"      python start_backend.py")
        
        print(f"\n📦 Virtual environment location: {venv_path}")
        print(f"🔧 To add more packages: {venv_path}/Scripts/pip install package_name")
        print(f"🗑️  To remove environment: delete {venv_path} folder")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
