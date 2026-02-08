"""
Simple Medical RAG Application - Environment Setup (No Pip Upgrade)
Skips pip upgrade to avoid issues
"""
import os
import sys
import platform
import subprocess
import shutil
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

def create_simple_venv():
    """Create a simple virtual environment"""
    app_dir = Path(__file__).parent
    venv_path = app_dir / "medrag_venv"
    
    if venv_path.exists():
        print(f"✅ Virtual environment already exists at: {venv_path}")
        return venv_path
    
    print(f"📦 Creating virtual environment at: {venv_path}")
    print(f"   Using Python: {sys.executable}")
    
    # Create venv
    cmd = [sys.executable, "-m", "venv", str(venv_path)]
    run_command(cmd)
    
    print(f"✅ Virtual environment created successfully")
    return venv_path

def install_basic_requirements(venv_path):
    """Install requirements without pip upgrade"""
    app_dir = Path(__file__).parent
    requirements_file = app_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    # Get pip executable
    if platform.system() == "Windows":
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        pip_exe = venv_path / "bin" / "pip"
    
    print(f"📦 Installing dependencies from: {requirements_file}")
    print("   (Skipping pip upgrade to avoid issues)")
    
    # Install requirements directly without pip upgrade
    print("Installing project dependencies...")
    try:
        run_command([str(pip_exe), "install", "-r", str(requirements_file)])
        print("✅ Project dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install project dependencies (code {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ Failed to install project dependencies: {e}")
        return False

def create_run_scripts(venv_path):
    """Create convenient run scripts"""
    app_dir = Path(__file__).parent
    
    if platform.system() == "Windows":
        # Windows batch files
        backend_script = app_dir / "run_backend_simple.bat"
        frontend_script = app_dir / "run_frontend_simple.bat"
        
        with open(backend_script, 'w') as f:
            f.write(f"""@echo off
echo Starting Medical RAG Backend...
call "{venv_path}\\Scripts\\activate.bat"
python start_backend.py
pause
""")
        
        with open(frontend_script, 'w') as f:
            f.write(f"""@echo off
echo Starting Medical RAG Frontend...
call "{venv_path}\\Scripts\\activate.bat"
python start_frontend.py
pause
""")
            
        print(f"✅ Created Windows batch files:")
        print(f"   Backend:  {backend_script}")
        print(f"   Frontend: {frontend_script}")
    
    else:
        # Linux/Mac shell scripts
        backend_script = app_dir / "run_backend_simple.sh"
        frontend_script = app_dir / "run_frontend_simple.sh"
        
        with open(backend_script, 'w') as f:
            f.write(f"""#!/bin/bash
echo "Starting Medical RAG Backend..."
source "{venv_path}/bin/activate"
python start_backend.py
""")
        
        with open(frontend_script, 'w') as f:
            f.write(f"""#!/bin/bash
echo "Starting Medical RAG Frontend..."
source "{venv_path}/bin/activate"
python start_frontend.py
""")
        
        # Make executable
        os.chmod(backend_script, 0o755)
        os.chmod(frontend_script, 0o755)
        
        print(f"✅ Created shell scripts:")
        print(f"   Backend:  {backend_script}")
        print(f"   Frontend: {frontend_script}")

def main():
    print("🏥 Medical RAG Application - Simple Setup")
    print("=" * 50)
    print("(This version skips pip upgrade to avoid issues)")
    
    try:
        # Create virtual environment
        venv_path = create_simple_venv()
        
        # Install dependencies
        if not install_basic_requirements(venv_path):
            return 1
        
        # Create run scripts
        create_run_scripts(venv_path)
        
        # Show next steps
        if platform.system() == "Windows":
            activation_cmd = f"{venv_path}\\Scripts\\activate.bat"
        else:
            activation_cmd = f"source {venv_path}/bin/activate"
        
        print("\n" + "=" * 50)
        print("🎉 Simple setup completed successfully!")
        print(f"🐍 Python version: {sys.version}")
        
        print("\n📋 Next steps:")
        print(f"   1. Use the convenient scripts:")
        print(f"      Backend:  run_backend_simple.bat")
        print(f"      Frontend: run_frontend_simple.bat")
        print(f"\n   2. Or manually activate environment:")
        print(f"      {activation_cmd}")
        print(f"      python start_backend.py")
        
        print(f"\n📦 Virtual environment: {venv_path}")
        print(f"🔧 To install more packages: {venv_path}/Scripts/pip install package_name")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
