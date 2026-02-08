# Medical RAG Application Deployment Guide

## Python Version Compatibility

This application supports **Python 3.9, 3.10, 3.11, and 3.12**.

### Version Selection Strategy

1. **Docker Deployment**: Uses Python 3.11 for optimal performance and library compatibility
2. **Virtual Environment**: Automatically detects available Python versions on your system and lets you choose

### Deployment Options

## Option 1: Virtual Environment (Recommended for Development)

The setup script will automatically detect Python versions on your system:

```bash
python setup_environment.py
```

On Windows with multiple Python versions:
- The script will show available versions (e.g., 3.9, 3.11, 3.12)
- You can select your preferred version for consistency with Docker
- Recommended: Use Python 3.11 to match Docker environment

## Option 2: Docker (Recommended for Production)

Uses Python 3.11 in containerized environment:

```bash
docker-compose up -d
```

### Version Consistency Tips

1. **For Development**: Use Python 3.11 locally to match Docker environment
2. **For Testing**: The application works with any supported Python version (3.9+)
3. **For Production**: Docker ensures consistent Python 3.11 environment

### System Requirements

- **Minimum**: Python 3.9+
- **Recommended**: Python 3.11 (matches Docker)
- **GPU**: Optional, improves performance significantly
- **RAM**: 8GB+ recommended for RAG operations
- **Storage**: ~2GB for models and indexes

### Quick Start

1. **Check your Python versions**:
   ```bash
   # Windows
   py --list
   
   # Linux/Mac
   python3 --version
   ```

2. **Setup with preferred version**:
   ```bash
   python setup_environment.py
   ```

3. **Run the application**:
   ```bash
   # Virtual Environment
   run_backend_venv.bat
   run_frontend_venv.bat
   
   # Docker
   docker-compose up -d
   ```

### Troubleshooting

- **Version conflicts**: Use the setup script to select a specific Python version
- **Missing dependencies**: The setup automatically handles all requirements
- **GPU issues**: Check `GPU_OPTIMIZATION_GUIDE.md` for detailed GPU setup

### Performance Notes

- Python 3.11 offers ~10-15% better performance than 3.9
- GPU acceleration works with all supported Python versions
- Docker deployment provides consistent performance across systems
