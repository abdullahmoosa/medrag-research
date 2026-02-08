# 🏥 Medical RAG Setup Guide

This guide provides multiple ways to set up and run the Medical RAG application with proper dependency management.

## 📋 Prerequisites

- **Python 3.8+** installed
- **Git** for cloning repositories  
- **Ollama** running locally or in Docker
- **GPU drivers** (optional, for CUDA acceleration)

## 🚀 Setup Methods

### Method 1: Virtual Environment (Recommended for Development)

#### 1. Automatic Setup
```bash
cd src/app
python setup_environment.py
```

This script will:
- ✅ Check Python version compatibility
- ✅ Create virtual environment in `src/app/venv/`
- ✅ Install all dependencies
- ✅ Check GPU availability
- ✅ Create convenient run scripts

#### 2. Manual Setup
```bash
cd src/app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### 3. Run the Application
After setup, use the generated scripts:

**Windows:**
```cmd
# Backend (GPU machine)
run_backend_venv.bat

# Frontend (any machine)  
run_frontend_venv.bat
```

**Linux/Mac:**
```bash
# Backend (GPU machine)
./run_backend_venv.sh

# Frontend (any machine)
./run_frontend_venv.sh
```

Or manually:
```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Start backend
python start_backend.py --host 0.0.0.0 --port 8000

# Start frontend (in another terminal)
python start_frontend.py --host 0.0.0.0 --port 3000
```

### Method 2: Docker (Recommended for Production)

#### 1. Quick Start with Docker Compose
```bash
cd src/app

# Copy environment template
cp .env.example .env
# Edit .env file as needed

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

This will start:
- 🤖 **Ollama** on port 11434
- 🖥️ **Backend API** on port 8000  
- 🌐 **Frontend** on port 3000
- 📦 **Redis** on port 6379 (optional)

#### 2. Backend Only (Docker)
```bash
# Build backend image
docker build -t medrag-backend .

# Run backend container
docker run -d \
  --name medrag-backend \
  -p 8000:8000 \
  -v $(pwd)/../../indexes:/app/indexes:ro \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  medrag-backend
```

#### 3. GPU Support (Docker)
Make sure you have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed:

```bash
# Uncomment GPU sections in docker-compose.yml
# Then start with GPU support
docker-compose up -d
```

### Method 3: System-Wide Installation (Not Recommended)

```bash
cd src/app

# Install dependencies globally
pip install -r requirements.txt

# Run directly
python start_backend.py
python start_frontend.py
```

## 🔧 Configuration

### Environment Variables
Create `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Key variables to configure:
```env
OLLAMA_BASE_URL=http://localhost:11434
CUDA_VISIBLE_DEVICES=0
DEFAULT_INDEX_NAME=medcorp_medembed
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

### Index Setup
Make sure your indexes are available:
```bash
ls ../../indexes/
# Should show: medcorp_medembed/ medembed/ textbooks_medembed_flat/
```

### Ollama Models
Download required models:
```bash
ollama pull thewindmom/llama3-med42-8b
ollama pull deepseek-ai/deepseek-r1:8b
# ollama pull oscardp96/medcpt-query:latest  # If using MedCPT
```

## 🧪 Testing Your Setup

### 1. Test Backend API
```bash
# Test with the included script
python test_distributed.py --backend-url http://localhost:8000

# Or manually test endpoints
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are symptoms of diabetes?"}'
```

### 2. Test Frontend
Open browser: `http://localhost:3000`
1. Configure backend URL: `http://localhost:8000`
2. Click "Connect" 
3. Send test message

### 3. Test Docker Setup
```bash
# Check all containers are running
docker-compose ps

# Test backend health
curl http://localhost:8000/api/health

# Test frontend
curl http://localhost:3000
```

## 🌐 Network Deployment

### Running on Different Machines

1. **Backend (GPU machine):**
   ```bash
   # Find your IP
   hostname -I  # Linux
   ipconfig     # Windows
   
   # Start backend with network access
   python start_backend.py --host 0.0.0.0 --port 8000
   ```

2. **Frontend (client machines):**
   ```bash
   # Start frontend
   python start_frontend.py --host 0.0.0.0 --port 3000
   
   # Or serve static files with any web server
   cd frontend
   python -m http.server 3000
   ```

3. **Configure in browser:**
   - Backend URL: `http://<gpu-machine-ip>:8000`
   - Frontend URL: `http://<frontend-machine-ip>:3000`

### Firewall Configuration
```bash
# Ubuntu/Debian
sudo ufw allow 8000  # Backend
sudo ufw allow 3000  # Frontend

# Windows
# Allow ports through Windows Defender Firewall
```

## 📊 Performance Optimization

### For Backend (GPU Machine)
```env
# .env configuration for performance
CUDA_VISIBLE_DEVICES=0
MAX_WORKERS=4
BATCH_SIZE=1
```

### For Docker
```yaml
# docker-compose.yml - resource limits
services:
  medrag-backend:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
```

## 🐛 Troubleshooting

### Virtual Environment Issues
```bash
# Remove and recreate venv
rm -rf venv
python setup_environment.py

# Or manually:
python -m venv venv --clear
```

### Docker Issues
```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check logs
docker-compose logs medrag-backend
docker-compose logs ollama
```

### GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Network Issues
```bash
# Test connectivity
curl http://backend-ip:8000/api/health
telnet backend-ip 8000

# Check firewall
sudo ufw status  # Linux
netsh advfirewall show allprofiles  # Windows
```

## 🔒 Security Notes

### Production Deployment
- Use HTTPS with SSL certificates
- Implement API authentication
- Restrict CORS origins
- Use environment variables for secrets
- Run behind reverse proxy (nginx/caddy)

### Example Production Docker Compose
```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./ssl:/etc/nginx/ssl:ro
      - ./nginx-prod.conf:/etc/nginx/nginx.conf:ro
```

## 📚 Next Steps

1. **Customize Configuration**: Edit `.env` and config files
2. **Add Custom Indexes**: Place in `indexes/` directory
3. **Integrate New Models**: Update model configurations
4. **Scale Deployment**: Use Docker Swarm or Kubernetes
5. **Monitor Performance**: Add logging and metrics

## 🤝 Development

### Adding Dependencies
```bash
# Virtual environment
source venv/bin/activate
pip install new-package
pip freeze > requirements.txt

# Docker
# Add to requirements.txt and rebuild
docker-compose build medrag-backend
```

### Code Changes
- **Backend changes**: Restart backend service
- **Frontend changes**: Refresh browser (no restart needed)
- **Docker changes**: Rebuild and restart containers

## 📞 Support

- **Issues**: Check logs and error messages
- **Performance**: Monitor GPU usage and memory
- **API Documentation**: Visit `/docs` on backend
- **Configuration**: Refer to `.env.example` for all options

Happy coding! 🚀
