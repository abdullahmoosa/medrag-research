# Medical RAG Chat Application - Distributed Architecture 🏥

A distributed medical RAG chat application with separated frontend and backend components. The backend runs on a GPU-enabled server with indexes, while the frontend can run on any machine and connect via HTTP API.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────┐    HTTP API    ┌─────────────────────────────────┐
│         Frontend Client         │◀──────────────▶│         Backend Server          │
│         (Any Machine)           │                │       (GPU + Indexes)          │
├─────────────────────────────────┤                ├─────────────────────────────────┤
│ • Web Interface (HTML/JS)       │                │ • FastAPI REST API              │
│ • Chat UI & Configuration       │                │ • RAG Service                   │
│ • Source Citations Display      │                │ • HybridIndex (FAISS+BM25)     │
│ • Real-time Connection Status   │                │ • MedEmbed Embeddings           │
│ • Cross-platform Compatible     │                │ • Ollama LLM Integration        │
└─────────────────────────────────┘                └─────────────────────────────────┘
            Port 3000                                           Port 8000
```

### Key Benefits:
- **Scalability**: Multiple frontend clients can connect to one backend
- **Flexibility**: Frontend runs on any device (laptop, tablet, phone)  
- **Resource Optimization**: GPU and large indexes only needed on backend
- **Network Deployment**: Frontend and backend can be on different networks

## 📁 Directory Structure

```
src/app/
├── backend.py              # Backend API server (runs on GPU machine)
├── start_backend.py        # Backend startup script  
├── start_backend.bat       # Windows backend startup
├── start_frontend.py       # Frontend startup script
├── start_frontend.bat      # Windows frontend startup
├── rag_service.py          # Core RAG logic
├── config.py              # Configuration management
├── models.py              # API data models
├── requirements.txt       # Dependencies
└── frontend/
    ├── index.html         # Web interface
    └── serve.py           # Frontend HTTP server
```

## 🚀 Quick Start Guide

### Step 1: Start Backend Server (GPU Machine)

On your GPU-enabled machine with the indexes:

```bash
# Navigate to app directory
cd src/app

# Install dependencies
pip install -r requirements.txt

# Start backend server
python start_backend.py --host 0.0.0.0 --port 8000
```

**Windows users:**
```cmd
cd src\app
start_backend.bat
```

The backend will be available at: `http://your-gpu-machine:8000`

### Step 2: Start Frontend Client (Any Machine)

On any client machine:

```bash
# Navigate to app directory  
cd src/app

# Start frontend server
python start_frontend.py --host 0.0.0.0 --port 3000
```

**Windows users:**
```cmd
cd src\app  
start_frontend.bat
```

The frontend will be available at: `http://localhost:3000`

### Step 3: Connect Frontend to Backend

1. Open frontend in browser: `http://localhost:3000`
2. Configure backend URL: `http://your-gpu-machine:8000`
3. Click "Connect" button
4. Start asking medical questions!

## 🔧 Configuration

### Backend Configuration

The backend can be configured via:

**Command Line:**
```bash
python start_backend.py --host 0.0.0.0 --port 8000 --workers 1
```

**Environment Variables:**
```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export CUDA_VISIBLE_DEVICES="0"
```

### Frontend Configuration

**Command Line:**
```bash
python start_frontend.py --port 3000 --backend-url http://gpu-server:8000
```

**Web Interface:**
- Backend server URL
- Index selection  
- Retrieval parameters
- Model selection

## 🌐 Network Setup

### Local Network Deployment

1. **Find Backend IP Address:**
   ```bash
   # Linux/Mac
   hostname -I
   
   # Windows  
   ipconfig
   ```

2. **Configure Firewall:**
   - Backend: Open port 8000
   - Frontend: Open port 3000

3. **Connect from Frontend:**
   - Use: `http://<backend-ip>:8000`
   - Example: `http://192.168.1.100:8000`

### Internet Deployment

For internet access, consider:
- **Backend**: Use reverse proxy (nginx) with SSL
- **Frontend**: Can be served from any web server
- **Security**: Implement authentication and HTTPS

## 🔌 API Endpoints

The backend provides these REST API endpoints:

### Core Endpoints
- `POST /api/chat` - Send chat messages
- `GET /api/health` - Health check
- `POST /api/config` - Update configuration  
- `GET /api/config` - Get current config

### Information Endpoints
- `GET /api/available-indexes` - List available indexes
- `GET /api/stats` - Service statistics
- `GET /docs` - Swagger API documentation

### Example API Usage:

```bash
# Health check
curl http://gpu-server:8000/api/health

# Send chat message
curl -X POST http://gpu-server:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the symptoms of diabetes?"}'

# Update configuration
curl -X POST http://gpu-server:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"mode": "dense", "k": 20}'
```

## 📱 Multi-Device Access

### Desktop Computers
- Full web interface with all features
- Optimal experience with large screens
- Real-time configuration changes

### Tablets & Mobile
- Responsive design adapts to screen size
- Touch-friendly interface
- All core functionality available

### Multiple Users
- Each client gets independent chat session
- Shared backend resources (GPU, indexes)
- Configurable per-client parameters

## 🛠️ Development

### Adding Frontend Features
1. Edit `frontend/index.html`
2. Add API calls as needed
3. No backend restart required

### Adding Backend Features  
1. Modify `backend.py` or `rag_service.py`
2. Update API models in `models.py`
3. Restart backend to apply changes

### Custom Frontend
You can create your own frontend in any technology:
- React/Vue/Angular web apps
- Mobile apps (iOS/Android)
- Desktop applications
- Command-line tools

Just connect to the backend API endpoints.

## 🔒 Security Considerations

### Production Deployment
- **Authentication**: Add API keys or OAuth
- **HTTPS**: Use SSL certificates
- **CORS**: Restrict origins to specific domains
- **Rate Limiting**: Prevent API abuse
- **Firewall**: Limit network access

### Example Secure Configuration:
```python
# In production backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📊 Performance

### Backend (GPU Machine)
- **RAM**: 8-16GB (for indexes)
- **GPU**: Optional, improves embedding speed
- **CPU**: Multi-core recommended
- **Storage**: SSD for faster index loading

### Frontend (Client Machines)  
- **RAM**: 2GB minimum
- **Network**: Stable internet connection
- **Browser**: Modern browser with JavaScript

### Typical Response Times:
- **Network latency**: 10-100ms
- **Retrieval**: 0.1-0.5 seconds  
- **Generation**: 1-5 seconds
- **Total**: 1-6 seconds per query

## 🐛 Troubleshooting

### Backend Issues
```bash
# Check if backend is running
curl http://backend-ip:8000/api/health

# Check Ollama connection
curl http://backend-ip:11434/api/tags

# Check logs
python start_backend.py --log-level debug
```

### Frontend Issues  
```bash
# Test backend connectivity
curl http://backend-ip:8000/api/health

# Check browser console for errors
# Try different backend URL
```

### Network Issues
```bash  
# Test port connectivity
telnet backend-ip 8000

# Check firewall settings
# Verify IP addresses and ports
```

## 📚 Examples

### Python Client Example:
```python
import requests

backend_url = "http://gpu-server:8000"

# Send message
response = requests.post(f"{backend_url}/api/chat", json={
    "message": "What are the symptoms of hypertension?",
    "conversation_id": "python_client"
})

data = response.json()
print(f"Response: {data['response']}")
print(f"Sources: {len(data['sources'])}")
```

### JavaScript/Node.js Client:
```javascript
const backend_url = 'http://gpu-server:8000';

fetch(`${backend_url}/api/chat`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message: 'What are the symptoms of diabetes?',
    conversation_id: 'js_client'
  })
})
.then(response => response.json())
.then(data => {
  console.log('Response:', data.response);
  console.log('Sources:', data.sources.length);
});
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch for frontend or backend
3. Test with both components
4. Submit pull request

## 📞 Support

- **Documentation**: See individual component READMEs
- **API Docs**: Visit `/docs` endpoint on backend
- **Issues**: Report on GitHub repository

---

This distributed architecture allows you to leverage powerful GPU resources on one machine while providing flexible, multi-device access through lightweight frontend clients! 🚀
