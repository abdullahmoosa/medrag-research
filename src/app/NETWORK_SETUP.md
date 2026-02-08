# Network Access Guide for Medical RAG Application

## 🌐 Running Frontend from Other PCs

The Medical RAG application is designed to work across networks. Here's how to set it up:

### Current Configuration
- **Backend Port**: 8547
- **Frontend Port**: 3829
- **Backend URL**: http://localhost:8547

## Step-by-Step Setup

### 1. Find Your Server's IP Address

On the GPU server (where backend runs), find the IP address:

**Windows:**
```powershell
ipconfig
```
Look for "IPv4 Address" under your main network adapter (usually WiFi or Ethernet).

**Linux/Mac:**
```bash
ip addr show
# or
ifconfig
```

Example IP: `192.168.1.100`

### 2. Configure Backend for Network Access

The backend is already configured to accept connections from other machines (`0.0.0.0:8547`).

**Start the backend:**
```bash
# On GPU server
run_backend_venv.bat
```

### 3. Configure Frontend for Remote Access

You have two options:

#### Option A: Run Frontend on Client PC

1. **Copy the frontend files** to the client PC:
   - Copy `src/app/frontend/` folder to client PC
   - Copy `src/app/start_frontend.py` to client PC

2. **Install Python** on client PC (if not already installed)

3. **Start frontend** pointing to your server:
   ```bash
   # On client PC
   python start_frontend.py --backend-url http://192.168.1.100:8547 --port 3829
   ```

4. **Access the application**: http://localhost:3829

#### Option B: Access Frontend Through Network

1. **Start frontend on server:**
   ```bash
   # On GPU server
   run_frontend_venv.bat
   ```

2. **Access from client PC**: http://192.168.1.100:3829

### 4. Update Frontend Configuration

If you want to permanently change the default backend URL, edit the frontend:

1. Open `src/app/frontend/index.html`
2. Find line with: `value="http://localhost:8547"`
3. Change to: `value="http://YOUR_SERVER_IP:8547"`

## Network Requirements

### Firewall Configuration

**Windows (on GPU server):**
```powershell
# Allow backend port
netsh advfirewall firewall add rule name="MedRAG Backend" dir=in action=allow protocol=TCP localport=8547

# Allow frontend port (if running frontend on server)
netsh advfirewall firewall add rule name="MedRAG Frontend" dir=in action=allow protocol=TCP localport=3829
```

**Linux (on GPU server):**
```bash
# Allow backend port
sudo ufw allow 8547/tcp

# Allow frontend port (if running frontend on server) 
sudo ufw allow 3829/tcp
```

### Port Forwarding (if using router)

If clients are on different networks, configure router port forwarding:
- External Port: 8547 → Internal Port: 8547 (Backend)
- External Port: 3829 → Internal Port: 3829 (Frontend, optional)

## Testing Network Access

### Test Backend Connectivity

From client PC:
```bash
# Test if backend is accessible
curl http://192.168.1.100:8547/api/health

# Or in browser
http://192.168.1.100:8547/docs
```

### Test Frontend Connectivity

In browser from client PC:
```
http://192.168.1.100:3829
```

## Troubleshooting

### Common Issues

1. **Connection Refused:**
   - Check if backend is running: `http://SERVER_IP:8547/docs`
   - Check firewall settings
   - Verify IP address is correct

2. **Frontend Can't Connect to Backend:**
   - Update backend URL in frontend interface
   - Check if both frontend and backend are running
   - Test backend directly: `http://SERVER_IP:8547/api/health`

3. **Slow Performance:**
   - Ensure GPU server has good network connection
   - Consider running frontend locally on client PCs

### Network Diagnostics

```bash
# Test network connectivity
ping SERVER_IP

# Test specific ports
telnet SERVER_IP 8547  # Backend
telnet SERVER_IP 3829  # Frontend
```

## Security Considerations

### Production Deployment

For production use, consider:

1. **HTTPS Setup**: Use reverse proxy (nginx/Apache) with SSL
2. **Authentication**: Add user authentication
3. **Rate Limiting**: Implement API rate limiting
4. **VPN**: Use VPN for secure access across networks

### Basic Security

- Change default ports if needed
- Use strong firewall rules
- Monitor access logs
- Keep software updated

## Example Network Topology

```
[Client PC 1] ←→ [Network/WiFi] ←→ [GPU Server]
     ↑                                  ↓
[Client PC 2] ←→ [Network/WiFi] ←→ [Backend:8547]
     ↑                                  ↓
[Client PC 3] ←→ [Network/WiFi] ←→ [Frontend:3829]
```

## Quick Setup Commands

**On GPU Server:**
```bash
# Start both services
run_backend_venv.bat   # Terminal 1
run_frontend_venv.bat  # Terminal 2 (optional)
```

**On Client PC (Option A):**
```bash
# Copy frontend files and run locally
python start_frontend.py --backend-url http://192.168.1.100:8547
```

**Access URLs:**
- Backend API: http://192.168.1.100:8547/docs
- Frontend: http://192.168.1.100:3829
- Health Check: http://192.168.1.100:8547/api/health

Replace `192.168.1.100` with your actual server IP address.
