# Exposing Frontend (5173) + Backend (8000) to Internet

## Option 1: ngrok - Two Separate Tunnels (Easiest)

### Step 1: Expose Backend
```bash
# Terminal 1: Start backend
python application/backend/main.py

# Terminal 2: Expose backend
ngrok http 8000
```
Copy the backend URL (e.g., `https://abc123.ngrok.io`)

### Step 2: Expose Frontend
```bash
# Terminal 3: Start frontend (if not running)
cd application/frontend/dashboard
npm run dev

# Terminal 4: Expose frontend
ngrok http 5173
```
Copy the frontend URL (e.g., `https://xyz789.ngrok.io`)

### Step 3: Update Frontend to Use Public Backend URL

Create `.env` file in `application/frontend/dashboard/`:
```env
VITE_API_BASE_URL=https://abc123.ngrok.io
```

Restart frontend dev server.

**Note**: ngrok free URLs change on restart. For permanent URLs, use paid plan or Option 2.

---

## Option 2: ngrok - Single Tunnel with Reverse Proxy (Better)

Use one ngrok URL that serves both frontend and backend.

### Install nginx (Windows)
```powershell
# Using Chocolatey
choco install nginx

# Or download from: http://nginx.org/en/download.html
```

### Create nginx config (`nginx.conf`):
```nginx
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name localhost;

        # Serve frontend
        location / {
            proxy_pass http://localhost:5173;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
        }

        # Proxy backend API
        location /api/ {
            proxy_pass http://localhost:8000/api/;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        location /predict {
            proxy_pass http://localhost:8000/predict;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }

        location /chat {
            proxy_pass http://localhost:8000/chat;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }

        location /news {
            proxy_pass http://localhost:8000/news;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }

        location /models {
            proxy_pass http://localhost:8000/models;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }

        location /storage {
            proxy_pass http://localhost:8000/storage;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }
    }
}
```

### Run:
```bash
# Start nginx
nginx -c path/to/nginx.conf

# Expose nginx (port 80) with ngrok
ngrok http 80
```

### Update Frontend
Set `VITE_API_BASE_URL` to the ngrok URL (same origin, so can be empty or same URL).

---

## Option 3: Cloudflare Tunnel (Free, Permanent)

### Expose Both Ports:
```bash
# Install cloudflared (if not installed)
# Windows: Download from https://github.com/cloudflare/cloudflared/releases

# Authenticate
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create my-app

# Run tunnel (exposes both ports)
cloudflared tunnel --url http://localhost:5173 --url http://localhost:8000
```

This gives you two permanent URLs. Update frontend `.env`:
```env
VITE_API_BASE_URL=https://your-backend-url.trycloudflare.com
```

---

## Option 4: Simple Node.js Proxy (No nginx needed)

Create `proxy-server.js` in project root:

```javascript
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();

// Proxy backend API
app.use('/api', createProxyMiddleware({
  target: 'http://localhost:8000',
  changeOrigin: true,
  pathRewrite: { '^/api': '' }
}));

app.use('/predict', createProxyMiddleware({
  target: 'http://localhost:8000',
  changeOrigin: true
}));

app.use('/chat', createProxyMiddleware({
  target: 'http://localhost:8000',
  changeOrigin: true
}));

app.use('/news', createProxyMiddleware({
  target: 'http://localhost:8000',
  changeOrigin: true
}));

app.use('/models', createProxyMiddleware({
  target: 'http://localhost:8000',
  changeOrigin: true
}));

app.use('/storage', createProxyMiddleware({
  target: 'http://localhost:8000',
  changeOrigin: true
}));

// Proxy frontend
app.use('/', createProxyMiddleware({
  target: 'http://localhost:5173',
  changeOrigin: true,
  ws: true // WebSocket support for Vite HMR
}));

app.listen(3000, () => {
  console.log('Proxy server running on http://localhost:3000');
});
```

### Install dependencies:
```bash
npm install express http-proxy-middleware
```

### Run:
```bash
# Terminal 1: Backend
python application/backend/main.py

# Terminal 2: Frontend
cd application/frontend/dashboard && npm run dev

# Terminal 3: Proxy
node proxy-server.js

# Terminal 4: Expose proxy
ngrok http 3000
```

Update frontend `.env`:
```env
VITE_API_BASE_URL=https://your-ngrok-url.ngrok.io
```

---

## Recommended Quick Setup

**For immediate testing**: Use Option 1 (two ngrok tunnels)

**For production-like setup**: Use Option 4 (Node.js proxy) - easiest, no nginx needed

**For permanent solution**: Use Option 3 (Cloudflare Tunnel)

---

## Quick Start: Proxy Server (Already Set Up)

Dependencies are installed. Run these commands:

```powershell
# Terminal 1: Start backend
python application/backend/main.py

# Terminal 2: Start frontend
cd application/frontend/dashboard
npm run dev

# Terminal 3: Start proxy server
node proxy-server.js

# Terminal 4: Expose to internet
ngrok http 3000
```

Copy the ngrok HTTPS URL (e.g., `https://abc123.ngrok.io`) and access your app there!

**Note**: Frontend will automatically use the same origin for API calls, so no `.env` changes needed.

