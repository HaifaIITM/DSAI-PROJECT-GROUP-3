# Exposing Your API to the Internet

## Option 1: ngrok (Quick Testing)

1. **Install ngrok**: Download from https://ngrok.com/download
2. **Start your backend**:
   ```bash
   python application/backend/main.py
   ```
3. **In another terminal, run ngrok**:
   ```bash
   ngrok http 8000
   ```
4. **Copy the HTTPS URL** (e.g., `https://abc123.ngrok.io`)

**Note**: Free ngrok URLs change on restart. For permanent URLs, use paid plan or other options.

---

## Option 2: Cloudflare Tunnel (Free, Permanent)

1. **Install cloudflared**:
   - Windows: Download from https://github.com/cloudflare/cloudflared/releases
   - Or use: `winget install --id Cloudflare.cloudflared`

2. **Authenticate**:
   ```bash
   cloudflared tunnel login
   ```

3. **Create a tunnel**:
   ```bash
   cloudflared tunnel create my-api
   ```

4. **Run tunnel**:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
   This gives you a permanent `*.trycloudflare.com` URL.

5. **For permanent custom domain** (optional):
   - Configure DNS in Cloudflare dashboard
   - Use config file for persistent tunnels

---

## Option 3: localtunnel (Alternative Quick Option)

1. **Install**:
   ```bash
   npm install -g localtunnel
   ```

2. **Run**:
   ```bash
   lt --port 8000
   ```

---

## Option 4: Deploy to Cloud Services

### Railway (Easiest)
1. Push code to GitHub
2. Connect repo at https://railway.app
3. Deploy automatically

### Render
1. Connect GitHub repo at https://render.com
2. Set build command and start command
3. Free tier available

### Fly.io
1. Install flyctl: `iwr https://fly.io/install.ps1 -useb | iex`
2. Run: `fly launch`
3. Deploy: `fly deploy`

---

## Option 5: Docker + Cloud Deployment

Your Dockerfile is ready. Deploy to:
- **Fly.io**: `fly launch` (auto-detects Dockerfile)
- **Railway**: Connect repo, auto-detects Docker
- **DigitalOcean App Platform**: Connect repo, uses Dockerfile

---

## Security Notes

1. **Update CORS** in `main.py` to allow your public URL:
   ```python
   allowed_origins = [
       "https://your-ngrok-url.ngrok.io",
       "https://your-custom-domain.com",
       # ... existing origins
   ]
   ```

2. **Add authentication** if exposing sensitive endpoints

3. **Use HTTPS** (all tunnel services provide this automatically)

---

## Recommended for Development
- **ngrok** - fastest to test

## Recommended for Production
- **Cloudflare Tunnel** - free, permanent, reliable
- **Railway/Render** - full deployment platform

