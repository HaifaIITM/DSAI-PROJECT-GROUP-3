/**
 * Simple proxy server to combine frontend (5173) and backend (8000)
 * Expose this with ngrok: ngrok http 3000
 */
import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';

const app = express();
const PORT = 3000;

// Proxy backend API endpoints
const backendProxy = createProxyMiddleware({
  target: 'http://localhost:8000',
  changeOrigin: true,
  logLevel: 'debug'
});

app.use('/api', backendProxy);
app.use('/predict', backendProxy);
app.use('/chat', backendProxy);
app.use('/news', backendProxy);
app.use('/models', backendProxy);
app.use('/storage', backendProxy);
app.use('/docs', backendProxy);
app.use('/openapi.json', backendProxy);

// Proxy frontend (must be last)
app.use('/', createProxyMiddleware({
  target: 'http://localhost:5173',
  changeOrigin: true,
  ws: true, // WebSocket support for Vite HMR
  logLevel: 'debug'
}));

app.listen(PORT, () => {
  console.log(`\n🚀 Proxy server running on http://localhost:${PORT}`);
  console.log(`   Frontend: http://localhost:5173`);
  console.log(`   Backend:  http://localhost:8000`);
  console.log(`\n📡 Expose with: ngrok http ${PORT}\n`);
});

