# Hugging Face Spaces Deployment

This application is deployed on Hugging Face Spaces using Docker.

## 🚀 Quick Start

1. **Create a new Space** on Hugging Face with Docker SDK
2. **Push the repository root** so the Space sees the top-level `Dockerfile`, `.dockerignore`, `application/`, `config/`, `src/`, and `data/experiments/`
3. **Set environment variables** in Space settings:
   - `OLLAMA_URL` (optional): Ollama API URL for RAG
   - `OLLAMA_MODEL` (optional): Model name (default: `gpt-oss:120b-cloud`)
   - `HF_SPACE_ID`: Automatically set by HF Spaces

## 📁 Required Files

The following files must be in the Space root:

```
.
├── Dockerfile
├── .dockerignore
├── application/
│   ├── README.md (this file)
│   ├── app.py
│   ├── backend/
│   │   ├── main.py
│   │   ├── rag.py
│   │   ├── storage.py
│   │   ├── util.py
│   │   ├── requirements.txt
│   │   └── data/ (created at runtime)
│   └── frontend/
│       └── dashboard/
│           ├── package.json
│           └── src/
├── config/
├── src/
├── production_predictor.py
└── data/
    └── experiments/ (model checkpoints)
```

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `7860` |
| `HOST` | Server host | `0.0.0.0` |
| `OPENAI_API_KEY` | OpenAI API key (required for RAG) | None |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `HF_SPACE_ID` | HF Space identifier | Auto-set |

## 📦 Model Files

**Important**: Model checkpoint files must be uploaded to the Space:

1. Upload `data/experiments/hybrid/` directory containing:
   - `fold_3/model_target_h1/`
   - `fold_8/model_target_h5/`
   - `fold_3/model_target_h20/`

2. Or use Git LFS for large files:
   ```bash
   git lfs track "data/experiments/**/*.pkl"
   git lfs track "data/experiments/**/*.npy"
   ```

## 🏗️ Build Process

The Dockerfile uses a multi-stage build:

1. **Frontend Stage**: Builds React app with Vite
2. **Backend Stage**: Installs Python dependencies and copies built frontend

## 🌐 Access

Once deployed, your Space will be available at:
- `https://<your-username>-<space-name>.hf.space`

The API endpoints are:
- `GET /` - Health check
- `GET /predict` - Get predictions
- `POST /chat` - RAG chat endpoint
- `GET /news` - Get headlines
- `GET /models/info` - Model information
- `GET /storage/info` - Storage information

## 🔍 Troubleshooting

### Models Not Loading
- Ensure `data/experiments/hybrid/` is uploaded
- Check file paths in logs
- Verify model files are not corrupted

### RAG Not Working
- Check `OPENAI_API_KEY` environment variable is set
- Verify `OPENAI_MODEL` is valid (default: `gpt-4o-mini`)
- Check OpenAI API usage and rate limits
- RAG will gracefully degrade if API key is missing

### Frontend Not Loading
- Check that `npm run build` completed successfully
- Verify static files are in `backend/static/`
- Check browser console for errors

## 📝 Notes

- HF Spaces provides 16GB RAM and 2 CPU cores
- Cold starts may take 30-60 seconds
- Data directory is ephemeral (use HF datasets for persistence)
- Consider using HF datasets for model storage

