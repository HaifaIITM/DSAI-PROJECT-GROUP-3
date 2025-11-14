# Hugging Face Spaces Deployment Guide

This guide explains how to deploy the Hybrid ESN-Ridge Stock Predictor to Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co
2. **Model Files**: Ensure model checkpoints are available
3. **Git Repository**: Code should be in a Git repo (or upload directly)

## 🚀 Deployment Steps

### Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure:
   - **SDK**: Docker
   - **Hardware**: CPU Basic (or GPU if needed for models)
   - **Visibility**: Public or Private
   - **Name**: e.g., `spy-predictor`

### Step 2: Upload Files

Upload the following structure to your Space:

```
your-space/
├── Dockerfile
├── .dockerignore
├── README.md
├── app.py (optional entry point)
├── backend/
│   ├── main.py
│   ├── rag.py
│   ├── storage.py
│   ├── util.py
│   ├── requirements.txt
│   └── data/ (created at runtime)
├── frontend/
│   └── dashboard/
│       ├── package.json
│       └── src/
├── config/
├── src/
├── production_predictor.py
└── data/
    └── experiments/
        └── hybrid/
            ├── fold_3/
            │   ├── model_target_h1/
            │   └── model_target_h20/
            └── fold_8/
                └── model_target_h5/
```

### Step 3: Set Environment Variables

In Space Settings → Variables, add:

- `OPENAI_API_KEY` (required): Your OpenAI API key for RAG functionality
- `OPENAI_MODEL` (optional): Model name (default: `gpt-4o-mini`)
- `HF_SPACE_ID`: Automatically set by HF Spaces

### Step 4: Build and Deploy

HF Spaces will automatically:
1. Build the Docker image
2. Install dependencies
3. Build the frontend
4. Start the application

Monitor the build logs in the Space interface.

## 📁 File Structure

### Required Files

- **Dockerfile**: Multi-stage build (frontend + backend)
- **.dockerignore**: Excludes unnecessary files
- **backend/main.py**: FastAPI application
- **backend/requirements.txt**: Python dependencies
- **frontend/dashboard/package.json**: Node dependencies

### Model Files

Model checkpoints must be in:
```
data/experiments/hybrid/fold_X/model_target_hY/
```

**Note**: For large files (>10MB), use Git LFS:
```bash
git lfs track "data/experiments/**/*.pkl"
git lfs track "data/experiments/**/*.npy"
git add .gitattributes
```

## 🔧 Configuration

### Port Configuration

The application uses port `7860` by default (HF Spaces standard). This is configured via:
- `PORT` environment variable (auto-set by HF Spaces)
- `HOST=0.0.0.0` (required for Docker)

### CORS Configuration

CORS is configured to allow:
- `https://*.hf.space` domains
- Localhost for development
- All origins when `HF_SPACE_ID` is set

### Static Files

The frontend is built during Docker build and served from:
- `backend/static/` (created from `frontend/dashboard/dist/`)

## 🌐 Accessing Your Space

Once deployed, your Space will be available at:
```
https://<your-username>-<space-name>.hf.space
```

### API Endpoints

- `GET /` - Frontend dashboard
- `GET /api/health` - Health check
- `GET /predict` - Get predictions
- `POST /chat` - RAG chat endpoint
- `GET /news` - Get headlines
- `GET /models/info` - Model information
- `GET /storage/info` - Storage information
- `GET /docs` - API documentation (Swagger UI)

## 🐛 Troubleshooting

### Build Fails

1. **Check Dockerfile syntax**: Ensure all paths are correct
2. **Verify dependencies**: Check `requirements.txt` and `package.json`
3. **Check logs**: Review build logs in HF Spaces interface

### Models Not Loading

1. **Verify paths**: Check `data/experiments/hybrid/` exists
2. **Check file sizes**: Large files may need Git LFS
3. **Review logs**: Check startup logs for model loading errors

### Frontend Not Loading

1. **Check build**: Verify `npm run build` completed
2. **Verify static files**: Check `backend/static/` exists
3. **Check browser console**: Look for 404 errors

### RAG Not Working

1. **Check API Key**: Verify `OPENAI_API_KEY` is set correctly
2. **Check Model**: Ensure `OPENAI_MODEL` is valid (default: `gpt-4o-mini`)
3. **API Limits**: Check OpenAI API usage and rate limits
4. **Fallback**: RAG will gracefully degrade if unavailable

## 📊 Resource Limits

HF Spaces provides:
- **CPU Basic**: 2 vCPU, 16GB RAM
- **CPU Upgrade**: 4 vCPU, 32GB RAM
- **GPU**: Various GPU options (paid)

Monitor resource usage in Space settings.

## 🔄 Updating Your Space

1. **Push changes** to your Git repository
2. HF Spaces will **automatically rebuild**
3. Monitor **build logs** for errors
4. Test the **deployed application**

## 📝 Notes

- **Cold starts**: First request may take 30-60 seconds
- **Data persistence**: `backend/data/` is ephemeral (use HF datasets for persistence)
- **Model storage**: Consider using HF model hub for large models
- **Environment variables**: Can be updated without rebuild

## 🆘 Support

For issues:
1. Check build logs in HF Spaces
2. Review application logs
3. Test locally with Docker
4. Check HF Spaces documentation

