# Multi-stage build for Hugging Face Spaces deployment
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend/dashboard

# Copy frontend files
COPY frontend/dashboard/package*.json ./
RUN npm ci

COPY frontend/dashboard/ ./
RUN npm run build

# Python backend stage
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install additional dependencies for production
RUN pip install --no-cache-dir \
    requests \
    aiofiles

# Copy project root files (for production_predictor)
COPY production_predictor.py ./
COPY config/ ./config/
COPY src/ ./src/
COPY data/experiments/ ./data/experiments/

# Copy HF entrypoint
COPY app.py ./app.py

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/dashboard/dist ./backend/static

# Create data directories
RUN mkdir -p backend/data/{headlines,predictions,embeddings,features}

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV HOST=0.0.0.0

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/')" || exit 1

# Run the application (use app.py for HF Spaces compatibility)
CMD ["python", "app.py"]

