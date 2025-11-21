FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend/dashboard

COPY application/frontend/dashboard/package*.json ./
RUN npm ci

COPY application/frontend/dashboard/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY application/backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

RUN pip install --no-cache-dir \
    requests \
    aiofiles

COPY production_predictor.py ./
COPY config/ ./config/
COPY src/ ./src/
COPY data/experiments/ ./data/experiments/

COPY application/backend/ ./backend/
COPY --from=frontend-builder /app/frontend/dashboard/dist ./backend/static
COPY application/app.py ./app.py

RUN mkdir -p backend/data/{headlines,predictions,embeddings,features}

ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV HOST=0.0.0.0

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/')" || exit 1

CMD ["python", "app.py"]

