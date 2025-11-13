# Hybrid Model API Backend

FastAPI backend for real-time stock predictions using Hybrid ESN-Ridge models.

## Features

- **Live Data**: Fetches SPY data from Yahoo Finance
- **Multi-Horizon Predictions**: h1 (1-day), h5 (5-day), h20 (20-day)
- **Recent News**: Last 3 days of SPY news headlines (accumulated over time)
- **Production Models**: Uses best-performing models (fold_3, fold_8)
- **Persistent Storage**: Automatically stores headlines, predictions, embeddings, and features
- **Data Accumulation**: Headlines accumulate over time (solving yfinance 3-day limitation)

## Quick Start

### 1. Install Dependencies

```bash
cd application/backend
pip install -r requirements.txt
```

### 2. Run the API

```bash
python main.py
```

The API will start on `http://localhost:8000`

### 3. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### GET /

Health check endpoint.

**Response**:
```json
{
  "status": "online",
  "service": "Hybrid ESN-Ridge Stock Predictor API",
  "version": "1.0.0"
}
```

### GET /predict

Get predictions for last 30 days + recent news (last 3 days).

**Response**:
```json
{
  "symbol": "SPY",
  "predictions": [
    {
      "date": "2025-11-11",
      "h1_prediction": 0.0023,
      "h1_signal": "BUY",
      "h5_prediction": 0.0045,
      "h5_signal": "BUY",
      "h20_prediction": 0.0168,
      "h20_signal": "BUY",
      "actual_close": 450.32
    },
    ...
  ],
  "recent_news": [
    {
      "date": "2025-11-09 10:30",
      "title": "S&P 500 Reaches New High",
      "publisher": "Reuters",
      "link": "https://..."
    },
    ...
  ],
  "generated_at": "2025-11-11T15:30:00"
}
```

### GET /models/info

Get information about loaded models.

**Response**:
```json
{
  "models": {
    "h1": {
      "fold": 3,
      "sharpe": 1.25,
      "path": "data/experiments/hybrid/fold_3/model_target_h1"
    },
    "h5": {
      "fold": 8,
      "sharpe": 2.94,
      "path": "data/experiments/hybrid/fold_8/model_target_h5"
    },
    "h20": {
      "fold": 3,
      "sharpe": 6.813,
      "path": "data/experiments/hybrid/fold_3/model_target_h20"
    }
  },
  "status": "ready"
}
```

### GET /news

Get stored news headlines (accumulated over time, not just last 3 days).

**Query Parameters**:
- `days_back` (optional): Number of days to look back (default: 7)

**Example**:
```bash
curl http://localhost:8000/news?days_back=30
```

**Response**:
```json
[
  {
    "date": "2025-11-09 10:30",
    "title": "S&P 500 Reaches New High",
    "publisher": "Reuters",
    "link": "https://..."
  }
]
```

### GET /storage/info

Get information about stored data.

**Response**:
```json
{
  "headlines": {
    "file": "application/backend/data/headlines/spy_headlines.csv",
    "exists": true,
    "count": 150,
    "date_range": {
      "start": "2025-10-01",
      "end": "2025-11-11"
    }
  },
  "predictions": {
    "directory": "application/backend/data/predictions",
    "count": 25
  },
  "embeddings": {
    "directory": "application/backend/data/embeddings",
    "count": 25
  },
  "features": {
    "directory": "application/backend/data/features",
    "count": 25
  }
}
```

## How It Works

### 1. Data Pipeline

```
yfinance → SPY OHLCV data (last 300 days)
         ↓
Fetch & Store News Headlines (last 3 days from yfinance)
         ↓
Load Stored Headlines CSV (accumulated over time)
         ↓
Technical Indicators (10 features):
- ret_1, ret_2, ret_5
- vol_20, ma_10, ma_20, ma_gap
- rsi_14, vol_z, dow
         ↓
Headline Embeddings (28 features):
- FinBERT sentence embeddings (from stored headlines)
- PCA reduced (12 + 14 dimensions)
         ↓
Total: 38 features (10 + 28)
         ↓
Z-score normalization (252-day rolling window)
         ↓
Production Predictor → Predictions (h1, h5, h20)
         ↓
Store: Predictions, Embeddings, Features
         ↓
Return JSON response
```

### 2. Storage System

The backend automatically stores all data:

```
application/backend/data/
├── headlines/
│   └── spy_headlines.csv          # Accumulated headlines
├── predictions/
│   └── predictions_batch_*.json    # Timestamped predictions
├── embeddings/
│   └── embeddings_*.npz            # Headline embeddings
└── features/
    ├── features_*.npz               # Final 38 features (numpy)
    └── features_*.csv               # Final 38 features (CSV)
```

**What gets stored:**
- **Headlines**: Automatically saved when fetched (accumulates over time)
- **Predictions**: Every prediction call with metadata
- **Embeddings**: Headline PCA features (28 dimensions)
- **Features**: Final 38 normalized features (both NPZ and CSV)

### 3. News Accumulation

Since yfinance only provides last 3 days of headlines, the system:
1. Fetches headlines from yfinance (last 3 days)
2. Saves new headlines to storage (CSV file)
3. Loads all stored headlines for feature computation
4. Over time, builds a historical database of headlines

This solves the yfinance limitation by accumulating headlines over multiple API calls.

## Configuration

### Change Port

Edit `main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8080)  # Change port here
```

### Change Symbol

Edit `fetch_spy_data()` and `fetch_spy_news()`:
```python
ticker = yf.Ticker("AAPL")  # Change symbol
```

### Change Prediction Window

Edit `/predict` endpoint:
```python
last_30_days = min(60, len(df))  # Change to 60 days
```

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t hybrid-api .
docker run -p 8000:8000 hybrid-api
```

### Using Gunicorn

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Performance

- **Model Load Time**: ~0.5s (once at startup)
- **Data Fetch Time**: ~1-2s (yfinance API)
- **Prediction Time**: ~0.03s (all 3 horizons)
- **Total Response Time**: ~1-2s

## Error Handling

The API handles:
- ✅ Missing data from yfinance
- ✅ Model loading failures
- ✅ Invalid dates
- ✅ Missing technical indicators
- ✅ Network timeouts

All errors return appropriate HTTP status codes and error messages.

## Testing

### Using curl

```bash
# Health check
curl http://localhost:8000/

# Get predictions
curl http://localhost:8000/predict

# Get model info
curl http://localhost:8000/models/info
```

### Using Python

```python
import requests

# Get predictions
response = requests.get("http://localhost:8000/predict")
data = response.json()

print(f"Symbol: {data['symbol']}")
print(f"Latest prediction: {data['predictions'][-1]}")
print(f"Recent news: {len(data['recent_news'])} items")
```

## Data Storage

### Accessing Stored Data

**Python:**
```python
from storage import DataStorage

storage = DataStorage()

# Get headlines from last 30 days
headlines = storage.get_headlines(days_back=30)

# Get latest predictions
latest = storage.get_latest_predictions(n=5)

# Check storage status
info = storage.get_storage_info()
```

**API:**
```bash
# Get stored headlines
curl http://localhost:8000/news?days_back=30

# Check storage info
curl http://localhost:8000/storage/info
```

### File Formats

- **Headlines**: CSV format (`spy_headlines.csv`)
- **Predictions**: JSON format with metadata
- **Embeddings**: NPZ format (numpy compressed)
- **Features**: Both NPZ and CSV formats

## Limitations

1. **yfinance Headlines**: yfinance only provides last 3 days, but system accumulates them over time in storage.
2. **Historical Data**: yfinance may have delays (15-20 minutes).
3. **Rate Limits**: yfinance has rate limits on API calls.
4. **Single Symbol**: Currently only supports SPY.
5. **File-based Storage**: Currently uses file system (can be migrated to database for production).

## Future Enhancements

- [x] Persistent headline storage (accumulates over time)
- [x] Prediction storage with metadata
- [x] Embedding and feature storage
- [ ] Real-time headline embeddings from news API
- [ ] Multi-symbol support
- [ ] WebSocket for live updates
- [ ] Caching layer for faster responses
- [ ] Database migration (PostgreSQL) for historical predictions
- [ ] Authentication/API keys
- [ ] Rate limiting
- [ ] Historical prediction analysis endpoint

## Troubleshooting

**Models not loading**:
```bash
# Ensure models are trained
python scripts/training/train_all_hybrid_models.py
```

**yfinance errors**:
```bash
# Update yfinance
pip install --upgrade yfinance
```

**Import errors**:
```bash
# Ensure you're in project root
cd ../..
python application/backend/main.py
```

## Support

- API Documentation: http://localhost:8000/docs
- Project README: ../../README.md
- Production Predictor: ../../production_predictor.py

