# Hybrid Model API Backend

FastAPI backend for real-time stock predictions using Hybrid ESN-Ridge models.

## Features

- **Live Data**: Fetches SPY data from Yahoo Finance
- **Multi-Horizon Predictions**: h1 (1-day), h5 (5-day), h20 (20-day)
- **Recent News**: Last 3 days of SPY news headlines
- **Production Models**: Uses best-performing models (fold_3, fold_8)

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

## How It Works

### 1. Data Pipeline

```
yfinance → SPY OHLCV data (last 90 days)
         ↓
Technical Indicators (10 features):
- ret_1, ret_2, ret_5
- vol_20, ma_10, ma_20, ma_gap
- rsi_14, vol_z, dow
         ↓
Headline Embeddings (28 features):
- FinBERT sentence embeddings (optional)
         ↓
Total: 38 features (10 + 28)
```

### 2. Feature Normalization

```
Z-score normalization using 252-day rolling window
```

### 3. Prediction

```
Features → Production Predictor → Predictions (h1, h5, h20)
                                 ↓
                              Trading Signals (BUY/SELL)
```

### 4. News Fetching

```
yfinance API → SPY news (last 3 days) → Sorted by date ascending
```

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

## Limitations

1. **Headline Embeddings**: Currently uses zero embeddings. For production, integrate live news API and compute FinBERT embeddings.
2. **Historical Data**: yfinance may have delays (15-20 minutes).
3. **Rate Limits**: yfinance has rate limits on API calls.
4. **Single Symbol**: Currently only supports SPY.

## Future Enhancements

- [ ] Real-time headline embeddings from news API
- [ ] Multi-symbol support
- [ ] WebSocket for live updates
- [ ] Caching layer for faster responses
- [ ] Database for historical predictions
- [ ] Authentication/API keys
- [ ] Rate limiting

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

