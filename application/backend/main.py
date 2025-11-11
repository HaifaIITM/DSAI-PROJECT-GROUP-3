"""
FastAPI Backend for Hybrid Model Predictions

Endpoints:
- GET /predict: Returns predictions for last 30 days + recent news
"""
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

import yfinance as yf
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from production_predictor import ProductionPredictor
from src.data.embeddings import compute_headline_features

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid ESN-Ridge Stock Predictor API",
    description="Production API for multi-horizon stock predictions using Hybrid ESN-Ridge models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (loaded once at startup)
predictor = None


# Response models
class NewsItem(BaseModel):
    date: str
    title: str
    publisher: str
    link: str


class PredictionItem(BaseModel):
    date: str
    h1_prediction: float
    h1_signal: str
    h5_prediction: float
    h5_signal: str
    h20_prediction: float
    h20_signal: str
    actual_close: float


class PredictionResponse(BaseModel):
    symbol: str
    predictions: List[PredictionItem]
    recent_news: List[NewsItem]
    generated_at: str


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators (10 features) from OHLCV data.
    Same logic as src/data/features.py
    """
    out = df.copy()
    
    # Use Adj Close if available, else Close
    price_col = "Adj Close" if "Adj Close" in out.columns else "Close"
    out = out.rename(columns={price_col: "PX"})
    
    # Returns & lags
    out["ret_1"] = np.log(out["PX"]).diff()
    out["ret_2"] = out["ret_1"].shift(1).rolling(2).sum()
    out["ret_5"] = out["ret_1"].shift(1).rolling(5).sum()
    
    # Realized volatility (20d, annualized)
    out["vol_20"] = out["ret_1"].rolling(20).std() * np.sqrt(252.0)
    
    # Moving averages & gap
    out["ma_10"] = out["PX"].rolling(10).mean()
    out["ma_20"] = out["PX"].rolling(20).mean()
    out["ma_gap"] = out["PX"] / out["ma_20"] - 1.0
    
    # RSI-14
    delta = out["PX"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    
    # Volume z-score (60d)
    if "Volume" in out.columns:
        out["vol_z"] = (out["Volume"] - out["Volume"].rolling(60).mean()) / out["Volume"].rolling(60).std()
    else:
        out["vol_z"] = 0.0
    
    # Day of week
    out["dow"] = out.index.dayofweek
    
    return out


def fetch_spy_data(days_back: int = 90) -> pd.DataFrame:
    """
    Fetch SPY data from yfinance for last N days.
    Need extra days for technical indicator warmup (60d for vol_z).
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    ticker = yf.Ticker("SPY")
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    
    if df.empty:
        raise HTTPException(status_code=500, detail="Failed to fetch SPY data from yfinance")
    
    return df


def fetch_spy_news(days_back: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch recent SPY news from yfinance.
    Returns list of news items for last N days.
    """
    try:
        ticker = yf.Ticker("SPY")
        news = ticker.news
        
        if not news:
            return []
        
        # Filter news to last N days
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_timestamp = cutoff_date.timestamp()
        
        recent_news = []
        for item in news:
            # New yfinance structure: data is nested in 'content' key
            content = item.get('content', {})
            pub_date = content.get('pubDate', '')  # ISO format: '2025-11-10T14:15:00Z'
            
            if pub_date:
                try:
                    # Parse ISO format and convert to timestamp
                    dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    pub_time = dt.timestamp()
                    
                    if pub_time >= cutoff_timestamp:
                        recent_news.append({
                            'date': dt.strftime('%Y-%m-%d %H:%M'),
                            'title': content.get('title', 'No title'),
                            'publisher': content.get('provider', {}).get('displayName', 'Unknown'),
                            'link': content.get('canonicalUrl', {}).get('url', '#')
                        })
                except Exception as e:
                    print(f"Error parsing news date: {e}")
                    continue
        
        # Sort by date ascending
        recent_news.sort(key=lambda x: x['date'])
        
        return recent_news
    
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Z-score normalization of features.
    Uses last 252 days as reference (rolling window).
    """
    normalized = df[feature_cols].copy()
    
    # Rolling z-score normalization (252-day window)
    for col in feature_cols:
        rolling_mean = normalized[col].rolling(252, min_periods=60).mean()
        rolling_std = normalized[col].rolling(252, min_periods=60).std()
        normalized[col] = (normalized[col] - rolling_mean) / (rolling_std + 1e-8)
    
    # Fill any remaining NaNs with 0
    normalized = normalized.fillna(0)
    
    return normalized.values


@app.on_event("startup")
async def startup_event():
    """Load production models on startup"""
    global predictor
    print("Loading production models...")
    try:
        # Use absolute path to models (relative to project root)
        models_base_dir = os.path.join(project_root, "data", "experiments")
        predictor = ProductionPredictor(base_dir=models_base_dir)
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Hybrid ESN-Ridge Stock Predictor API",
        "version": "1.0.0"
    }


@app.get("/predict", response_model=PredictionResponse)
async def get_predictions():
    """
    Get predictions for SPY for last 30 days + recent news (last 3 days).
    
    Returns:
        - predictions: Array of predictions for each day (all horizons: h1, h5, h20)
        - recent_news: Last 3 days of news headlines (ascending by date)
    """
    try:
        # 1. Fetch SPY data (need ~90 days for technical indicators)
        print("Fetching SPY data...")
        df = fetch_spy_data(days_back=90)
        
        # 2. Compute technical features (10 features)
        print("Computing technical features...")
        df = compute_technical_features(df)
        
        # 3. Get headline embeddings (28 features)
        # For production, we use the most recent headline or average of recent headlines
        print("Computing headline features...")
        # Simplified: Use zero embeddings for now (can be enhanced with live news API)
        # In production, you'd call compute_headline_features with real headlines
        headline_features = np.zeros((len(df), 28))
        
        # 4. Combine technical + headline features (38 total)
        technical_cols = ["ret_1", "ret_2", "ret_5", "vol_20", "ma_10", "ma_20", 
                         "ma_gap", "rsi_14", "vol_z", "dow"]
        
        # Check if all required columns exist
        missing_cols = [col for col in technical_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=500, 
                detail=f"Missing technical features: {missing_cols}"
            )
        
        # Normalize technical features
        X_technical = normalize_features(df, technical_cols)
        
        # Combine: 10 technical + 28 headline = 38 features
        X_features = np.concatenate([X_technical, headline_features], axis=1)
        
        # 5. Generate predictions for all horizons
        print("Generating predictions...")
        predictions_all = predictor.predict_all(X_features)
        
        # 6. Get last 30 days
        last_30_days = min(30, len(df))
        recent_dates = df.index[-last_30_days:]
        recent_prices = df["PX"].iloc[-last_30_days:].values
        
        predictions_list = []
        for i, date in enumerate(recent_dates):
            idx = len(df) - last_30_days + i
            
            predictions_list.append({
                "date": date.strftime('%Y-%m-%d'),
                "h1_prediction": float(predictions_all['h1'][idx]),
                "h1_signal": "BUY" if predictions_all['h1'][idx] > 0 else "SELL",
                "h5_prediction": float(predictions_all['h5'][idx]),
                "h5_signal": "BUY" if predictions_all['h5'][idx] > 0 else "SELL",
                "h20_prediction": float(predictions_all['h20'][idx]),
                "h20_signal": "BUY" if predictions_all['h20'][idx] > 0 else "SELL",
                "actual_close": float(recent_prices[i])
            })
        
        # 7. Fetch recent news (last 3 days)
        print("Fetching recent news...")
        news = fetch_spy_news(days_back=3)
        
        return {
            "symbol": "SPY",
            "predictions": predictions_list,
            "recent_news": news,
            "generated_at": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "models": predictor.get_model_info(),
        "status": "ready"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("Starting Hybrid ESN-Ridge Prediction API")
    print("="*60)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )

