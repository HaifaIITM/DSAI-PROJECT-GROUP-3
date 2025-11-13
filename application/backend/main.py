"""
FastAPI Backend for Hybrid Model Predictions

Endpoints:
- GET /predict: Returns predictions for last 30 days + recent news
"""
import sys
import os
from datetime import datetime
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Add backend directory to path for local imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from production_predictor import ProductionPredictor
from util import (
    fetch_spy_data,
    fetch_spy_news,
    prepare_features_for_prediction
)

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
        # 1. Fetch SPY data (need ~90 days for technical indicators, more for normalization)
        print("Fetching SPY data...")
        try:
            # Fetch more days for better normalization (252 days for rolling window)
            df = fetch_spy_data(days_back=300)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        # 2. Prepare features using complete pipeline (matches training process)
        print("Preparing features (technical + headlines)...")
        try:
            # Check if headlines CSV exists in project root
            headlines_csv = os.path.join(project_root, "data", "spy_news.csv")
            if not os.path.exists(headlines_csv):
                headlines_csv = None
            
            X_features, feature_dates = prepare_features_for_prediction(
                df=df,
                headlines_csv=headlines_csv,
                use_headlines=True
            )
        except Exception as e:
            print(f"Error in feature preparation: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature preparation error: {str(e)}"
            )
        
        # 3. Generate predictions for all horizons
        print("Generating predictions...")
        predictions_all = predictor.predict_all(X_features)
        
        # 4. Get last 30 days
        last_30_days = min(30, len(X_features))
        
        # Get dates and prices for last 30 days
        recent_dates = feature_dates[-last_30_days:]
        
        # Get corresponding prices from original dataframe
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df_prices = df.rename(columns={price_col: "PX"})
        recent_prices = df_prices.loc[recent_dates, "PX"].values
        
        # Align predictions (take last 30 from predictions)
        pred_start_idx = max(0, len(X_features) - last_30_days)
        
        predictions_list = []
        for i, date in enumerate(recent_dates):
            pred_idx = pred_start_idx + i
            
            predictions_list.append({
                "date": date.strftime('%Y-%m-%d'),
                "h1_prediction": float(predictions_all['h1'][pred_idx]),
                "h1_signal": "BUY" if predictions_all['h1'][pred_idx] > 0 else "SELL",
                "h5_prediction": float(predictions_all['h5'][pred_idx]),
                "h5_signal": "BUY" if predictions_all['h5'][pred_idx] > 0 else "SELL",
                "h20_prediction": float(predictions_all['h20'][pred_idx]),
                "h20_signal": "BUY" if predictions_all['h20'][pred_idx] > 0 else "SELL",
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


@app.get("/news")
async def get_news():
    """Get recent news for SPY"""
    news = fetch_spy_news(days_back=2)
    return news


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

