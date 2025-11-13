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
from fastapi import FastAPI, HTTPException, Query
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
    prepare_features_for_prediction,
    SMALL_MODEL,
    LARGE_MODEL,
    FEATURE_COLS_FULL
)
from storage import DataStorage

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

# Global storage instance
storage = DataStorage()


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
    Get fresh predictions for SPY for last 3 days (today, yesterday, day before yesterday).
    
    Returns:
        - predictions: Array of predictions for last 3 days (all horizons: h1, h5, h20)
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
        
        # 2. Fetch and store recent news headlines
        print("Fetching and storing news headlines...")
        recent_news = fetch_spy_news(days_back=3, storage=storage)
        saved_count = storage.save_headlines_batch(recent_news)
        print(f"Saved {saved_count} new headlines to storage")
        
        # Use stored headlines CSV for feature preparation
        headlines_csv = storage.get_headlines_csv_path()
        if not os.path.exists(headlines_csv):
            # Fallback to project root headlines CSV
            headlines_csv = os.path.join(project_root, "data", "spy_news.csv")
            if not os.path.exists(headlines_csv):
                headlines_csv = None
        
        # 3. Prepare features using complete pipeline (matches training process)
        print("Preparing features (technical + headlines)...")
        try:
            X_features, feature_dates, intermediates = prepare_features_for_prediction(
                df=df,
                headlines_csv=headlines_csv,
                use_headlines=True,
                return_intermediates=True
            )
        except Exception as e:
            print(f"Error in feature preparation: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature preparation error: {str(e)}"
            )
        
        # 4. Generate predictions for all horizons
        print("Generating predictions...")
        predictions_all = predictor.predict_all(X_features)
        
        # 5. Get last 3 days (today, yesterday, day before yesterday)
        # Get the most recent trading days
        last_3_days = min(3, len(X_features))
        recent_dates = feature_dates[-last_3_days:]
        
        # Get corresponding prices from original dataframe
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df_prices = df.rename(columns={price_col: "PX"})
        recent_prices = df_prices.loc[recent_dates, "PX"].values
        
        # Align predictions (take last 3 from predictions)
        pred_start_idx = max(0, len(X_features) - last_3_days)
        
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
        
        # Sort by date (oldest first)
        predictions_list.sort(key=lambda x: x['date'])
        
        # 6. Store predictions, embeddings, and features
        print("Storing predictions, embeddings, and features...")
        
        # Store predictions (only last 3 days)
        prediction_metadata = {
            "symbol": "SPY",
            "model_info": predictor.get_model_info(),
            "n_samples": len(X_features),
            "prediction_days": last_3_days,
            "date_range": {
                "start": str(recent_dates[0]),
                "end": str(recent_dates[-1])
            },
            "prediction_type": "fresh_3_days"
        }
        storage.save_predictions_batch(predictions_list, metadata=prediction_metadata)
        
        # Store embeddings (if available) - only for last 3 days
        if intermediates and intermediates.get("headline_features") is not None:
            headline_embeddings = intermediates["headline_features"]
            # Only store embeddings for last 3 days
            headline_embeddings_last3 = headline_embeddings[-last_3_days:]
            feature_dates_last3 = feature_dates[-last_3_days:]
            
            try:
                storage.save_embeddings(
                    embeddings=headline_embeddings_last3,
                    dates=feature_dates_last3,
                    metadata={
                        "embedding_type": "headline_pca",
                        "feature_names": intermediates.get("headline_feature_names"),
                        "small_model": SMALL_MODEL,
                        "large_model": LARGE_MODEL,
                        "prediction_type": "fresh_3_days"
                    }
                )
                print("✓ Embeddings stored successfully (last 3 days)")
            except Exception as e:
                print(f"⚠ Warning: Could not store embeddings: {e}")
        else:
            print("⚠ Warning: No headline embeddings to store (using zero-filled embeddings)")
        
        # Store final 38 features (only for last 3 days)
        feature_names = FEATURE_COLS_FULL if intermediates is None else intermediates.get("feature_names", FEATURE_COLS_FULL)
        # Only store features for the last 3 days
        X_features_last3 = X_features[-last_3_days:]
        feature_dates_last3 = feature_dates[-last_3_days:]
        
        storage.save_features(
            features=X_features_last3,
            dates=feature_dates_last3,
            feature_names=feature_names,
            metadata={
                "normalization": "rolling_zscore_252d",
                "n_features": X_features_last3.shape[1],
                "n_samples": X_features_last3.shape[0],
                "prediction_type": "fresh_3_days"
            }
        )
        
        print("✓ All data stored successfully")
        
        # 7. Get recent news for response (already fetched and stored above)
        news = recent_news
        
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
async def get_news(days_back: int = Query(7, ge=1, description="Number of days to look back (must be >= 1)")):
    """
    Get stored news headlines for SPY
    
    Args:
        days_back: Number of days to look back (must be >= 1, default: 7)
    """
    df_news = storage.get_headlines(days_back=days_back)
    
    if len(df_news) == 0:
        return []
    
    # Convert to list of dicts
    news_list = []
    for _, row in df_news.iterrows():
        news_list.append({
            "date": row['date'].strftime('%Y-%m-%d %H:%M') if hasattr(row['date'], 'strftime') else str(row['date']),
            "title": row['title'],
            "publisher": row.get('publisher', 'Unknown'),
            "link": row.get('link', '#')
        })
    
    return news_list


@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "models": predictor.get_model_info(),
        "status": "ready"
    }


@app.get("/storage/info")
async def get_storage_info():
    """Get information about stored data"""
    return storage.get_storage_info()


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

