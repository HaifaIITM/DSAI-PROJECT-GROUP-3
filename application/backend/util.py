"""
Utility functions for fetching and processing financial data and news.
"""
import os
import sys
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

try:
    from src.data.embeddings import compute_headline_features
    from config.settings import (
        SMALL_MODEL, LARGE_MODEL, SMALL_PCA_DIM, LARGE_PCA_DIM,
        AGG_METHOD, RANDOM_SEED, FEATURE_COLS_FULL
    )
except ImportError as e:
    print(f"Warning: Could not import embeddings module: {e}")
    compute_headline_features = None
    # Default settings if config not available
    SMALL_MODEL = "all-MiniLM-L6-v2"
    LARGE_MODEL = "sentence-transformers/all-mpnet-base-v2"
    SMALL_PCA_DIM = 12
    LARGE_PCA_DIM = 14
    AGG_METHOD = "mean"
    RANDOM_SEED = 42
    FEATURE_COLS_FULL = [
        "ret_1","ret_2","ret_5","vol_20","ma_10","ma_20","ma_gap","rsi_14","vol_z","dow",
        "pca_1","pca_2","pca_3","pca_4","pca_5","pca_6","pca_7","pca_8","pca_9","pca_10","pca_11","pca_12","has_news",
        "pca_1_large","pca_2_large","pca_3_large","pca_4_large","pca_5_large","pca_6_large",
        "pca_7_large","pca_8_large","pca_9_large","pca_10_large","pca_11_large","pca_12_large",
        "pca_13_large","pca_14_large","has_news_large"
    ]


def get_daily_headlines(
    ticker_symbol: str = 'SPY',
    days_back: int = 20,
    return_format: str = 'dict'
) -> Dict[str, List[str]]:
    """
    Fetch news headlines for a ticker symbol, grouped by date.
    
    Args:
        ticker_symbol: Stock ticker symbol (default: 'SPY')
        days_back: Number of days to look back (default: 20)
        return_format: Return format - 'dict' or 'list' (default: 'dict')
    
    Returns:
        If return_format='dict': Dictionary with dates as keys and lists of headlines as values
        If return_format='list': List of dictionaries with 'date' and 'headline' keys
    
    Example:
        >>> headlines = get_daily_headlines('SPY', days_back=7)
        >>> print(headlines['2025-11-10'])
        ['Market Minute 11-10-25...', 'ETF Futures Higher...']
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        
        if not news:
            return {} if return_format == 'dict' else []
        
        # Calculate cutoff date (timezone-aware UTC)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        daily_headlines = {}
        
        for item in news:
            # Extract content from nested structure
            content = item.get('content', {})
            pub_date_str = content.get('pubDate', '')
            
            if not pub_date_str:
                continue
            
            try:
                # Parse ISO format date
                publish_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                
                # Ensure timezone-aware
                if publish_date.tzinfo is None:
                    publish_date = publish_date.replace(tzinfo=timezone.utc)
                
                # Filter news items within the date range
                if publish_date >= cutoff_date:
                    date_str = publish_date.strftime('%Y-%m-%d')
                    title = content.get('title', 'No title')
                    
                    if date_str not in daily_headlines:
                        daily_headlines[date_str] = []
                    daily_headlines[date_str].append(title)
            
            except (ValueError, KeyError) as e:
                # Skip items with invalid dates
                continue
        
        # Convert to list format if requested
        if return_format == 'list':
            result = []
            for date in sorted(daily_headlines.keys()):
                for headline in daily_headlines[date]:
                    result.append({
                        'date': date,
                        'headline': headline
                    })
            return result
        
        return daily_headlines
    
    except Exception as e:
        print(f"Error fetching news for {ticker_symbol}: {e}")
        return {} if return_format == 'dict' else []


def print_daily_headlines(ticker_symbol: str = 'SPY', days_back: int = 20) -> None:
    """
    Fetch and print news headlines grouped by date.
    
    Args:
        ticker_symbol: Stock ticker symbol (default: 'SPY')
        days_back: Number of days to look back (default: 20)
    
    Example:
        >>> print_daily_headlines('SPY', days_back=7)
        --- Headlines for 2025-11-10 ---
        - Market Minute 11-10-25...
        - ETF Futures Higher...
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    print(f"Retrieving news headlines for {ticker_symbol} from the last {days_back} days "
          f"(since {cutoff_date.strftime('%Y-%m-%d')}).\n")
    
    daily_headlines = get_daily_headlines(ticker_symbol, days_back, return_format='dict')
    
    if not daily_headlines:
        print(f"No news found for {ticker_symbol} or could not retrieve news.")
        return
    
    # Sort dates and print headlines
    for date in sorted(daily_headlines.keys()):
        print(f"\n--- Headlines for {date} ---")
        for headline in daily_headlines[date]:
            print(f"- {headline}")


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
    
    Raises:
        ValueError: If data fetch fails or returns empty DataFrame
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    ticker = yf.Ticker("SPY")
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    
    if df.empty:
        raise ValueError("Failed to fetch SPY data from yfinance")
    
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
    Matches training normalization approach (StandardScaler equivalent).
    """
    normalized = df[feature_cols].copy()
    
    # Rolling z-score normalization (252-day window, matching training window)
    for col in feature_cols:
        rolling_mean = normalized[col].rolling(252, min_periods=60).mean()
        rolling_std = normalized[col].rolling(252, min_periods=60).std()
        normalized[col] = (normalized[col] - rolling_mean) / (rolling_std + 1e-8)
    
    # Fill any remaining NaNs with 0
    normalized = normalized.fillna(0)
    
    return normalized.values


def prepare_headlines_csv_from_news(
    news_items: List[Dict[str, Any]], 
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert news items from yfinance to DataFrame format expected by compute_headline_features.
    
    Args:
        news_items: List of news dicts with 'date', 'title', etc.
        output_path: Optional path to save CSV (for debugging)
    
    Returns:
        DataFrame with 'published_utc' and 'title' columns
    """
    if not news_items:
        return pd.DataFrame(columns=['published_utc', 'title'])
    
    records = []
    for item in news_items:
        # Parse date string (format: 'YYYY-MM-DD HH:MM')
        date_str = item.get('date', '')
        try:
            # Convert to datetime and then to UTC timestamp string
            dt = pd.to_datetime(date_str)
            records.append({
                'published_utc': dt.isoformat(),
                'title': item.get('title', 'No title')
            })
        except Exception as e:
            print(f"Warning: Could not parse date '{date_str}': {e}")
            continue
    
    df = pd.DataFrame(records)
    
    if output_path and len(df) > 0:
        df.to_csv(output_path, index=False)
    
    return df


def prepare_features_for_prediction(
    df: pd.DataFrame,
    headlines_csv: Optional[str] = None,
    use_headlines: bool = True
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Complete feature preparation pipeline matching training process.
    
    Steps:
    1. Compute technical features (10 features)
    2. Compute headline embeddings (28 features) if available
    3. Merge technical + headline features
    4. Normalize using rolling z-score (252-day window)
    5. Return features in exact order expected by model (FEATURE_COLS_FULL)
    
    Args:
        df: DataFrame with OHLCV data (must have enough history for indicators)
        headlines_csv: Optional path to headlines CSV file
        use_headlines: If True, attempt to compute headline features
    
    Returns:
        Tuple of (normalized feature matrix, corresponding dates)
        - Feature matrix: (n_samples, 38 features) in FEATURE_COLS_FULL order
        - Dates: DatetimeIndex corresponding to each row in feature matrix
    """
    # Step 1: Compute technical features
    df_features = compute_technical_features(df)
    
    # Step 2: Compute headline embeddings if available
    headline_feats = None
    if use_headlines and compute_headline_features is not None:
        try:
            # Try to use provided headlines CSV
            if headlines_csv and os.path.exists(headlines_csv):
                headline_feats = compute_headline_features(
                    target_dates=df_features.index,
                    headlines_csv=headlines_csv,
                    small_model=SMALL_MODEL,
                    large_model=LARGE_MODEL,
                    small_pca_dim=SMALL_PCA_DIM,
                    large_pca_dim=LARGE_PCA_DIM,
                    random_state=RANDOM_SEED,
                    agg_method=AGG_METHOD
                )
            else:
                # Try to fetch recent news and create temporary CSV
                print("[INFO] No headlines CSV provided, fetching recent news...")
                recent_news = fetch_spy_news(days_back=90)  # Get more days for better coverage
                
                if recent_news:
                    # Create temporary headlines DataFrame
                    headlines_df = prepare_headlines_csv_from_news(recent_news)
                    
                    if len(headlines_df) > 0:
                        # Save to temp file for compute_headline_features
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                            temp_path = f.name
                            headlines_df.to_csv(temp_path, index=False)
                        
                        try:
                            headline_feats = compute_headline_features(
                                target_dates=df_features.index,
                                headlines_csv=temp_path,
                                small_model=SMALL_MODEL,
                                large_model=LARGE_MODEL,
                                small_pca_dim=SMALL_PCA_DIM,
                                large_pca_dim=LARGE_PCA_DIM,
                                random_state=RANDOM_SEED,
                                agg_method=AGG_METHOD
                            )
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
        
        except Exception as e:
            print(f"[WARN] Could not compute headline features: {e}")
            print("Continuing with technical features only")
    
    # Step 3: Merge technical + headline features
    if headline_feats is not None:
        # Merge on date index
        df_features = df_features.join(headline_feats, how="left")
        # Fill NaNs with 0 (for dates without news)
        headline_cols = headline_feats.columns
        df_features[headline_cols] = df_features[headline_cols].fillna(0.0)
    else:
        # Create zero-filled headline features
        print("[WARN] Using zero-filled headline features (no news data available)")
        headline_cols = [
            "pca_1", "pca_2", "pca_3", "pca_4", "pca_5", "pca_6",
            "pca_7", "pca_8", "pca_9", "pca_10", "pca_11", "pca_12",
            "has_news",
            "pca_1_large", "pca_2_large", "pca_3_large", "pca_4_large",
            "pca_5_large", "pca_6_large", "pca_7_large", "pca_8_large",
            "pca_9_large", "pca_10_large", "pca_11_large", "pca_12_large",
            "pca_13_large", "pca_14_large",
            "has_news_large"
        ]
        for col in headline_cols:
            df_features[col] = 0.0
    
    # Step 4: Ensure all required features exist
    missing_cols = [col for col in FEATURE_COLS_FULL if col not in df_features.columns]
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")
    
    # Step 5: Normalize features (rolling z-score, 252-day window)
    X_normalized = normalize_features(df_features, FEATURE_COLS_FULL)
    
    # Step 6: Return features and corresponding dates
    # (normalize_features already returns in FEATURE_COLS_FULL order)
    return X_normalized, df_features.index