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


def fetch_spy_news(days_back: int = 3, storage=None) -> List[Dict[str, Any]]:
    """
    Fetch recent SPY news from yfinance and optionally save to storage.
    
    Args:
        days_back: Number of days to look back
        storage: Optional DataStorage instance to save headlines
    
    Returns:
        List of news items for last N days
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
                        news_item = {
                            'date': dt.strftime('%Y-%m-%d %H:%M'),
                            'title': content.get('title', 'No title'),
                            'publisher': content.get('provider', {}).get('displayName', 'Unknown'),
                            'link': content.get('canonicalUrl', {}).get('url', '#')
                        }
                        recent_news.append(news_item)
                        
                        # Save to storage if provided
                        if storage:
                            storage.save_headline(news_item)
                
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
    use_headlines: bool = True,
    return_intermediates: bool = False
) -> Tuple[np.ndarray, pd.DatetimeIndex, Optional[Dict[str, Any]]]:
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
        return_intermediates: If True, return intermediate data (embeddings, etc.)
    
    Returns:
        Tuple of (normalized feature matrix, corresponding dates, intermediates)
        - Feature matrix: (n_samples, 38 features) in FEATURE_COLS_FULL order
        - Dates: DatetimeIndex corresponding to each row in feature matrix
        - Intermediates: Dict with embeddings and other intermediate data (if return_intermediates=True)
    """
    # Step 1: Compute technical features
    df_features = compute_technical_features(df)
    
    # Step 2: Compute headline embeddings if available
    headline_feats = None
    raw_embeddings = None
    if use_headlines and compute_headline_features is not None:
        try:
            # Try to use provided headlines CSV
            if headlines_csv and os.path.exists(headlines_csv):
                # Check if CSV needs conversion (storage format vs embeddings format)
                # Storage format: date,title,publisher,link
                # Embeddings format: published_utc,title
                try:
                    # Try to read and check format
                    test_df = pd.read_csv(headlines_csv, nrows=1)
                    if 'published_utc' not in test_df.columns and 'date' in test_df.columns:
                        # Need to convert storage format to embeddings format
                        print("[INFO] Converting headlines CSV format for embeddings...")
                        import tempfile
                        storage_df = pd.read_csv(headlines_csv, parse_dates=['date'])
                        # Ensure properly formatted datetime
                        storage_df['date'] = pd.to_datetime(storage_df['date'], errors='coerce')
                        
                        # Convert to embeddings format
                        # embeddings.py does: pd.to_datetime(df["published_utc"]).dt.tz_convert(None)
                        # tz_convert requires timezone-aware timestamps, so we need to add timezone
                        # Add UTC timezone to make it timezone-aware (if not already)
                        if storage_df['date'].dt.tz is None:
                            # Localize naive datetime to UTC
                            storage_df['date'] = storage_df['date'].dt.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
                        elif storage_df['date'].dt.tz is not None:
                            # Convert existing timezone to UTC
                            storage_df['date'] = storage_df['date'].dt.tz_convert('UTC')
                        
                        # Format as ISO with 'Z' suffix for UTC (pandas will parse as UTC)
                        embeddings_df = pd.DataFrame({
                            'published_utc': storage_df['date'].dt.strftime('%Y-%m-%dT%H:%M:%S') + 'Z',
                            'title': storage_df['title']
                        })
                        # Ensure published_utc is a string
                        embeddings_df['published_utc'] = embeddings_df['published_utc'].astype(str)
                        # Drop rows with missing titles
                        embeddings_df = embeddings_df.dropna(subset=['title'])
                        
                        num_headlines = len(embeddings_df)
                        print(f"[INFO] Converted {num_headlines} headlines for embedding computation")
                        
                        # Adjust PCA dimensions based on available samples
                        # PCA requires n_components <= min(n_samples, n_features)
                        # Small model embedding dim: 384, Large model: 768
                        # But we need to account for number of headlines
                        # Use min of requested dims and available samples
                        adjusted_small_pca = min(SMALL_PCA_DIM, num_headlines)
                        adjusted_large_pca = min(LARGE_PCA_DIM, num_headlines)
                        
                        if adjusted_small_pca < SMALL_PCA_DIM:
                            print(f"[INFO] Adjusting small PCA from {SMALL_PCA_DIM} to {adjusted_small_pca} (only {num_headlines} headlines)")
                        if adjusted_large_pca < LARGE_PCA_DIM:
                            print(f"[INFO] Adjusting large PCA from {LARGE_PCA_DIM} to {adjusted_large_pca} (only {num_headlines} headlines)")
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                            temp_path = f.name
                            embeddings_df.to_csv(temp_path, index=False)
                        
                        try:
                            headline_feats = compute_headline_features(
                                target_dates=df_features.index,
                                headlines_csv=temp_path,
                                small_model=SMALL_MODEL,
                                large_model=LARGE_MODEL,
                                small_pca_dim=adjusted_small_pca,
                                large_pca_dim=adjusted_large_pca,
                                random_state=RANDOM_SEED,
                                agg_method=AGG_METHOD
                            )
                            
                            # If PCA dimensions were reduced, pad with zeros to match expected feature count
                            if headline_feats is not None and (adjusted_small_pca < SMALL_PCA_DIM or adjusted_large_pca < LARGE_PCA_DIM):
                                print(f"[INFO] Padding headline features to match expected dimensions...")
                                # Get expected column names
                                expected_small_cols = [f"pca_{i+1}" for i in range(SMALL_PCA_DIM)] + ["has_news"]
                                expected_large_cols = [f"pca_{i+1}_large" for i in range(LARGE_PCA_DIM)] + ["has_news_large"]
                                
                                # Add missing columns with zeros
                                for col in expected_small_cols:
                                    if col not in headline_feats.columns:
                                        headline_feats[col] = 0.0
                                
                                for col in expected_large_cols:
                                    if col not in headline_feats.columns:
                                        headline_feats[col] = 0.0
                                
                                # Reorder columns to match expected order
                                headline_feats = headline_feats[expected_small_cols + expected_large_cols]
                                print(f"[INFO] Headline features padded to {len(headline_feats.columns)} dimensions")
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                    else:
                        # Already in correct format - check number of headlines for PCA adjustment
                        try:
                            check_df = pd.read_csv(headlines_csv)
                            num_headlines = len(check_df)
                            
                            # Adjust PCA dimensions based on available samples
                            adjusted_small_pca = min(SMALL_PCA_DIM, num_headlines)
                            adjusted_large_pca = min(LARGE_PCA_DIM, num_headlines)
                            
                            if adjusted_small_pca < SMALL_PCA_DIM or adjusted_large_pca < LARGE_PCA_DIM:
                                print(f"[INFO] Adjusting PCA dimensions: small={adjusted_small_pca}, large={adjusted_large_pca} (only {num_headlines} headlines)")
                            
                            headline_feats = compute_headline_features(
                                target_dates=df_features.index,
                                headlines_csv=headlines_csv,
                                small_model=SMALL_MODEL,
                                large_model=LARGE_MODEL,
                                small_pca_dim=adjusted_small_pca,
                                large_pca_dim=adjusted_large_pca,
                                random_state=RANDOM_SEED,
                                agg_method=AGG_METHOD
                            )
                            
                            # If PCA dimensions were reduced, pad with zeros to match expected feature count
                            if headline_feats is not None and (adjusted_small_pca < SMALL_PCA_DIM or adjusted_large_pca < LARGE_PCA_DIM):
                                print(f"[INFO] Padding headline features to match expected dimensions...")
                                expected_small_cols = [f"pca_{i+1}" for i in range(SMALL_PCA_DIM)] + ["has_news"]
                                expected_large_cols = [f"pca_{i+1}_large" for i in range(LARGE_PCA_DIM)] + ["has_news_large"]
                                
                                for col in expected_small_cols:
                                    if col not in headline_feats.columns:
                                        headline_feats[col] = 0.0
                                
                                for col in expected_large_cols:
                                    if col not in headline_feats.columns:
                                        headline_feats[col] = 0.0
                                
                                headline_feats = headline_feats[expected_small_cols + expected_large_cols]
                                print(f"[INFO] Headline features padded to {len(headline_feats.columns)} dimensions")
                        except Exception as e:
                            print(f"[WARN] Error checking headlines count: {e}")
                            # Fallback to original dimensions (might fail, but try anyway)
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
                except Exception as e:
                    print(f"[WARN] Error reading/converting headlines CSV: {e}")
                    import traceback
                    traceback.print_exc()
                    headline_feats = None
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
            import traceback
            traceback.print_exc()
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
    
    # Step 6: Prepare return values
    intermediates = None
    if return_intermediates:
        # Log whether headline features were computed
        if headline_feats is not None:
            print(f"[INFO] Headline features computed: {headline_feats.shape[0]} samples, {headline_feats.shape[1]} features")
        else:
            print("[WARN] No headline features computed - will use zero-filled embeddings")
        
        intermediates = {
            "headline_features": headline_feats.values if headline_feats is not None else None,
            "headline_feature_names": list(headline_feats.columns) if headline_feats is not None else None,
            "raw_embeddings": raw_embeddings,
            "feature_names": FEATURE_COLS_FULL,
            "headline_feats_computed": headline_feats is not None
        }
    
    # Return features and corresponding dates (and intermediates if requested)
    return X_normalized, df_features.index, intermediates