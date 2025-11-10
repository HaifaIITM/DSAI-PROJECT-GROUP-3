import os
import numpy as np
import pandas as pd
from .loader import load_yahoo_csv
from .embeddings import compute_headline_features

def compute_features(df: pd.DataFrame, symbol: str, rsi_window: int = 14) -> pd.DataFrame:
    out = df.copy()
    price_col = "Adj Close" if "Adj Close" in out.columns else "Close"
    out = out.rename(columns={price_col: "PX"})

    # returns & lags
    out["ret_1"] = np.log(out["PX"]).diff()
    out["ret_2"] = out["ret_1"].shift(1).rolling(2).sum()
    out["ret_5"] = out["ret_1"].shift(1).rolling(5).sum()

    # realized vol (20d, annualized)
    out["vol_20"] = out["ret_1"].rolling(20).std() * np.sqrt(252.0)

    # MAs & gap
    out["ma_10"] = out["PX"].rolling(10).mean()
    out["ma_20"] = out["PX"].rolling(20).mean()
    out["ma_gap"] = out["PX"] / out["ma_20"] - 1.0

    # RSI-14
    delta = out["PX"].diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # volume z-score (60d)
    if "Volume" in out.columns:
        out["vol_z"] = (out["Volume"] - out["Volume"].rolling(60).mean()) / out["Volume"].rolling(60).std()
    else:
        out["vol_z"] = np.nan

    # calendar
    out["dow"] = out.index.dayofweek

    # targets (forward)
    out["target_h1"]  = out["ret_1"].shift(-1)
    out["target_h5"]  = out["ret_1"].rolling(5).sum().shift(-5)
    out["target_h20"] = out["ret_1"].rolling(20).sum().shift(-20)

    out = out.dropna().copy()
    out.insert(0, "Symbol", symbol)
    return out

def process_and_save(raw_csv_path: str, symbol: str, out_dir: str, headlines_csv: str = None) -> str:
    """
    Process raw price data and optionally merge with headline embeddings.
    
    Args:
        raw_csv_path: Path to raw OHLCV CSV
        symbol: Ticker symbol
        out_dir: Output directory
        headlines_csv: Optional path to headlines CSV for embedding features
    """
    os.makedirs(out_dir, exist_ok=True)
    raw = load_yahoo_csv(raw_csv_path)
    feats = compute_features(raw, symbol)
    
    # Optionally add headline embeddings
    if headlines_csv and os.path.exists(headlines_csv):
        try:
            from config.settings import (
                SMALL_MODEL, LARGE_MODEL, SMALL_PCA_DIM, LARGE_PCA_DIM,
                AGG_METHOD, RANDOM_SEED
            )
            
            print(f"\n--- Computing headline embeddings for {symbol} ---")
            headline_feats = compute_headline_features(
                target_dates=feats.index,
                headlines_csv=headlines_csv,
                small_model=SMALL_MODEL,
                large_model=LARGE_MODEL,
                small_pca_dim=SMALL_PCA_DIM,
                large_pca_dim=LARGE_PCA_DIM,
                random_state=RANDOM_SEED,
                agg_method=AGG_METHOD
            )
            
            if headline_feats is not None:
                # Merge on date index
                feats = feats.join(headline_feats, how="left")
                # Fill any NaNs with 0 (in case date ranges don't perfectly overlap)
                headline_cols = headline_feats.columns
                feats[headline_cols] = feats[headline_cols].fillna(0.0)
                print(f"[OK] Merged {len(headline_cols)} headline features")
            else:
                print("[WARN] Headline features not available, using technical features only")
                
        except Exception as e:
            print(f"[WARN] Could not add headline features: {e}")
            print("Continuing with technical features only")
    
    out_path = os.path.join(out_dir, f"{symbol}_features.csv")
    feats.to_csv(out_path, index=True)
    return out_path
