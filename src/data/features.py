import os
import numpy as np
import pandas as pd
from .loader import load_yahoo_csv

def _compute_market_sentiment_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Compute market-based sentiment/momentum proxy from price action.
    Positive = bullish sentiment, Negative = bearish/risk-off
    """
    # Momentum signal (positive momentum = positive signal)
    momentum_5d = df["ret_1"].rolling(5).sum()
    momentum_20d = df["ret_1"].rolling(20).sum()
    mom_z = (momentum_5d - momentum_5d.rolling(60).mean()) / momentum_5d.rolling(60).std()
    
    # Trend strength (price vs MA)
    trend = (df["PX"] - df["ma_20"]) / df["ma_20"]
    trend_z = (trend - trend.rolling(60).mean()) / trend.rolling(60).std()
    
    # Vol regime (low vol = positive, high vol = negative for risk-adjusted returns)
    vol_current = df["ret_1"].rolling(10).std()
    vol_baseline = df["ret_1"].rolling(60).std()
    vol_signal = -(vol_current - vol_baseline) / vol_baseline  # Invert: low vol is good
    vol_z = (vol_signal - vol_signal.rolling(60).mean()) / vol_signal.rolling(60).std()
    
    # RSI momentum (away from extremes)
    rsi_centered = (df.get("rsi_14", 50) - 50) / 50  # Range [-1, 1]
    rsi_z = (rsi_centered - rsi_centered.rolling(60).mean()) / rsi_centered.rolling(60).std()
    
    # Composite index (momentum-focused)
    components = pd.DataFrame({
        'mom': mom_z.fillna(0),
        'trend': trend_z.fillna(0),
        'vol': vol_z.fillna(0),
        'rsi': rsi_z.fillna(0)
    })
    
    # Weighted composite (emphasize momentum and trend)
    sentiment_proxy = (
        0.4 * components['mom'] + 
        0.3 * components['trend'] + 
        0.2 * components['vol'] + 
        0.1 * components['rsi']
    )
    
    return sentiment_proxy.fillna(0)

def compute_features(df: pd.DataFrame, symbol: str, rsi_window: int = 14, risk_df: pd.DataFrame = None) -> pd.DataFrame:
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

    # NLP risk index (market-based proxy if no real headlines)
    if risk_df is not None and not risk_df.empty:
        # Use real NLP data if available
        out = out.join(risk_df[['Risk_z']], how='left')
        out['risk_index'] = out['Risk_z'].ffill().fillna(0)
        out = out.drop(columns=['Risk_z'], errors='ignore')
    else:
        # Market-based sentiment proxy: combines vol, gaps, and momentum
        out['risk_index'] = _compute_market_sentiment_proxy(out)

    # targets (forward)
    out["target_h1"]  = out["ret_1"].shift(-1)
    out["target_h5"]  = out["ret_1"].rolling(5).sum().shift(-5)
    out["target_h20"] = out["ret_1"].rolling(20).sum().shift(-20)

    out = out.dropna().copy()
    out.insert(0, "Symbol", symbol)
    return out

def process_and_save(raw_csv_path: str, symbol: str, out_dir: str, risk_df: pd.DataFrame = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    raw = load_yahoo_csv(raw_csv_path)
    feats = compute_features(raw, symbol, risk_df=risk_df)
    out_path = os.path.join(out_dir, f"{symbol}_features.csv")
    feats.to_csv(out_path, index=True)
    return out_path
