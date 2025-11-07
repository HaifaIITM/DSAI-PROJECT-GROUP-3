import os
import numpy as np
import pandas as pd
from .loader import load_yahoo_csv

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

    # NLP risk index (if provided)
    if risk_df is not None and not risk_df.empty:
        # Align by date index, forward-fill missing dates
        out = out.join(risk_df[['Risk_z']], how='left')
        out['risk_index'] = out['Risk_z'].ffill().fillna(0)
        out = out.drop(columns=['Risk_z'], errors='ignore')
    else:
        out['risk_index'] = 0.0

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
