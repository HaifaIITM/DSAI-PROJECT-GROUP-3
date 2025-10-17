import os
from typing import List, Dict, Tuple
import yfinance as yf
import pandas as pd
from ..utils.io import ensure_dir

CANON = ["Open","High","Low","Close","Adj Close","Volume"]

def download_symbol(symbol: str, start: str, end: str, interval="1d", auto_adjust=False, save_dir="data/raw") -> Tuple[pd.DataFrame, Dict]:
    ensure_dir(save_dir)
    df = yf.download(
        symbol, start=start, end=end, interval=interval,
        auto_adjust=auto_adjust, progress=False, threads=False
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}.")
    cols = [c for c in CANON if c in df.columns]
    df = df[cols].copy()
    df.index.name = "Date"
    path = os.path.join(save_dir, f"{symbol.replace('^','')}_{start}_to_{end}_{interval}.csv")
    df.to_csv(path)
    meta = dict(symbol=symbol, rows=len(df),
                start=str(df.index.min().date()),
                end=str(df.index.max().date()),
                columns=cols, path=path)
    return df, meta

def download_many(symbols: List[str], start: str, end: str, interval="1d", save_dir="data/raw") -> List[Dict]:
    results = []
    for s in symbols:
        _, meta = download_symbol(s, start, end, interval=interval, auto_adjust=False, save_dir=save_dir)
        results.append(meta)
    return results
