import numpy as np
import pandas as pd

CANON = ["Open","High","Low","Close","Adj Close","Volume"]

def load_yahoo_csv(path: str) -> pd.DataFrame:
    # Try standard
    try:
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date").sort_index()
        else:
            df = pd.read_csv(path, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.sort_index()
        if not any(c in df.columns for c in CANON):
            raise ValueError("No canonical OHLCV columns.")
        df = df[~df.index.isna()]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        cols = [c for c in CANON if c in df.columns]
        return df[cols].dropna(how="all")
    except Exception:
        # Multi-row header variant (Price/Ticker/Date)
        raw = pd.read_csv(path, header=None)
        hdr_rows = np.where(raw.iloc[:,0].astype(str).str.strip() == "Date")[0]
        if len(hdr_rows) == 0:
            raise ValueError(f"Could not find 'Date' header row in {path}")
        hdr = int(hdr_rows[0])
        header = raw.iloc[hdr].tolist()
        body = raw.iloc[hdr+1:].copy()
        body.columns = header
        body["Date"] = pd.to_datetime(body["Date"], errors="coerce")
        body = body.set_index("Date").sort_index()
        for c in body.columns:
            body[c] = pd.to_numeric(body[c], errors="coerce")
        cols = [c for c in CANON if c in body.columns]
        return body[cols].dropna(how="all")
