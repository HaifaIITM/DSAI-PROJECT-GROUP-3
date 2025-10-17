from datetime import date, timedelta

# ---- Ticker Symbols ----
SYMBOLS = [
    "^GSPC", "SPY",                # Index & ETF
    "BTC-USD", "ETH-USD",          # crypto
    "^NSEI", "^NSEBANK",           # India indices
    "RELIANCE.NS", "TCS.NS",       # India stocks
    "EURUSD=X", "USDINR=X",        # FX
    "GC=F", "CL=F",                # commodities
    "^VIX"                         # volatility
]

TODAY = date.today()
START = (TODAY - timedelta(days=365*20 + 5*6)).isoformat()  # ~20y w/ leap buffer
END = TODAY.isoformat()
INTERVAL = "1d"

# data dirs
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROC_DIR = f"{DATA_DIR}/processed"
SPLIT_DIR = f"{DATA_DIR}/splits"
EXP_DIR = f"{DATA_DIR}/experiments"

# walk-forward
TRAIN_DAYS = 252 * 10
TEST_DAYS  = 252
STEP_DAYS  = 252

# splitting policy
ANCHOR_TICKER = "SPY"   # primary series for modeling
USE_INTERSECTION = True # True: intersect ANCHOR_TICKER with '^GSPC' to align; False: use ANCHOR_TICKER alone

# features/targets
FEATURE_COLS = [
    "ret_1","ret_2","ret_5",
    "vol_20","ma_10","ma_20","ma_gap",
    "rsi_14","vol_z","dow"
]
TARGET_COLS  = ["target_h1", "target_h5", "target_h20"]
