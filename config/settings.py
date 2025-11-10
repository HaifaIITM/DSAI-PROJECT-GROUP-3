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

# Headline embeddings
HEADLINES_CSV = "spy_news.csv"
SMALL_MODEL = "all-MiniLM-L6-v2"
LARGE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SMALL_PCA_DIM = 12
LARGE_PCA_DIM = 14
AGG_METHOD = "mean"  # daily aggregation
RANDOM_SEED = 42

# features/targets
# Full feature set (38 features: 10 technical + 28 headline embeddings)
FEATURE_COLS_FULL = [
    "ret_1","ret_2","ret_5",
    "vol_20","ma_10","ma_20","ma_gap",
    "rsi_14","vol_z","dow",
    # Small model PCA (12 dims)
    "pca_1", "pca_2", "pca_3", "pca_4", "pca_5", "pca_6",
    "pca_7", "pca_8", "pca_9", "pca_10", "pca_11", "pca_12",
    "has_news",
    # Large model PCA (14 dims)
    "pca_1_large", "pca_2_large", "pca_3_large", "pca_4_large",
    "pca_5_large", "pca_6_large", "pca_7_large", "pca_8_large",
    "pca_9_large", "pca_10_large", "pca_11_large", "pca_12_large",
    "pca_13_large", "pca_14_large",
    "has_news_large"
]

# Technical features only (10 features)
FEATURE_COLS_TECH = [
    "ret_1","ret_2","ret_5",
    "vol_20","ma_10","ma_20","ma_gap",
    "rsi_14","vol_z","dow"
]

# Active feature set (switch between FEATURE_COLS_FULL and FEATURE_COLS_TECH)
FEATURE_COLS = FEATURE_COLS_FULL  # <--- Using 38 features (10 technical + 28 headline embeddings)

TARGET_COLS  = ["target_h1", "target_h5", "target_h20"]
