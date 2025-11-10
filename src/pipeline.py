import os
import pandas as pd
from typing import Dict, List, Tuple

from config.settings import (
    SYMBOLS, START, END, INTERVAL,
    RAW_DIR, PROC_DIR, SPLIT_DIR, EXP_DIR,
    TRAIN_DAYS, TEST_DAYS, STEP_DAYS,
    ANCHOR_TICKER, USE_INTERSECTION,
    FEATURE_COLS, TARGET_COLS
)

from .utils.io import ensure_dir, save_json, read_json
from .data.download import download_many
from .data.features import process_and_save
from .splits.walkforward import build_splits, materialize_fold
from .models.registry import get_model
from .eval.metrics import evaluate_predictions, sign_backtest

# --------- DATA ---------
def run_download() -> List[Dict]:
    ensure_dir(RAW_DIR)
    return download_many(SYMBOLS, START, END, interval=INTERVAL, save_dir=RAW_DIR)

def _find_latest_raw(prefix: str) -> str:
    files = [f for f in os.listdir(RAW_DIR) if f.startswith(prefix + "_") and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No raw CSV found for '{prefix}' in {RAW_DIR}")
    files.sort()
    return os.path.join(RAW_DIR, files[-1])

def run_process() -> Dict[str, str]:
    ensure_dir(PROC_DIR)
    # Check if headlines CSV exists in project root
    from config.settings import HEADLINES_CSV
    headlines_path = HEADLINES_CSV if os.path.exists(HEADLINES_CSV) else None
    
    # Always generate GSPC & ANCHOR_TICKER features so we can intersect if needed
    gspc_raw = _find_latest_raw("GSPC")
    anchor_raw = _find_latest_raw(ANCHOR_TICKER)
    gspc_out = process_and_save(gspc_raw, "GSPC", PROC_DIR, headlines_csv=headlines_path)
    anchor_out = process_and_save(anchor_raw, ANCHOR_TICKER, PROC_DIR, headlines_csv=headlines_path)
    return {"GSPC": gspc_out, ANCHOR_TICKER: anchor_out}

def _read_proc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date").sort_index()
    else:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()
    return df[~df.index.isna()]

# --------- SPLITS ---------
def run_build_splits(proc_paths: Dict[str, str]) -> List[Dict]:
    ensure_dir(SPLIT_DIR)
    gspc = _read_proc(proc_paths["GSPC"])
    anchor = _read_proc(proc_paths[ANCHOR_TICKER])

    if USE_INTERSECTION:
        idx = gspc.index.intersection(anchor.index).sort_values()
        index_label = "intersection(GSPC,{})".format(ANCHOR_TICKER)
    else:
        idx = anchor.index
        index_label = f"{ANCHOR_TICKER}_only"

    folds = build_splits(idx, TRAIN_DAYS, TEST_DAYS, STEP_DAYS)
    save_json({"index": index_label,
               "train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "step_days": STEP_DAYS,
               "folds": folds}, os.path.join(SPLIT_DIR, "splits.json"))
    return folds

def run_materialize_folds(proc_paths: Dict[str, str], folds: List[Dict]) -> None:
    anchor = _read_proc(proc_paths[ANCHOR_TICKER])
    if USE_INTERSECTION:
        gspc = _read_proc(proc_paths["GSPC"])
        idx = gspc.index.intersection(anchor.index).sort_values()
        anchor = anchor.loc[idx]
    for k, fold in enumerate(folds):
        fold_dir = os.path.join(SPLIT_DIR, f"fold_{k}")
        materialize_fold(anchor, FEATURE_COLS, TARGET_COLS, fold, fold_dir)

# --------- TRAIN / EVAL (single fold) ---------
def run_baseline(model_name: str, fold_id: int, horizon: str = "target_h1") -> Dict:
    fold_dir = os.path.join(SPLIT_DIR, f"fold_{fold_id}")
    train = pd.read_csv(os.path.join(fold_dir, "train.csv"), index_col=0, parse_dates=True)
    test  = pd.read_csv(os.path.join(fold_dir, "test.csv"),  index_col=0, parse_dates=True)

    X_tr = train[[c for c in train.columns if c.startswith("z_")]].values
    X_te = test[[c for c in test.columns if c.startswith("z_")]].values
    y_tr = train[horizon].values
    y_te = test[horizon].values

    model = get_model(model_name)  # e.g., "ridge" or "esn" or "lstm"
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)

    # save preds
    exp_dir = os.path.join(EXP_DIR, f"{model_name}", f"fold_{fold_id}")
    ensure_dir(exp_dir)
    out = pd.DataFrame({"y_true": y_te, "y_pred": y_hat}, index=test.index)
    out.to_csv(os.path.join(exp_dir, f"preds_{horizon}.csv"))

    # metrics
    m = evaluate_predictions(y_te, y_hat).to_dict()
    b = sign_backtest(y_te, y_hat, cost_per_trade=0.0001)
    result = dict(model=model_name, fold=fold_id, horizon=horizon, **m, backtest=b)
    save_json(result, os.path.join(exp_dir, f"metrics_{horizon}.json"))
    return result
