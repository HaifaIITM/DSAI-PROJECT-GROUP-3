# src/eval/aggregate.py
import os, json
import pandas as pd
from config import settings

def collect_all_metrics(model: str, exp_id: str):
    """Walk experiment folders and collect all metrics JSON into a single DataFrame."""
    base = os.path.join(settings.EXP_DIR, model, exp_id)
    rows = []
    if not os.path.exists(base):
        raise FileNotFoundError(base)
    for d in os.listdir(base):
        if not d.startswith("fold_"):
            continue
        fold_dir = os.path.join(base, d)
        for f in os.listdir(fold_dir):
            if f.startswith("metrics_") and f.endswith(".json"):
                with open(os.path.join(fold_dir, f), "r") as fh:
                    rows.append(json.load(fh))
    if not rows:
        return pd.DataFrame()
    # flatten backtest
    flat = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != "backtest"}
        base.update(r.get("backtest", {}))
        flat.append(base)
    df = pd.DataFrame(flat)
    df = df.sort_values(["horizon","fold"]).reset_index(drop=True)
    return df

def summarize_over_folds(df: pd.DataFrame):
    """Aggregate metrics over folds for each horizon."""
    if df.empty:
        return df
    gb = df.groupby("horizon").agg({
        "rmse":["mean","std"],
        "mae":["mean","std"],
        "r2":["mean","std"],
        "dir_acc":["mean","std"],
        "avg_daily_pnl":["mean","std"],
        "vol":["mean","std"],
        "sharpe":["mean","std"],
        "hit_ratio":["mean","std"],
        "turnover":["mean","std"],
    })
    return gb
