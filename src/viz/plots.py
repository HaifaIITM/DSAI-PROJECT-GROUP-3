# src/viz/plots.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Iterable, Dict, Any

from config import settings
from src.eval.metrics import sign_backtest, basic_metrics


# ---------- Loaders / Aggregation ----------
def _metrics_path(model: str, fold_id: int, horizon: str) -> str:
    return os.path.join(settings.EXP_DIR, model, f"fold_{fold_id}", f"metrics_{horizon}.json")

def _preds_path(model: str, fold_id: int, horizon: str) -> str:
    return os.path.join(settings.EXP_DIR, model, f"fold_{fold_id}", f"preds_{horizon}.csv")

def load_metrics(model: str, fold_id: int, horizon: str) -> Dict[str, Any]:
    """Load metrics JSON for a model/fold/horizon."""
    path = _metrics_path(model, fold_id, horizon)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, "r") as f:
        return json.load(f)

def load_preds(model: str, fold_id: int, horizon: str) -> pd.DataFrame:
    """Load predictions CSV (y_true, y_pred) indexed by Date."""
    path = _preds_path(model, fold_id, horizon)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing preds file: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def collect_metrics(models: Iterable[str], fold_id: int, horizon: str) -> pd.DataFrame:
    """
    Aggregate metrics across models (one fold, one horizon).
    Returns tidy DataFrame with columns:
      ['model','fold','horizon','rmse','mae','r2','dir_acc','avg_daily_pnl','vol','sharpe','hit_ratio','turnover']
    """
    rows = []
    for m in models:
        M = load_metrics(m, fold_id, horizon)
        row = dict(
            model=m, fold=fold_id, horizon=horizon,
            rmse=M.get("rmse"), mae=M.get("mae"), r2=M.get("r2"), dir_acc=M.get("dir_acc"),
        )
        # backtest dict present under key "backtest"
        b = M.get("backtest", {})
        row.update(
            avg_daily_pnl=b.get("avg_daily_pnl"),
            vol=b.get("vol"),
            sharpe=b.get("sharpe"),
            hit_ratio=b.get("hit_ratio"),
            turnover=b.get("turnover"),
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rmse", "mae", "model"]).reset_index(drop=True)


# ---------- Plots ----------
def plot_metric_bars(df: pd.DataFrame, metric: str, title: str | None = None):
    """
    Bar chart for a single metric across models (e.g., rmse / mae / r2 / dir_acc / sharpe).
    Uses matplotlib defaults (no custom colors).
    """
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not in df columns: {list(df.columns)}")
    x = np.arange(len(df))
    plt.figure(figsize=(8,4))
    plt.bar(x, df[metric].values)
    plt.xticks(x, df["model"].tolist(), rotation=0)
    plt.ylabel(metric.upper())
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_cum_pnl(models: Iterable[str], fold_id: int, horizon: str, cost_per_trade: float = 0.0001):
    """
    Plot cumulative PnL curves for multiple models on the same axes
    using the simple sign backtest (recomputed from preds for transparency).
    """
    plt.figure(figsize=(10,5))
    for m in models:
        preds = load_preds(m, fold_id, horizon)
        y_true = preds["y_true"].values
        y_pred = preds["y_pred"].values
        # recompute backtest to get full pnl series
        pos = np.sign(y_pred)
        pos_change = np.abs(np.diff(pos, prepend=0))
        pnl = pos * y_true - cost_per_trade * pos_change
        cum_pnl = pnl.cumsum()
        plt.plot(preds.index, cum_pnl, label=m)
    plt.title(f"Cumulative PnL — fold {fold_id}, {horizon}, cost={cost_per_trade*1e4:.1f} bps")
    plt.xlabel("Date"); plt.ylabel("Cum. PnL (log-return units)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pred_vs_true(model: str, fold_id: int, horizon: str, point_alpha: float = 0.6):
    """
    Scatter: predicted vs. true target.
    """
    preds = load_preds(model, fold_id, horizon)
    y_true = preds["y_true"].values
    y_pred = preds["y_pred"].values
    lim = np.percentile(np.abs(np.concatenate([y_true, y_pred])), 99)  # robust axis
    lim = max(lim, 1e-3)
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=point_alpha, s=12)
    xs = np.linspace(-lim, lim, 100)
    plt.plot(xs, xs)  # y=x reference
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{model} — {horizon} — fold {fold_id}")
    plt.tight_layout()
    plt.show()

def plot_residual_hist(model: str, fold_id: int, horizon: str, bins: int = 30):
    """
    Histogram of residuals (y_true - y_pred).
    """
    preds = load_preds(model, fold_id, horizon)
    resid = preds["y_true"].values - preds["y_pred"].values
    plt.figure(figsize=(7,4))
    plt.hist(resid, bins=bins, alpha=0.85)
    plt.title(f"Residuals Histogram — {model}, {horizon}, fold {fold_id}")
    plt.xlabel("Residual"); plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_lastN_ts(model: str, fold_id: int, horizon: str, last_n: int = 250):
    """
    Plot time series of y_true and y_pred (last N points of the test period).
    """
    preds = load_preds(model, fold_id, horizon).iloc[-last_n:].copy()
    plt.figure(figsize=(10,4))
    plt.plot(preds.index, preds["y_true"].values, label="True")
    plt.plot(preds.index, preds["y_pred"].values, label="Pred")
    plt.title(f"{model} — {horizon} — last {last_n} test days")
    plt.xlabel("Date"); plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- turn in-memory results into a DataFrame ----------
def results_to_df(*results: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten result dicts returned by run_baseline(...) into a tidy DataFrame.
    """
    rows = []
    for r in results:
        base = {k: v for k, v in r.items() if k not in ("backtest",)}
        bt = r.get("backtest", {})
        base.update(bt)
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["rmse","mae","model"]).reset_index(drop=True)
