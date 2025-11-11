# src/train/runner.py
import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import settings
from src.models.registry import get_model
from src.train.utils import dict_product, param_slug, ensure_dir, set_seed
from src.eval.metrics import evaluate_predictions, sign_backtest

def _load_fold(fold_id: int):
    fold_dir = os.path.join(settings.SPLIT_DIR, f"fold_{fold_id}")
    train = pd.read_csv(os.path.join(fold_dir, "train.csv"), index_col=0, parse_dates=True)
    test  = pd.read_csv(os.path.join(fold_dir, "test.csv"),  index_col=0, parse_dates=True)
    X_tr = train[[c for c in train.columns if c.startswith("z_")]].values
    X_te = test [[c for c in test.columns if c.startswith("z_")]].values
    return train, test, X_tr, X_te, fold_dir

def run_experiment(model_name: str, fold_id: int, horizon: str, model_kwargs: dict, exp_name: str | None = None, save_model: bool = False) -> dict:
    """
    Train one model on one fold/horizon with given hyperparams.
    Saves preds/metrics under: data/experiments/<model>/<exp_id>/fold_<k>/
    
    Args:
        model_name: Name of model to train
        fold_id: Fold number
        horizon: Target horizon (target_h1, target_h5, target_h20)
        model_kwargs: Hyperparameters for the model
        exp_name: Optional experiment name
        save_model: If True, save the trained model to disk (for models with save() method)
    """
    set_seed(model_kwargs.get("seed", 0))
    train, test, X_tr, X_te, _ = _load_fold(fold_id)
    y_tr = train[horizon].values
    y_te = test[horizon].values

    model = get_model(model_name, **model_kwargs)
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)

    # exp folder
    exp_id = exp_name or param_slug(model_kwargs)
    out_dir = os.path.join(settings.EXP_DIR, model_name, exp_id, f"fold_{fold_id}")
    ensure_dir(out_dir)

    # save preds
    preds = pd.DataFrame({"y_true": y_te, "y_pred": y_hat}, index=test.index)
    preds.to_csv(os.path.join(out_dir, f"preds_{horizon}.csv"))

    # metrics + backtest
    m = evaluate_predictions(y_te, y_hat).to_dict()
    b = sign_backtest(y_te, y_hat, cost_per_trade=0.0001)
    result = dict(model=model_name, exp_id=exp_id, fold=fold_id, horizon=horizon, **m, backtest=b)

    with open(os.path.join(out_dir, f"metrics_{horizon}.json"), "w") as f:
        json.dump(result, f, indent=2)
    
    # Save trained model if requested and supported
    if save_model and hasattr(model, 'save'):
        model_dir = os.path.join(out_dir, f"model_{horizon}")
        model.save(model_dir)
    
    return result

def run_sweep(model_name: str, param_grid: dict, folds, horizons, exp_prefix: str | None = None) -> pd.DataFrame:
    """
    Loop over grid × folds × horizons. Returns a tidy DataFrame and saves
    a summary CSV under data/experiments/<model>/<exp_id>/.
    """
    rows = []
    grid = list(dict_product(param_grid))
    pbar = tqdm(total=len(grid)*len(folds)*len(horizons), desc=f"sweep:{model_name}", ncols=100)

    for params in grid:
        # fixed exp_id for this param set:
        exp_id = (exp_prefix + "_") if exp_prefix else ""
        exp_id += param_slug(params)
        for k in folds:
            for h in horizons:
                res = run_experiment(model_name, k, h, params, exp_name=exp_id)
                rows.append(res)
                pbar.update(1)

        # write interim summary per exp_id (aggregated over folds for each horizon)
        _write_interim_summary(model_name, exp_id, rows)

    pbar.close()
    df = _flatten_rows(rows)
    return df

def _flatten_rows(rows):
    flat = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != "backtest"}
        for bk, bv in r.get("backtest", {}).items():
            base[bk] = bv
        flat.append(base)
    return pd.DataFrame(flat)

def _write_interim_summary(model: str, exp_id: str, rows: list):
    df = _flatten_rows([r for r in rows if r["model"]==model and r["exp_id"]==exp_id])
    if df.empty:
        return
    out_dir = os.path.join(settings.EXP_DIR, model, exp_id)
    ensure_dir(out_dir)
    # group by horizon, aggregate over folds
    gb = df.groupby(["horizon"]).agg({
        "rmse":["mean","std"], "mae":["mean","std"], "r2":["mean","std"],
        "dir_acc":["mean","std"], "avg_daily_pnl":["mean","std"],
        "vol":["mean","std"], "sharpe":["mean","std"], "hit_ratio":["mean","std"], "turnover":["mean","std"],
    })
    gb.to_csv(os.path.join(out_dir, "summary_by_horizon.csv"))
