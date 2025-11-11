#!/usr/bin/env python
# coding: utf-8

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path (script is in scripts/training/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import settings
from config.experiments import (
    ESN_GRID,
    FOLDS,
    HORIZONS,
    LSTM_GRID,
    TCN_GRID,
    TRANSFORMER_GRID,
)
from src.eval.aggregate import collect_all_metrics
from src.pipeline import (
    run_baseline,
    run_build_splits,
    run_download,
    run_materialize_folds,
    run_process,
)
from src.train.runner import run_sweep
from src.viz.plots import collect_metrics, plot_metric_bars


MODELS_TO_RUN = ["esn", "lstm", "transformer", "tcn"]
GRIDS = {
    "esn": ESN_GRID,
    "lstm": LSTM_GRID,
    "transformer": TRANSFORMER_GRID,
    "tcn": TCN_GRID,
}


def list_exp_ids(model: str) -> List[str]:
    base = Path(settings.EXP_DIR) / model
    if not base.exists():
        return []
    return sorted([d.name for d in base.iterdir() if d.is_dir()])


def rank_experiments(
    model: str,
    horizon: str,
    key: str = "sharpe",
    ascending: bool = False,
    top_k: int = 3,
) -> pd.DataFrame:
    rows = []
    for exp_id in list_exp_ids(model):
        df = collect_all_metrics(model, exp_id)
        if df.empty or horizon not in df["horizon"].unique():
            continue
        sub = df[df["horizon"] == horizon].copy()
        agg = sub.mean(numeric_only=True).to_dict()
        agg["model"] = model
        agg["exp_id"] = exp_id
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    tab = pd.DataFrame(rows)
    tab = tab.sort_values(key, ascending=ascending).reset_index(drop=True)
    return tab.head(top_k)


def load_metrics_exp(model: str, exp_id: str, fold_id: int, horizon: str) -> Dict:
    path = (
        Path(settings.EXP_DIR)
        / model
        / exp_id
        / f"fold_{fold_id}"
        / f"metrics_{horizon}.json"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_cum_pnl_from_exp(
    winners_df: pd.DataFrame,
    fold_id: int,
    horizon: str,
    cost_per_trade: float = 0.0001,
) -> None:
    if winners_df.empty:
        print("No winners to plot cumulative PnL.")
        return

    plt.figure(figsize=(10, 5))
    for _, row in winners_df.iterrows():
        model, exp_id = row["model"], row["exp_id"]
        preds_path = (
            Path(settings.EXP_DIR)
            / model
            / exp_id
            / f"fold_{fold_id}"
            / f"preds_{horizon}.csv"
        )
        if not preds_path.exists():
            print(f"Missing preds file for {model} ({exp_id}). Skipping.")
            continue
        dfp = pd.read_csv(preds_path, index_col=0, parse_dates=True)
        y_true = dfp["y_true"].values
        y_pred = dfp["y_pred"].values
        pos = np.sign(y_pred)
        pos_change = np.abs(np.diff(pos, prepend=0))
        pnl = pos * y_true - cost_per_trade * pos_change
        plt.plot(dfp.index, pnl.cumsum(), label=f"{model} ({exp_id[:10]}…)")

    plt.title(f"Cumulative PnL — winners (fold {fold_id}, {horizon})")
    plt.xlabel("Date")
    plt.ylabel("Cum. PnL (log-returns)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def summarize_findings(
    horizon: str, leader_rmse: pd.DataFrame, leader_sharpe: pd.DataFrame
) -> str:
    lines = [f"# Milestone 4 Findings — Horizon: {horizon}\n"]
    lines.append("## Best by RMSE (per model)\n")
    if not leader_rmse.empty:
        for _, row in leader_rmse.iterrows():
            lines.append(
                "- **{model}**  | RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.3f}, "
                "DirAcc={dir_acc:.3f}  | exp_id={exp_id}".format(**row)
            )
    else:
        lines.append("- (no results)")

    lines.append("\n## Best by Sharpe (per model)\n")
    if not leader_sharpe.empty:
        for _, row in leader_sharpe.iterrows():
            lines.append(
                "- **{model}**  | Sharpe={sharpe:.3f}, PnL={avg_daily_pnl:.6f}, "
                "Turnover={turnover:.3f}, DirAcc={dir_acc:.3f}  | "
                "RMSE={rmse:.6f}  | exp_id={exp_id}".format(**row)
            )
    else:
        lines.append("- (no results)")

    if not leader_sharpe.empty:
        best_trader = leader_sharpe.sort_values("sharpe", ascending=False).iloc[0]
        lines.append(
            "\n**Bottom line:** For trading (Sharpe), **{model}** looks best. "
            "For pure magnitude accuracy (RMSE), refer to the RMSE table above.".format(
                **best_trader
            )
        )
    return "\n".join(lines)


def print_dataframe(df: pd.DataFrame, title: str) -> None:
    print(title)
    if df.empty:
        print("(no results)")
    else:
        print(df.to_string(index=False))
    print()


def main() -> None:
    downloads = run_download()
    print("Downloaded datasets:")
    for entry in downloads:
        print(
            f" - {entry['symbol']:>8} | {entry['rows']:>5} rows | "
            f"{entry['start']} -> {entry['end']} | {entry['path']}"
        )

    proc_paths = run_process()
    print("\nProcessed dataset paths:")
    for symbol, path in proc_paths.items():
        print(f" - {symbol}: {path}")

    folds = run_build_splits(proc_paths)
    print(f"\nBuilt {len(folds)} folds.")
    if folds:
        print(f"Example fold[0]: {folds[0]}")

    run_materialize_folds(proc_paths, folds)
    print(f"Materialized folds in: {settings.SPLIT_DIR}\n")

    print("=== Baseline comparisons (fold=0, horizon=target_h1) ===")
    baseline_models = ["esn", "ridge", "lstm", "transformer", "tcn"]
    baseline_results = {}
    for model_name in baseline_models:
        result = run_baseline(model_name=model_name, fold_id=0, horizon="target_h1")
        baseline_results[model_name] = result
        print(f"{model_name}: {result}")
    print()

    models = ["ridge", "esn", "lstm", "transformer", "tcn"]
    fold_id = 0
    horizon = "target_h1"
    metrics_df = collect_metrics(models, fold_id, horizon)
    print_dataframe(metrics_df, "Baseline metrics table:")

    plot_metric_bars(metrics_df, metric="rmse", title=f"RMSE — fold {fold_id}, {horizon}")
    plot_metric_bars(metrics_df, metric="mae", title=f"MAE — fold {fold_id}, {horizon}")
    plot_metric_bars(
        metrics_df, metric="dir_acc", title=f"Directional Accuracy — fold {fold_id}, {horizon}"
    )
    plot_metric_bars(
        metrics_df, metric="sharpe", title=f"Sharpe (sign backtest) — fold {fold_id}, {horizon}"
    )

    print("=== Ridge baseline on longer horizons ===")
    for horizon in ["target_h5", "target_h20"]:
        result = run_baseline(model_name="ridge", fold_id=0, horizon=horizon)
        print(f"{horizon}: {result}")
    print()

    print("=== Hyperparameter sweep configuration ===")
    print(f"Folds: {FOLDS}")
    print(f"Horizons: {HORIZONS}")
    for model in MODELS_TO_RUN:
        grid_size = int(np.prod([len(v) for v in GRIDS[model].values()]))
        print(f"{model} grid size = {grid_size}")
    print()

    sweep_results = {}
    for model in MODELS_TO_RUN:
        print(f"=== Running sweep: {model} ===")
        df = run_sweep(
            model_name=model,
            param_grid=GRIDS[model],
            folds=FOLDS,
            horizons=HORIZONS,
            exp_prefix="m4",
        )
        sweep_results[model] = df
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(df.head(3).to_string(index=False))
        print()

    H = HORIZONS[0]
    leader_rmse_rows = []
    leader_sharpe_rows = []
    for model in MODELS_TO_RUN:
        r_rmse = rank_experiments(model, H, key="rmse", ascending=True, top_k=1)
        r_sharpe = rank_experiments(model, H, key="sharpe", ascending=False, top_k=1)
        if not r_rmse.empty:
            leader_rmse_rows.append(r_rmse.iloc[0])
        if not r_sharpe.empty:
            leader_sharpe_rows.append(r_sharpe.iloc[0])

    leader_rmse = pd.DataFrame(leader_rmse_rows)
    leader_sharpe = pd.DataFrame(leader_sharpe_rows)
    print_dataframe(
        leader_rmse[["model", "exp_id", "rmse", "mae", "r2", "dir_acc", "sharpe"]]
        if not leader_rmse.empty
        else leader_rmse,
        "== Best by RMSE (per model) ==",
    )
    print_dataframe(
        leader_sharpe[["model", "exp_id", "rmse", "mae", "r2", "dir_acc", "sharpe"]]
        if not leader_sharpe.empty
        else leader_sharpe,
        "== Best by Sharpe (per model) ==",
    )

    winners = leader_sharpe[["model", "exp_id"]].dropna() if not leader_sharpe.empty else pd.DataFrame()
    print_dataframe(winners, "Winners (Sharpe):")

    if not winners.empty:
        fold_id = FOLDS[0]
        rows = []
        for _, row in winners.iterrows():
            metrics = load_metrics_exp(row["model"], row["exp_id"], fold_id, H)
            base = {k: v for k, v in metrics.items() if k != "backtest"}
            base.update(metrics.get("backtest", {}))
            base["model"] = row["model"]
            base["exp_id"] = row["exp_id"]
            rows.append(base)
        df_win = pd.DataFrame(rows)
        print_dataframe(
            df_win[["model", "exp_id", "rmse", "mae", "r2", "dir_acc", "sharpe", "avg_daily_pnl", "turnover"]],
            "Winner metrics:",
        )

        plot_metric_bars(
            df_win.rename(columns={"model": "model"}),
            metric="rmse",
            title=f"RMSE — winners (fold {fold_id}, {H})",
        )
        plot_metric_bars(
            df_win.rename(columns={"model": "model"}),
            metric="dir_acc",
            title=f"Directional Accuracy — winners (fold {fold_id}, {H})",
        )
        plot_metric_bars(
            df_win.rename(columns={"model": "model"}),
            metric="sharpe",
            title=f"Sharpe — winners (fold {fold_id}, {H})",
        )

        plot_cum_pnl_from_exp(winners, fold_id, H, cost_per_trade=0.0001)

        focus_model = winners.iloc[0]["model"]
        focus_exp = winners.iloc[0]["exp_id"]
        print(f"Focus: {focus_model} {focus_exp}")

        preds_path = (
            Path(settings.EXP_DIR)
            / focus_model
            / focus_exp
            / f"fold_{fold_id}"
            / f"preds_{H}.csv"
        )
        if preds_path.exists():
            dfp = pd.read_csv(preds_path, index_col=0, parse_dates=True)
            y_true = dfp["y_true"].values
            y_pred = dfp["y_pred"].values

            plt.figure(figsize=(5, 5))
            plt.scatter(y_true, y_pred, s=12, alpha=0.6)
            lim = np.percentile(np.abs(np.concatenate([y_true, y_pred])), 99)
            lim = max(lim, 1e-3)
            xs = np.linspace(-lim, lim, 100)
            plt.plot(xs, xs)
            plt.title(f"{focus_model} — {H} — fold {fold_id}")
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.tight_layout()
            plt.show()

            resid = y_true - y_pred
            plt.figure(figsize=(7, 4))
            plt.hist(resid, bins=30, alpha=0.85)
            plt.title(f"Residuals — {focus_model} — {H} — fold {fold_id}")
            plt.xlabel("Residual")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

            last_n = 250
            plt.figure(figsize=(10, 4))
            plt.plot(dfp.index[-last_n:], dfp["y_true"].values[-last_n:], label="True")
            plt.plot(dfp.index[-last_n:], dfp["y_pred"].values[-last_n:], label="Pred")
            plt.title(f"{focus_model} — {H} — last {last_n} test days")
            plt.xlabel("Date")
            plt.ylabel("Return")
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print(f"Missing preds file at {preds_path}")

        findings_text = summarize_findings(H, leader_rmse, leader_sharpe)
        print(findings_text)
        out_txt = Path(settings.EXP_DIR) / f"milestone4_findings_{H}.txt"
        out_txt.write_text(findings_text, encoding="utf-8")
        print(f"\nSaved findings to: {out_txt}")
    else:
        print("No Sharpe winners identified; skipping detailed analysis.")

    print("\n=== Milestone 5: cross-fold baselines ===")
    folds_to_run = [0, 1, 2]
    horizons = ["target_h1", "target_h5", "target_h20"]
    for fold_id in folds_to_run:
        for horizon in horizons:
            result_esn = run_baseline(model_name="esn", fold_id=fold_id, horizon=horizon)
            result_ridge = run_baseline(model_name="ridge", fold_id=fold_id, horizon=horizon)
            result_lstm = run_baseline(model_name="lstm", fold_id=fold_id, horizon=horizon)
            result_tf = run_baseline(model_name="transformer", fold_id=fold_id, horizon=horizon)
            result_tcn = run_baseline(model_name="tcn", fold_id=fold_id, horizon=horizon)
            print(
                f"fold={fold_id}, horizon={horizon} | "
                 f"esn={result_esn} | ridge={result_ridge} | "
                f"lstm={result_lstm} | transformer={result_tf} | tcn={result_tcn}"
            )


if __name__ == "__main__":
    main()

