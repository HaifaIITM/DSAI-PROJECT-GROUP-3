# State-Space Echo State Networks for Multi-Horizon Financial Forecasting

> **Course:** DS & AI Lab
> **Repo status:** **Milestones 1–4 completed**
> – M1: Problem Definition & Literature Review
> – M2: Dataset Preparation, Features, Walk-Forward Splits
> – M3: Model Architecture (ESN) + Deep Baselines (LSTM/Transformer/TCN) + Viz
> – **M4: Training runs, hyper-parameter sweeps, fold/horizon aggregation, findings**

---

## 🔎 Project Overview

We study **Echo State Networks (ESNs)** as **state-space models (SSMs)** for **multi-horizon financial forecasting** at daily frequency. ESNs maintain a latent “reservoir” state (leaky, random recurrent) and train only a **ridge readout**, which controls variance under noisy, nonstationary returns. We benchmark against **LSTM**, **Transformer encoder**, and **Temporal ConvNet (TCN)** under a **leakage-safe walk-forward** protocol.

* **Tags:** DL · Time-Series · Finance · SSM · Reservoir Computing
* **Targets:** forward log-returns at **1, 5, 20** days
* **Validation:** rolling **10y train / 1y test / 1y step**; **train-only** normalization
* **Open data:** Yahoo Finance (primary); optional exogenous series planned.

---

## ✅ What’s New in Milestone 4

* **Experiment config & grids:** `config/experiments.py` with small, reproducible grids for ESN/LSTM/Transformer/TCN.
* **Training runner:** `src/train/runner.py` to run **grid sweeps** across **folds × horizons**, save artifacts, and auto-aggregate.
* **Unique experiment IDs:** filesystem-safe **param slugs** to separate runs; seeds fixed for reproducibility.
* **Aggregation utilities:** fold/horizon summaries in `src/eval/aggregate.py`; leaderboard helpers in notebook.
* **Findings auto-report:** notebook cell writes `data/experiments/milestone4_findings_<h>.txt`.

---

## 🗂️ Repository Structure

```
esn-finance/
├─ main.ipynb
├─ requirements.txt
├─ .gitignore
├─ config/
│  ├─ settings.py                 # tickers, dates, dirs, split sizes, features/targets
│  └─ experiments.py              # NEW: default grids, folds, horizons (M4)
├─ src/
│  ├─ pipeline.py                 # download → process → splits → materialize → run_baseline
│  ├─ utils/
│  │  └─ io.py
│  ├─ data/
│  │  ├─ download.py              # yfinance downloaders
│  │  ├─ loader.py                # robust Yahoo CSV reader (two formats)
│  │  └─ features.py              # feature engineering + targets
│  ├─ splits/
│  │  └─ walkforward.py           # split plan + leakage-safe scaling per fold
│  ├─ models/
│  │  ├─ registry.py              # {"ridge","esn","lstm","transformer","tcn"} → class
│  │  ├─ ridge_readout.py
│  │  ├─ esn.py                   # leaky reservoir + ridge readout
│  │  ├─ lstm.py                  # sequence-to-one, left-padded windows
│  │  ├─ transformer.py           # encoder + positional encoding
│  │  └─ tcn.py                   # causal dilated ConvNet
│  ├─ train/
│  │  ├─ __init__.py
│  │  ├─ utils.py                 # seeds, dict grid, param slug
│  │  └─ runner.py                # run_experiment / run_sweep (M4 core)
│  ├─ eval/
│  │  ├─ metrics.py               # RMSE/MAE/R²/DirAcc + sign backtest (toy)
│  │  └─ aggregate.py             # collect & summarize metrics across folds
│  └─ viz/
│     └─ plots.py                 # metrics table, bars, residuals, cum-PnL, time series
└─ data/
   ├─ raw/                        # OHLCV CSVs (git-ignored)
   ├─ processed/                  # *_features.csv (PX, features, targets)
   ├─ splits/                     # splits.json + fold_k/{train.csv,test.csv,scaler.json}
   └─ experiments/                # <model>/<exp_id>/fold_k/{preds_*.csv, metrics_*.json}
```

---

## ⚙️ Setup

```bash
# Recommended
conda create -n esn-finance python=3.11 -y
conda activate esn-finance
pip install -r requirements.txt
```

**requirements.txt (core):**

```
numpy
pandas
scikit-learn
matplotlib
yfinance
tqdm
torch
```

---

## 🚀 Quickstart (M2–M3)

Open **`main.ipynb`** and run:

1. **Download** raw data → `data/raw/`
2. **Process** features/targets → `data/processed/`
3. **Build** `splits.json` → `data/splits/`
4. **Materialize** fold files with **train-only** scalers → `data/splits/fold_k/`
5. **Train/Eval** quick baselines: `"ridge"`, `"esn"`, `"lstm"`, `"transformer"`, `"tcn"`

Visualization helpers live in `src/viz/plots.py`.

---

## 🧪 Milestone 4: Run Sweeps & Report

Inside **`main.ipynb`**:

* Configure grids/folds/horizons via `config/experiments.py`.
* Use `run_sweep(model_name, grid, folds, horizons, exp_prefix)` from `src/train/runner.py`.
* Artifacts are saved under `data/experiments/<model>/<exp_id>/fold_<k>/`.
* Build **leaderboards** and **findings** automatically (cells included in notebook).

**Artifacts per fold:**

* `preds_<h>.csv` (y_true, y_pred indexed by date)
* `metrics_<h>.json` (RMSE/MAE/R²/DirAcc + backtest stats)
* `summary_by_horizon.csv` (mean±std across folds written per exp_id)

---

## 📦 Data & Features

* **Symbols**: `^GSPC`, `SPY`, `BTC-USD`, `ETH-USD`, `^NSEI`, `^NSEBANK`, `RELIANCE.NS`, `TCS.NS`, `EURUSD=X`, `USDINR=X`, `GC=F`, `CL=F`, `^VIX`.
* **Features** (computed on Adj Close if present): `ret_1`, `ret_2`, `ret_5`, `vol_20`, `ma_10`, `ma_20`, `ma_gap`, `rsi_14`, `vol_z`, `dow`.
* **Targets**: `target_h1`, `target_h5`, `target_h20` (forward log-returns).
* **Leakage control**: walk-forward splits; **scaler fit on train only**; scaler params stored.

---

## 🧠 Models

* **Ridge**: strong linear floor on standardized features.
* **ESN**: leaky reservoir (`tanh`), spectral radius, washout, **ridge readout**.
* **LSTM**: sequence-to-one; left-padded windows for full alignment.
* **Transformer (encoder)**: positional encoding; last-token head.
* **TCN**: causal 1-D dilated ConvNet with residual blocks.

All expose a uniform API: `.fit(X, y)` / `.predict(X)` and are registered in `src/models/registry.py`.

---

## 📊 Snapshot: Initial Cross-Model Comparison (Fold 0, SPY, h=1)

| model       | fold | horizon   | rmse    | mae     | r2       | dir_acc   | avg_daily_pnl | vol     | sharpe    | hit_ratio | turnover |
| ----------- | ---- | --------- | ------- | ------- | -------- | --------- | ------------- | ------- | --------- | --------- | -------- |
| ridge       | 0    | target_h1 | 0.00834 | 0.00599 | -0.00520 | 0.492     | 0.000077      | 0.00832 | 0.147     | 0.492     | 0.687    |
| lstm        | 0    | target_h1 | 0.00939 | 0.00703 | -0.27641 | 0.508     | 0.000208      | 0.00833 | 0.396     | 0.508     | 0.337    |
| **esn**     | 0    | target_h1 | 0.01010 | 0.00785 | -0.47538 | 0.528     | **0.000841**  | 0.00828 | **1.612** | 0.528     | 0.853    |
| transformer | 0    | target_h1 | 0.01437 | 0.01109 | -1.98638 | 0.480     | -0.000127     | 0.00833 | -0.242    | 0.480     | 0.456    |
| tcn         | 0    | target_h1 | 0.02857 | 0.01933 | -10.8069 | **0.552** | 0.000433      | 0.00831 | 0.826     | 0.552     | 0.631    |

**Interpretation (early):**

* **Best magnitude (RMSE/MAE):** *Ridge* (linear floor is hard to beat on daily returns).
* **Best risk-adjusted trading signal:** *ESN* (highest Sharpe and avg PnL with decent DirAcc).
* **TCN** shows strong **directional accuracy** but poor calibration (high RMSE).
* **Transformer** underperforms with current small-data/short-training settings.

> Results are **single-fold** sanity checks; full evaluation will average across folds/horizons (M5).

---

## 🔁 Reproducibility & Artifacts

* **Seeds** fixed in trainers; **scaler.json** per fold stores means/scales.
* **Param slugs** (hash of canonicalized params) name experiment folders to avoid overwrites.
* Every experiment is a **pure function** of (data split, hyper-params, seed).

---

## 🧭 Roadmap

* **M1 (Oct 3)** ✅ Problem & Literature
* **M2 (Oct 10)** ✅ Data, features, walk-forward splits
* **M3 (Oct 17)** ✅ ESN + deep baselines + viz
* **M4 (Oct 31)** ✅ Training sweeps, aggregation, findings
* **M5 (Nov 7)** ▶ Robust evaluation across folds/horizons; error analysis
* **M6 (Nov 14)** ▶ Minimal demo/API (HF Spaces or Gradio), docs, final report

---

## 🔒 Ethics & Responsible Use

This repository is for **education/research** only and **not** investment advice. Backtests are simplified (toy costs) and not indicative of live performance. We avoid look-ahead bias and data snooping; all scaling is train-only.

---

## 📚 References (core)

* Jaeger, H. (2001); Lukoševičius & Jaeger (2009): Echo State Networks / Reservoir Computing
* Jaeger, H., et al. (2007): Leaky-Integrator ESNs
* Durbin, J., & Koopman, S. (2012): *Time Series Analysis by State Space Methods*
* Hyndman, R., & Athanasopoulos, G.: *Forecasting: Principles and Practice*
* Gneiting, T., & Raftery, A. (2007): Proper scoring rules
* Diebold, F., & Mariano, R. (1995): Comparing predictive accuracy
* Bailey, D., et al. (2014): Backtest overfitting

---

## 👥 Contributors

Team **GROUP-3** — feel free to open an Issue for questions, bugs, or feature requests.
