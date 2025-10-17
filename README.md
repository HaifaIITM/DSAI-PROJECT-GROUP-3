# State-Space Echo State Networks for Multi-Horizon Financial Forecasting

> **Course:** DS & AI Lab
> **Repo status:** **Milestones 1–3 completed**
> – M1: Problem Definition & Literature Review
> – M2: Dataset Preparation, Features, Walk-Forward Splits
> – M3: Model Architecture (ESN) + Deep Baselines (LSTM/Transformer/TCN) + Viz

---

## 🔎 Project Overview

We study **Echo State Networks (ESNs)** as **state-space models (SSMs)** for **multi-horizon financial forecasting** at daily frequency. The ESN uses a fixed, leaky random reservoir (latent state transition) and trains only a **ridge readout**—a low-variance approach well-suited to noisy, nonstationary markets. We compare against strong **deep baselines** (LSTM, Transformer encoder, Temporal ConvNet/TCN) under a **leakage-safe, walk-forward** protocol.

* **Tags:** DL · Time-Series · Finance · SSM · Reservoir Computing
* **Targets:** forward log-returns at **1, 5, 20** days
* **Validation:** rolling **10y train / 1y test / 1y step**; **train-only** normalization

**Open data sources:** Yahoo Finance (primary), with optional sources (Stooq, Alpha Vantage, FRED) for future exogenous inputs.

---

## ✅ What’s New (since Milestone 1)

* **M2 (Data):**

  * ~20 years of daily OHLCV for multiple assets via `yfinance`
  * Robust CSV loader (handles Yahoo’s single-header & multi-row header variants)
  * Interpretable feature set (u_t): returns, volatility, moving averages, RSI-14, volume z-score, weekday
  * **Walk-forward splits** with **train-only** `StandardScaler`, artifacts saved per fold

* **M3 (Models & Viz):**

  * **ESN implementation** (leaky reservoir with spectral-radius control; ridge readout; washout)
  * **Deep baselines**: LSTM, Transformer encoder, TCN (causal dilated ConvNet)
  * **Model registry** (swap models by name) and **viz module** for metrics & PnL plots

---

## 🗂️ Repository Structure

```
esn-finance/
├─ main.ipynb                         # single entry-point orchestrating the pipeline
├─ requirements.txt                   # includes torch, sklearn, yfinance, etc.
├─ .gitignore
├─ config/
│  └─ settings.py                     # tickers, dates, dirs, split sizes, features/targets
├─ src/
│  ├─ pipeline.py                     # run_download / run_process / run_build_splits / run_materialize_folds / run_baseline
│  ├─ utils/
│  │  └─ io.py                        # I/O helpers (mkdirs, json)
│  ├─ data/
│  │  ├─ download.py                  # yfinance downloaders
│  │  ├─ loader.py                    # robust Yahoo CSV reader (two formats)
│  │  └─ features.py                  # feature engineering + targets
│  ├─ splits/
│  │  └─ walkforward.py               # split plan + fold materialization (scalers)
│  ├─ models/
│  │  ├─ registry.py                  # {"ridge","esn","lstm","transformer","tcn"} → class
│  │  ├─ ridge_readout.py             # linear baseline
│  │  ├─ esn.py                       # echo state network (leaky reservoir + ridge readout)
│  │  ├─ lstm.py                      # LSTM baseline
│  │  ├─ transformer.py               # Transformer encoder baseline
│  │  └─ tcn.py                       # Temporal ConvNet baseline
│  ├─ eval/
│  │  └─ metrics.py                   # RMSE/MAE/R²/DirAcc + sign backtest (toy)
│  └─ viz/
│     └─ plots.py                     # metrics table, bars, residuals, cum-PnL, time series
└─ data/
   ├─ raw/                            # canonical OHLCV CSVs (ignored by git)
   ├─ processed/                      # *_features.csv (PX, features, targets)
   ├─ splits/                         # splits.json + fold_k/{train.csv,test.csv,scaler.json}
   └─ experiments/                    # model outputs by run (preds_*.csv, metrics_*.json)
```

---

## ⚙️ Setup

```bash
# Conda (recommended)
conda create -n esn-finance python=3.11 -y
conda activate esn-finance
pip install -r requirements.txt

# or venv
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

**`requirements.txt` (core):**

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

## 🚀 Quickstart (Milestones 2–3)

Open **`main.ipynb`** and run the cells in order:

1. **Download** raw data → `data/raw/`
2. **Process** to features/targets → `data/processed/`
3. **Build** walk-forward split plan → `data/splits/splits.json`
4. **Materialize folds** with train-only scalers → `data/splits/fold_k/`
5. **Train/eval** a model (choose any: `"ridge"`, `"esn"`, `"lstm"`, `"transformer"`, `"tcn"`)

Example (inside `main.ipynb`):

```python
from src.pipeline import (
    run_download, run_process, run_build_splits, run_materialize_folds, run_baseline
)

# 1–4) Data → features → splits → scaled folds
run_download()
run_process()
run_build_splits()
run_materialize_folds()

# 5) Baselines (Fold 0, horizon target_h1)
result_ridge = run_baseline(model_name="ridge",       fold_id=0, horizon="target_h1")
result_esn   = run_baseline(model_name="esn",         fold_id=0, horizon="target_h1")
result_lstm  = run_baseline(model_name="lstm",        fold_id=0, horizon="target_h1")
result_tf    = run_baseline(model_name="transformer", fold_id=0, horizon="target_h1")
result_tcn   = run_baseline(model_name="tcn",         fold_id=0, horizon="target_h1")
```

### Visualize

```python
from src.viz.plots import collect_metrics, plot_metric_bars, plot_cum_pnl, plot_pred_vs_true, plot_residual_hist, plot_lastN_ts

models  = ["ridge","esn","lstm","transformer","tcn"]
fold_id = 0
h       = "target_h1"

dfm = collect_metrics(models, fold_id, h)  # table of RMSE/MAE/R2/DirAcc + backtest stats
display(dfm)

plot_metric_bars(dfm, "rmse",     f"RMSE — fold {fold_id}, {h}")
plot_metric_bars(dfm, "dir_acc",  f"Directional Accuracy — fold {fold_id}, {h}")
plot_cum_pnl(models, fold_id, h, cost_per_trade=0.0001)

plot_pred_vs_true("esn", fold_id, h)
plot_residual_hist("esn", fold_id, h)
plot_lastN_ts("esn", fold_id, h, last_n=250)
```

---

## 📦 Data & Features

* **Symbols** include: `^GSPC`, `SPY`, `BTC-USD`, `ETH-USD`, `^NSEI`, `^NSEBANK`, `RELIANCE.NS`, `TCS.NS`, `EURUSD=X`, `USDINR=X`, `GC=F`, `CL=F`, `^VIX`.
* **Features (u_t)** (computed on `Adj Close` if available; else `Close`):

  * `ret_1`, `ret_2`, `ret_5` (log-returns, short momentum)
  * `vol_20` (realized vol, annualized), `ma_10`, `ma_20`, `ma_gap`
  * `rsi_14`, `vol_z` (volume z-score), `dow` (weekday)
* **Targets:** `target_h1`, `target_h5`, `target_h20` (future log-returns)

**Leakage control:** Walk-forward splits; scalers fit on train only; fold artifacts stored (`scaler.json`).

---

## 🧠 Models (Milestone 3)

* **Ridge** (baseline floor): linear readout on standardized features
* **ESN**: leaky reservoir (`tanh`), spectral-radius control, ridge readout, washout
  *Pros:* fast, low-variance, state-space interpretation
* **LSTM**: sequence-to-one; left-padded windows (default `seq_len=32`)
* **Transformer**: encoder w/ positional encoding; last-token head
* **TCN**: causal dilated 1-D conv with residual blocks; efficient receptive fields

All expose a **uniform API**: `.fit(X, y)` / `.predict(X)` and are registered in `src/models/registry.py`.

---

## 📊 Current Sanity-Check Results (Fold 0, SPY, h=1)

From the initial **Ridge** baseline on fold 0:

```
Test Metrics (Fold 0)
target_h1: RMSE=0.008599 | MAE=0.006228 | R^2=0.000284 | DirAcc=0.504
target_h5: RMSE=0.018104 | MAE=0.013711 | R^2=-0.050412 | DirAcc=0.448
target_h20: RMSE=0.035407 | MAE=0.028105 | R^2=-0.127060 | DirAcc=0.440

Simple Backtest (h=1, 1bp cost)
Avg daily PnL: 0.000185
Vol (std):     0.008607
Sharpe:        0.341
Hit ratio:     0.504
Turnover:      0.679 trades/day
```

> These numbers serve as a **sanity check** / lower bound before ESN and deep models. The sign-PnL is a **toy diagnostic** (not a trading claim).

---

## 🧭 Roadmap

* **M1 (Oct 3)** ✅ Problem & Literature
* **M2 (Oct 10)** ✅ Data download · Robust loader · Features/targets · Walk-forward splits
* **M3 (Oct 17)** ✅ ESN implementation · LSTM/Transformer/TCN baselines · Viz module
* **M4 (Oct 31)** ▶ Training runs across folds/horizons · Hyperparameter sweeps · Ablations
* **M5 (Nov 7)** ▶ Evaluation (RMSE/MAE/R²/DirAcc) · Error analysis · Statistical tests
* **M6 (Nov 14)** ▶ Minimal demo/API · Docs · Final report

---

## 🔒 Ethics & Responsible Use

* This repository is **for education/research**. Nothing here is investment advice.
* Backtests are simplified, include costs only in toy form, and are not indicative of live performance.
* We explicitly avoid look-ahead bias, overfitting practices, and data snooping.

---

## 📚 References (core)

* Jaeger (2001); Lukoševičius & Jaeger (2009): Echo State Networks / Reservoir Computing
* Jaeger et al. (2007): Leaky-Integrator ESNs
* Durbin & Koopman (2012): State Space Methods
* Hyndman & Athanasopoulos: Forecasting (FPP3)
* Gneiting & Raftery (2007): Proper scoring rules
* Diebold & Mariano (1995): Predictive accuracy tests
* Bailey et al. (2014): Backtest overfitting

---

## 👥 Contributors

Team **GROUP-3**

For questions or issues, please open a GitHub Issue.
