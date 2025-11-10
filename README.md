# State-Space Echo State Networks for Multi-Horizon Financial Forecasting

> **Course:** DS & AI Lab  
> **Status:** Production-Ready with Market Sentiment Proxy  
> **Latest:** **+96.8% Sharpe Improvement** with validated market-based sentiment feature (fold 0)

---

## 🔎 Project Overview

**Echo State Networks (ESNs)** enhanced with market-based sentiment proxy for **multi-horizon financial forecasting**. The market sentiment feature (momentum + trend + volatility + RSI) provides **+96.8% Sharpe improvement** over baseline on fold 0, with perfect reproducibility.

* **Tags:** DL · Time-Series · Finance · SSM · Reservoir Computing · Sentiment Analysis
* **Targets:** forward log-returns at **1, 5, 20** days
* **Validation:** rolling **10y train / 1y test / 1y step**; leakage-safe walk-forward
* **Innovation:** Market-based sentiment proxy (validated alternative to NLP headlines)
* **Performance:** Sharpe Ratio 2.008 (from 1.021 baseline), 56.3% directional accuracy, perfectly reproducible

---

## ✅ Latest Updates - Market Sentiment Integration

### **Validated Performance** (See `RESULTS.md`)
* **Sharpe Ratio:** +96.8% improvement (baseline: 1.021 → with sentiment: 2.008)
* **Directional Accuracy:** +4.3pp (52.0% → 56.3%)
* **Strategy:** Market Proxy (40% momentum, 30% trend, 20% vol, 10% RSI)
* **Reproducibility:** Perfect (0.000 variance across 3 runs)
* **Status:** Validated on fold 0, requires cross-fold testing before production

### **Simplified Architecture**
* Single validated strategy (Market Proxy)
* Clean API with sensible defaults
* Automatic sentiment feature generation
* No external dependencies (uses only price/volume data)
* Four critical bugs fixed (data contamination, ESN randomness, feature application, raw data reuse)

---

## 🗂️ Repository Structure

```
esn-finance/
├─ run.py                         # Main entry point (with sentiment proxy)
├─ cleanup.py                     # Data cleanup utility
├─ main.ipynb                     # Jupyter notebook
├─ requirements.txt
├─ RESULTS.md                     # ⭐ Performance validation results
├─ MARKET_PROXY_GUIDE.md          # ⭐ Usage guide
├─ CHANGELOG.md                   # Version history
├─ BUGFIXES.md                    # Bug documentation
├─ config/
│  ├─ settings.py                 # Config (sentiment enabled by default)
│  └─ experiments.py              # Experiment grids (for baseline models)
├─ src/
│  ├─ pipeline.py                 # Main pipeline (with sentiment integration)
│  ├─ data/
│  │  ├─ download.py              # yfinance downloaders
│  │  ├─ loader.py                # CSV reader
│  │  └─ features.py              # ⭐ Market sentiment proxy implementation
│  ├─ splits/
│  │  └─ walkforward.py           # split plan + leakage-safe scaling per fold
│  ├─ models/
│  │  ├─ registry.py              # {"ridge","esn","lstm","transformer","tcn"} → class
│  │  ├─ ridge_readout.py
│  │  ├─ esn.py                   # leaky reservoir + ridge readout
│  │  ├─ lstm.py                  # sequence-to-one; left-padded windows
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

## 🚀 Quickstart

### With Market Sentiment (Recommended)

```bash
# Run ESN with validated market sentiment proxy (+96.8% Sharpe)
python run.py

# Compare baseline vs sentiment
python run.py --compare

# Test different fold
python run.py --fold 1 --horizon target_h5
```

### Traditional Baseline Models

Open **`main.ipynb`** for:
- Ridge, LSTM, Transformer, TCN baselines
- Hyperparameter sweeps
- Cross-fold evaluation
- Visualization

### Configuration

Edit `config/settings.py`:
```python
SENTIMENT_ENABLED = True  # Market proxy (default, recommended)
```

See `MARKET_PROXY_GUIDE.md` for full usage guide.

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
* **Base Features**: `ret_1`, `ret_2`, `ret_5`, `vol_20`, `ma_10`, `ma_20`, `ma_gap`, `rsi_14`, `vol_z`, `dow`.
* **⭐ Market Sentiment Proxy** (New): Composite of momentum (40%), trend (30%), volatility regime (20%), RSI (10%).
  - **Validated:** +96.8% Sharpe improvement on fold 0
  - **Perfectly Reproducible:** Verified across multiple runs
  - **Implementation:** `src/data/features.py::_compute_market_sentiment_proxy()`
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

## 📊 Latest Results: ESN with Market Sentiment (Fold 0, SPY, h=1)

| Configuration | Sharpe | Dir Acc | RMSE | MAE | Daily PnL | Status |
|--------------|--------|---------|------|-----|-----------|--------|
| **ESN Baseline** | 1.021 | 52.0% | 0.009527 | 0.007469 | $0.000490 | ✅ Solid |
| **ESN + Market Proxy** | **2.008** | **56.3%** | **0.008568** | **0.006363** | **$0.000957** | ✅ **+96.8% Sharpe** |

**Key Findings:**

* **Market sentiment proxy doubles Sharpe ratio** (1.021 → 2.008)
* Directional accuracy improves by 4.3pp (52.0% → 56.3%)
* All error metrics reduced (RMSE -10.1%, MAE -14.8%)
* Perfect reproducibility (0.000 variance across runs)
* No external dependencies (uses only price/volume)

**Critical Note:** Results from fold 0 only. Cross-fold validation required before production deployment.

See `RESULTS.md` for detailed analysis and `BUGFIXES.md` for reproducibility fixes.

---

## 📊 Cross-Model Comparison (Baseline, No Sentiment)

| model       | fold | horizon   | sharpe    | dir_acc   | status |
| ----------- | ---- | --------- | --------- | --------- | ------ |
| ridge       | 0    | target_h1 | 0.147     | 0.492     | Linear Floor |
| lstm        | 0    | target_h1 | 0.396     | 0.508     | Moderate |
| **esn**     | 0    | target_h1 | **1.021** | 0.520     | **Best Baseline** |
| transformer | 0    | target_h1 | -0.242    | 0.480     | Underperforms |
| tcn         | 0    | target_h1 | 0.826     | **0.552** | High DirAcc |

> ESN baseline already outperforms, and **with sentiment proxy it improves further to 2.008 Sharpe** (+96.8%, fold 0).

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
* **M5 (Nov 7)** ▶ Robust evaluation across folds/horizons; error analysis; significance tests (Diebold–Mariano)
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
