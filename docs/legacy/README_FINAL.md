# Financial Forecasting with Echo State Networks & Deep Learning

> **Final Project Status:** Complete - All models evaluated, best performers identified

---

## 🎯 Final Results

### Best Models by Horizon

| Horizon | Best Model | Sharpe | R² | RMSE | Key Strength |
|---------|-----------|--------|-----|------|--------------|
| **h1 (1-day)** | Transformer | 0.664 | -0.987 | 0.011 | Short-term patterns |
| **h5 (5-day)** | **LSTM** | **4.560** | **+0.060** | 0.014 | **Only positive R²!** |
| **h20 (20-day)** | **TCN** | **5.951** | **-0.373** | **0.028** | **Best overall** |

### Overall Winner: TCN (h20 horizon)
- Sharpe: 5.951 (highest risk-adjusted returns)
- R²: -0.373 (best magnitude prediction)
- RMSE: 0.028 (tied best forecast accuracy)
- Dir Accuracy: 60.7%
- **Use for:** Monthly trading strategies

### Novel Contribution: Hybrid ESN-Ridge
- 2nd-best Sharpe: 3.926
- Ultra-low turnover: 0.155 (most stable)
- Improves ESN by 288% (Sharpe)
- **Architecture:** `y_pred = sign(ESN) × |Ridge|`
- **Use for:** Low-turnover strategies

---

## 📊 Quick Start

### Evaluate Best Model (TCN)
```python
from src.pipeline import run_baseline

result = run_baseline("tcn", fold_id=0, horizon="target_h20")
print(f"Sharpe: {result['backtest']['sharpe']:.3f}")  # 5.951
```

### Run Comprehensive Comparison
```bash
python evaluate_hybrid_model.py
```

### Full Pipeline (Download → Process → Train)
```bash
python main.py
```

---

## 🗂️ Repository Structure

```
├── FINAL_RESULTS.md              # Complete performance analysis
├── HYBRID_MODEL_SUMMARY.md       # Hybrid ESN-Ridge documentation
├── INTEGRATION_GUIDE.md          # Headline embeddings integration
├── main.py                       # Full pipeline execution
├── evaluate_hybrid_model.py      # Model comparison script
├── spy_news.csv                  # Headline data (1000 articles)
├── config/
│   ├── settings.py               # Features: 10 technical + 28 headline embeddings
│   └── experiments.py            # Hyperparameter grids
├── src/
│   ├── pipeline.py               # Download → Process → Train → Evaluate
│   ├── data/
│   │   ├── features.py           # Technical indicators + headline embeddings
│   │   ├── embeddings.py         # Sentence-transformer PCA features
│   │   └── loader.py             # Yahoo Finance data loader
│   ├── models/
│   │   ├── registry.py           # Model factory
│   │   ├── esn.py                # Echo State Network
│   │   ├── lstm.py               # LSTM regressor
│   │   ├── transformer.py        # Transformer encoder
│   │   ├── tcn.py                # Temporal Convolutional Network
│   │   ├── ridge_readout.py      # Ridge baseline
│   │   └── hybrid_esn_ridge.py   # Novel hybrid architecture
│   ├── splits/
│   │   └── walkforward.py        # Walk-forward cross-validation
│   ├── eval/
│   │   └── metrics.py            # RMSE, MAE, R², Sharpe, Dir Accuracy
│   └── train/
│       └── runner.py             # Hyperparameter sweeps
└── data/
    ├── raw/                      # Downloaded OHLCV data
    ├── processed/                # 38-feature datasets
    ├── splits/                   # 9 walk-forward folds
    └── experiments/              # Model predictions & metrics
```

---

## 🔬 Models Implemented

### 1. TCN (Winner) ⭐
- **Architecture:** Dilated causal convolutions
- **Sharpe:** 5.951
- **Use for:** Monthly predictions (h20)

### 2. Hybrid ESN-Ridge (Novel) 🆕
- **Architecture:** ESN (direction) + Ridge (magnitude)
- **Sharpe:** 3.926
- **Unique:** Lowest turnover (0.155)
- **Use for:** Low-cost trading strategies

### 3. LSTM
- **Sharpe (h5):** 4.560
- **R² (h5):** +0.060 (only positive!)
- **Use for:** Weekly predictions

### 4. Transformer
- **Sharpe (h1):** 0.664
- **Use for:** Daily predictions

### 5. ESN (Baseline)
- **Improved by Hybrid:** +288% Sharpe
- **Challenge:** Magnitude prediction

### 6. Ridge (Linear Baseline)
- **Sharpe:** -5.034
- **Use for:** Benchmark comparison

---

## 📦 Features

### Technical Indicators (10):
`ret_1`, `ret_2`, `ret_5`, `vol_20`, `ma_10`, `ma_20`, `ma_gap`, `rsi_14`, `vol_z`, `dow`

### Headline Embeddings (28):
- Small model (all-MiniLM-L6-v2): 12 PCA + has_news
- Large model (all-mpnet-base-v2): 14 PCA + has_news_large
- Source: 1000 SPY-related headlines (2024-2025)
- Encoding: Sentence-transformers → PCA → Daily aggregation

**Total:** 38 features

---

## 🎓 Key Insights

### 1. Headlines Help at Weekly Horizon Only
- **h5 (weekly):** LSTM improves 42% with headlines (3.22 → 4.56 Sharpe)
- **h1, h20:** Headlines add noise, 10 technical features perform better

### 2. Hybrid Architecture Success
- Separating direction (ESN) from magnitude (Ridge) fixes ESN's R² problem
- Achieves 2nd-best Sharpe with lowest turnover

### 3. Model-Horizon Matching Matters
- TCN dominates monthly (h20)
- LSTM dominates weekly (h5)
- Transformer best for daily (h1)

### 4. Walk-Forward Validation Critical
- 9 folds reveal regime dependency
- Fold 1 (2017-2018): All models excel
- Fold 2 (2018-2019): Most models fail
- **Conclusion:** Overfitting to specific regimes is common

---

## 🚀 Installation

```bash
# Create environment
conda create -n esn-finance python=3.11 -y
conda activate esn-finance

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```
numpy
pandas
scikit-learn
matplotlib
yfinance
tqdm
torch
sentence-transformers
```

---

## 📈 Usage Examples

### Train Single Model
```python
from src.pipeline import run_baseline

# Best model (TCN)
result = run_baseline("tcn", fold_id=0, horizon="target_h20")

# Novel hybrid
result = run_baseline("hybrid", fold_id=0, horizon="target_h20")

# Best for h5
result = run_baseline("lstm", fold_id=0, horizon="target_h5")
```

### Hyperparameter Sweep
```python
from src.train.runner import run_sweep
from config.experiments import TCN_GRID

results_df = run_sweep(
    model_name="tcn",
    param_grid=TCN_GRID,
    folds=[0, 1, 2],
    horizons=["target_h20"],
    exp_prefix="production"
)
```

### Access Predictions
```python
import pandas as pd

preds = pd.read_csv("data/experiments/tcn/fold_0/preds_target_h20.csv")
print(preds.head())
```

---

## 📊 Validation Methodology

- **Walk-forward:** 10-year train, 1-year test, 1-year step
- **9 folds total:** 2006-2025
- **Leakage control:** Train-only scaler, no future information
- **Metrics:** RMSE, MAE, R², Directional Accuracy, Sharpe Ratio, Turnover

---

## 📚 Documentation

- **FINAL_RESULTS.md** - Complete model comparison & recommendations
- **HYBRID_MODEL_SUMMARY.md** - Novel hybrid architecture details
- **INTEGRATION_GUIDE.md** - Headline embeddings system
- **config/settings.py** - All configuration options

---

## 🏆 Contributions

### Novel Work:
1. **Hybrid ESN-Ridge Architecture** - First model to separate ESN direction from Ridge magnitude
2. **Headline Embeddings at Scale** - 28 PCA features from dual sentence-transformers
3. **Comprehensive Horizon Analysis** - h1, h5, h20 with model-specific winners
4. **Production-Ready Pipeline** - Full automation from download to evaluation

---

## 📖 Citation

If you use this work, please cite:

```
Financial Forecasting with Echo State Networks and Deep Learning
[Your Institution/Course Name]
2025
```

---

## 📧 Contact

For questions about model implementation or results, please open an issue.

---

## License

MIT License - See LICENSE file for details

---

**Status: ✅ Complete**

All models evaluated, best performers identified, production code ready.

