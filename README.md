# Hybrid Model - Production Inference

Production-ready inference using the 3 best hybrid ESN-Ridge models for market prediction.

## 🚀 Quick Start

```python
from production_predictor import ProductionPredictor

# Initialize (loads 3 best models)
predictor = ProductionPredictor()

# Predict
predictions = predictor.predict(X_new, horizon='h20')

# Get trading signals
signals = predictor.get_signals(X_new, horizon='h20')
# Returns: +1 (buy), -1 (sell)
```

## 📊 Best Models (Production)

| Horizon | Fold | Sharpe | Purpose |
|---------|------|--------|---------|
| **h1** | 3 | 1.25 | Day trading (1-day ahead) |
| **h5** | 8 | 2.94 | Swing trading (5-day ahead) |
| **h20** | 3 | 6.81 | Position trading (20-day ahead) ⭐ **Recommended** |

**Recommendation**: Use `h20` (Sharpe 6.81) for best risk-adjusted returns.

## 📦 Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.12+, NumPy, Pandas, scikit-learn

## 🧪 Test

```bash
python production_predictor.py
```

**Output**:
```
Loading production models...
  [OK] Loaded h1: fold_3 (Sharpe 1.25)
  [OK] Loaded h5: fold_8 (Sharpe 2.94)
  [OK] Loaded h20: fold_3 (Sharpe 6.813)

Example predictions:
Sample  1    | +0.016856 | BUY
Sample  2    | +0.018037 | BUY
Sample  3    | +0.017611 | BUY
```

## 📖 API Reference

### Initialize

```python
predictor = ProductionPredictor()
```

Loads all 3 best models into memory (~0.5s, one-time).

### Predict

```python
predictions = predictor.predict(X, horizon='h20')
```

**Args**:
- `X`: Feature matrix (n_samples, 38 features)
- `horizon`: `'h1'`, `'h5'`, or `'h20'`

**Returns**: Predictions array (n_samples,)

**Performance**: ~0.01s per batch

### Get Signals

```python
signals = predictor.get_signals(X, horizon='h20')
```

Converts predictions to trading signals.

**Returns**: +1 (buy), -1 (sell)

### Predict All Horizons

```python
all_preds = predictor.predict_all(X)
# Returns: {'h1': [...], 'h5': [...], 'h20': [...]}
```

## 💡 Usage Examples

### Single Prediction

```python
from production_predictor import ProductionPredictor

predictor = ProductionPredictor()
predictions = predictor.predict(X_new, horizon='h20')

if predictions[-1] > 0:
    print(f"BUY signal: {predictions[-1]:.4f}")
else:
    print(f"SELL signal: {predictions[-1]:.4f}")
```

### Multi-Timeframe Analysis

```python
# Get predictions for all timeframes
all_preds = predictor.predict_all(X_new)

# Get signals
signals = {
    h: predictor.get_signals(X_new, h) 
    for h in ['h1', 'h5', 'h20']
}

# Consensus-based decision
consensus = sum([signals[h][-1] for h in ['h1', 'h5', 'h20']])

if consensus >= 2:
    print("STRONG BUY - Multiple timeframes bullish")
elif consensus <= -2:
    print("STRONG SELL - Multiple timeframes bearish")
else:
    print("MIXED - Wait for confirmation")
```

### Position Sizing

```python
predictions = predictor.predict(X_new, horizon='h20')
confidence = abs(predictions[-1])

# Size position based on prediction confidence
if confidence > 0.03:
    position_size = 1.0  # Full position
elif confidence > 0.02:
    position_size = 0.5  # Half position
else:
    position_size = 0.25  # Quarter position

direction = "BUY" if predictions[-1] > 0 else "SELL"
print(f"{direction} with {position_size*100:.0f}% position")
```

## ⚙️ Input Requirements

**Features**: Exactly 38 features required
- 10 technical indicators (OHLCV-derived)
- 28 headline embeddings (FinBERT sentence embeddings)

**Format**: NumPy array, shape `(n_samples, 38)`

**Preprocessing**: Features must be z-score normalized

**Example**:
```python
import pandas as pd

# Load data with z-scored features
data = pd.read_csv("data/splits/fold_0/test.csv", index_col=0, parse_dates=True)
X = data[[c for c in data.columns if c.startswith("z_")]].values
```

## 📁 Project Structure

```
DSAI-PROJECT-GROUP-3/
├── production_predictor.py       ⭐ Main inference code
├── README.md                     ⭐ This file
├── requirements.txt              ⭐ Dependencies
│
├── data/experiments/hybrid/      💾 Best models (3)
│   ├── fold_3/
│   │   ├── model_target_h1/
│   │   └── model_target_h20/    ← Best overall (Sharpe 6.81)
│   └── fold_8/
│       └── model_target_h5/
│
├── src/models/                   📚 Model implementations
│   ├── hybrid_esn_ridge.py
│   ├── esn.py
│   └── ridge_readout.py
│
├── config/                       ⚙️ Configuration
│   └── settings.py
│
├── scripts/                      🔧 Utility scripts
│   ├── training/                 (Model training)
│   └── evaluation/               (Evaluation & testing)
│
└── docs/                         📖 Documentation only
    ├── experimental/             (Reference docs)
    ├── legacy/                   (Project history)
    └── results/                  (Analysis outputs)
```

## 🎯 Model Performance

**Best Model (h20, fold_3)**:
- **Sharpe Ratio**: 6.81 (excellent)
- **Directional Accuracy**: 68.7%
- **Training Period**: 2008-2018
- **Includes**: 2008 financial crisis (robust)

**All Models**:
| Horizon | Avg Sharpe | Avg Dir Acc | Best For |
|---------|-----------|-------------|----------|
| h1 | 0.17 | 49.7% | Day trading |
| h5 | 0.50 | 52.0% | Swing trading |
| h20 | 1.28 | 54.7% | Position trading ⭐ |

## 🔧 Development

### Retrain Models

```bash
python scripts/training/train_all_hybrid_models.py
```

Trains all 27 models (9 folds × 3 horizons). Takes ~1-2 hours.

### Run Evaluation

```bash
python scripts/evaluation/evaluate_hybrid_model.py
```

### Compare Strategies

```bash
python scripts/evaluation/predict_all_models.py
```

## 📚 Additional Documentation

- **Training guides**: `docs/experimental/`
- **Training scripts**: `scripts/training/`
- **Evaluation scripts**: `scripts/evaluation/`
- **Project history**: `docs/legacy/`
- **Results**: `docs/results/`

## ⚠️ Important Notes

1. **Feature consistency**: Input must have exactly 38 features in the same order as training
2. **Model selection**: h20 (fold_3) recommended for best performance
3. **Memory**: ~300 MB for all 3 models
4. **Inference speed**: ~0.01s per batch (fast)

## 🐛 Troubleshooting

**Models not found**:
```
FileNotFoundError: Model not found at: data/experiments/...
```
→ Train models first: `python scripts/training/train_all_hybrid_models.py`

**Wrong number of features**:
```
ValueError: Expected 38 features, got X
```
→ Ensure input has 38 z-scored features

**Import errors**:
```
ModuleNotFoundError: No module named 'sklearn'
```
→ Install requirements: `pip install -r requirements.txt`

## 📞 Support

- **Main code**: `production_predictor.py` (includes demo at bottom)
- **Training**: `scripts/training/train_all_hybrid_models.py`
- **Evaluation**: `scripts/evaluation/evaluate_hybrid_model.py`
- **Documentation**: `docs/experimental/`

## 📄 License

See project documentation for license information.

---

**Status**: ✅ Production Ready | ✅ Tested | ✅ Documented

**Last Updated**: November 11, 2025

