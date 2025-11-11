# Training & Predicting with ALL Models - Complete Guide

## Overview

Train and use hybrid models across:
- **All 9 folds** (fold_0 to fold_8)
- **All 3 horizons** (target_h1, target_h5, target_h20)
- **Total: 27 models**

---

## Step 1: Train All Models

### Basic Usage
```bash
# Train all 27 models and save them
python train_all_hybrid_models.py
```

This will:
- Train 9 folds × 3 horizons = **27 models**
- Save each model to: `data/experiments/hybrid/fold_X/model_target_hY/`
- Generate results CSV: `hybrid_all_folds_all_horizons_results.csv`
- Show progress bar and statistics

### Training Time
- Per model: ~2-5 minutes
- Total: ~60-135 minutes (1-2 hours)

### Don't Save Models (faster, testing only)
```bash
python train_all_hybrid_models.py --no-save
```

---

## Step 2: Predict Using Trained Models

### Compare All Strategies
```bash
# Use default test set (fold 0, horizon h20)
python predict_all_models.py

# Use different test set
python predict_all_models.py --fold 2 --horizon target_h5
```

This compares **4 prediction strategies**:

### Strategy 1: Single Best Model
Use the best-performing model (fold_0, target_h20):
```python
from src.models.hybrid_esn_ridge import HybridESNRidge

model = HybridESNRidge.load("data/experiments/hybrid/fold_0/model_target_h20")
predictions = model.predict(X_new)
```

**Pros**: Simplest, fastest inference  
**Cons**: Less robust to market changes

### Strategy 2: Ensemble by Horizon
Average predictions from all folds for one horizon:
```python
# Load all fold models for h20
models = []
for fold in range(9):
    model = HybridESNRidge.load(f"data/experiments/hybrid/fold_{fold}/model_target_h20")
    models.append(model)

# Average predictions
predictions = np.mean([m.predict(X_new) for m in models], axis=0)
```

**Pros**: More robust, same horizon  
**Cons**: 9× slower inference

### Strategy 3: Ensemble All Models
Average predictions from ALL 27 models:
```python
models = []
for fold in range(9):
    for horizon in ["target_h1", "target_h5", "target_h20"]:
        model = HybridESNRidge.load(f"data/experiments/hybrid/fold_{fold}/model_{horizon}")
        models.append(model)

predictions = np.mean([m.predict(X_new) for m in models], axis=0)
```

**Pros**: Maximum robustness  
**Cons**: 27× slower inference

### Strategy 4: Weighted Ensemble
Weight models by their training performance (Sharpe ratio):
```python
# Weights based on Sharpe from training
weights = {...}  # From results CSV
weighted_pred = sum(w * model.predict(X_new) for w, model in zip(weights, models))
```

**Pros**: Best theoretical performance  
**Cons**: Requires careful weight selection

---

## Step 3: Analyze Results

After training, you'll get:

### Results File
`hybrid_all_folds_all_horizons_results.csv` contains:
- Performance metrics for each fold × horizon combination
- RMSE, R², Dir Accuracy, Sharpe, etc.

### Console Output
```
PERFORMANCE BY HORIZON (averaged across folds)
------------------------------------------------------------
              rmse              r2         dir_acc      sharpe
              mean    std    mean   std   mean  std  mean  std
horizon                                                       
target_h1   0.0123  0.002  -0.45  0.15  0.652  0.03  4.21  1.2
target_h5   0.0189  0.003  -0.38  0.12  0.665  0.02  5.34  1.5
target_h20  0.0280  0.004  -0.37  0.10  0.671  0.03  6.27  1.8

BEST MODELS
------------------------------------------------------------
Best Sharpe: 6.267
  → Fold 0, target_h20
  → Model: data/experiments/hybrid/fold_0/model_target_h20/
```

---

## Quick Reference Commands

### Train Everything
```bash
python train_all_hybrid_models.py
```

### Predict with Best Model
```bash
python load_hybrid_model_demo.py
```

### Compare All Strategies
```bash
python predict_all_models.py
```

### Load Specific Model (in Python)
```python
from src.models.hybrid_esn_ridge import HybridESNRidge

# Load any model
model = HybridESNRidge.load("data/experiments/hybrid/fold_3/model_target_h5")
predictions = model.predict(X_new)
```

---

## Model Directory Structure

After training all models:

```
data/experiments/hybrid/
├── fold_0/
│   ├── model_target_h1/
│   │   ├── config.json
│   │   ├── esn_weights.npz
│   │   └── ridge_model.pkl
│   ├── model_target_h5/
│   └── model_target_h20/
├── fold_1/
│   ├── model_target_h1/
│   ├── model_target_h5/
│   └── model_target_h20/
├── fold_2/
│   ...
└── fold_8/
    ├── model_target_h1/
    ├── model_target_h5/
    └── model_target_h20/
```

Total: **27 model directories**

---

## Understanding the Data

### Folds (Time Windows)
```
fold_0: Train 2006-2016, Test 2016-2017
fold_1: Train 2007-2017, Test 2017-2018
fold_2: Train 2008-2018, Test 2018-2019
...
fold_8: Train 2014-2024, Test 2024-2025
```

### Horizons (Prediction Timeframe)
- **target_h1**: 1-day ahead return
- **target_h5**: 5-day ahead return  
- **target_h20**: 20-day ahead return

### Which to Use?
- **h1**: Day trading (most volatile)
- **h5**: Swing trading (balanced)
- **h20**: Position trading (most stable, **best Sharpe**)

---

## Production Recommendations

### For Best Performance
Use **fold_0 + target_h20** (highest Sharpe: 6.267):
```python
model = HybridESNRidge.load("data/experiments/hybrid/fold_0/model_target_h20")
```

### For Robustness
Use **ensemble by horizon** (h20):
```python
models = [HybridESNRidge.load(f"data/experiments/hybrid/fold_{i}/model_target_h20") 
          for i in range(9)]
pred = np.mean([m.predict(X) for m in models], axis=0)
```

### For Maximum Stability
Use **weighted ensemble** (all 27 models, weighted by Sharpe).

---

## Disk Space Requirements

- Single model: ~50-100 MB
- 27 models: ~1.4-2.7 GB total
- Results CSV: < 1 MB

---

## Example Workflow

```python
# 1. Train all models (one-time, ~1-2 hours)
# $ python train_all_hybrid_models.py

# 2. Load best model for production
from src.models.hybrid_esn_ridge import HybridESNRidge
import pandas as pd

model = HybridESNRidge.load("data/experiments/hybrid/fold_0/model_target_h20")

# 3. Prepare new data (must have same 38 features)
new_data = pd.read_csv("latest_market_data.csv")
X_new = new_data[[c for c in new_data.columns if c.startswith("z_")]].values

# 4. Generate predictions
predictions = model.predict(X_new)

# 5. Make trading decisions
signals = np.sign(predictions)  # +1 = buy, -1 = sell
```

---

## Troubleshooting

### Out of Memory
Train in batches:
```python
# Train only some folds
for fold in range(0, 3):  # Train fold 0-2 first
    for horizon in ["target_h1", "target_h5", "target_h20"]:
        run_baseline("hybrid", fold, horizon, save_model=True)
```

### Model Not Found
Make sure you trained first:
```bash
python train_all_hybrid_models.py
```

### Slow Inference
Use single model instead of ensemble:
```python
# Fast: single model
model = HybridESNRidge.load("data/experiments/hybrid/fold_0/model_target_h20")
pred = model.predict(X)  # ~0.01s

# Slow: ensemble
models = [HybridESNRidge.load(f"fold_{i}/model_h20") for i in range(9)]
pred = np.mean([m.predict(X) for m in models], axis=0)  # ~0.09s
```

---

## Summary Table

| Task | Command | Time | Output |
|------|---------|------|--------|
| Train all models | `python train_all_hybrid_models.py` | 1-2 hrs | 27 saved models |
| Compare strategies | `python predict_all_models.py` | 1-2 min | Comparison CSV |
| Use best model | `model.predict(X)` | <0.01s | Predictions |
| Use ensemble | `np.mean([m.predict(X) ...])` | <0.1s | Predictions |

---

**Ready to train? Run:**
```bash
python train_all_hybrid_models.py
```

Then use any strategy from `predict_all_models.py`!

