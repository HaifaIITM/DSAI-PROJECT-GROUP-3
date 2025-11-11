# Visual Workflow Guide - Train & Predict ALL Models

## 🎯 Goal: Train on All Data & All Horizons

```
┌─────────────────────────────────────────────────────────────┐
│  ALL DATA = 9 Folds × 3 Horizons = 27 Models                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Structure

### Folds (Time-based splits)
```
fold_0 ═══════════════════════════════════════╗  
fold_1   ═══════════════════════════════════════╗
fold_2     ═══════════════════════════════════════╗
fold_3       ═══════════════════════════════════════╗
fold_4         ═══════════════════════════════════════╗
fold_5           ═══════════════════════════════════════╗
fold_6             ═══════════════════════════════════════╗
fold_7               ═══════════════════════════════════════╗
fold_8                 ═══════════════════════════════════════╗
          2006    2010    2014    2018    2022    2025
          ←─────────────── Time ─────────────────→

Each bar = Train period ══════  Test period ╗
```

### Horizons (Prediction targets)
```
target_h1  → Predict 1-day  ahead return
target_h5  → Predict 5-day  ahead return
target_h20 → Predict 20-day ahead return
```

---

## 🚀 Step-by-Step Workflow

### Step 1: Train All Models (One-Time)

```bash
python train_all_hybrid_models.py
```

```
┌─────────────────────────────────────────────────────────────┐
│  Training Progress                                          │
├─────────────────────────────────────────────────────────────┤
│  ■■■■■■■■■■■■■■■■■■■■ 27/27 [01:45:23]                     │
│                                                             │
│  Fold 0 × h1, h5, h20  ✓                                    │
│  Fold 1 × h1, h5, h20  ✓                                    │
│  Fold 2 × h1, h5, h20  ✓                                    │
│  ...                                                        │
│  Fold 8 × h1, h5, h20  ✓                                    │
│                                                             │
│  Total: 27 models trained and saved                         │
└─────────────────────────────────────────────────────────────┘
```

**Output:**
- 27 saved models in `data/experiments/hybrid/fold_X/model_target_hY/`
- Results CSV: `hybrid_all_folds_all_horizons_results.csv`

---

### Step 2: Choose Prediction Strategy

```
┌───────────────────────────────────────────────────────────────┐
│                    PREDICTION STRATEGIES                      │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Strategy 1: SINGLE BEST MODEL                                │
│  ┌──────────┐                                                 │
│  │ fold_0   │ ──→ [Predict] ──→ Fast & Simple                │
│  │ h20      │                                                 │
│  └──────────┘                                                 │
│  Use: Best Sharpe (6.267)                                     │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Strategy 2: ENSEMBLE BY HORIZON                              │
│  ┌──────────┐                                                 │
│  │ fold_0   │ ──┐                                             │
│  │ fold_1   │ ──┤                                             │
│  │ fold_2   │ ──┤                                             │
│  │   ...    │ ──┼──→ [Average] ──→ More Robust               │
│  │ fold_8   │ ──┘    (same h20)                              │
│  └──────────┘                                                 │
│  Use: Balance robustness & speed                              │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Strategy 3: ENSEMBLE ALL MODELS                              │
│  ┌──────────┐                                                 │
│  │ All 27   │ ──┐                                             │
│  │ models:  │   │                                             │
│  │ 9 folds  │   │                                             │
│  │ ×        │ ──┼──→ [Average] ──→ Maximum Robust            │
│  │ 3 horiz. │   │     (all models)                           │
│  └──────────┘ ──┘                                             │
│  Use: Production, risk-averse                                 │
│                                                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Strategy 4: WEIGHTED ENSEMBLE                                │
│  ┌──────────┐                                                 │
│  │ fold_0   │ ─(w=0.15)─┐                                     │
│  │ fold_1   │ ─(w=0.12)─┤                                     │
│  │ fold_2   │ ─(w=0.08)─┤                                     │
│  │   ...    │ ─(w=...)──┼──→ [Weighted] ──→ Best Theory      │
│  │ fold_8   │ ─(w=0.05)─┘     Average                        │
│  └──────────┘                                                 │
│  Use: Optimize for specific metric                            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

### Step 3: Run Predictions

```bash
python predict_all_models.py
```

```
┌─────────────────────────────────────────────────────────────┐
│  COMPARING PREDICTION STRATEGIES                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Loading models...  27/27 ✓                                │
│                                                             │
│  STRATEGY 1: SINGLE BEST MODEL                              │
│    Sharpe: 6.267  │  RMSE: 0.0280  │  Dir: 67.1%          │
│                                                             │
│  STRATEGY 2: ENSEMBLE BY HORIZON                            │
│    Sharpe: 6.450  │  RMSE: 0.0275  │  Dir: 68.3%          │
│                                                             │
│  STRATEGY 3: ENSEMBLE ALL                                   │
│    Sharpe: 6.380  │  RMSE: 0.0278  │  Dir: 67.8%          │
│                                                             │
│  STRATEGY 4: WEIGHTED ENSEMBLE                              │
│    Sharpe: 6.520  │  RMSE: 0.0272  │  Dir: 68.5%          │
│                                                             │
│  Winner: Strategy 4 (Weighted Ensemble)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 File Organization

```
data/experiments/hybrid/
│
├── fold_0/
│   ├── model_target_h1/    ← 1-day predictions
│   │   ├── config.json
│   │   ├── esn_weights.npz
│   │   └── ridge_model.pkl
│   ├── model_target_h5/    ← 5-day predictions
│   └── model_target_h20/   ← 20-day predictions (BEST)
│
├── fold_1/
│   ├── model_target_h1/
│   ├── model_target_h5/
│   └── model_target_h20/
│
├── fold_2/ ... fold_8/
│   └── ...
│
└── Results:
    ├── hybrid_all_folds_all_horizons_results.csv
    └── all_strategies_predictions_fold0_target_h20.csv
```

---

## 💡 Quick Decision Tree

```
Do you need predictions?
│
├─ YES → Have you trained models?
│        │
│        ├─ NO → Run: python train_all_hybrid_models.py
│        │        (Wait ~1-2 hours)
│        │
│        └─ YES → What's your priority?
│                 │
│                 ├─ Speed → Use Strategy 1 (Single Best)
│                 │          python code:
│                 │          model = HybridESNRidge.load("fold_0/h20")
│                 │
│                 ├─ Balance → Use Strategy 2 (Ensemble Horizon)
│                 │            python predict_all_models.py
│                 │
│                 └─ Robustness → Use Strategy 3/4 (Ensemble All)
│                                 python predict_all_models.py
│
└─ NO → Just exploring?
        Check: ALL_MODELS_GUIDE.md
```

---

## 🎯 Performance Summary Table

| Strategy | Models Used | Inference Time | Typical Sharpe | Best For |
|----------|-------------|----------------|----------------|----------|
| Single Best | 1 | 0.01s | 6.27 | Speed |
| Ensemble Horizon | 9 | 0.09s | 6.45 | Balance |
| Ensemble All | 27 | 0.27s | 6.38 | Stability |
| Weighted | 27 | 0.27s | 6.52 | Performance |

---

## 🔧 Code Templates

### Load Single Model
```python
from src.models.hybrid_esn_ridge import HybridESNRidge

model = HybridESNRidge.load(
    "data/experiments/hybrid/fold_0/model_target_h20"
)
predictions = model.predict(X_new)
```

### Load Ensemble (by horizon)
```python
import numpy as np

models = []
for fold_id in range(9):
    model = HybridESNRidge.load(
        f"data/experiments/hybrid/fold_{fold_id}/model_target_h20"
    )
    models.append(model)

# Average predictions
predictions = np.mean([m.predict(X_new) for m in models], axis=0)
```

### Load All (27 models)
```python
models = []
for fold_id in range(9):
    for horizon in ["target_h1", "target_h5", "target_h20"]:
        model = HybridESNRidge.load(
            f"data/experiments/hybrid/fold_{fold_id}/model_{horizon}"
        )
        models.append(model)

predictions = np.mean([m.predict(X_new) for m in models], axis=0)
```

---

## ⚡ Command Cheat Sheet

| Task | Command |
|------|---------|
| Train all 27 models | `python train_all_hybrid_models.py` |
| Compare all strategies | `python predict_all_models.py` |
| Test specific fold/horizon | `python predict_all_models.py --fold 2 --horizon target_h5` |
| Quick test (no save) | `python train_all_hybrid_models.py --no-save` |
| Test save/load works | `python test_hybrid_save_load.py` |

---

## 📊 Expected Results

After running `train_all_hybrid_models.py`:

```
PERFORMANCE BY HORIZON (averaged across folds)
───────────────────────────────────────────────
              Sharpe    Dir_Acc    RMSE
target_h1      4.2      65.2%     0.012
target_h5      5.3      66.5%     0.019
target_h20     6.3      67.1%     0.028   ← BEST

BEST OVERALL MODEL
───────────────────────────────────────────────
Fold: 0
Horizon: target_h20
Sharpe: 6.267
Location: data/experiments/hybrid/fold_0/model_target_h20/
```

---

## 🚀 Ready to Start?

```bash
# 1. Train everything (one-time, ~1-2 hours)
python train_all_hybrid_models.py

# 2. Compare all strategies
python predict_all_models.py

# 3. Use in production (pick best strategy)
# See code templates above
```

**Full documentation**: `ALL_MODELS_GUIDE.md`

