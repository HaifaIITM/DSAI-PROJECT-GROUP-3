# Hybrid Model Save/Load - Quick Reference

## 🚀 Quick Start (2 Steps)

### Step 1: Train & Save
```bash
python evaluate_hybrid_model.py
```
Model saved to: `data/experiments/hybrid/fold_0/model_target_h20/`

### Step 2: Load & Use
```python
from src.models.hybrid_esn_ridge import HybridESNRidge

model = HybridESNRidge.load("data/experiments/hybrid/fold_0/model_target_h20")
predictions = model.predict(X_new)
```

---

## 📝 API Cheat Sheet

### Save Model
```python
model.save("path/to/save/directory")
```

### Load Model
```python
model = HybridESNRidge.load("path/to/saved/model")
```

### In Training Pipeline
```python
from src.pipeline import run_baseline

result = run_baseline(
    model_name="hybrid",
    fold_id=0,
    horizon="target_h20",
    save_model=True  # ← Add this
)
```

---

## 📂 File Structure

```
data/experiments/hybrid/fold_0/model_target_h20/
├── config.json          # Hyperparameters
├── esn_weights.npz      # ESN weights (~50-100 MB)
└── ridge_model.pkl      # Ridge model (~1 MB)
```

---

## 🧪 Test It

```bash
# Quick test (synthetic data)
python test_hybrid_save_load.py

# Full demo (real data)
python load_hybrid_model_demo.py
```

---

## 📚 Documentation

- **Full Guide**: `HYBRID_MODEL_SAVE_GUIDE.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Examples**: `load_hybrid_model_demo.py`

---

## ✅ What Works

- ✓ Save trained models to disk
- ✓ Load models for inference (no retraining needed)
- ✓ Predictions match exactly (0 difference)
- ✓ Works with all horizons (h1, h5, h20)
- ✓ Works with all folds (0-8)
- ✓ Backward compatible

---

## 💡 Common Use Cases

### 1. Production Deployment
```python
# Train once
model.fit(X_train, y_train)
model.save("production_model")

# Deploy - load at startup
model = HybridESNRidge.load("production_model")

# Inference (fast, no retraining)
predictions = model.predict(new_data)
```

### 2. Model Comparison
```python
model_v1 = HybridESNRidge.load("models/config_1")
model_v2 = HybridESNRidge.load("models/config_2")

pred_v1 = model_v1.predict(X_test)
pred_v2 = model_v2.predict(X_test)
```

### 3. Ensemble
```python
models = [HybridESNRidge.load(f"fold_{i}/model") for i in range(9)]
predictions = np.mean([m.predict(X) for m in models], axis=0)
```

---

## 🔧 Key Parameters

When saving in `run_baseline()` or `run_experiment()`:
```python
save_model=True  # Enable saving (default: False)
```

When saving manually:
```python
model.save(save_dir="path/to/directory")
```

When loading:
```python
model = HybridESNRidge.load(save_dir="path/to/directory")
```

---

## 🎯 Status

**Implementation**: ✅ Complete  
**Testing**: ✅ All tests passing  
**Documentation**: ✅ Complete  

Ready to use!

