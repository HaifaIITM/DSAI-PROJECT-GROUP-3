# 🚀 START HERE - Production Hybrid Model

## For Production Use

**You only need 1 file:**

### ⭐ `production_predictor.py`

```python
from production_predictor import ProductionPredictor

predictor = ProductionPredictor()
predictions = predictor.predict(X_new, horizon='h20')
```

**That's it!**

---

## Documentation (Choose Based on Need)

### 🎯 I want to USE the model (Production)
**Read**: `PRODUCTION_README.md`  
**File**: `production_predictor.py`  
**Test**: `python production_predictor.py`

### 🔧 I want to RETRAIN models
**Location**: `docs/development/train_all_hybrid_models.py`  
**Run**: `python docs/development/train_all_hybrid_models.py`

### 📖 I want REFERENCE docs
**Location**: `docs/experimental/`  
**Files**: Implementation details, guides, etc.

---

## Quick Decision Tree

```
What do you need?
│
├─ Use model for predictions? 
│  └─ Read: PRODUCTION_README.md
│     Use: production_predictor.py
│
├─ Retrain models?
│  └─ Go to: docs/development/
│
└─ Understand implementation?
   └─ Go to: docs/experimental/
```

---

## File Structure (Clean)

```
Root (PRODUCTION ONLY):
├── production_predictor.py         ⭐ MAIN FILE
├── PRODUCTION_README.md            📖 Start here
├── PRODUCTION_GUIDE.md             📚 Full docs
└── PRODUCTION_FILES_CHECKLIST.md  📋 Organization

docs/development/ (RETRAINING):
├── train_all_hybrid_models.py
├── evaluate_hybrid_model.py
└── ... other training scripts

docs/experimental/ (REFERENCE):
├── ALL_MODELS_GUIDE.md
├── IMPLEMENTATION_SUMMARY.md
└── ... other guides

data/experiments/hybrid/ (MODELS):
├── fold_3/model_target_h1/         ✅ Best h1
├── fold_3/model_target_h20/        ✅ Best h20 ⭐
└── fold_8/model_target_h5/         ✅ Best h5
```

---

## 🎯 Most Common Use Case

```python
# 1. Import
from production_predictor import ProductionPredictor

# 2. Initialize (once)
predictor = ProductionPredictor()

# 3. Predict (fast, repeated)
predictions = predictor.predict(X_new, horizon='h20')
signals = predictor.get_signals(X_new, horizon='h20')

# Done!
```

---

## Best Model

**h20 (Sharpe 6.81)** - Position trading, most stable

```python
predictions = predictor.predict(X_new, horizon='h20')
```

---

## Test It Now

```bash
python production_predictor.py
```

See predictions and signals in action!

---

**Next**: Read `PRODUCTION_README.md` or just use `production_predictor.py`

