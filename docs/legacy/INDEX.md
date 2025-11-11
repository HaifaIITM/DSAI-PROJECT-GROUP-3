# Project File Index

## 🎯 PRODUCTION FILES (Use These)

### Essential Production Files
| File | Purpose | Priority |
|------|---------|----------|
| **START_HERE.md** | Navigation guide | ⭐ Read first |
| **production_predictor.py** | Main inference code | ⭐ Use this |
| **PRODUCTION_README.md** | Quick reference | ⭐ Production docs |
| **PRODUCTION_GUIDE.md** | Complete API guide | 📖 Details |
| **PRODUCTION_FILES_CHECKLIST.md** | File organization | 📋 Reference |
| **README_PRODUCTION.md** | Production overview | 📖 Alternative start |
| **DOCUMENTATION_CLEANUP_SUMMARY.md** | This cleanup summary | ℹ️ Info |

### Required Data
- **data/experiments/hybrid/fold_3/model_target_h1/** (h1 model)
- **data/experiments/hybrid/fold_3/model_target_h20/** (h20 model - best)
- **data/experiments/hybrid/fold_8/model_target_h5/** (h5 model)

### Required Code
- **src/models/hybrid_esn_ridge.py** (model class)
- **src/models/esn.py** (ESN implementation)
- **src/models/ridge_readout.py** (Ridge implementation)
- **config/settings.py** (configuration)

---

## 🛠️ DEVELOPMENT FILES (For Retraining)

**Location**: `docs/development/`

| File | Purpose |
|------|---------|
| **train_all_hybrid_models.py** | Train all 27 models |
| **predict_all_models.py** | Compare strategies |
| **evaluate_hybrid_model.py** | Model evaluation |
| **load_hybrid_model_demo.py** | Demo script |
| **test_hybrid_save_load.py** | Save/load testing |

---

## 📖 REFERENCE DOCUMENTATION (Read Only)

**Location**: `docs/experimental/`

| File | Content |
|------|---------|
| **ALL_MODELS_GUIDE.md** | Training all models guide |
| **WORKFLOW_VISUAL_GUIDE.md** | Visual workflow diagrams |
| **HYBRID_MODEL_SAVE_GUIDE.md** | Save/load implementation |
| **IMPLEMENTATION_SUMMARY.md** | Technical implementation |
| **QUICK_REFERENCE.md** | Quick reference card |

---

## 📄 PROJECT DOCUMENTATION (Reports)

**Location**: `docs/`

- Milestone-1.pdf through Milestone-5.pdf
- Milestone 4 - Intro.pdf

---

## ⚠️ LEGACY/OTHER FILES IN ROOT (Not Production)

### Original Project Files
| File | Status | Notes |
|------|--------|-------|
| README.md | Original | Original project README |
| README_FINAL.md | Legacy | Previous final README |
| main.py | Training | Original training script |
| main.ipynb | Notebook | Jupyter notebook version |

### Analysis/Results Files
| File | Status | Notes |
|------|--------|-------|
| hybrid_all_folds_all_horizons_results.csv | Results | All models results |
| hybrid_evaluation_fold0_target_h20.csv | Results | Evaluation results |
| FINAL_RESULTS.md | Summary | Final results summary |
| HYBRID_MODEL_SUMMARY.md | Summary | Hybrid model summary |

### Other Documentation
| File | Status | Notes |
|------|--------|-------|
| INTEGRATION_GUIDE.md | Legacy | Integration guide |
| GIT_TRACKING.md | Info | Git tracking info |
| requirements.txt | Config | Dependencies |

### Data Files
| File | Status | Notes |
|------|--------|-------|
| spy_news.csv | Data | News headlines data |

---

## 🎯 Quick Decision Guide

### I want to USE the model (Production)
**Read**: `START_HERE.md` → `PRODUCTION_README.md`  
**Use**: `production_predictor.py`  
**Ignore**: Everything else

### I want to RETRAIN models
**Go to**: `docs/development/`  
**Run**: `python docs/development/train_all_hybrid_models.py`

### I want to UNDERSTAND implementation
**Go to**: `docs/experimental/`  
**Start**: `IMPLEMENTATION_SUMMARY.md`

### I want PROJECT REPORTS
**Go to**: `docs/`  
**Read**: `Milestone-*.pdf`

---

## 📊 File Count Summary

| Category | Location | Count | Purpose |
|----------|----------|-------|---------|
| **Production** | Root | 7 | Use these ⭐ |
| **Development** | docs/development/ | 5 | Retraining |
| **Reference** | docs/experimental/ | 5 | Documentation |
| **Legacy** | Root | ~10 | Original project |
| **Results** | Root | 4 | Analysis outputs |
| **Models** | data/experiments/ | 3 | Inference |

**Recommendation**: Focus on the 7 production files in root, ignore everything else for deployment.

---

## ✅ Clean Production Deployment

**Only copy these for production**:

```
production/
├── production_predictor.py
├── PRODUCTION_README.md
├── src/models/
│   ├── hybrid_esn_ridge.py
│   ├── esn.py
│   └── ridge_readout.py
├── config/
│   └── settings.py
└── data/experiments/hybrid/
    ├── fold_3/model_target_h1/
    ├── fold_3/model_target_h20/
    └── fold_8/model_target_h5/
```

**Size**: ~300 MB  
**Files**: ~15 total

Everything else is **optional** for production.

---

## 🚀 Next Steps

1. **Read**: `START_HERE.md`
2. **Test**: `python production_predictor.py`
3. **Use**: Import and predict!

---

**Last Updated**: November 11, 2025  
**Status**: ✅ Organized | ✅ Documented | ✅ Production Ready

