# Production Files Checklist

Guide to what to keep for production vs what was experimental.

## ✅ KEEP for Production

### Core Production Files
- ✅ **`production_predictor.py`** - Main inference code (PRODUCTION READY)
- ✅ **`PRODUCTION_README.md`** - Quick start guide
- ✅ **`PRODUCTION_GUIDE.md`** - Full documentation

### Required Models (3 only)
- ✅ `data/experiments/hybrid/fold_3/model_target_h1/`
- ✅ `data/experiments/hybrid/fold_3/model_target_h20/`
- ✅ `data/experiments/hybrid/fold_8/model_target_h5/`

### Core Library (needed by production_predictor.py)
- ✅ `src/models/hybrid_esn_ridge.py`
- ✅ `src/models/esn.py`
- ✅ `src/models/ridge_readout.py`
- ✅ `config/settings.py`
- ✅ `requirements.txt`

**Total production size**: ~300 MB (3 models) + ~50 KB (code)

---

## 🗑️ OPTIONAL - Experimental/Development

### Training Scripts (not needed for inference)
- ⚠️ `train_all_hybrid_models.py` - Training script
- ⚠️ `evaluate_hybrid_model.py` - Evaluation script
- ⚠️ `main.py` - Original training pipeline

### Comparison/Analysis Tools
- ⚠️ `predict_all_models.py` - Strategy comparison
- ⚠️ `load_hybrid_model_demo.py` - Demo script
- ⚠️ `test_hybrid_save_load.py` - Testing script

### Documentation (for reference)
- ⚠️ `ALL_MODELS_GUIDE.md` - Training all models guide
- ⚠️ `WORKFLOW_VISUAL_GUIDE.md` - Visual workflow
- ⚠️ `HYBRID_MODEL_SAVE_GUIDE.md` - Save/load guide
- ⚠️ `IMPLEMENTATION_SUMMARY.md` - Implementation details
- ⚠️ `QUICK_REFERENCE.md` - Quick reference

### Extra Model Files (24 unused models)
- ⚠️ `data/experiments/hybrid/fold_0/` (except if you want fold_0)
- ⚠️ `data/experiments/hybrid/fold_1/`
- ⚠️ `data/experiments/hybrid/fold_2/`
- ⚠️ `data/experiments/hybrid/fold_4/`
- ⚠️ `data/experiments/hybrid/fold_5/`
- ⚠️ `data/experiments/hybrid/fold_6/`
- ⚠️ `data/experiments/hybrid/fold_7/`
- ⚠️ Other horizons in fold_3/fold_8 not listed above

**Can delete to save**: ~2 GB (24 unused models)

---

## 📦 Minimal Production Package

```
production_deployment/
│
├── production_predictor.py       ← Main code
├── PRODUCTION_README.md           ← Quick start
│
├── src/
│   └── models/
│       ├── hybrid_esn_ridge.py
│       ├── esn.py
│       └── ridge_readout.py
│
├── config/
│   └── settings.py
│
├── data/experiments/hybrid/
│   ├── fold_3/
│   │   ├── model_target_h1/      ← 3 models only
│   │   └── model_target_h20/
│   └── fold_8/
│       └── model_target_h5/
│
└── requirements.txt
```

**Size**: ~300 MB  
**Files**: ~15 files (vs 1000+ in full project)

---

## 🚀 Deployment Options

### Option 1: Full Project (Development)
Keep everything - good for retraining and experimentation.
- **Size**: ~3 GB
- **Use**: Development environment

### Option 2: Minimal Production
Only files listed above in "KEEP for Production".
- **Size**: ~300 MB
- **Use**: Production deployment

### Option 3: Docker (Recommended)
```dockerfile
FROM python:3.12-slim

# Copy only production files
COPY production_predictor.py /app/
COPY src/models/*.py /app/src/models/
COPY config/settings.py /app/config/
COPY data/experiments/hybrid/fold_3/model_target_h* /app/data/experiments/hybrid/fold_3/
COPY data/experiments/hybrid/fold_8/model_target_h5 /app/data/experiments/hybrid/fold_8/

WORKDIR /app
RUN pip install numpy pandas scikit-learn

CMD ["python", "production_predictor.py"]
```

---

## 🔄 Migration Steps

### To Production-Only Setup

1. **Copy production files** to new directory:
```bash
mkdir production_deployment
cp production_predictor.py production_deployment/
cp PRODUCTION_README.md production_deployment/
cp -r src/models production_deployment/src/
cp config/settings.py production_deployment/config/
```

2. **Copy only 3 best models**:
```bash
cp -r data/experiments/hybrid/fold_3/model_target_h1 production_deployment/data/experiments/hybrid/fold_3/
cp -r data/experiments/hybrid/fold_3/model_target_h20 production_deployment/data/experiments/hybrid/fold_3/
cp -r data/experiments/hybrid/fold_8/model_target_h5 production_deployment/data/experiments/hybrid/fold_8/
```

3. **Test**:
```bash
cd production_deployment
python production_predictor.py
```

4. **Done** - Deploy the `production_deployment/` folder

---

## ✅ Verification Checklist

After setting up production:

- [ ] `production_predictor.py` exists
- [ ] Can import: `from production_predictor import ProductionPredictor`
- [ ] Models load without errors
- [ ] `predictor.predict(X_test, horizon='h20')` works
- [ ] All 3 models accessible: h1, h5, h20
- [ ] Demo runs successfully: `python production_predictor.py`

---

## 📊 Space Savings

| Setup | Size | Models | Purpose |
|-------|------|--------|---------|
| **Full project** | 3 GB | 27 | Development |
| **Production minimal** | 300 MB | 3 | Deployment |
| **Docker image** | 200 MB | 3 | Cloud deployment |

**Recommendation**: Use minimal production setup or Docker for deployment.

---

## 🎯 What You Need

**For inference only (production)**:
```
✅ production_predictor.py
✅ 3 model folders
✅ src/models/ (3 files)
✅ config/settings.py
✅ requirements.txt
```

**Everything else is optional** - keep for development/retraining if needed.

---

## 🚀 Quick Deploy

```bash
# 1. Test production code works
python production_predictor.py

# 2. Copy to production server (only needed files)
scp -r production_deployment/ user@server:/app/

# 3. Run on server
ssh user@server "cd /app && python production_predictor.py"
```

**Done!** Production deployment with only essential files.

