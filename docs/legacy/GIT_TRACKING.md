# Git Tracking Guide

## What's Tracked (Committed to Repository)

### Source Code ✅
- `src/` - All Python modules (models, pipeline, eval, etc.)
- `config/` - Configuration files
- `main.py` - Main pipeline script
- `evaluate_hybrid_model.py` - Evaluation script

### Documentation ✅
- `README_FINAL.md` - Main documentation
- `FINAL_RESULTS.md` - Complete performance analysis
- `HYBRID_MODEL_SUMMARY.md` - Hybrid model documentation
- `INTEGRATION_GUIDE.md` - Headline embeddings guide
- `GIT_TRACKING.md` - This file
- `docs/` - Milestone PDFs

### Data (Minimal) ✅
- `spy_news.csv` - Headline data (480KB, needed for reproduction)
- `data/*/.gitkeep` - Directory structure markers
- `architecture/architecture.png` - Architecture diagram

### Configuration ✅
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

---

## What's Ignored (Not Committed)

### Large Data Files ❌
```
data/raw/*.csv              # Downloaded OHLCV data (~5MB+)
data/processed/*.csv        # Feature-engineered datasets (~10MB+)
data/splits/**/*.csv        # Train/test splits (~20MB+)
data/experiments/**/*       # Model predictions & metrics (~50MB+)
```

**Why:** Large files, can be regenerated with `python main.py`

### Generated Results ❌
```
baseline_comparison_*.csv
hybrid_evaluation_*.csv
hybrid_predictions_*.csv
strategy_comparison_*.csv
*.png (except architecture diagrams)
```

**Why:** Temporary analysis outputs, regenerated during evaluation

### Python Cache ❌
```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
```

**Why:** Build artifacts, automatically regenerated

### Environment ❌
```
venv/
env/
.env
```

**Why:** Local development environments, each user creates their own

### IDE Files ❌
```
.vscode/
.idea/
.ipynb_checkpoints/
```

**Why:** Editor-specific settings

---

## How to Reproduce Results

Since data files are ignored, reproduce the full pipeline:

```bash
# 1. Clone repository
git clone <repository-url>
cd DSAI-PROJECT-GROUP-3

# 2. Install dependencies
conda create -n esn-finance python=3.11 -y
conda activate esn-finance
pip install -r requirements.txt

# 3. Run full pipeline (downloads data, trains models)
python main.py

# 4. Evaluate best models
python evaluate_hybrid_model.py
```

**Output:**
- `data/raw/` - Downloaded OHLCV CSVs (~5MB)
- `data/processed/` - Feature datasets with 38 features (~10MB)
- `data/splits/` - 9 walk-forward folds (~20MB)
- `data/experiments/` - Model predictions & metrics (~50MB)

---

## Repository Size

**Tracked (in Git):**
- Source code: ~100KB
- Documentation: ~50KB
- Headlines CSV: ~480KB
- **Total: <1MB**

**Generated (Local Only):**
- Data files: ~80MB
- Model outputs: ~50MB
- **Total: ~130MB**

**Why this approach:**
- Keeps repository lightweight (<1MB)
- Anyone can reproduce full results
- No storage of redundant/regenerable data

---

## Special Cases

### Included Despite Being .csv
```
spy_news.csv  ✅ Tracked
```
**Why:** External data (not regenerable), small size (480KB), required for reproduction

### Excluded Despite Being Code
```
Untitled*.ipynb  ❌ Ignored (temporary notebooks)
test_*.py        ❌ Ignored (experimental scripts, if created)
```
**Why:** Temporary/exploratory code not part of final system

---

## For Collaborators

### Before Committing
1. Run `git status` to check what's being committed
2. Ensure no large data files are staged
3. Check that only source code and documentation are included

### After Cloning
1. Run `python main.py` to generate all data files
2. Verify `data/` directories populate correctly
3. All models should reproduce documented results

---

## Git Commands Quick Reference

```bash
# Check repository status
git status

# See what's ignored
git status --ignored

# Check repository size
git count-objects -vH

# Add new files (respects .gitignore)
git add .
git commit -m "Your message"
git push

# Clean ignored files (careful!)
git clean -Xn  # Preview what would be removed
git clean -Xf  # Actually remove ignored files
```

---

**Last Updated:** Final project completion
**Repository Size:** <1MB tracked, ~130MB local (regenerable)

