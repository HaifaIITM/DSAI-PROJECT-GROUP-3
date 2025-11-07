# NLP + ESN Integration - Notebook Cells

Add these cells to your Jupyter notebook to use NLP risk index with ESN:

## Cell 1: Enable NLP Features

```python
# Enable NLP risk index feature
from config import settings
settings.NLP_ENABLED = True
settings.NLP_TICKER = "SPY"
settings.NLP_LOOKBACK_DAYS = 365
print("✓ NLP features enabled")
```

## Cell 2: Process with NLP

```python
# This will automatically generate NLP risk index
proc_paths = run_process()
print("✓ Features processed (including risk_index)")
```

## Cell 3: Build Splits with NLP

```python
# Rebuild splits to include risk_index in z-scored features
folds = run_build_splits(proc_paths)
run_materialize_folds(proc_paths, folds)
print("✓ Splits materialized with z_risk_index feature")
```

## Cell 4: Train ESN with NLP

```python
# Train ESN (now includes risk_index as a feature)
result_esn_nlp = run_baseline(model_name="esn", fold_id=0, horizon="target_h1")
print(f"Sharpe Ratio: {result_esn_nlp['backtest']['sharpe']:.3f}")
print(f"Dir. Accuracy: {result_esn_nlp['dir_acc']:.3f}")
```

## Cell 5: Compare With/Without NLP

```python
# Disable NLP and retrain for comparison
settings.NLP_ENABLED = False
proc_paths_baseline = run_process()
folds_baseline = run_build_splits(proc_paths_baseline)
run_materialize_folds(proc_paths_baseline, folds_baseline)
result_esn_baseline = run_baseline(model_name="esn", fold_id=0, horizon="target_h1")

# Compare
import pandas as pd
comparison = pd.DataFrame([
    {"Model": "ESN (baseline)", "Sharpe": result_esn_baseline['backtest']['sharpe'], 
     "Dir.Acc": result_esn_baseline['dir_acc'], "RMSE": result_esn_baseline['rmse']},
    {"Model": "ESN + NLP", "Sharpe": result_esn_nlp['backtest']['sharpe'],
     "Dir.Acc": result_esn_nlp['dir_acc'], "RMSE": result_esn_nlp['rmse']}
])
display(comparison)
```

## Cell 6: Verify Risk Index Feature

```python
# Check that risk_index is in the feature matrix
import pandas as pd
train = pd.read_csv("data/splits/fold_0/train.csv", index_col=0)
test = pd.read_csv("data/splits/fold_0/test.csv", index_col=0)

z_cols = [c for c in train.columns if c.startswith("z_")]
print(f"Number of features: {len(z_cols)}")
print(f"Features: {z_cols}")
print(f"✓ z_risk_index included: {'z_risk_index' in z_cols}")

# Show risk index statistics
if 'z_risk_index' in train.columns:
    print(f"\nRisk Index Stats (train):")
    print(train['z_risk_index'].describe())
```

## Full Example in One Cell

```python
from config import settings
from src.pipeline import run_download, run_process, run_build_splits, run_materialize_folds, run_baseline

# Enable NLP
settings.NLP_ENABLED = True

# Run pipeline
run_download()
proc_paths = run_process()
folds = run_build_splits(proc_paths)
run_materialize_folds(proc_paths, folds)

# Train ESN with NLP features
result = run_baseline(model_name="esn", fold_id=0, horizon="target_h1")

# Display results
print(f"\n{'='*60}")
print(f"ESN + NLP Risk Index Results")
print(f"{'='*60}")
print(f"Sharpe Ratio: {result['backtest']['sharpe']:.3f}")
print(f"Directional Accuracy: {result['dir_acc']:.3f}")
print(f"RMSE: {result['rmse']:.6f}")
print(f"{'='*60}\n")
```

---

**Note**: Make sure you have the required NLP packages installed:
```bash
pip install vaderSentiment sentence-transformers spacy
python -m spacy download en_core_web_sm
```

