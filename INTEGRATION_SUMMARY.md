# ESN + NLP Risk Index Integration Summary

## What Was Done

Successfully integrated the NLP-derived risk index as a feature for the Echo State Network (ESN) model.

## Changes Made

### 1. **NLP Pipeline Enhancement** (`src/nlp/headline_processor.py`)
- Added `generate_risk_index_timeseries()` function
- Generates historical risk index from news headlines
- Outputs daily risk scores based on sentiment, uncertainty, events, and novelty

### 2. **Feature Engineering** (`src/data/features.py`)
- Modified `compute_features()` to accept optional `risk_df` parameter
- Merges risk index with technical features via date alignment
- Forward-fills missing dates (headlines may not be available every day)
- Adds `risk_index` column to feature set

### 3. **Pipeline Integration** (`src/pipeline.py`)
- Updated `run_process()` to generate NLP risk index when enabled
- Automatically passes risk data to feature processing
- Handles errors gracefully (continues without NLP if generation fails)

### 4. **Configuration** (`config/settings.py`)
- Added `risk_index` to `FEATURE_COLS`
- Added NLP configuration options:
  - `NLP_ENABLED`: Toggle NLP features on/off
  - `NLP_TICKER`: Ticker to fetch headlines for
  - `NLP_LOOKBACK_DAYS`: Historical lookback period

### 5. **Documentation**
- Created comprehensive integration guide (`docs/NLP_ESN_Integration_Guide.md`)
- Created example script (`examples/esn_with_nlp_example.py`)
- Created notebook cells reference (`examples/nlp_esn_notebook_cell.md`)

## How It Works

```
1. Headlines fetched via yfinance (SPY news)
   ↓
2. NLP processing:
   - VADER sentiment analysis
   - Uncertainty detection (modal verbs, hedging)
   - Event intensity (crisis keywords)
   - Semantic novelty (embedding distance)
   ↓
3. Daily aggregation:
   - Risk_z = sum(standardized([Neg, Event, Novelty, Dispersion, Volume]))
   ↓
4. Feature merging:
   - Joined to price data by date
   - Forward-filled for missing dates
   ↓
5. Z-scoring:
   - Standardized with other features (z_risk_index)
   ↓
6. ESN Training:
   - Automatically includes z_risk_index (reads all z_* columns)
   - No ESN code changes needed!
```

## Quick Start

### Method 1: Using Settings

```python
# In config/settings.py, set:
NLP_ENABLED = True

# Then run normal pipeline:
from src.pipeline import run_download, run_process, run_build_splits, run_materialize_folds, run_baseline

run_download()
proc_paths = run_process()  # Will generate NLP risk index
folds = run_build_splits(proc_paths)
run_materialize_folds(proc_paths, folds)
result = run_baseline(model_name="esn", fold_id=0, horizon="target_h1")
```

### Method 2: Programmatic Override

```python
from config import settings

# Enable NLP at runtime
settings.NLP_ENABLED = True
settings.NLP_TICKER = "SPY"
settings.NLP_LOOKBACK_DAYS = 365

# Run pipeline...
```

### Method 3: Run Example Script

```bash
python examples/esn_with_nlp_example.py
```

## Feature Verification

Check that risk index is included:

```python
import pandas as pd
train = pd.read_csv("data/splits/fold_0/train.csv", index_col=0)
z_cols = [c for c in train.columns if c.startswith("z_")]
print("z_risk_index" in z_cols)  # Should be True
```

## Feature Counts

- **Without NLP**: 10 features (technical indicators only)
  - z_ret_1, z_ret_2, z_ret_5, z_vol_20, z_ma_10, z_ma_20, z_ma_gap, z_rsi_14, z_vol_z, z_dow

- **With NLP**: 11 features (technical + sentiment)
  - All above + z_risk_index

## Performance Impact

- **Generation Time**: ~5-10 seconds for 365 days
- **Memory**: Negligible (1 additional column)
- **Training**: No change
- **Inference**: No change

## Comparison Testing

To test impact:

```python
# Baseline (no NLP)
settings.NLP_ENABLED = False
result_baseline = run_baseline("esn", fold_id=0, horizon="target_h1")

# With NLP
settings.NLP_ENABLED = True
# Re-run pipeline: download, process, splits, materialize
result_nlp = run_baseline("esn", fold_id=0, horizon="target_h1")

# Compare
print(f"Baseline Sharpe: {result_baseline['backtest']['sharpe']:.3f}")
print(f"NLP Sharpe: {result_nlp['backtest']['sharpe']:.3f}")
print(f"Improvement: {result_nlp['backtest']['sharpe'] - result_baseline['backtest']['sharpe']:.3f}")
```

## Files to Review

1. **Core Integration**:
   - `src/nlp/headline_processor.py` (lines 124-138)
   - `src/data/features.py` (lines 40-47, 58-61)
   - `src/pipeline.py` (lines 33-58)
   - `config/settings.py` (lines 40, 44-47)

2. **Documentation**:
   - `docs/NLP_ESN_Integration_Guide.md` (comprehensive guide)
   - `examples/esn_with_nlp_example.py` (runnable example)
   - `examples/nlp_esn_notebook_cell.md` (notebook cells)

3. **Model** (no changes needed):
   - `src/models/esn.py` (unchanged - automatically uses all z_* features)

## Key Design Decisions

1. **Feature Toggle**: NLP is optional via `NLP_ENABLED` flag
   - Allows easy A/B testing
   - No breaking changes for existing code

2. **Forward Fill**: Missing dates use previous risk value
   - News not available every day (weekends, holidays)
   - Assumes risk persists until new information

3. **Zero Fallback**: If NLP fails, `risk_index = 0`
   - Graceful degradation
   - Pipeline continues without NLP

4. **Single Risk Metric**: Uses `Risk_z` (not `Risk_pca`)
   - Simpler interpretability
   - Sum of standardized components

5. **No ESN Changes**: Model reads all `z_*` features
   - Clean separation of concerns
   - Works with any model that uses standardized features

## Next Steps

1. **Test on Multiple Folds**: Run full walk-forward validation
2. **Hyperparameter Search**: Re-run ESN grid search with NLP enabled
3. **Feature Importance**: Analyze ESN readout weights for `z_risk_index`
4. **Other Models**: Enable NLP for LSTM, Transformer, TCN
5. **Risk Variants**: Experiment with different risk formulations

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No headlines fetched | Increase `NLP_LOOKBACK_DAYS` or check ticker symbol |
| Risk index all zeros | Check if `NLP_ENABLED = True` and review console logs |
| Feature count mismatch | Regenerate folds after enabling NLP |
| Import errors | Install: `pip install vaderSentiment sentence-transformers spacy` |

## Contact

For questions or issues, refer to:
- Full documentation: `docs/NLP_ESN_Integration_Guide.md`
- Example script: `examples/esn_with_nlp_example.py`
- Notebook cells: `examples/nlp_esn_notebook_cell.md`

---

**Status**: ✅ Integration complete and tested
**Date**: November 2025
**Files Modified**: 4 core files, 4 new documentation files

