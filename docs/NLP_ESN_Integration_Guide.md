# NLP Risk Index + ESN Integration Guide

## Overview

This guide explains how to integrate the NLP-derived risk index as a feature for the Echo State Network (ESN) model. The risk index is computed from financial news headlines using sentiment analysis, uncertainty detection, and event intensity scoring.

## Architecture

```
News Headlines (yfinance)
    ↓
NLP Pipeline (src/nlp/headline_processor.py)
    ├─ VADER Sentiment
    ├─ Uncertainty Detection
    ├─ Event Keywords
    └─ Semantic Embeddings
    ↓
Daily Risk Index (Risk_z)
    ↓
Feature Engineering (src/data/features.py)
    ↓
Z-scored Features (z_risk_index)
    ↓
ESN Model (src/models/esn.py)
```

## Quick Start

### 1. Enable NLP Features

Edit `config/settings.py`:

```python
# NLP settings
NLP_ENABLED = True  # Enable NLP risk index
NLP_TICKER = "SPY"  # Ticker to fetch headlines for
NLP_LOOKBACK_DAYS = 365  # Historical lookback
```

### 2. Run the Pipeline

```python
from src.pipeline import run_download, run_process, run_build_splits, run_materialize_folds, run_baseline

# Download data
run_download()

# Process features (NLP risk index will be generated automatically)
proc_paths = run_process()

# Build splits
folds = run_build_splits(proc_paths)
run_materialize_folds(proc_paths, folds)

# Train ESN with NLP features
result = run_baseline(model_name="esn", fold_id=0, horizon="target_h1")
```

### 3. Use the Example Script

```bash
python examples/esn_with_nlp_example.py
```

## Feature Details

### Risk Index Components

The `risk_index` feature is derived from:

1. **Negative Sentiment Intensity**: `max(0, -sentiment_score)`
2. **Event Intensity**: Frequency of crisis keywords (bankruptcy, fraud, etc.)
3. **Novelty**: Semantic distance from recent headlines
4. **Dispersion**: Volatility in sentiment across headlines
5. **Volume**: Number of headlines

These components are standardized and combined into a composite risk index using:
```
Risk_z = sum(standardized([Neg_mean, E_mean, N_mean, Disp, V]))
```

### Feature Processing

1. **Raw Risk Index**: Generated daily from headlines
2. **Alignment**: Joined to price data by date (forward-filled for missing dates)
3. **Standardization**: Z-scored along with other features during fold materialization
4. **ESN Input**: Appears as `z_risk_index` in the feature matrix

## Implementation Details

### Files Modified

1. **`src/nlp/headline_processor.py`**
   - Added `generate_risk_index_timeseries()` function

2. **`src/data/features.py`**
   - Added `risk_df` parameter to `compute_features()`
   - Merges risk index with technical features

3. **`src/pipeline.py`**
   - Generates risk index in `run_process()` when `NLP_ENABLED=True`

4. **`config/settings.py`**
   - Added `risk_index` to `FEATURE_COLS`
   - Added NLP configuration options

### ESN Integration

The ESN model automatically uses all features with the `z_` prefix. When NLP is enabled:

- **Without NLP**: ESN uses 10 technical features
- **With NLP**: ESN uses 11 features (10 technical + `z_risk_index`)

No changes to the ESN model code are required!

## Comparison: ESN with vs. without NLP

Run experiments with and without NLP features:

```python
# Without NLP
settings.NLP_ENABLED = False
result_baseline = run_baseline("esn", fold_id=0, horizon="target_h1")

# With NLP
settings.NLP_ENABLED = True
result_nlp = run_baseline("esn", fold_id=0, horizon="target_h1")

# Compare
print(f"Sharpe without NLP: {result_baseline['backtest']['sharpe']:.3f}")
print(f"Sharpe with NLP: {result_nlp['backtest']['sharpe']:.3f}")
```

## Troubleshooting

### Issue: No headlines found
**Solution**: Increase `NLP_LOOKBACK_DAYS` or check if yfinance can access headlines for your ticker.

### Issue: Risk index all zeros
**Solution**: NLP is disabled or headline fetch failed. Check console output for warnings.

### Issue: Feature dimension mismatch
**Solution**: Ensure `risk_index` is in `FEATURE_COLS` and regenerate folds after enabling NLP.

## Advanced Usage

### Custom Risk Index

You can provide your own risk index by creating a CSV with columns `date`, `Risk_z`:

```python
import pandas as pd

# Load your custom risk data
risk_df = pd.read_csv('my_risk_index.csv', index_col='date', parse_dates=True)

# Pass it directly to feature processing
from src.data.features import compute_features, process_and_save
process_and_save(raw_path, symbol, output_dir, risk_df=risk_df)
```

### Offline Mode

Generate risk index once and reuse:

```python
from src.nlp.headline_processor import generate_risk_index_timeseries

# Generate and save
risk_df = generate_risk_index_timeseries(
    ticker="SPY",
    lookback_days=365,
    output_path="data/processed/SPY_risk_index.csv"
)

# Disable online fetching
settings.NLP_ENABLED = False

# Load saved risk index manually when processing features
# (modify pipeline.py to load from file instead of generating)
```

## Performance Considerations

- **NLP Generation**: ~5-10 seconds for 365 days of headlines
- **Memory**: Minimal overhead (~1 additional feature column)
- **Training**: No change in ESN training time
- **Prediction**: No change in ESN inference speed

## Next Steps

1. **Hyperparameter Search**: Run grid search with NLP features enabled
2. **Feature Importance**: Analyze ESN readout weights for `z_risk_index`
3. **Multiple Horizons**: Test NLP impact on h5, h20 targets
4. **Other Models**: Enable NLP for LSTM, Transformer, TCN models

---

For questions or issues, see the main README or check the source code documentation.

