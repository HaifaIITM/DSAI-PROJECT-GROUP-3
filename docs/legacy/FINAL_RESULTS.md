# Final Model Performance Comparison

## Executive Summary

**Best Overall Model: TCN (38 features, h20 horizon)**
- Sharpe: 5.951
- R²: -0.373
- RMSE: 0.028
- Directional Accuracy: 60.7%

---

## Complete Results (Fold 0, h20 Horizon)

### All Models Performance

| Rank | Model | Features | Sharpe | R² | RMSE | Dir_Acc | Turnover |
|------|-------|----------|--------|-----|------|---------|----------|
| 1 | **TCN** | 38 | **5.951** | **-0.373** | **0.028** | 60.7% | 0.440 |
| 2 | **Hybrid ESN-Ridge** | 38 | 3.926 | -0.420 | 0.028 | 57.1% | **0.155** |
| 3 | **LSTM** | 38 | -0.199 | -0.557 | 0.030 | 50.4% | 0.226 |
| 4 | **ESN baseline** | 38 | -2.085 | -2.070 | 0.042 | 42.5% | 0.536 |
| 5 | **Ridge** | 38 | -5.034 | -0.767 | 0.032 | 38.1% | 0.377 |

---

## Model-Specific Analysis

### 1. TCN (Winner)

**Configuration:**
- Features: 38 (10 technical + 28 headline embeddings)
- Architecture: Temporal Convolutional Network with dilated causal convolutions

**Performance:**
- Sharpe: 5.951 (best)
- R²: -0.373 (best)
- RMSE: 0.028 (tied best)
- Dir Accuracy: 60.7%

**Strengths:**
- Highest risk-adjusted returns
- Best magnitude prediction (R² closest to 0)
- Consistent performance across metrics
- Handles 38-dimensional input well

**Use for:**
- h20 (monthly) predictions
- Trading strategies requiring high Sharpe
- Applications needing both direction and magnitude

---

### 2. Hybrid ESN-Ridge (2nd Place)

**Configuration:**
- Features: 38 (10 technical + 28 headline embeddings)
- Architecture: ESN (1600 neurons, SR=0.85, leak=0.3) for direction + Ridge for magnitude
- Final prediction: sign(ESN) × |Ridge|

**Performance:**
- Sharpe: 3.926 (2nd best)
- R²: -0.420 (2nd best)
- RMSE: 0.028 (tied best)
- Dir Accuracy: 57.1%

**Strengths:**
- Ultra-low turnover (0.155) - most stable signals
- Good magnitude prediction (R² much better than pure ESN)
- Novel hybrid architecture

**Improvement over pure ESN:**
- Sharpe: 3.926 vs -2.085 (+288% improvement)
- R²: -0.420 vs -2.070 (5× better)
- RMSE: 0.028 vs 0.042 (33% better)

**Use for:**
- Applications requiring stable signals (low turnover)
- When ESN directional strength is desired with better magnitude
- Alternative to TCN with different architectural approach

---

### 3. LSTM

**Performance:**
- Sharpe: -0.199 (unprofitable at h20)
- R²: -0.557
- RMSE: 0.030

**Note:** LSTM excels at h5 horizon, not h20. See horizon-specific results below.

---

## Feature Set Comparison (h20 Horizon)

### 38 Features vs 10 Features

**Best with 38 features:**
- TCN: Sharpe 5.951

**Best with 10 features:**
- TCN: Sharpe 6.232

**Key Finding:** For TCN at h20, **10 technical features perform slightly better** (6.23 vs 5.95). Headlines add complexity without benefit at monthly horizon for TCN.

**Hybrid ESN-Ridge comparison:**
- 38 features: Sharpe 3.926
- 10 features: N/A (Hybrid requires 38 features for directional patterns)

---

## Horizon-Specific Best Models

### h1 (1-day ahead):
**Winner: Transformer (38 feat)**
- Sharpe: 0.664
- R²: -0.987
- RMSE: 0.011

### h5 (5-day ahead):
**Winner: LSTM (38 feat)**
- Sharpe: 4.560
- R²: 0.060 (ONLY positive R²!)
- RMSE: 0.014
- Dir Accuracy: 62.7%

### h20 (20-day ahead):
**Winner: TCN (38 feat)**
- Sharpe: 5.951
- R²: -0.373
- RMSE: 0.028
- Dir Accuracy: 60.7%

---

## Key Insights

### 1. Headline Embeddings Impact

**Help at h5 (5-day):**
- LSTM: 3.216 → 4.560 Sharpe (+42%)

**Minimal impact or hurt at h1 and h20:**
- TCN h20: 6.232 (10 feat) vs 5.951 (38 feat)
- Most models perform worse with 38 features at short/long horizons

**Conclusion:** Headlines are useful for **weekly predictions only** (h5).

---

### 2. Hybrid Architecture Value

**Hybrid ESN-Ridge vs Pure ESN:**
- Sharpe: 3.926 vs -2.085 (massive improvement)
- R²: -0.420 vs -2.070 (5× better magnitude)
- RMSE: 0.028 vs 0.042 (33% better)

**Key Discovery:** Separating direction (ESN) from magnitude (Ridge) fixes ESN's catastrophic magnitude prediction while preserving reasonable directional accuracy.

---

### 3. Model Selection Guide

| Use Case | Best Model | Horizon | Sharpe | Features |
|----------|-----------|---------|--------|----------|
| Monthly trading | TCN | h20 | 5.951 | 38 |
| Weekly trading | LSTM | h5 | 4.560 | 38 |
| Low turnover strategy | Hybrid | h20 | 3.926 | 38 |
| Actual forecasting (positive R²) | LSTM | h5 | 4.560 | 38 |
| Simplicity + performance | TCN | h20 | 6.232 | 10 |

---

## Production Recommendations

### Primary Model: TCN (38 features, h20)
```python
from src.pipeline import run_baseline

result = run_baseline("tcn", fold_id=0, horizon="target_h20")
# Expected: Sharpe 5.951, R² -0.373
```

### Alternative: Hybrid ESN-Ridge (38 features, h20)
```python
result = run_baseline("hybrid", fold_id=0, horizon="target_h20")
# Expected: Sharpe 3.926, R² -0.420, Turnover 0.155
```

### For Weekly Predictions: LSTM (38 features, h5)
```python
result = run_baseline("lstm", fold_id=0, horizon="target_h5")
# Expected: Sharpe 4.560, R² 0.060 (only positive R²!)
```

---

## Files

- **Models:** `src/models/` (esn.py, lstm.py, tcn.py, transformer.py, ridge_readout.py, hybrid_esn_ridge.py)
- **Registry:** `src/models/registry.py`
- **Evaluation:** `evaluate_hybrid_model.py`
- **Baselines:** Use `src/pipeline.py::run_baseline()`
- **Data:** `data/processed/` (38-feature datasets with headline embeddings)

---

## Data & Features

### Technical Features (10):
`ret_1`, `ret_2`, `ret_5`, `vol_20`, `ma_10`, `ma_20`, `ma_gap`, `rsi_14`, `vol_z`, `dow`

### Headline Embeddings (28):
- Small model (all-MiniLM-L6-v2): 12 PCA components + has_news flag
- Large model (all-mpnet-base-v2): 14 PCA components + has_news flag

### Total: 38 features

---

## Validation Methodology

- **Walk-forward validation:** 10-year train, 1-year test, 1-year step
- **9 folds total:** Results shown for fold 0 (2016-2017 test period)
- **Leakage control:** Train-only scaler, no future information
- **Metrics:** RMSE, MAE, R², Directional Accuracy, Sharpe, Turnover

---

## Conclusion

**TCN is the clear winner for monthly (h20) predictions** with Sharpe 5.951 and best R² (-0.373). 

**Hybrid ESN-Ridge** offers a compelling alternative with ultra-low turnover (0.155) and novel architecture, achieving strong performance (Sharpe 3.926) while fixing ESN's magnitude prediction problem.

For different horizons:
- **h5 (weekly):** Use LSTM (Sharpe 4.560, R² +0.060)
- **h20 (monthly):** Use TCN (Sharpe 5.951, R² -0.373)

