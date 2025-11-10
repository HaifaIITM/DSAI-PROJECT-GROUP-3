# Critical Bug Fixes - ESN Comparison Pipeline

## Summary
This document details the **three critical bugs** discovered and fixed in the ESN sentiment comparison pipeline that were causing unstable and unreliable results.

---

## Bug #1: Data Contamination Between Runs

### Problem
- `compare_models()` was cleaning `data/processed` and `data/splits` before EACH run (baseline and sentiment)
- Each run re-downloaded raw data from yfinance
- yfinance returns slightly different data each time (late quotes, timeouts, cached updates)
- **Result:** Baseline and sentiment were trained on **different underlying price data**
- Comparisons were meaningless - measured data drift, not feature impact

### Evidence
```
Run 1: Baseline Sharpe 1.021 → Sentiment Sharpe 0.354 (−65%)
Run 2: Baseline Sharpe 1.021 → Sentiment Sharpe 1.976 (+93%)
Run 3: Baseline Sharpe 1.021 → Sentiment Sharpe 0.289 (−72%)
```
Baseline stayed constant, but sentiment jumped wildly even with seed=42.

### Fix
Modified `compare_models()` to:
1. Download raw data **once** at the start
2. Process baseline (sentiment disabled) using that data
3. Process sentiment (sentiment enabled) using **same** data
4. Both models now see identical price history

**File:** `run.py::compare_models()`

---

## Bug #2: ESN Randomness (Multiple Root Causes)

### Problem 2a: Code Path Mismatch
- Baseline called: `run_process() → run_build_splits() → run_materialize_folds()`
- Sentiment called: `run_download() → run_process() → run_build_splits() → run_materialize_folds()`
- Different code paths → different numpy random state consumption → different ESN reservoir initialization
- Even with `np.random.seed(42)`, results varied because the seed was consumed differently

### Problem 2b: Global Numpy State Not Set
- ESN's `__init__` used `np.random.default_rng(seed)` (local RNG)
- But data processing used global `np.random` functions
- Setting seed in one place didn't control the other

### Evidence
```
Run 1: Baseline Sharpe -0.436
Run 2: Baseline Sharpe  2.981 (same fold, same data!)
Run 3: Baseline Sharpe  1.350
```
Extreme variance even with seed parameter.

### Fix
1. Set `np.random.seed(42)` at **start** of both pipeline functions
2. Made baseline call `run_download()` (same code path as sentiment)
3. Ensures both pipelines execute **identical operations** before ESN training

**Files:** `run.py::run_with_sentiment()`, `run.py::run_baseline_only()`, `run.py::compare_models()`

---

## Bug #3: Sentiment Feature Not Applied

### Problem
- `SENTIMENT_ENABLED` flag was **never checked** in `compute_features()`
- The `risk_index` feature was **ALWAYS** generated, even when disabled
- Baseline was secretly using the sentiment feature!
- Both runs had identical results despite different feature counts (10 vs 11)

### Evidence
```
Run 1: Baseline Sharpe 2.390 → Sentiment Sharpe 2.390 (IDENTICAL)
Run 2: Baseline Sharpe 1.372 → Sentiment Sharpe 1.372 (IDENTICAL)
```
Results matched to 6+ decimal places - the 11th feature had **zero effect**.

### Fix
Modified feature generation to respect the flag:

**In `src/data/features.py`:**
```python
# Old code - ALWAYS creates risk_index
if risk_df is not None:
    out['risk_index'] = ...
else:
    out['risk_index'] = _compute_market_sentiment_proxy(out)

# New code - Only creates when enabled
if SENTIMENT_ENABLED:
    if risk_df is not None:
        out['risk_index'] = ...
    else:
        out['risk_index'] = _compute_market_sentiment_proxy(out)
```

**In `src/pipeline.py`:**
```python
# Dynamic feature list based on sentiment flag
features = FEATURE_COLS.copy()
if not SENTIMENT_ENABLED and "risk_index" in features:
    features.remove("risk_index")
```

**Files:** `src/data/features.py::compute_features()`, `src/pipeline.py::run_materialize_folds()`

---

## Verification

### Before Fixes
```
python run.py --compare
# Run 1: Sentiment +631.9% Sharpe
# Run 2: Sentiment −79.0% Sharpe  
# Run 3: Sentiment +288.6% Sharpe
# Run 4: Sentiment −99.3% Sharpe
```
**Completely unstable and unreliable.**

### After Fixes
```
python run.py --baseline
# Baseline: Sharpe 1.021 (consistent)

python run.py
# Run 1: Sentiment Sharpe 0.354
# Run 2: Sentiment Sharpe 0.354
# Run 3: Sentiment Sharpe 0.354
# Run 4: Sentiment Sharpe 0.354
```
**Perfect reproducibility.**

```
python run.py --compare
# Downloads once, processes twice
# Baseline: 1.021 Sharpe (10 features, no sentiment)
# Sentiment: 0.354 Sharpe (11 features, with sentiment)
# Fair comparison on identical data
```

---

## Key Learnings

1. **Data Contamination:** Never re-download between comparison runs - cache raw data first
2. **Randomness Control:** Set global numpy seed at pipeline start, before any processing
3. **Code Path Synchronization:** Ensure baseline and test follow identical execution paths
4. **Feature Flags:** Always verify flags are actually checked in the code, not just config
5. **Reproducibility Testing:** Run the same command 3-5 times to verify stability

---

## Current Status

✅ **All three bugs fixed**  
✅ **Reproducible results** (seed=42)  
✅ **Fair comparison** (identical raw data)  
✅ **Clean separation** (10 features baseline, 11 features sentiment)

The pipeline now provides **reliable, reproducible comparisons** between baseline and sentiment-enhanced ESN models.

---

## Performance Reality Check

After fixing all bugs, the **true performance on fold 0** is:

| Metric | Baseline | Market Proxy | Change |
|--------|----------|--------------|--------|
| Sharpe | 1.021 | 0.354 | **−65%** |
| Dir.Acc | 52.0% | 48.4% | −3.6pp |

The market sentiment proxy **degrades performance** on this fold, contradicting the original "+300% Sharpe" claim. This suggests:
- Original results were from contaminated data
- Need to test across all 9 folds for robust evaluation
- The proxy may help on some folds but hurt on others

**Recommendation:** Run cross-fold validation before making any performance claims.

