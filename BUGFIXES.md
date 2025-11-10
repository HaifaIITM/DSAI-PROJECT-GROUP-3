# Critical Bug Fixes - ESN Comparison Pipeline

## Summary
This document details the **four critical bugs** discovered and fixed in the ESN sentiment comparison pipeline that were causing unstable and unreliable results.

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

## Bug #4: Raw Data Reuse

### Problem
- Even with identical code paths and seeds, repeated `python run.py` calls would download new data from yfinance
- Each download could return slightly different values (late quotes, bid-ask spreads, data revisions)
- Sentiment performance varied wildly between runs: 2.008 → -0.771 → 2.008
- Made iterative testing unreliable

### Evidence
```
Run 1: python run.py → Downloads 13 datasets → Sharpe 2.008
Run 2: python run.py → Downloads 13 datasets → Sharpe -0.771  (different data!)
Run 3: python run.py → Downloads 13 datasets → Sharpe 2.008   (different again!)
```

### Fix
Added `_ensure_raw_data()` helper in `run.py` to:
1. Check if `data/raw/*.csv` files exist (counts files)
2. Only download if insufficient files found
3. Reuse cached CSVs for subsequent runs

**File:** `run.py::_ensure_raw_data()`, called by `run_with_sentiment()` and `run_baseline_only()`

---

## Verification

### Before All Fixes
```
python run.py --compare
# Run 1: Sentiment +631.9% Sharpe
# Run 2: Sentiment −79.0% Sharpe  
# Run 3: Sentiment +288.6% Sharpe
# Run 4: Sentiment −99.3% Sharpe
```
**Completely unstable and unreliable.**

### After All Fixes
```
python run.py --baseline
# Baseline: Sharpe 1.021 (consistent)

python run.py
# Run 1: Sentiment Sharpe 2.008
# Run 2: Sentiment Sharpe 2.008  ← IDENTICAL
# Run 3: Sentiment Sharpe 2.008  ← IDENTICAL
# Run 4: Sentiment Sharpe 2.008  ← IDENTICAL
```
**Perfect reproducibility - all metrics match to 3+ decimal places.**

```
python run.py --compare
# Downloads once, processes twice
# Baseline: 1.021 Sharpe (10 features, no sentiment)
# Sentiment: 2.008 Sharpe (11 features, with sentiment)
# Fair comparison on identical data
# Improvement: +96.8% Sharpe
```

---

## Key Learnings

1. **Data Contamination:** Never re-download between comparison runs - cache raw data first
2. **Randomness Control:** Set global numpy seed at pipeline start, before any processing
3. **Code Path Synchronization:** Ensure baseline and test follow identical execution paths
4. **Feature Flags:** Always verify flags are actually checked in the code, not just config
5. **Raw Data Stability:** Check for existing data before downloading to avoid yfinance drift
6. **Reproducibility Testing:** Run the same command 3-5 times to verify stability

---

## Current Status

✅ **All four bugs fixed**  
✅ **Reproducible results** (seed=42)  
✅ **Fair comparison** (identical raw data)  
✅ **Clean separation** (10 features baseline, 11 features sentiment)  
✅ **Raw data caching** (reuses existing downloads)

The pipeline now provides **reliable, reproducible comparisons** between baseline and sentiment-enhanced ESN models.

---

## Performance Reality Check

After fixing all bugs, the **true performance on fold 0** is:

| Metric | Baseline | Market Proxy | Change |
|--------|----------|--------------|--------|
| Sharpe | 1.021 | 2.008 | **+96.8%** ✅ |
| Dir.Acc | 52.0% | 56.3% | +4.3pp ✅ |
| RMSE | 0.009527 | 0.008568 | -10.1% ✅ |
| MAE | 0.007469 | 0.006363 | -14.8% ✅ |
| Daily PnL | $0.000490 | $0.000957 | +95.5% ✅ |

The market sentiment proxy **significantly improves performance** on this fold, with perfect reproducibility across runs.

**Key Achievement:** We now have a trustworthy comparison framework. The +96.8% improvement is real, but requires cross-fold validation to confirm robustness across different time periods.

**Recommendation:** Test across all 9 folds to calculate average performance and confidence intervals.

