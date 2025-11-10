# Testing Guide: Sentiment Strategies

## The Caching Issue (Now Fixed)

**Problem:** Data was being cached between tests, causing:
- Baseline metrics to change between runs
- Invalid comparisons (comparing contaminated baselines)
- Misleading improvement percentages

**Solution:** `test_strategies.py` now automatically cleans data between each test.

## Automated Testing (Recommended)

Run all three strategies with automatic cleanup:

```bash
python test_strategies.py
```

**What it does:**
1. Cleans `data/processed` and `data/splits`
2. Runs Market Proxy test
3. Cleans again
4. Runs VIX Inverted test
5. Cleans again
6. Runs Combined test

**Expected output:**
- Each test should show **identical baseline Sharpe** (~0.3-0.7)
- If baselines differ significantly, ESN randomness is still an issue

## Manual Testing (For Individual Strategies)

### Step 1: Clean Data

```bash
python cleanup.py
```

### Step 2: Test Strategy

```bash
# Market Proxy
python run.py --compare --strategy market_proxy

# Clean again
python cleanup.py

# VIX Inverted
python run.py --compare --strategy vix_inverted

# Clean again
python cleanup.py

# Combined (70/30)
python run.py --compare --strategy combined --vix-weight 0.3
```

## Tuning Combined Strategy

Test different VIX weights (must clean between each):

```bash
# 80% Market / 20% VIX
python cleanup.py
python run.py --compare --strategy combined --vix-weight 0.2

# 60% Market / 40% VIX
python cleanup.py
python run.py --compare --strategy combined --vix-weight 0.4

# 50% Market / 50% VIX
python cleanup.py
python run.py --compare --strategy combined --vix-weight 0.5
```

## PowerShell Commands

For Windows PowerShell users:

```powershell
# Automated test
python test_strategies.py

# Manual cleanup
python cleanup.py

# Individual tests
python run.py --compare --strategy market_proxy
python run.py --compare --strategy vix_inverted
python run.py --compare --strategy combined --vix-weight 0.3
```

## Interpreting Results

### Good Results (Baseline Consistency)

```
Test 1 Baseline: Sharpe 0.680
Test 2 Baseline: Sharpe 0.682  ← Within ~5% of Test 1
Test 3 Baseline: Sharpe 0.675  ← Within ~5% of Test 1
```

**Action:** Results are reliable, choose best strategy.

### Bad Results (ESN Randomness)

```
Test 1 Baseline: Sharpe 0.680
Test 2 Baseline: Sharpe 1.854  ← 173% different!
Test 3 Baseline: Sharpe -0.319 ← Negative!
```

**Action:** ESN seed is not working. Need to fix reservoir initialization.

## Expected Performance (From Clean Tests)

Based on first run data:

| Strategy | Baseline | With Strategy | Improvement | Dir. Acc | Verdict |
|----------|----------|---------------|-------------|----------|---------|
| **Market Proxy** | 0.680 | 1.399 | **+106%** | 55.6% | ✓✓✓ Best for momentum |
| **VIX Inverted** | 0.680 | ? | **TBD** | ? | Test needed |
| **Combined** | 0.680 | ? | **TBD** | ? | Test needed |

## Troubleshooting

### "Permission Denied" when cleaning

**Problem:** Files are in use by another process.

**Solutions:**
1. Close any Python processes
2. Close Jupyter notebooks
3. Close VS Code/file explorers viewing the data folders
4. Restart terminal

### Baseline still changing

**Problem:** ESN randomness despite fixed seed.

**Check:**
```python
# In src/models/esn.py, line 36
seed: int = 0,  # Should be present
```

**Debug:** Add print statement in ESN.__init__:
```python
print(f"ESN initialized with seed={seed}")
```

### Out of memory

**Problem:** Multiple large datasets in memory.

**Solution:** Clean between tests:
```bash
python cleanup.py
```

## Best Practices

1. ✓ **Always clean before testing** - Prevents contamination
2. ✓ **Run full test suite** - `python test_strategies.py`
3. ✓ **Check baseline consistency** - Should be within ±5%
4. ✓ **Test across multiple folds** - Verify stability
5. ✓ **Document your results** - Keep a testing log

## Next Steps After Testing

1. **Identify best strategy** from clean results
2. **Set in config/settings.py:**
   ```python
   SENTIMENT_ENABLED = True
   SENTIMENT_STRATEGY = "market_proxy"  # or "vix_inverted" or "combined"
   ```
3. **Test across all 9 folds:**
   ```bash
   for i in {0..8}; do
       python cleanup.py
       python run.py --fold $i --strategy YOUR_BEST_STRATEGY
   done
   ```
4. **Calculate average metrics** across folds
5. **Deploy best configuration** to production

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python test_strategies.py` | Test all 3 strategies (recommended) |
| `python cleanup.py` | Clean data manually |
| `python run.py --compare` | Compare with current settings |
| `python run.py --strategy X` | Run specific strategy |
| `python run.py --fold N` | Test specific fold |

