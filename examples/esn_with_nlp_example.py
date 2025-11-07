"""
Example: Training ESN with NLP Risk Index Feature

This script demonstrates how to integrate the NLP-derived risk index
as a feature for the Echo State Network (ESN) model.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.pipeline import run_download, run_process, run_build_splits, run_materialize_folds, run_baseline

# ======================================
# Step 1: Enable NLP Feature
# ======================================
# Set NLP_ENABLED = True in config/settings.py, or override it here:
settings.NLP_ENABLED = True
settings.NLP_TICKER = "SPY"  # Ticker to fetch headlines for
settings.NLP_LOOKBACK_DAYS = 365  # Historical lookback

print("=" * 60)
print("ESN + NLP Risk Index Integration Example")
print("=" * 60)

# ======================================
# Step 2: Download Raw Data
# ======================================
print("\n[1/5] Downloading market data...")
res = run_download()
print(f"Downloaded {len(res)} datasets")

# ======================================
# Step 3: Process Features (including NLP risk index)
# ======================================
print("\n[2/5] Processing features and generating NLP risk index...")
proc_paths = run_process()
print(f"Processed: {list(proc_paths.keys())}")

# ======================================
# Step 4: Build Walk-Forward Splits
# ======================================
print("\n[3/5] Building walk-forward splits...")
folds = run_build_splits(proc_paths)
print(f"Created {len(folds)} folds")

# ======================================
# Step 5: Materialize Folds (z-score features)
# ======================================
print("\n[4/5] Materializing folds...")
run_materialize_folds(proc_paths, folds)
print("Folds materialized with z-scored features (including risk_index)")

# ======================================
# Step 6: Train ESN with NLP Features
# ======================================
print("\n[5/5] Training ESN with NLP risk index...")
result = run_baseline(model_name="esn", fold_id=0, horizon="target_h1")

print("\n" + "=" * 60)
print("Results:")
print("=" * 60)
print(f"Model: {result['model']}")
print(f"RMSE: {result['rmse']:.6f}")
print(f"MAE: {result['mae']:.6f}")
print(f"R²: {result['r2']:.3f}")
print(f"Dir. Accuracy: {result['dir_acc']:.3f}")
print(f"Sharpe Ratio: {result['backtest']['sharpe']:.3f}")
print(f"Hit Ratio: {result['backtest']['hit_ratio']:.3f}")
print("=" * 60)

print("\nℹ️  The risk_index feature is now included in z_risk_index column")
print("ℹ️  ESN automatically uses all z_* features, including z_risk_index")

