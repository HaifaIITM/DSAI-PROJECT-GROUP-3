#!/usr/bin/env python
"""
ESN sweep focused on fixing magnitude prediction (R²) while keeping directional edge
Strategy: Stronger regularization + state clipping
"""
import os
import sys
sys.path.append(os.getcwd())

from config import settings
from src.train.runner import run_sweep

# Focus on h20 config that had Sharpe 6.267 but R² -14.48
# hidden_size=1600, spectral_radius=0.85, leak_rate=0.3
# Now sweep much stronger regularization + add state clipping

ESN_MAGNITUDE_FIX = {
    "hidden_size":   [1600],  # Keep winning size
    "spectral_radius":[0.85], # Keep winning value
    "leak_rate":     [0.3],   # Keep winning value
    "ridge_alpha":   [10, 30, 100, 300],  # MUCH stronger regularization
    "washout":       [100],
    "density":       [0.1],
    "state_clip":    [1.0, 5.0],  # Clip reservoir states to prevent explosion
    "seed":          [0],
}

if __name__ == "__main__":
    print(f"Running ESN magnitude fix with {len(settings.FEATURE_COLS)} features")
    print(f"Grid size: {len(ESN_MAGNITUDE_FIX['ridge_alpha']) * len(ESN_MAGNITUDE_FIX['state_clip'])}")
    
    if len(settings.FEATURE_COLS) != 38:
        print(f"WARNING: Expected 38 features, got {len(settings.FEATURE_COLS)}")
        sys.exit(1)
    
    # Focus on h20 where we had Sharpe 6.267 but terrible R²
    results = run_sweep(
        model_name="esn",
        param_grid=ESN_MAGNITUDE_FIX,
        folds=[0],
        horizons=["target_h20"],
        exp_prefix="m4_magnitude_fix"
    )
    
    print("\n=== ESN Magnitude Fix Results (h20) ===")
    if not results.empty:
        # Sort by R² (ascending, less negative is better)
        results_sorted = results.sort_values("r2", ascending=False)
        print("\nTop 5 by R² (magnitude accuracy):")
        print(results_sorted[["exp_id", "rmse", "r2", "sharpe", "dir_acc"]].head(5).to_string(index=False))
        
        print("\n\nTop 5 by Sharpe (trading performance):")
        results_sharpe = results.sort_values("sharpe", ascending=False)
        print(results_sharpe[["exp_id", "rmse", "r2", "sharpe", "dir_acc"]].head(5).to_string(index=False))
        
        # Find best trade-off (good Sharpe + best R²)
        print("\n\nBest trade-off (Sharpe > 3 with best R²):")
        good_sharpe = results[results["sharpe"] > 3.0].sort_values("r2", ascending=False)
        if not good_sharpe.empty:
            print(good_sharpe[["exp_id", "rmse", "r2", "sharpe", "dir_acc"]].head(3).to_string(index=False))
        else:
            print("No configs with Sharpe > 3")

