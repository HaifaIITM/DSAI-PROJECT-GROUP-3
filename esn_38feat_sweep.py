#!/usr/bin/env python
"""
ESN hyperparameter sweep optimized for 38 features
Testing larger reservoir sizes (1200, 1600, 2000)
"""
import os
import sys
sys.path.append(os.getcwd())

from config import settings
from src.train.runner import run_sweep

# Larger reservoirs for 38-dim input
ESN_GRID_38FEAT = {
    "hidden_size":   [1200, 1600, 2000],  # Scale up for 38 features
    "spectral_radius":[0.85, 0.95],
    "leak_rate":     [0.3, 1.0],
    "ridge_alpha":   [0.3, 1.0, 3.0],
    "washout":       [100],
    "density":       [0.1],
    "state_clip":    [None],
    "seed":          [0],
}

if __name__ == "__main__":
    # Ensure we're using 38 features
    print(f"Running ESN sweep with {len(settings.FEATURE_COLS)} features")
    print(f"Grid size: {len(ESN_GRID_38FEAT['hidden_size']) * len(ESN_GRID_38FEAT['spectral_radius']) * len(ESN_GRID_38FEAT['leak_rate']) * len(ESN_GRID_38FEAT['ridge_alpha'])}")
    
    if len(settings.FEATURE_COLS) != 38:
        print(f"WARNING: Expected 38 features, got {len(settings.FEATURE_COLS)}")
        print("Switch to FEATURE_COLS_FULL in config/settings.py")
        sys.exit(1)
    
    # Run sweep on fold 0, all horizons
    results = run_sweep(
        model_name="esn",
        param_grid=ESN_GRID_38FEAT,
        folds=[0],
        horizons=["target_h1", "target_h5", "target_h20"],
        exp_prefix="m4_38feat"
    )
    
    print("\n=== ESN 38-Feature Sweep Results ===")
    if not results.empty:
        # Show top 3 by Sharpe for each horizon
        for h in ["target_h1", "target_h5", "target_h20"]:
            sub = results[results["horizon"] == h].sort_values("sharpe", ascending=False).head(3)
            if not sub.empty:
                print(f"\n{h} - Top 3 by Sharpe:")
                print(sub[["exp_id", "rmse", "r2", "sharpe", "dir_acc"]].to_string(index=False))

