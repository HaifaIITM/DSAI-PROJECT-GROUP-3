#!/usr/bin/env python
"""
Compare baseline model performance with 10 technical features vs 38 features (tech + headlines)
"""
import os
import sys
import pandas as pd
sys.path.append(os.getcwd())

from config import settings
from src.pipeline import run_baseline, run_materialize_folds, run_process

def run_baseline_comparison():
    """Run baseline models with current feature configuration"""
    
    # Load processed data
    proc_paths = {
        "GSPC": f"{settings.PROC_DIR}/GSPC_features.csv",
        "SPY": f"{settings.PROC_DIR}/SPY_features.csv"
    }
    
    # Re-materialize folds with current FEATURE_COLS setting
    print(f"=== Materializing folds with {len(settings.FEATURE_COLS)} features ===")
    print(f"Features: {settings.FEATURE_COLS[:3]}... (showing first 3)")
    
    from src.splits.walkforward import build_splits
    import pandas as pd
    
    # Build splits (same as main pipeline)
    anchor_df = pd.read_csv(proc_paths[settings.ANCHOR_TICKER], index_col=0, parse_dates=True)
    folds = build_splits(
        anchor_df.index,
        train_days=settings.TRAIN_DAYS,
        test_days=settings.TEST_DAYS,
        step_days=settings.STEP_DAYS
    )
    
    run_materialize_folds(proc_paths, folds)
    print(f"Materialized {len(folds)} folds in: {settings.SPLIT_DIR}\n")
    
    # Run baselines on fold 0, multiple horizons
    models = ["ridge", "esn", "lstm", "transformer", "tcn"]
    horizons = ["target_h1", "target_h5", "target_h20"]
    
    results = []
    for horizon in horizons:
        print(f"\n=== Horizon: {horizon} ===")
        for model_name in models:
            result = run_baseline(model_name=model_name, fold_id=0, horizon=horizon)
            result_flat = {
                'model': model_name,
                'horizon': horizon,
                'n_features': len(settings.FEATURE_COLS),
                'rmse': result['rmse'],
                'mae': result['mae'],
                'r2': result['r2'],
                'dir_acc': result['dir_acc'],
                **result['backtest']
            }
            results.append(result_flat)
            print(f"{model_name:12} | RMSE={result['rmse']:.6f} | R²={result['r2']:.3f} | "
                  f"Sharpe={result['backtest']['sharpe']:.3f} | DirAcc={result['dir_acc']:.3f}")
    
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    print(f"Current FEATURE_COLS setting: {len(settings.FEATURE_COLS)} features\n")
    
    results_df = run_baseline_comparison()
    
    # Save results
    output_file = f"baseline_comparison_{len(settings.FEATURE_COLS)}feat.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n=== Saved results to {output_file} ===")
    
    print("\n=== Summary Table ===")
    summary = results_df.pivot_table(
        index=['model', 'horizon'],
        values=['rmse', 'r2', 'sharpe', 'dir_acc'],
        aggfunc='first'
    ).round(4)
    print(summary)

