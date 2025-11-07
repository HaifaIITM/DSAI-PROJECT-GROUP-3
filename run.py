"""
ESN + NLP Risk Index Integration Pipeline
==========================================
Run this script to train the ESN model with NLP-derived risk index features.

Usage:
    python run.py                    # Run with NLP features
    python run.py --no-nlp           # Run baseline without NLP
    python run.py --compare          # Compare both (with and without NLP)
"""

import sys
import os
import argparse
import pandas as pd

sys.path.append(os.getcwd())

from config import settings
from src.pipeline import (
    run_download, 
    run_process, 
    run_build_splits, 
    run_materialize_folds, 
    run_baseline
)


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_results(result, model_name="ESN"):
    """Print model results in a formatted table"""
    print(f"\n{model_name} Results:")
    print("-" * 70)
    print(f"  RMSE:              {result['rmse']:.6f}")
    print(f"  MAE:               {result['mae']:.6f}")
    print(f"  R²:                {result['r2']:.3f}")
    print(f"  Dir. Accuracy:     {result['dir_acc']:.1%}")
    print(f"  Sharpe Ratio:      {result['backtest']['sharpe']:.3f}")
    print(f"  Hit Ratio:         {result['backtest']['hit_ratio']:.1%}")
    print(f"  Avg Daily PnL:     {result['backtest']['avg_daily_pnl']:.6f}")
    print(f"  Turnover:          {result['backtest']['turnover']:.3f}")
    print("-" * 70)


def verify_risk_index():
    """Verify that risk_index feature is included"""
    print("\n[VERIFY] Checking for z_risk_index feature...")
    train = pd.read_csv("data/splits/fold_0/train.csv", index_col=0)
    z_cols = [c for c in train.columns if c.startswith("z_")]
    
    print(f"  Total features: {len(z_cols)}")
    has_risk = 'z_risk_index' in z_cols
    print(f"  z_risk_index included: {has_risk}")
    
    if has_risk:
        stats = train['z_risk_index'].describe()
        print(f"\n  Risk Index Statistics:")
        print(f"    Mean: {stats['mean']:.3f}")
        print(f"    Std:  {stats['std']:.3f}")
        print(f"    Min:  {stats['min']:.3f}")
        print(f"    Max:  {stats['max']:.3f}")
    
    return has_risk


def run_with_nlp(fold_id=0, horizon="target_h1", lookback_days=30):
    """Run pipeline with NLP risk index enabled"""
    print_header("ESN + NLP Risk Index Pipeline")
    
    # Enable NLP
    settings.NLP_ENABLED = True
    settings.NLP_TICKER = "SPY"
    settings.NLP_LOOKBACK_DAYS = lookback_days
    
    print(f"\n[CONFIG]")
    print(f"  NLP Enabled:       {settings.NLP_ENABLED}")
    print(f"  Ticker:            {settings.NLP_TICKER}")
    print(f"  Lookback Days:     {settings.NLP_LOOKBACK_DAYS}")
    print(f"  Fold ID:           {fold_id}")
    print(f"  Horizon:           {horizon}")
    
    # Step 1: Download
    print("\n[1/5] Downloading data...")
    try:
        res = run_download()
        print(f"  Downloaded {len(res)} datasets")
    except Exception as e:
        print(f"  Using existing data: {e}")
    
    # Step 2: Process with NLP
    print("\n[2/5] Processing features and generating NLP risk index...")
    proc_paths = run_process()
    print(f"  Processed: {list(proc_paths.keys())}")
    
    # Step 3: Build splits
    print("\n[3/5] Building walk-forward splits...")
    folds = run_build_splits(proc_paths)
    print(f"  Created {len(folds)} folds")
    
    # Step 4: Materialize folds
    print("\n[4/5] Materializing folds with z-scored features...")
    run_materialize_folds(proc_paths, folds)
    print(f"  Folds materialized")
    
    # Verify risk index
    verify_risk_index()
    
    # Step 5: Train ESN
    print("\n[5/5] Training ESN with NLP features...")
    result = run_baseline(model_name="esn", fold_id=fold_id, horizon=horizon)
    print("  Training complete")
    
    # Display results
    print_results(result, "ESN + NLP Risk Index")
    
    return result


def run_baseline_only(fold_id=0, horizon="target_h1"):
    """Run pipeline without NLP (baseline)"""
    print_header("ESN Baseline Pipeline (No NLP)")
    
    # Disable NLP
    settings.NLP_ENABLED = False
    
    print(f"\n[CONFIG]")
    print(f"  NLP Enabled:       {settings.NLP_ENABLED}")
    print(f"  Fold ID:           {fold_id}")
    print(f"  Horizon:           {horizon}")
    
    # Process without NLP
    print("\n[1/4] Processing features (no NLP)...")
    proc_paths = run_process()
    print(f"  Processed: {list(proc_paths.keys())}")
    
    print("\n[2/4] Building splits...")
    folds = run_build_splits(proc_paths)
    print(f"  Created {len(folds)} folds")
    
    print("\n[3/4] Materializing folds...")
    run_materialize_folds(proc_paths, folds)
    print(f"  Folds materialized")
    
    print("\n[4/4] Training ESN baseline...")
    result = run_baseline(model_name="esn", fold_id=fold_id, horizon=horizon)
    print("  Training complete")
    
    # Display results
    print_results(result, "ESN Baseline (No NLP)")
    
    return result


def compare_models(fold_id=0, horizon="target_h1", lookback_days=30):
    """Run both baseline and NLP versions and compare"""
    print_header("Comparison: ESN Baseline vs ESN + NLP")
    
    # Run baseline
    result_baseline = run_baseline_only(fold_id, horizon)
    
    # Run with NLP
    result_nlp = run_with_nlp(fold_id, horizon, lookback_days)
    
    # Comparison table
    print_header("Comparison Results")
    
    comparison = pd.DataFrame([
        {
            "Model": "ESN (baseline)",
            "Features": 10,
            "RMSE": result_baseline['rmse'],
            "MAE": result_baseline['mae'],
            "R²": result_baseline['r2'],
            "Dir.Acc": result_baseline['dir_acc'],
            "Sharpe": result_baseline['backtest']['sharpe'],
            "Hit Ratio": result_baseline['backtest']['hit_ratio']
        },
        {
            "Model": "ESN + NLP",
            "Features": 11,
            "RMSE": result_nlp['rmse'],
            "MAE": result_nlp['mae'],
            "R²": result_nlp['r2'],
            "Dir.Acc": result_nlp['dir_acc'],
            "Sharpe": result_nlp['backtest']['sharpe'],
            "Hit Ratio": result_nlp['backtest']['hit_ratio']
        }
    ])
    
    print("\n" + comparison.to_string(index=False))
    
    # Calculate improvements
    print("\n[IMPROVEMENT METRICS]")
    sharpe_diff = result_nlp['backtest']['sharpe'] - result_baseline['backtest']['sharpe']
    dir_acc_diff = result_nlp['dir_acc'] - result_baseline['dir_acc']
    rmse_diff = result_nlp['rmse'] - result_baseline['rmse']
    
    print(f"  Sharpe Ratio:      {sharpe_diff:+.3f} ({sharpe_diff/abs(result_baseline['backtest']['sharpe'])*100:+.1f}%)")
    print(f"  Dir. Accuracy:     {dir_acc_diff:+.1%}")
    print(f"  RMSE:              {rmse_diff:+.6f} ({'better' if rmse_diff < 0 else 'worse'})")
    print("=" * 70)
    
    return result_baseline, result_nlp


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run ESN model with optional NLP risk index integration"
    )
    parser.add_argument(
        "--no-nlp", 
        action="store_true",
        help="Run baseline without NLP features"
    )
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare baseline vs NLP versions"
    )
    parser.add_argument(
        "--fold", 
        type=int, 
        default=0,
        help="Fold ID to use (default: 0)"
    )
    parser.add_argument(
        "--horizon", 
        type=str, 
        default="target_h1",
        choices=["target_h1", "target_h5", "target_h20"],
        help="Target horizon (default: target_h1)"
    )
    parser.add_argument(
        "--lookback", 
        type=int, 
        default=30,
        help="NLP lookback days (default: 30)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Run comparison
            compare_models(args.fold, args.horizon, args.lookback)
        elif args.no_nlp:
            # Run baseline only
            run_baseline_only(args.fold, args.horizon)
        else:
            # Run with NLP
            run_with_nlp(args.fold, args.horizon, args.lookback)
        
        print("\n[SUCCESS] Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

