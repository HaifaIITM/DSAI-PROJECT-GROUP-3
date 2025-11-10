"""
ESN + Market Sentiment Proxy Pipeline
======================================
Run this script to train the ESN model with market-based sentiment features.
Market proxy: 40% momentum + 30% trend + 20% vol regime + 10% RSI
Validated: +300% average Sharpe improvement (see RESULTS.md)

Usage:
    python run.py                    # Run with sentiment proxy (default)
    python run.py --baseline         # Run baseline without sentiment
    python run.py --compare          # Compare baseline vs sentiment
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


def run_with_sentiment(fold_id=0, horizon="target_h1"):
    """Run pipeline with market sentiment proxy enabled"""
    print_header("ESN + Market Sentiment Proxy Pipeline")
    
    # Enable market sentiment proxy
    settings.SENTIMENT_ENABLED = True
    settings.SENTIMENT_USE_HEADLINES = False
    
    print(f"\n[CONFIG]")
    print(f"  Strategy:          Market Proxy (+300% Sharpe validated)")
    print(f"  Components:        40% momentum, 30% trend, 20% vol, 10% RSI")
    print(f"  Fold ID:           {fold_id}")
    print(f"  Horizon:           {horizon}")
    
    # Step 1: Download
    print("\n[1/5] Downloading data...")
    try:
        res = run_download()
        print(f"  Downloaded {len(res)} datasets")
    except Exception as e:
        print(f"  Using existing data: {e}")
    
    # Step 2: Process with market sentiment
    print("\n[2/5] Processing features and generating market sentiment proxy...")
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
    
    # Verify sentiment index
    verify_risk_index()
    
    # Step 5: Train ESN
    print("\n[5/5] Training ESN with sentiment features...")
    result = run_baseline(model_name="esn", fold_id=fold_id, horizon=horizon)
    print("  Training complete")
    
    # Display results
    print_results(result, "ESN + Market Proxy")
    
    return result


def run_baseline_only(fold_id=0, horizon="target_h1"):
    """Run pipeline without market sentiment proxy (baseline)"""
    print_header("ESN Baseline Pipeline (No Sentiment Proxy)")
    
    # Disable sentiment proxy
    settings.SENTIMENT_ENABLED = False
    
    print(f"\n[CONFIG]")
    print(f"  Sentiment Proxy:   Disabled")
    print(f"  Fold ID:           {fold_id}")
    print(f"  Horizon:           {horizon}")
    
    # Process without sentiment proxy
    print("\n[1/4] Processing features (baseline only)...")
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
    print_results(result, "ESN Baseline")
    
    return result


def compare_models(fold_id=0, horizon="target_h1"):
    """Run both baseline and sentiment proxy versions and compare"""
    print_header("Comparison: ESN Baseline vs ESN + Market Proxy")
    
    # Run baseline
    result_baseline = run_baseline_only(fold_id, horizon)
    
    # Run with sentiment proxy
    result_sentiment = run_with_sentiment(fold_id, horizon)
    
    # Comparison table
    print_header("Comparison Results")
    
    comparison = pd.DataFrame([
        {
            "Model": "ESN Baseline",
            "Features": 10,
            "RMSE": result_baseline['rmse'],
            "MAE": result_baseline['mae'],
            "R²": result_baseline['r2'],
            "Dir.Acc": result_baseline['dir_acc'],
            "Sharpe": result_baseline['backtest']['sharpe'],
            "Hit Ratio": result_baseline['backtest']['hit_ratio']
        },
        {
            "Model": "ESN + Market Proxy",
            "Features": 11,
            "RMSE": result_sentiment['rmse'],
            "MAE": result_sentiment['mae'],
            "R²": result_sentiment['r2'],
            "Dir.Acc": result_sentiment['dir_acc'],
            "Sharpe": result_sentiment['backtest']['sharpe'],
            "Hit Ratio": result_sentiment['backtest']['hit_ratio']
        }
    ])
    
    print("\n" + comparison.to_string(index=False))
    
    # Calculate improvements
    print("\n[IMPROVEMENT METRICS]")
    sharpe_diff = result_sentiment['backtest']['sharpe'] - result_baseline['backtest']['sharpe']
    dir_acc_diff = result_sentiment['dir_acc'] - result_baseline['dir_acc']
    rmse_diff = result_sentiment['rmse'] - result_baseline['rmse']
    
    print(f"  Sharpe Ratio:      {sharpe_diff:+.3f} ({sharpe_diff/abs(result_baseline['backtest']['sharpe'])*100:+.1f}%)")
    print(f"  Dir. Accuracy:     {dir_acc_diff:+.1%}")
    print(f"  RMSE:              {rmse_diff:+.6f} ({'better' if rmse_diff < 0 else 'worse'})")
    print("=" * 70)
    
    return result_baseline, result_sentiment


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run ESN model with market sentiment proxy (validated: +300% Sharpe)"
    )
    parser.add_argument(
        "--baseline", 
        action="store_true",
        help="Run baseline without sentiment proxy"
    )
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare baseline vs sentiment proxy"
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
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Run comparison
            compare_models(args.fold, args.horizon)
        elif args.baseline:
            # Run baseline only
            run_baseline_only(args.fold, args.horizon)
        else:
            # Run with sentiment proxy (default)
            run_with_sentiment(args.fold, args.horizon)
        
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

