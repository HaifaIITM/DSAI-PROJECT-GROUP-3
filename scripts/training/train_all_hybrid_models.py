#!/usr/bin/env python
"""
Train Hybrid models on ALL folds and ALL horizons.

This script:
1. Trains hybrid models on all 9 folds (fold_0 to fold_8)
2. For all 3 horizons (target_h1, target_h5, target_h20)
3. Saves all 27 models to disk
4. Generates comprehensive results
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path (script is in scripts/training/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from config import settings
from src.pipeline import run_baseline


def train_all_models(save_models=True):
    """
    Train hybrid models on all folds and horizons.
    
    Args:
        save_models: If True, save all trained models to disk
    """
    # Configuration
    folds = list(range(9))  # fold_0 to fold_8
    horizons = ["target_h1", "target_h5", "target_h20"]
    
    # Ensure we use all features
    settings.FEATURE_COLS = settings.FEATURE_COLS_FULL
    
    print("="*80)
    print("TRAINING HYBRID MODELS - ALL FOLDS × ALL HORIZONS")
    print("="*80)
    print(f"Folds: {len(folds)} ({min(folds)} to {max(folds)})")
    print(f"Horizons: {len(horizons)} {horizons}")
    print(f"Total models to train: {len(folds) * len(horizons)}")
    print(f"Features: {len(settings.FEATURE_COLS)} (10 technical + 28 headline embeddings)")
    print(f"Save models: {save_models}")
    print("="*80)
    print()
    
    # Store results
    results = []
    
    # Train all combinations
    total = len(folds) * len(horizons)
    with tqdm(total=total, desc="Training", ncols=100) as pbar:
        for fold_id in folds:
            for horizon in horizons:
                try:
                    # Train model
                    result = run_baseline(
                        model_name="hybrid",
                        fold_id=fold_id,
                        horizon=horizon,
                        save_model=save_models
                    )
                    
                    # Store result
                    results.append({
                        'fold': fold_id,
                        'horizon': horizon,
                        'rmse': result['rmse'],
                        'mae': result['mae'],
                        'r2': result['r2'],
                        'dir_acc': result['dir_acc'],
                        'sharpe': result['backtest']['sharpe'],
                        'avg_pnl': result['backtest']['avg_daily_pnl'],
                        'vol': result['backtest']['vol'],
                        'hit_ratio': result['backtest']['hit_ratio'],
                        'turnover': result['backtest']['turnover']
                    })
                    
                    pbar.set_postfix({
                        'fold': fold_id,
                        'horizon': horizon.split('_')[1],
                        'sharpe': f"{result['backtest']['sharpe']:.2f}"
                    })
                    
                except Exception as e:
                    print(f"\n[ERROR] fold={fold_id}, horizon={horizon}: {e}")
                    results.append({
                        'fold': fold_id,
                        'horizon': horizon,
                        'error': str(e)
                    })
                
                pbar.update(1)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    return pd.DataFrame(results)


def analyze_results(results_df):
    """Analyze and display results across all folds and horizons."""
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print("\n1. OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total models trained: {len(results_df)}")
    print(f"Successful: {len(results_df[~results_df['sharpe'].isna()])}")
    if 'error' in results_df.columns:
        print(f"Failed: {results_df['error'].notna().sum()}")
    
    # Aggregate by horizon
    print("\n2. PERFORMANCE BY HORIZON (averaged across folds)")
    print("-" * 80)
    by_horizon = results_df.groupby('horizon').agg({
        'rmse': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'dir_acc': ['mean', 'std'],
        'sharpe': ['mean', 'std'],
        'turnover': ['mean', 'std']
    }).round(4)
    print(by_horizon)
    
    # Aggregate by fold
    print("\n3. PERFORMANCE BY FOLD (averaged across horizons)")
    print("-" * 80)
    by_fold = results_df.groupby('fold').agg({
        'rmse': 'mean',
        'r2': 'mean',
        'dir_acc': 'mean',
        'sharpe': 'mean',
        'turnover': 'mean'
    }).round(4)
    print(by_fold)
    
    # Best models
    print("\n4. BEST MODELS")
    print("-" * 80)
    
    # Best by Sharpe
    best_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
    print(f"Best Sharpe: {best_sharpe['sharpe']:.3f}")
    print(f"  → Fold {int(best_sharpe['fold'])}, {best_sharpe['horizon']}")
    print(f"  → Model: data/experiments/hybrid/fold_{int(best_sharpe['fold'])}/model_{best_sharpe['horizon']}/")
    
    # Best by R²
    best_r2 = results_df.loc[results_df['r2'].idxmax()]
    print(f"\nBest R²: {best_r2['r2']:.3f}")
    print(f"  → Fold {int(best_r2['fold'])}, {best_r2['horizon']}")
    
    # Best by Dir Accuracy
    best_dir = results_df.loc[results_df['dir_acc'].idxmax()]
    print(f"\nBest Dir Accuracy: {best_dir['dir_acc']:.1%}")
    print(f"  → Fold {int(best_dir['fold'])}, {best_dir['horizon']}")
    
    # Save results to docs/results/
    output_dir = os.path.join(project_root, "docs", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "hybrid_all_folds_all_horizons_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n[OK] Detailed results saved to: {output_file}")
    
    return by_horizon, by_fold


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train hybrid models on all folds and horizons")
    parser.add_argument("--no-save", action="store_true", help="Don't save models to disk")
    args = parser.parse_args()
    
    # Train all models
    results_df = train_all_models(save_models=not args.no_save)
    
    # Analyze results
    by_horizon, by_fold = analyze_results(results_df)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. View detailed results: docs/results/hybrid_all_folds_all_horizons_results.csv")
    print("2. Load any model:")
    print("   from src.models.hybrid_esn_ridge import HybridESNRidge")
    print("   model = HybridESNRidge.load('data/experiments/hybrid/fold_X/model_target_hY')")
    print("3. Use ensemble predictions (see predict_all_models.py)")
    print("="*80)

