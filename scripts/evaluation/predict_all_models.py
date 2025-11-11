#!/usr/bin/env python
"""
Load and predict using ALL trained hybrid models.

Shows different strategies:
1. Single best model
2. Ensemble across folds (for one horizon)
3. Ensemble across all folds and horizons
4. Weighted ensemble
"""
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path (script is in scripts/evaluation/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.models.hybrid_esn_ridge import HybridESNRidge
from src.eval.metrics import evaluate_predictions, sign_backtest
from config import settings


def load_all_models(horizons=None, folds=None):
    """
    Load all trained models.
    
    Args:
        horizons: List of horizons to load (default: all 3)
        folds: List of folds to load (default: all 9)
    
    Returns:
        Dict mapping (fold, horizon) -> model
    """
    if horizons is None:
        horizons = ["target_h1", "target_h5", "target_h20"]
    if folds is None:
        folds = list(range(9))
    
    models = {}
    missing = []
    
    print("Loading models...")
    for fold_id in folds:
        for horizon in horizons:
            model_dir = os.path.join(
                settings.EXP_DIR,
                "hybrid",
                f"fold_{fold_id}",
                f"model_{horizon}"
            )
            
            if os.path.exists(model_dir):
                try:
                    model = HybridESNRidge.load(model_dir)
                    models[(fold_id, horizon)] = model
                    print(f"  ✓ Loaded fold={fold_id}, horizon={horizon}")
                except Exception as e:
                    print(f"  ✗ Error loading fold={fold_id}, horizon={horizon}: {e}")
                    missing.append((fold_id, horizon))
            else:
                print(f"  - Not found: fold={fold_id}, horizon={horizon}")
                missing.append((fold_id, horizon))
    
    print(f"\nLoaded {len(models)}/{len(folds)*len(horizons)} models")
    if missing:
        print(f"Missing {len(missing)} models. Train them first with: python train_all_hybrid_models.py")
    
    return models


def predict_single_best(models, X_test, y_test, test_index):
    """Strategy 1: Use single best model (highest Sharpe on training)."""
    print("\n" + "="*80)
    print("STRATEGY 1: SINGLE BEST MODEL")
    print("="*80)
    
    # Based on your evaluation, fold_0 + target_h20 has best Sharpe
    best_key = (0, "target_h20")
    
    if best_key not in models:
        print(f"Best model not found: fold={best_key[0]}, horizon={best_key[1]}")
        return None
    
    model = models[best_key]
    predictions = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_predictions(y_test, predictions)
    backtest = sign_backtest(y_test, predictions, cost_per_trade=0.0001)
    
    print(f"Model: fold={best_key[0]}, horizon={best_key[1]}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    print(f"Dir Acc: {metrics['dir_acc']:.1%}")
    print(f"Sharpe: {backtest['sharpe']:.3f}")
    
    return predictions


def predict_ensemble_by_horizon(models, X_test, y_test, test_index, horizon="target_h20"):
    """Strategy 2: Ensemble across all folds for one horizon."""
    print("\n" + "="*80)
    print(f"STRATEGY 2: ENSEMBLE ACROSS FOLDS (horizon={horizon})")
    print("="*80)
    
    # Get all models for this horizon
    horizon_models = [(k, v) for k, v in models.items() if k[1] == horizon]
    
    if not horizon_models:
        print(f"No models found for horizon={horizon}")
        return None
    
    print(f"Using {len(horizon_models)} models")
    
    # Generate predictions from each model
    all_preds = []
    for (fold_id, h), model in horizon_models:
        preds = model.predict(X_test)
        all_preds.append(preds)
        print(f"  Model fold={fold_id}")
    
    # Average predictions
    ensemble_pred = np.mean(all_preds, axis=0)
    
    # Evaluate
    metrics = evaluate_predictions(y_test, ensemble_pred)
    backtest = sign_backtest(y_test, ensemble_pred, cost_per_trade=0.0001)
    
    print(f"\nEnsemble Results:")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    print(f"Dir Acc: {metrics['dir_acc']:.1%}")
    print(f"Sharpe: {backtest['sharpe']:.3f}")
    
    return ensemble_pred


def predict_ensemble_all(models, X_test, y_test, test_index):
    """Strategy 3: Ensemble across ALL folds and horizons."""
    print("\n" + "="*80)
    print("STRATEGY 3: ENSEMBLE ACROSS ALL FOLDS & HORIZONS")
    print("="*80)
    
    print(f"Using {len(models)} models")
    
    # Generate predictions from ALL models
    all_preds = []
    for (fold_id, horizon), model in models.items():
        preds = model.predict(X_test)
        all_preds.append(preds)
    
    # Average predictions
    ensemble_pred = np.mean(all_preds, axis=0)
    
    # Evaluate
    metrics = evaluate_predictions(y_test, ensemble_pred)
    backtest = sign_backtest(y_test, ensemble_pred, cost_per_trade=0.0001)
    
    print(f"Ensemble Results:")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    print(f"Dir Acc: {metrics['dir_acc']:.1%}")
    print(f"Sharpe: {backtest['sharpe']:.3f}")
    
    return ensemble_pred


def predict_weighted_ensemble(models, X_test, y_test, test_index, weights=None):
    """Strategy 4: Weighted ensemble (weight by training Sharpe/performance)."""
    print("\n" + "="*80)
    print("STRATEGY 4: WEIGHTED ENSEMBLE")
    print("="*80)
    
    # If no weights provided, use uniform
    if weights is None:
        # Load results to get Sharpe ratios
        results_file = os.path.join(project_root, "docs", "results", "hybrid_all_folds_all_horizons_results.csv")
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            
            # Use Sharpe as weights (higher Sharpe = higher weight)
            # Normalize to [0, 1] and ensure positive
            sharpes = results_df.set_index(['fold', 'horizon'])['sharpe']
            sharpes = sharpes.clip(lower=0)  # Remove negative Sharpe
            weights = sharpes / sharpes.sum()
            print(f"Using Sharpe-based weights from {results_file}")
        else:
            print("No results file found, using uniform weights")
            weights = {k: 1.0 / len(models) for k in models.keys()}
    
    # Generate weighted predictions
    ensemble_pred = np.zeros(len(X_test))
    total_weight = 0
    
    for (fold_id, horizon), model in models.items():
        preds = model.predict(X_test)
        weight = weights.get((fold_id, horizon), 0)
        ensemble_pred += weight * preds
        total_weight += weight
        print(f"  Model fold={fold_id}, horizon={horizon}, weight={weight:.4f}")
    
    # Normalize
    if total_weight > 0:
        ensemble_pred /= total_weight
    
    # Evaluate
    metrics = evaluate_predictions(y_test, ensemble_pred)
    backtest = sign_backtest(y_test, ensemble_pred, cost_per_trade=0.0001)
    
    print(f"\nWeighted Ensemble Results:")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    print(f"Dir Acc: {metrics['dir_acc']:.1%}")
    print(f"Sharpe: {backtest['sharpe']:.3f}")
    
    return ensemble_pred


def compare_strategies(test_fold=0, test_horizon="target_h20"):
    """Compare all prediction strategies."""
    print("="*80)
    print("COMPARING PREDICTION STRATEGIES")
    print("="*80)
    
    # Load test data
    fold_dir = os.path.join(settings.SPLIT_DIR, f"fold_{test_fold}")
    test = pd.read_csv(os.path.join(fold_dir, "test.csv"), index_col=0, parse_dates=True)
    X_test = test[[c for c in test.columns if c.startswith("z_")]].values
    y_test = test[test_horizon].values
    
    print(f"Test data: fold={test_fold}, horizon={test_horizon}")
    print(f"Samples: {len(y_test)}")
    print()
    
    # Load all trained models
    models = load_all_models()
    
    if not models:
        print("\n[ERROR] No models found. Train models first:")
        print("  python train_all_hybrid_models.py")
        return
    
    # Run all strategies
    results = {}
    
    # Strategy 1: Single best
    pred1 = predict_single_best(models, X_test, y_test, test.index)
    if pred1 is not None:
        results['single_best'] = pred1
    
    # Strategy 2: Ensemble by horizon
    pred2 = predict_ensemble_by_horizon(models, X_test, y_test, test.index, test_horizon)
    if pred2 is not None:
        results['ensemble_horizon'] = pred2
    
    # Strategy 3: Ensemble all
    pred3 = predict_ensemble_all(models, X_test, y_test, test.index)
    if pred3 is not None:
        results['ensemble_all'] = pred3
    
    # Strategy 4: Weighted ensemble
    pred4 = predict_weighted_ensemble(models, X_test, y_test, test.index)
    if pred4 is not None:
        results['weighted_ensemble'] = pred4
    
    # Save all predictions
    output_df = pd.DataFrame({'y_true': y_test}, index=test.index)
    for name, preds in results.items():
        output_df[f'pred_{name}'] = preds
    
    # Save to docs/results/
    output_dir = os.path.join(project_root, "docs", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"all_strategies_predictions_fold{test_fold}_{test_horizon}.csv")
    output_df.to_csv(output_file)
    print(f"\n[OK] All predictions saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict using all trained models")
    parser.add_argument("--fold", type=int, default=0, help="Test fold (0-8)")
    parser.add_argument("--horizon", type=str, default="target_h20", 
                       choices=["target_h1", "target_h5", "target_h20"],
                       help="Target horizon")
    args = parser.parse_args()
    
    compare_strategies(test_fold=args.fold, test_horizon=args.horizon)
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("For production, choose based on:")
    print("  - Single Best: Simplest, fastest inference")
    print("  - Ensemble by Horizon: Balance of robustness and performance")
    print("  - Ensemble All: Maximum robustness, slower inference")
    print("  - Weighted: Best theoretical performance if weights are good")
    print("="*80)

