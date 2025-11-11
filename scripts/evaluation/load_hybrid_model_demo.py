#!/usr/bin/env python
"""
Demo: Load and use a saved HybridESNRidge model for inference.

This script demonstrates how to:
1. Load a trained hybrid model from disk
2. Load test data
3. Generate predictions
4. Evaluate performance
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


def load_and_predict(fold_id: int = 0, horizon: str = "target_h20"):
    """
    Load saved hybrid model and generate predictions.
    
    Args:
        fold_id: Fold number (must match saved model)
        horizon: Target horizon (must match saved model)
    """
    print("="*80)
    print("HYBRID MODEL INFERENCE DEMO")
    print("="*80)
    
    # Path to saved model
    model_dir = os.path.join(
        settings.EXP_DIR, 
        "hybrid", 
        f"fold_{fold_id}", 
        f"model_{horizon}"
    )
    
    if not os.path.exists(model_dir):
        print(f"[ERROR] Model not found at: {model_dir}")
        print("Please train and save a model first using evaluate_hybrid_model.py")
        return
    
    # Load the trained model
    print(f"\n[1] Loading model from: {model_dir}")
    model = HybridESNRidge.load(model_dir)
    
    # Load test data
    print(f"\n[2] Loading test data for fold {fold_id}...")
    fold_dir = os.path.join(settings.SPLIT_DIR, f"fold_{fold_id}")
    test = pd.read_csv(os.path.join(fold_dir, "test.csv"), index_col=0, parse_dates=True)
    
    X_te = test[[c for c in test.columns if c.startswith("z_")]].values
    y_te = test[horizon].values
    
    print(f"   Test samples: {len(X_te)}")
    print(f"   Features: {X_te.shape[1]}")
    
    # Generate predictions
    print(f"\n[3] Generating predictions...")
    y_pred = model.predict(X_te)
    
    # Evaluate
    print(f"\n[4] Evaluating predictions...")
    metrics = evaluate_predictions(y_te, y_pred)
    backtest = sign_backtest(y_te, y_pred, cost_per_trade=0.0001)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"RMSE:          {metrics['rmse']:.6f}")
    print(f"MAE:           {metrics['mae']:.6f}")
    print(f"R²:            {metrics['r2']:.6f}")
    print(f"Dir Accuracy:  {metrics['dir_acc']:.1%}")
    print()
    print("Backtest (sign-based trading):")
    print(f"  Sharpe:      {backtest['sharpe']:.3f}")
    print(f"  Avg PnL:     {backtest['avg_daily_pnl']:.6f}")
    print(f"  Volatility:  {backtest['vol']:.6f}")
    print(f"  Hit Ratio:   {backtest['hit_ratio']:.1%}")
    print(f"  Turnover:    {backtest['turnover']:.3f}")
    
    # Save predictions to docs/results/
    output_dir = os.path.join(project_root, "docs", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"loaded_model_predictions_fold{fold_id}_{horizon}.csv")
    preds_df = pd.DataFrame({
        "y_true": y_te,
        "y_pred": y_pred
    }, index=test.index)
    preds_df.to_csv(output_file)
    print(f"\n[OK] Predictions saved to: {output_file}")
    
    return model, y_pred, metrics, backtest


def predict_new_data(model: HybridESNRidge, X_new: np.ndarray):
    """
    Use a loaded model to predict on new data.
    
    Args:
        model: Loaded HybridESNRidge model
        X_new: New feature matrix (must have same number of features as training)
    
    Returns:
        Predictions array
    """
    print(f"\n[INFERENCE] Predicting on {len(X_new)} new samples...")
    predictions = model.predict(X_new)
    print(f"[OK] Generated {len(predictions)} predictions")
    return predictions


if __name__ == "__main__":
    # Example 1: Load model and predict on test set
    print("Example 1: Load and evaluate on test set\n")
    model, y_pred, metrics, backtest = load_and_predict(fold_id=0, horizon="target_h20")
    
    # Example 2: Use loaded model for inference on "new" data
    # (For demo purposes, we'll just use the same test data)
    print("\n" + "="*80)
    print("Example 2: Inference on 'new' data")
    print("="*80)
    
    fold_dir = os.path.join(settings.SPLIT_DIR, "fold_0")
    test = pd.read_csv(os.path.join(fold_dir, "test.csv"), index_col=0, parse_dates=True)
    X_new = test[[c for c in test.columns if c.startswith("z_")]].values[:10]  # First 10 samples
    
    new_preds = predict_new_data(model, X_new)
    print(f"\nPredictions for first 10 samples:")
    for i, pred in enumerate(new_preds):
        print(f"  Sample {i+1}: {pred:.6f}")
    
    print("\n" + "="*80)
    print("USAGE SUMMARY")
    print("="*80)
    print("To use your saved model:")
    print("  1. Train and save: python evaluate_hybrid_model.py")
    print("  2. Load model: model = HybridESNRidge.load('path/to/model')")
    print("  3. Predict: y_pred = model.predict(X_new)")
    print()
    print("Model saved to: data/experiments/hybrid/fold_<id>/model_<horizon>/")
    print("="*80)

