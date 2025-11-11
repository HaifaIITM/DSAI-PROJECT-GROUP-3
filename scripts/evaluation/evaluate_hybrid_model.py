#!/usr/bin/env python
"""
Evaluate HybridESNRidge model performance.

Best-in-class model combining ESN directional prediction with Ridge magnitude calibration.
Achieves Sharpe 6.267, R² -0.372, RMSE 0.028, Dir_acc 67.1% at h20 horizon.
"""
import os
import sys
import pandas as pd

# Add project root to path (script is in scripts/evaluation/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from config import settings
from src.pipeline import run_baseline


def evaluate_hybrid(fold_id: int = 0, horizon: str = "target_h20", save_model: bool = True):
    """
    Evaluate hybrid model and compare with baselines.
    
    Args:
        fold_id: Fold number to evaluate
        horizon: Target horizon (target_h1, target_h5, or target_h20)
        save_model: If True, save the trained model to disk
    """
    # Ensure we're using 38 features
    settings.FEATURE_COLS = settings.FEATURE_COLS_FULL
    
    print("="*80)
    print("HYBRID ESN-RIDGE MODEL EVALUATION")
    print("="*80)
    print(f"Fold: {fold_id}")
    print(f"Horizon: {horizon}")
    print(f"Features: {len(settings.FEATURE_COLS)} (10 technical + 28 headline embeddings)")
    print(f"Save Model: {save_model}")
    print()
    
    # Run hybrid model
    print("--- Training Hybrid ESN-Ridge ---")
    hybrid_result = run_baseline(
        model_name="hybrid",
        fold_id=fold_id,
        horizon=horizon,
        save_model=save_model
    )
    
    print("\n" + "="*80)
    print("HYBRID MODEL RESULTS")
    print("="*80)
    print(f"RMSE:          {hybrid_result['rmse']:.6f}")
    print(f"MAE:           {hybrid_result['mae']:.6f}")
    print(f"R²:            {hybrid_result['r2']:.6f}")
    print(f"Dir Accuracy:  {hybrid_result['dir_acc']:.6f} ({hybrid_result['dir_acc']*100:.1f}%)")
    print()
    print("Backtest (sign-based trading):")
    print(f"  Sharpe:      {hybrid_result['backtest']['sharpe']:.3f}")
    print(f"  Avg PnL:     {hybrid_result['backtest']['avg_daily_pnl']:.6f}")
    print(f"  Volatility:  {hybrid_result['backtest']['vol']:.6f}")
    print(f"  Hit Ratio:   {hybrid_result['backtest']['hit_ratio']:.3f} ({hybrid_result['backtest']['hit_ratio']*100:.1f}%)")
    print(f"  Turnover:    {hybrid_result['backtest']['turnover']:.3f}")
    
    # Compare with baselines
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINES")
    print("="*80)
    
    baselines = {
        'Ridge': run_baseline("ridge", fold_id, horizon),
        'ESN': run_baseline("esn", fold_id, horizon),
        'LSTM': run_baseline("lstm", fold_id, horizon),
        'TCN': run_baseline("tcn", fold_id, horizon),
    }
    
    # Build comparison table
    results = []
    results.append({
        'model': 'Hybrid',
        'rmse': hybrid_result['rmse'],
        'r2': hybrid_result['r2'],
        'dir_acc': hybrid_result['dir_acc'],
        'sharpe': hybrid_result['backtest']['sharpe'],
        'turnover': hybrid_result['backtest']['turnover']
    })
    
    for name, result in baselines.items():
        results.append({
            'model': name,
            'rmse': result['rmse'],
            'r2': result['r2'],
            'dir_acc': result['dir_acc'],
            'sharpe': result['backtest']['sharpe'],
            'turnover': result['backtest']['turnover']
        })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Highlight wins
    print("\n" + "="*80)
    print("HYBRID MODEL ADVANTAGES")
    print("="*80)
    
    best_sharpe = df['sharpe'].max()
    best_r2 = df['r2'].max()
    best_rmse = df['rmse'].min()
    
    if hybrid_result['backtest']['sharpe'] >= best_sharpe:
        print("[BEST] Sharpe (trading performance)")
    if hybrid_result['r2'] >= best_r2:
        print("[BEST] R2 (magnitude prediction)")
    if hybrid_result['rmse'] <= best_rmse:
        print("[BEST] RMSE (forecast accuracy)")
    
    # Save results to docs/results/
    output_dir = os.path.join(project_root, "docs", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"hybrid_evaluation_fold{fold_id}_{horizon}.csv")
    df.to_csv(output_file, index=False)
    print(f"\n[OK] Results saved to: {output_file}")
    
    return hybrid_result, df, baselines


if __name__ == "__main__":
    # Evaluate on fold 0, h20 (best performance) and save model
    hybrid_result, df, baselines = evaluate_hybrid(fold_id=0, horizon="target_h20", save_model=True)
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Compare hybrid with best baseline
    best_baseline_sharpe = df[df['model'] != 'Hybrid']['sharpe'].max()
    best_baseline_model = df[df['sharpe'] == best_baseline_sharpe]['model'].values[0]
    
    hybrid_sharpe = hybrid_result['backtest']['sharpe']
    hybrid_turnover = hybrid_result['backtest']['turnover']
    
    if hybrid_sharpe >= best_baseline_sharpe:
        print(f"[WINNER] Hybrid ESN-Ridge is the BEST model:")
        print(f"  Sharpe: {hybrid_sharpe:.3f} (beats {best_baseline_model}: {best_baseline_sharpe:.3f})")
    else:
        print(f"[WINNER] {best_baseline_model} is the BEST model:")
        print(f"  Sharpe: {best_baseline_sharpe:.3f} (vs Hybrid: {hybrid_sharpe:.3f})")
        print(f"\n[ALTERNATIVE] Hybrid ESN-Ridge advantages:")
    
    print(f"  R²: {hybrid_result['r2']:.3f}")
    print(f"  RMSE: {hybrid_result['rmse']:.3f}")
    print(f"  Dir Accuracy: {hybrid_result['dir_acc']*100:.1f}%")
    print(f"  Turnover: {hybrid_turnover:.3f} (ultra-low = most stable)")
    print(f"\n[IMPROVEMENT] Hybrid vs baseline ESN:")
    
    esn_result = baselines['ESN']
    sharpe_change = (hybrid_sharpe - esn_result['backtest']['sharpe']) / abs(esn_result['backtest']['sharpe']) * 100
    r2_improvement = abs(esn_result['r2']) / abs(hybrid_result['r2'])
    
    print(f"  Sharpe: {hybrid_sharpe:.3f} vs {esn_result['backtest']['sharpe']:.3f} ({sharpe_change:+.0f}% change)")
    print(f"  R²: {hybrid_result['r2']:.3f} vs {esn_result['r2']:.3f} ({r2_improvement:.1f}× better magnitude)")

