"""
Production Predictor - Hybrid Model Inference

Uses the 3 best-performing models:
- h1:  fold_3, Sharpe 1.25  (day trading)
- h5:  fold_8, Sharpe 2.94  (swing trading)
- h20: fold_3, Sharpe 6.81  (position trading)

Usage:
    predictor = ProductionPredictor()
    predictions = predictor.predict(X_new, horizon='h20')
"""
import os
import numpy as np
from typing import Literal

from src.models.hybrid_esn_ridge import HybridESNRidge
from config import settings


class ProductionPredictor:
    """
    Production-ready predictor using best hybrid models.
    
    Models are loaded once at initialization and cached for fast inference.
    """
    
    # Best model configurations
    BEST_MODELS = {
        'h1':  {'fold': 3, 'sharpe': 1.25},
        'h5':  {'fold': 8, 'sharpe': 2.94},
        'h20': {'fold': 3, 'sharpe': 6.813}
    }
    
    def __init__(self, base_dir: str = None):
        """
        Initialize predictor and load all 3 best models.
        
        Args:
            base_dir: Base directory for models (default: from settings)
        """
        self.base_dir = base_dir or settings.EXP_DIR
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all 3 best models into memory."""
        print("Loading production models...")
        
        for horizon, config in self.BEST_MODELS.items():
            fold = config['fold']
            model_path = os.path.join(
                self.base_dir,
                "hybrid",
                f"fold_{fold}",
                f"model_target_{horizon}"
            )
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Train models first: python train_all_hybrid_models.py"
                )
            
            self.models[horizon] = HybridESNRidge.load(model_path)
            print(f"  [OK] Loaded {horizon}: fold_{fold} (Sharpe {config['sharpe']})")
        
        print(f"All {len(self.models)} models loaded successfully.\n")
    
    def predict(
        self, 
        X: np.ndarray, 
        horizon: Literal['h1', 'h5', 'h20'] = 'h20'
    ) -> np.ndarray:
        """
        Generate predictions for given horizon.
        
        Args:
            X: Feature matrix (n_samples, 38 features)
            horizon: Prediction horizon ('h1', 'h5', or 'h20')
        
        Returns:
            Predictions array (n_samples,)
        
        Example:
            predictions = predictor.predict(X_new, horizon='h20')
            signals = np.sign(predictions)  # +1 buy, -1 sell
        """
        if horizon not in self.models:
            raise ValueError(
                f"Invalid horizon '{horizon}'. "
                f"Must be one of: {list(self.models.keys())}"
            )
        
        if X.shape[1] != 38:
            raise ValueError(
                f"Expected 38 features, got {X.shape[1]}. "
                f"Ensure you're using FEATURE_COLS_FULL."
            )
        
        model = self.models[horizon]
        predictions = model.predict(X)
        
        return predictions
    
    def predict_all(self, X: np.ndarray) -> dict:
        """
        Generate predictions for all 3 horizons.
        
        Args:
            X: Feature matrix (n_samples, 38 features)
        
        Returns:
            Dict with keys 'h1', 'h5', 'h20' containing predictions
        
        Example:
            all_preds = predictor.predict_all(X_new)
            short_term = all_preds['h1']
            medium_term = all_preds['h5']
            long_term = all_preds['h20']
        """
        return {
            horizon: self.predict(X, horizon=horizon)
            for horizon in ['h1', 'h5', 'h20']
        }
    
    def get_signals(
        self, 
        X: np.ndarray, 
        horizon: Literal['h1', 'h5', 'h20'] = 'h20'
    ) -> np.ndarray:
        """
        Generate trading signals (+1 buy, -1 sell, 0 neutral).
        
        Args:
            X: Feature matrix (n_samples, 38 features)
            horizon: Prediction horizon
        
        Returns:
            Signals array: +1 (buy), -1 (sell), 0 (neutral/no position)
        """
        predictions = self.predict(X, horizon=horizon)
        signals = np.sign(predictions)
        return signals
    
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            horizon: {
                'fold': config['fold'],
                'sharpe': config['sharpe'],
                'path': os.path.join(
                    self.base_dir, 
                    "hybrid", 
                    f"fold_{config['fold']}", 
                    f"model_target_{horizon}"
                )
            }
            for horizon, config in self.BEST_MODELS.items()
        }


# Convenience function for quick inference
def predict_production(X: np.ndarray, horizon: str = 'h20') -> np.ndarray:
    """
    Quick prediction function (loads models on each call).
    
    For repeated predictions, use ProductionPredictor class instead
    to avoid reloading models.
    
    Args:
        X: Feature matrix (n_samples, 38 features)
        horizon: 'h1', 'h5', or 'h20'
    
    Returns:
        Predictions array
    """
    predictor = ProductionPredictor()
    return predictor.predict(X, horizon=horizon)


if __name__ == "__main__":
    """Demo usage"""
    import pandas as pd
    
    print("="*80)
    print("PRODUCTION PREDICTOR - DEMO")
    print("="*80)
    print()
    
    # Initialize predictor (loads models once)
    predictor = ProductionPredictor()
    
    # Show model info
    print("Model Configuration:")
    print("-" * 80)
    for horizon, info in predictor.get_model_info().items():
        print(f"{horizon:4s}: fold_{info['fold']} | Sharpe {info['sharpe']:.2f}")
    print()
    
    # Load sample test data
    print("Loading sample data...")
    test_path = os.path.join(settings.SPLIT_DIR, "fold_0", "test.csv")
    test = pd.read_csv(test_path, index_col=0, parse_dates=True)
    X_sample = test[[c for c in test.columns if c.startswith("z_")]].values[:10]
    
    print(f"Sample shape: {X_sample.shape}")
    print()
    
    # Example 1: Single horizon prediction
    print("Example 1: Predict h20 (position trading)")
    print("-" * 80)
    pred_h20 = predictor.predict(X_sample, horizon='h20')
    signals_h20 = predictor.get_signals(X_sample, horizon='h20')
    
    print("Date          | Prediction  | Signal")
    print("-" * 45)
    for i, (pred, sig) in enumerate(zip(pred_h20[:5], signals_h20[:5])):
        signal_str = "BUY" if sig > 0 else "SELL" if sig < 0 else "HOLD"
        print(f"Sample {i+1:2d}    | {pred:+.6f} | {signal_str}")
    print()
    
    # Example 2: All horizons
    print("Example 2: Predict all horizons")
    print("-" * 80)
    all_preds = predictor.predict_all(X_sample)
    
    print("Sample | h1 (1-day)  | h5 (5-day)  | h20 (20-day)")
    print("-" * 60)
    for i in range(5):
        print(f"{i+1:6d} | {all_preds['h1'][i]:+.6f} | "
              f"{all_preds['h5'][i]:+.6f} | {all_preds['h20'][i]:+.6f}")
    print()
    
    # Example 3: Generate signals for all horizons
    print("Example 3: Trading signals across timeframes")
    print("-" * 80)
    signals_all = {h: predictor.get_signals(X_sample, h) for h in ['h1', 'h5', 'h20']}
    
    print("Sample | h1     | h5     | h20    | Consensus")
    print("-" * 60)
    for i in range(5):
        h1_sig = "BUY " if signals_all['h1'][i] > 0 else "SELL"
        h5_sig = "BUY " if signals_all['h5'][i] > 0 else "SELL"
        h20_sig = "BUY " if signals_all['h20'][i] > 0 else "SELL"
        consensus = sum([signals_all['h1'][i], signals_all['h5'][i], signals_all['h20'][i]])
        cons_str = "STRONG BUY" if consensus >= 2 else "STRONG SELL" if consensus <= -2 else "MIXED"
        
        print(f"{i+1:6d} | {h1_sig:4s} | {h5_sig:4s} | {h20_sig:4s} | {cons_str}")
    
    print()
    print("="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nFor production use:")
    print("  predictor = ProductionPredictor()")
    print("  predictions = predictor.predict(X_new, horizon='h20')")
    print("  signals = predictor.get_signals(X_new, horizon='h20')")

