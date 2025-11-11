#!/usr/bin/env python
"""
Quick test to verify hybrid model save/load functionality.
"""
import os
import sys
import numpy as np
import tempfile
import shutil
sys.path.append(os.getcwd())

from src.models.hybrid_esn_ridge import HybridESNRidge


def test_save_load():
    """Test that model can be saved and loaded correctly."""
    print("="*80)
    print("TESTING HYBRID MODEL SAVE/LOAD")
    print("="*80)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 38
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples) * 0.02
    X_test = np.random.randn(50, n_features)
    
    print(f"\n[1] Creating and training model...")
    print(f"    Training samples: {n_samples}")
    print(f"    Features: {n_features}")
    
    # Train model
    model = HybridESNRidge(
        hidden_size=200,  # Small for quick test
        spectral_radius=0.85,
        leak_rate=0.3,
        esn_alpha=0.3,
        ridge_alpha=1.0,
        washout=50,
        seed=42
    )
    model.fit(X_train, y_train)
    
    # Get predictions before saving
    print(f"\n[2] Generating predictions (before save)...")
    y_pred_before = model.predict(X_test)
    print(f"    Predictions shape: {y_pred_before.shape}")
    print(f"    Sample predictions: {y_pred_before[:5]}")
    
    # Save model to temp directory
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, "test_hybrid_model")
    
    print(f"\n[3] Saving model to: {save_path}")
    model.save(save_path)
    
    # Verify files were created
    expected_files = ["config.json", "esn_weights.npz", "ridge_model.pkl"]
    for fname in expected_files:
        fpath = os.path.join(save_path, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    [OK] {fname} ({size_kb:.2f} KB)")
        else:
            print(f"    [ERROR] {fname} MISSING!")
            return False
    
    # Load model
    print(f"\n[4] Loading model from disk...")
    loaded_model = HybridESNRidge.load(save_path)
    
    # Get predictions after loading
    print(f"\n[5] Generating predictions (after load)...")
    y_pred_after = loaded_model.predict(X_test)
    print(f"    Predictions shape: {y_pred_after.shape}")
    print(f"    Sample predictions: {y_pred_after[:5]}")
    
    # Compare predictions
    print(f"\n[6] Comparing predictions...")
    max_diff = np.max(np.abs(y_pred_before - y_pred_after))
    mean_diff = np.mean(np.abs(y_pred_before - y_pred_after))
    
    print(f"    Max absolute difference:  {max_diff:.2e}")
    print(f"    Mean absolute difference: {mean_diff:.2e}")
    
    # Check if predictions match
    tolerance = 1e-10
    if max_diff < tolerance:
        print(f"\n[PASS] TEST PASSED! Predictions match (diff < {tolerance})")
        success = True
    else:
        print(f"\n[FAIL] TEST FAILED! Predictions differ (diff = {max_diff})")
        success = False
    
    # Cleanup
    print(f"\n[7] Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print(f"    Removed: {temp_dir}")
    
    print("\n" + "="*80)
    if success:
        print("ALL TESTS PASSED [OK]")
    else:
        print("TESTS FAILED [FAIL]")
    print("="*80)
    
    return success


if __name__ == "__main__":
    try:
        success = test_save_load()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

