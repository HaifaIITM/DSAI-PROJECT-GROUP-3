#!/usr/bin/env python
"""
Quick test script to verify headline embedding integration.
"""
import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from config import settings
from src.data.embeddings import compute_headline_features

def test_headline_pipeline():
    """Test that headline embeddings can be generated."""
    print("=" * 60)
    print("Testing Headline Embedding Pipeline")
    print("=" * 60)
    
    # Check if headlines file exists
    if not os.path.exists(settings.HEADLINES_CSV):
        print(f"[X] Headlines file not found: {settings.HEADLINES_CSV}")
        print("Please ensure spy_news.csv is in the project root.")
        return False
    
    print(f"[OK] Found headlines file: {settings.HEADLINES_CSV}")
    
    # Create sample date range (2024 trading days)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")  # Business days
    print(f"[OK] Testing with {len(dates)} trading days")
    
    # Generate headline features
    try:
        headline_feats = compute_headline_features(
            target_dates=dates,
            headlines_csv=settings.HEADLINES_CSV,
            small_model=settings.SMALL_MODEL,
            large_model=settings.LARGE_MODEL,
            small_pca_dim=settings.SMALL_PCA_DIM,
            large_pca_dim=settings.LARGE_PCA_DIM,
            random_state=settings.RANDOM_SEED,
            agg_method=settings.AGG_METHOD
        )
        
        if headline_feats is not None:
            print(f"\n[OK] Successfully generated headline features!")
            print(f"   Shape: {headline_feats.shape}")
            print(f"   Columns: {list(headline_feats.columns)}")
            print(f"\nSample data (first 5 rows):")
            print(headline_feats.head())
            
            # Check for expected columns
            expected_small = [f"pca_{i+1}" for i in range(settings.SMALL_PCA_DIM)]
            expected_large = [f"pca_{i+1}_large" for i in range(settings.LARGE_PCA_DIM)]
            expected_flags = ["has_news", "has_news_large"]
            
            all_expected = expected_small + expected_large + expected_flags
            missing = [col for col in all_expected if col not in headline_feats.columns]
            
            if missing:
                print(f"\n[WARN] Warning: Missing expected columns: {missing}")
            else:
                print(f"\n[OK] All {len(all_expected)} expected columns present")
            
            return True
        else:
            print("[X] Failed to generate headline features")
            return False
            
    except Exception as e:
        print(f"[X] Error during headline feature generation:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_headline_pipeline()
    sys.exit(0 if success else 1)

