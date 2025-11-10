"""
Clean Generated Data
====================
Remove processed data and splits to force regeneration.
Use this before running tests to ensure clean baseline.

Usage:
    python cleanup.py
"""

import os
import shutil

def cleanup():
    """Remove generated data directories"""
    paths = [
        "data/processed",
        "data/splits"
    ]
    
    print("\n" + "="*70)
    print("  Cleaning Generated Data")
    print("="*70 + "\n")
    
    removed = []
    failed = []
    
    for path in paths:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                removed.append(path)
                print(f"✓ Removed: {path}")
            except Exception as e:
                failed.append((path, str(e)))
                print(f"✗ Failed to remove {path}: {e}")
        else:
            print(f"○ Not found: {path} (already clean)")
    
    print("\n" + "="*70)
    if removed:
        print(f"  Successfully cleaned {len(removed)} directories")
    if failed:
        print(f"  Failed to clean {len(failed)} directories")
    if not removed and not failed:
        print("  All directories already clean")
    print("="*70 + "\n")
    
    return len(removed), len(failed)

if __name__ == "__main__":
    removed, failed = cleanup()
    
    if failed > 0:
        print("⚠️  Some directories could not be removed.")
        print("   Try closing any programs that might be using the files.")
        exit(1)
    else:
        print("✅ Ready to run tests with clean baseline!")
        exit(0)

