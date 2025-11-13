"""
Simple test runner for API tests.

Usage:
    python run_tests.py
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Change to backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    
    # Run tests
    result = subprocess.run([sys.executable, "test_api.py"], cwd=backend_dir)
    sys.exit(result.returncode)

