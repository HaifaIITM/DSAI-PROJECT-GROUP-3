"""
Hugging Face Spaces entry point.
This file serves as an alternative entry point for HF Spaces.
"""
import os
import sys

# Add paths (app.py is in application/ directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

# Import and run the FastAPI app
from backend.main import app

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print("="*60)
    print("Starting Hybrid ESN-Ridge Prediction API (HF Spaces)")
    print(f"Host: {host}, Port: {port}")
    print("="*60)
    
    uvicorn.run(app, host=host, port=port, log_level="info")

