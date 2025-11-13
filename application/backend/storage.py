"""
Storage utilities for headlines, predictions, embeddings, and features.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class DataStorage:
    """
    Manages persistent storage for:
    - News headlines (accumulated over time)
    - Predictions with metadata
    - Embeddings used
    - Final features (38 features)
    """
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize storage with base directory.
        
        Args:
            storage_dir: Base directory for storage (default: application/backend/data)
        """
        if storage_dir is None:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            storage_dir = os.path.join(backend_dir, "data")
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.headlines_dir = self.storage_dir / "headlines"
        self.predictions_dir = self.storage_dir / "predictions"
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.features_dir = self.storage_dir / "features"
        
        # Create subdirectories
        for dir_path in [self.headlines_dir, self.predictions_dir, 
                        self.embeddings_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Headlines file
        self.headlines_file = self.headlines_dir / "spy_headlines.csv"
    
    def save_headline(self, headline_data: Dict[str, Any]) -> bool:
        """
        Save a single headline to storage (append if exists).
        
        Args:
            headline_data: Dict with keys: date, title, publisher, link
        
        Returns:
            True if saved successfully
        """
        try:
            # Ensure all required columns are present
            required_cols = ['date', 'title', 'publisher', 'link']
            for col in required_cols:
                if col not in headline_data:
                    headline_data[col] = '' if col != 'date' else datetime.now().strftime('%Y-%m-%d %H:%M')
            
            # Convert to DataFrame
            df_new = pd.DataFrame([headline_data])
            df_new['date'] = pd.to_datetime(df_new['date'], errors='coerce')
            
            # Drop rows with invalid dates
            df_new = df_new.dropna(subset=['date'])
            if len(df_new) == 0:
                return False
            
            # Load existing headlines
            if self.headlines_file.exists():
                try:
                    df_existing = pd.read_csv(self.headlines_file, parse_dates=['date'])
                    # Check if headline already exists (by date + title)
                    date_match = df_existing['date'].dt.date == df_new['date'].dt.date[0]
                    title_match = df_existing['title'] == df_new['title'].iloc[0]
                    if not df_existing[date_match & title_match].empty:
                        return False  # Already exists
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                except Exception as e:
                    print(f"Error reading existing headlines: {e}")
                    df_combined = df_new
            else:
                df_combined = df_new
            
            # Remove duplicates and sort by date
            df_combined = df_combined.drop_duplicates(subset=['date', 'title'])
            df_combined = df_combined.sort_values('date')
            
            # Ensure all required columns exist
            for col in required_cols:
                if col not in df_combined.columns:
                    df_combined[col] = ''
            
            # Save
            df_combined[required_cols].to_csv(self.headlines_file, index=False)
            return True
        
        except Exception as e:
            print(f"Error saving headline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_headlines_batch(self, headlines: List[Dict[str, Any]]) -> int:
        """
        Save multiple headlines at once.
        
        Args:
            headlines: List of headline dicts
        
        Returns:
            Number of new headlines saved
        """
        saved_count = 0
        for headline in headlines:
            if self.save_headline(headline):
                saved_count += 1
        return saved_count
    
    def get_headlines(self, days_back: Optional[int] = None, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get stored headlines with optional date filtering.
        
        Args:
            days_back: Get headlines from last N days
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with headlines
        """
        if not self.headlines_file.exists():
            return pd.DataFrame(columns=['date', 'title', 'publisher', 'link'])
        
        df = pd.read_csv(self.headlines_file, parse_dates=['date'])
        
        if days_back:
            cutoff = datetime.now() - pd.Timedelta(days=days_back)
            df = df[df['date'] >= cutoff]
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        return df.sort_values('date')
    
    def get_headlines_csv_path(self) -> str:
        """Get path to headlines CSV file."""
        return str(self.headlines_file)
    
    def save_prediction(self, prediction_data: Dict[str, Any], 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save prediction with metadata.
        
        Args:
            prediction_data: Dict with prediction results
            metadata: Optional metadata (model info, feature info, etc.)
        
        Returns:
            Path to saved prediction file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{timestamp}.json"
        filepath = self.predictions_dir / filename
        
        save_data = {
            "timestamp": timestamp,
            "generated_at": datetime.now().isoformat(),
            "prediction": prediction_data,
            "metadata": metadata or {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        return str(filepath)
    
    def save_predictions_batch(self, predictions: List[Dict[str, Any]],
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save multiple predictions in a single file.
        
        Args:
            predictions: List of prediction dicts
            metadata: Optional metadata
        
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_batch_{timestamp}.json"
        filepath = self.predictions_dir / filename
        
        save_data = {
            "timestamp": timestamp,
            "generated_at": datetime.now().isoformat(),
            "predictions": predictions,
            "metadata": metadata or {}
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        return str(filepath)
    
    def save_embeddings(self, embeddings: np.ndarray, dates: pd.DatetimeIndex,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save embeddings used for predictions.
        
        Args:
            embeddings: Embedding matrix (n_samples, n_features)
            dates: Corresponding dates
            metadata: Optional metadata (model names, PCA info, etc.)
        
        Returns:
            Path to saved embeddings file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{timestamp}.npz"
        filepath = self.embeddings_dir / filename
        
        # Save as numpy compressed format
        np.savez(
            filepath,
            embeddings=embeddings,
            dates=dates.values.astype(str),
            metadata=json.dumps(metadata or {})
        )
        
        return str(filepath)
    
    def save_features(self, features: np.ndarray, dates: pd.DatetimeIndex,
                     feature_names: List[str],
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save final 38 features used for predictions.
        
        Args:
            features: Feature matrix (n_samples, 38)
            dates: Corresponding dates
            feature_names: List of feature names (should be FEATURE_COLS_FULL)
            metadata: Optional metadata
        
        Returns:
            Path to saved features file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_{timestamp}.npz"
        filepath = self.features_dir / filename
        
        # Save as numpy compressed format
        np.savez(
            filepath,
            features=features,
            dates=dates.values.astype(str),
            feature_names=np.array(feature_names),
            metadata=json.dumps(metadata or {})
        )
        
        # Also save as CSV for easier inspection
        csv_filename = f"features_{timestamp}.csv"
        csv_filepath = self.features_dir / csv_filename
        
        df_features = pd.DataFrame(
            features,
            index=dates,
            columns=feature_names
        )
        df_features.to_csv(csv_filepath)
        
        return str(filepath)
    
    def get_latest_predictions(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Get latest N prediction files.
        
        Args:
            n: Number of latest files to retrieve
        
        Returns:
            List of prediction data dicts
        """
        prediction_files = sorted(
            self.predictions_dir.glob("prediction_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:n]
        
        results = []
        for filepath in prediction_files:
            with open(filepath, 'r') as f:
                results.append(json.load(f))
        
        return results
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about stored data."""
        info = {
            "headlines": {
                "file": str(self.headlines_file),
                "exists": self.headlines_file.exists(),
                "count": 0
            },
            "predictions": {
                "directory": str(self.predictions_dir),
                "count": len(list(self.predictions_dir.glob("*.json")))
            },
            "embeddings": {
                "directory": str(self.embeddings_dir),
                "count": len(list(self.embeddings_dir.glob("*.npz")))
            },
            "features": {
                "directory": str(self.features_dir),
                "count": len(list(self.features_dir.glob("*.npz")))
            }
        }
        
        if info["headlines"]["exists"]:
            df = pd.read_csv(self.headlines_file)
            info["headlines"]["count"] = len(df)
            if len(df) > 0:
                info["headlines"]["date_range"] = {
                    "start": str(df['date'].min()),
                    "end": str(df['date'].max())
                }
        
        return info

