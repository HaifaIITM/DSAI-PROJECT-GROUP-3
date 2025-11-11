import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path


def load_headlines(csv_path: str) -> pd.DataFrame:
    """
    Load headlines from CSV and normalize dates.
    Expected columns: published_utc, title
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Headlines file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["published_utc"]).dt.tz_convert(None).dt.normalize()
    df = df[["date", "title"]].dropna(subset=["title"])
    print(f"Loaded {len(df)} headlines from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def get_embeddings(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for a list of texts using sentence-transformers.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x  # fallback if tqdm not available
    
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


def make_daily_pca(
    model_name: str,
    n_components: int,
    headlines_df: pd.DataFrame,
    random_state: int = 42,
    agg_method: str = "mean"
) -> Tuple[pd.DataFrame, object]:
    """
    Embed headlines, reduce with PCA, and aggregate by date.
    
    Returns:
        (daily_pca_df, fitted_pca_object)
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "sentence-transformers and scikit-learn required for embeddings. "
            f"Install with: pip install sentence-transformers scikit-learn\n{e}"
        )
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = headlines_df["title"].tolist()
    emb = get_embeddings(model, texts)
    
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(emb)
    
    reduced_df = pd.DataFrame(
        reduced, 
        columns=[f"pca_{i+1}" for i in range(n_components)]
    )
    reduced_df["date"] = headlines_df["date"].values
    
    # Aggregate multiple headlines per day
    daily = reduced_df.groupby("date").agg(agg_method).reset_index()
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_components} components explain {explained_var:.2%} variance")
    
    return daily, pca


def align_with_dates(
    pca_df: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    suffix: str = ""
) -> pd.DataFrame:
    """
    Reindex PCA features to match trading calendar.
    Missing dates filled with 0.0, and add 'has_news' indicator.
    
    Args:
        pca_df: DataFrame with 'date' column and PCA features
        target_dates: DatetimeIndex of trading days
        suffix: Optional suffix for column names (e.g., "_large")
    """
    pca_df = pca_df.set_index("date").reindex(target_dates)
    pca_df.index.name = "date"
    
    # Fill missing with 0
    pca_df = pca_df.fillna(0.0)
    
    # Add has_news indicator
    has_news_col = f"has_news{suffix}"
    pca_df[has_news_col] = (pca_df.sum(axis=1) != 0).astype(int)
    
    # Rename columns with suffix if provided
    if suffix:
        rename_map = {col: f"{col}{suffix}" for col in pca_df.columns if col.startswith("pca_")}
        pca_df = pca_df.rename(columns=rename_map)
    
    return pca_df.reset_index()


def compute_headline_features(
    target_dates: pd.DatetimeIndex,
    headlines_csv: str,
    small_model: str = "all-MiniLM-L6-v2",
    large_model: str = "sentence-transformers/all-mpnet-base-v2",
    small_pca_dim: int = 12,
    large_pca_dim: int = 14,
    random_state: int = 42,
    agg_method: str = "mean"
) -> Optional[pd.DataFrame]:
    """
    Full pipeline: load headlines → embed with 2 models → PCA → align to trading days.
    
    Returns:
        DataFrame with date index + all PCA features + has_news flags, or None if headlines unavailable
    """
    if not os.path.exists(headlines_csv):
        print(f"[WARN] Headlines file not found: {headlines_csv}. Skipping embeddings.")
        return None
    
    try:
        headlines_df = load_headlines(headlines_csv)
        
        # Small model
        print(f"\n--- Small model ({small_model}) ---")
        small_daily, _ = make_daily_pca(
            small_model, small_pca_dim, headlines_df, random_state, agg_method
        )
        small_aligned = align_with_dates(small_daily, target_dates, suffix="")
        
        # Large model
        print(f"\n--- Large model ({large_model}) ---")
        large_daily, _ = make_daily_pca(
            large_model, large_pca_dim, headlines_df, random_state, agg_method
        )
        large_aligned = align_with_dates(large_daily, target_dates, suffix="_large")
        
        # Merge
        merged = small_aligned.merge(large_aligned, on="date", how="outer")
        merged = merged.set_index("date").sort_index()
        
        print(f"[OK] Generated {len(merged.columns)} headline features for {len(merged)} dates")
        return merged
        
    except Exception as e:
        print(f"[WARN] Error computing headline features: {e}")
        return None

