"""
EXPERIMENTAL: Headline-Based NLP Sentiment
===========================================
⚠️  NOT RECOMMENDED FOR PRODUCTION ⚠️

This module fetches news headlines and computes sentiment scores.
HOWEVER, testing shows it performs WORSE than market-based proxy.

Issues:
- Only 10-20 recent headlines available via yfinance (not historical)
- VIX integration underperforms market proxy
- Adds complexity without benefit

Recommendation: Use market-based proxy in features.py instead.
Status: Kept for experimental purposes only.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import re, time
from datetime import datetime, timedelta, timezone

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ------------------------------------------------
# 1. FETCH FRESH HEADLINES
# ------------------------------------------------
def fetch_headlines(ticker="SPY", lookback_days=7):
    """
    Fetch latest news headlines using yfinance.
    Returns DataFrame with ['title','publisher','providerPublishTime']
    """
    tk = yf.Ticker(ticker)
    news = tk.news
    df = pd.DataFrame(news)
    # Extract timestamp, title and publisher from the 'content' column
    df['date'] = pd.to_datetime(df['content'].apply(lambda x: x.get('pubDate')), errors='coerce') # Use pubDate and handle errors
    df['title'] = df['content'].apply(lambda x: x.get('title'))
    df['publisher'] = df['content'].apply(lambda x: x.get('provider', {}).get('displayName')) # Access displayName within provider

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days) # Make cutoff timezone-aware (UTC)
    df = df.dropna(subset=['date', 'title', 'publisher']) # Drop rows with missing values after extraction
    df = df[df['date'] >= cutoff]
    return df.sort_values('date', ascending=True).reset_index(drop=True)


# ------------------------------------------------
# 2. DEFINE NLP MODELS & LEXICONS
# ------------------------------------------------
vader = SentimentIntensityAnalyzer()
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

UNCERTAINTY = {"may","might","could","possibly","unclear","uncertain","likely","unlikely","suggests"}
EVENT_KEYWORDS = {"bankrupt","bankruptcy","fraud","recall","downgrade","layoff","miss",
                  "investigation","acquisition","lawsuit","fine","default","selloff","sanction"}

# ------------------------------------------------
# 3. FEATURE EXTRACTION FOR EACH HEADLINE
# ------------------------------------------------
def extract_features(df, rolling_mean_emb=None):
    feats = []
    for _, row in df.iterrows():
        text = row['title']
        tokens = [t.text.lower() for t in nlp(text)]
        sent = vader.polarity_scores(text)['compound']
        neg_intensity = max(0, -sent)
        uncertainty = sum(1 for w in tokens if w in UNCERTAINTY) / max(1, len(tokens))
        event_intensity = sum(1 for w in tokens if w in EVENT_KEYWORDS) / max(1, len(tokens))
        ent_count = len(tokens)
        emb = model.encode(text, normalize_embeddings=True)
        novelty = 0
        if rolling_mean_emb is not None:
            novelty = 1 - np.dot(emb, rolling_mean_emb)
        feats.append({
            'date': row['date'],
            'title': text,
            'sentiment': sent,
            'neg_intensity': neg_intensity,
            'uncertainty': uncertainty,
            'event_intensity': event_intensity,
            'novelty': novelty,
            'embedding': emb
        })
    return pd.DataFrame(feats)


# ------------------------------------------------
# 4. AGGREGATE DAILY FEATURES INTO RISK INDEX
# ------------------------------------------------
def aggregate_daily_features(df_feats):
    daily = []
    for date, group in df_feats.groupby(df_feats['date'].dt.date):
        n = len(group)
        S_mean = np.mean(group['sentiment'])
        Neg_mean = np.mean(group['neg_intensity'])
        U_mean = np.mean(group['uncertainty'])
        E_mean = np.mean(group['event_intensity'])
        N_mean = np.mean(group['novelty'])
        Disp = np.std(group['sentiment'])
        daily.append({
            'date': date,
            'V': n,
            'S_mean': S_mean,
            'Neg_mean': Neg_mean,
            'U_mean': U_mean,
            'E_mean': E_mean,
            'N_mean': N_mean,
            'Disp': Disp
        })
    daily_df = pd.DataFrame(daily).sort_values('date')
    # Normalize and compute composite Risk Index
    scaler = StandardScaler()
    Z = scaler.fit_transform(daily_df[['Neg_mean','E_mean','N_mean','Disp','V']])
    daily_df['Risk_z'] = Z.sum(axis=1)
    # Optional PCA-based index
    pca = PCA(n_components=1)
    daily_df['Risk_pca'] = pca.fit_transform(Z)
    return daily_df

# ------------------------------------------------
# 5. RUN PIPELINE
# ------------------------------------------------
def run_pipeline(ticker="SPY", lookback_days=14):
    headlines = fetch_headlines(ticker, lookback_days)
    if headlines.empty:
        raise ValueError("No headlines found for given ticker/period.")
    print(f"Fetched {len(headlines)} headlines for {ticker}")
    feats = extract_features(headlines)
    daily_df = aggregate_daily_features(feats)
    return daily_df, feats


def generate_risk_index_timeseries(ticker="SPY", lookback_days=30, output_path=None, use_vix_proxy=True):
    """
    Generate historical risk index and save to CSV.
    If use_vix_proxy=True, supplements headlines with VIX-based risk for full history.
    Returns DataFrame with date index and risk columns: Risk_z, Risk_pca
    """
    try:
        daily_df, _ = run_pipeline(ticker, lookback_days)
        risk_ts = daily_df[['date', 'Risk_z', 'Risk_pca']].copy()
        risk_ts['date'] = pd.to_datetime(risk_ts['date'])
        risk_ts = risk_ts.set_index('date').sort_index()
        
        # Merge with VIX if enabled for fuller coverage
        if use_vix_proxy:
            risk_ts = _merge_with_vix_proxy(risk_ts, ticker)
        
        if output_path:
            risk_ts.to_csv(output_path)
            print(f"Risk index saved to {output_path}")
        
        return risk_ts
    except Exception as e:
        print(f"Warning: NLP pipeline failed ({e}), using VIX-only proxy")
        if use_vix_proxy:
            return _vix_only_proxy(ticker, lookback_days, output_path)
        raise

def _merge_with_vix_proxy(nlp_risk: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Hybrid approach: VIX baseline + NLP adjustment for recent days.
    VIX provides full history, headlines enhance recent signal.
    """
    import yfinance as yf
    
    try:
        # Get full VIX history
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="max")
        
        if hist.empty:
            return nlp_risk
        
        # Fix timezone mismatch
        hist.index = hist.index.tz_localize(None)
        if nlp_risk.index.tz is not None:
            nlp_risk.index = nlp_risk.index.tz_localize(None)
        
        # Compute VIX-based sentiment (INVERTED: high VIX = contrarian buy signal)
        hist['vix_z'] = (hist['Close'] - hist['Close'].rolling(60).mean()) / hist['Close'].rolling(60).std()
        hist['vix_risk'] = -hist['vix_z']  # Invert: high VIX spike = bullish contrarian signal
        vix_risk = hist[['vix_risk']].dropna()
        
        # Merge both signals
        merged = vix_risk.join(nlp_risk, how='left')
        
        # Hybrid strategy:
        # 1. VIX-only periods: use VIX directly
        # 2. NLP available periods: blend 70% VIX + 30% NLP sentiment
        has_nlp = merged['Risk_z'].notna()
        
        merged['Risk_z_hybrid'] = merged['vix_risk'].copy()
        merged.loc[has_nlp, 'Risk_z_hybrid'] = (
            0.7 * merged.loc[has_nlp, 'vix_risk'] + 
            0.3 * merged.loc[has_nlp, 'Risk_z']
        )
        
        merged['Risk_pca_hybrid'] = merged['vix_risk'].copy()
        merged.loc[has_nlp, 'Risk_pca_hybrid'] = (
            0.7 * merged.loc[has_nlp, 'vix_risk'] + 
            0.3 * merged.loc[has_nlp, 'Risk_pca']
        )
        
        # Use hybrid signals
        merged['Risk_z'] = merged['Risk_z_hybrid']
        merged['Risk_pca'] = merged['Risk_pca_hybrid']
        
        nlp_days = has_nlp.sum()
        total_days = len(merged)
        print(f"  VIX baseline: {total_days} days | NLP enhanced: {nlp_days} days ({nlp_days/total_days*100:.1f}%)")
        
        return merged[['Risk_z', 'Risk_pca']].dropna()
    
    except Exception as e:
        print(f"Warning: VIX proxy merge failed ({e})")
        import traceback
        traceback.print_exc()
        return nlp_risk

def _vix_only_proxy(ticker: str, lookback_days: int, output_path=None) -> pd.DataFrame:
    """Fallback: use only VIX as risk proxy"""
    import yfinance as yf
    
    vix = yf.Ticker("^VIX")
    hist = vix.history(period="max")
    
    # Fix timezone
    hist.index = hist.index.tz_localize(None)
    
    # VIX-based sentiment (INVERTED: high VIX = contrarian buy opportunity)
    vix_z = (hist['Close'] - hist['Close'].rolling(60).mean()) / hist['Close'].rolling(60).std()
    hist['Risk_z'] = -vix_z  # Invert: fear = opportunity
    hist['Risk_pca'] = hist['Risk_z']
    
    risk_ts = hist[['Risk_z', 'Risk_pca']].dropna()
    
    print(f"  VIX-only fallback: {len(risk_ts)} days of data")
    
    if output_path:
        risk_ts.to_csv(output_path)
    
    return risk_ts


if __name__ == "__main__":
    daily_df, feats = run_pipeline("SPY", lookback_days=5)
    print(daily_df.tail())
