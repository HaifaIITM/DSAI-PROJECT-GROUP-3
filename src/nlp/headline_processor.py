
import yfinance as yf
import pandas as pd
import numpy as np
import re, time
from datetime import datetime, timedelta, timezone # Import timezone

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


def generate_risk_index_timeseries(ticker="SPY", lookback_days=30, output_path=None):
    """
    Generate historical risk index and save to CSV.
    Returns DataFrame with date index and risk columns: Risk_z, Risk_pca
    """
    daily_df, _ = run_pipeline(ticker, lookback_days)
    risk_ts = daily_df[['date', 'Risk_z', 'Risk_pca']].copy()
    risk_ts['date'] = pd.to_datetime(risk_ts['date'])
    risk_ts = risk_ts.set_index('date').sort_index()
    
    if output_path:
        risk_ts.to_csv(output_path)
        print(f"Risk index saved to {output_path}")
    
    return risk_ts


if __name__ == "__main__":
    daily_df, feats = run_pipeline("SPY", lookback_days=5)
    print(daily_df.tail())
