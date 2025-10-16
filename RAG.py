import yfinance as yf
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


# helper
def parse_date(x):
    return pd.to_datetime(x)



def fetch_yfinance_docs(ticker='^GSPC', query_date=None, window_days=7, max_news=5):
    """
    Returns a list of small textual documents (strings) about the ticker around query_date:
      - price history snippet (window_days before/after)
      - top news headlines (if available)
    """
    docs = []
    qd = parse_date(query_date) if query_date is not None else pd.Timestamp.today().normalize()
    t = yf.Ticker(ticker)
    # 1) price history snippet
    start = (qd - pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')
    end = (qd + pd.Timedelta(days=window_days)).strftime('%Y-%m-%d')
    try:
        hist = t.history(start=start, end=end)
        if not hist.empty:
            hist_s = hist[['Open','High','Low','Close','Volume']].tail(2*window_days+1).to_csv()
            docs.append(f"Price history for {ticker} from {start} to {end}:\n{hist_s}")
    except Exception as e:
        docs.append(f"Failed to fetch price history for {ticker}: {e}")

    # 2) news (yfinance.Ticker.news if available)
    try:
        news = getattr(t, 'news', None)
        if news is None:
            # some yfinance versions offer t.get_news() or t.news property
            news = t.get('news') if hasattr(t, 'get') else None
        news_list = t.news if hasattr(t, 'news') else []
    except Exception:
        # fallback: empty
        news_list = []

    # Select top headlines within a rough time window (yfinance news items often have 'providerPublishTime')
    i = 0
    for item in news_list:
        if i >= max_news:
            break
        title = item.get('title', '')
        summary = item.get('summary', '') or item.get('publisher', '')
        ts = item.get('providerPublishTime', None)
        when = pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d %H:%M:%S') if ts else ''
        docs.append(f"News: {title} ({when})\n{summary}")
        i += 1

    # 3) Add optional fundamentals / description
    info = {}
    try:
        info = t.info if hasattr(t, 'info') else {}
        if info:
            short_info = {k: info[k] for k in ['longName','sector','industry','summary'] if k in info}
            docs.append(f"Ticker info for {ticker}: {short_info}")
    except Exception:
        pass

    return docs


def construct_rag():
    """
    Constructs RAG components."""
    if EMBEDDER_AVAILABLE:
        embedder = SentenceTransformer('all-mpnet-base-v2')
    else:
        embedder = None

    rag_texts = []
    rag_meta = []

    tickers = ["^GSPC", "^SPY"]  

    for ticker in tickers:  # e.g. ["^GSPC", "^IXIC", "^N225"]
        model_preds = get_model_predictions(ticker)
        news = fetch_yfinance_docs(ticker)

        for item in model_preds:
            combined_text = f"""
            Ticker: {ticker}
            Date: {item['date']}
            Predicted: {item['pred']}
            Actual: {item['actual']}
            Error: {abs(item['pred'] - item['actual']):.4f}
            Direction: {item['direction']}
            News: {' | '.join(news[:3])}
            """
            rag_texts.append(combined_text.strip())
            rag_meta.append({
                "ticker": ticker,
                "date": item["date"],
                "pred": item["pred"],
                "actual": item["actual"]
            })

    if embedder is not None:
        rag_embs = embedder.encode(rag_texts, convert_to_numpy=True)
    else:
        # fallback: random vectors (deterministic) to enable cosine similarity behavior
        rng = np.random.RandomState(0)
        rag_embs = np.vstack([rng.randn(384) for _ in rag_texts])

    np.savez("rag_store.npz", embs=rag_embs, texts=rag_texts, meta=rag_meta)
    print(f"Saved RAG store with {len(rag_texts)} entries.")


def query_rag(query, top_k=5, ticker=None):
    """Retrieve top relevant documents for a natural language query."""
    data = np.load("rag_store.npz", allow_pickle=True)
    embs = data["embs"]
    texts = data["texts"].tolist()
    meta = data["meta"].tolist()

    embedder = SentenceTransformer('all-mpnet-base-v2')
    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embs)[0]

    # Optional ticker filter
    if ticker:
        mask = np.array([m["ticker"] == ticker for m in meta])
        sims = sims * mask

    top_idx = np.argsort(-sims)[:top_k]
    return [(texts[i], meta[i], sims[i]) for i in top_idx]


def summarize_rag(query, model='gpt-5'):
    retrieved = query_rag(query)
    context = "\n\n".join([r[0] for r in retrieved])
    prompt = f"""
    Context:\n{context}\n\n
    Task: {query}
    """
    # Call your LLM here
    print("=== Prompt for LLM ===")
    print(prompt)

