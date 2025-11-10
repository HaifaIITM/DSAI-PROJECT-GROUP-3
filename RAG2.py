# rag.py
import os
import json
import pandas as pd
from datetime import datetime
from uuid import uuid4

# embeddings/vectorstore / doc classes (langchain-community)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document

# LLM / chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

# simple surrogate explainer
from sklearn.linear_model import Ridge
import numpy as np

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
WINDOW_SIZE = 10  # number of past rows to include in each input window
TOP_K_ANALOGUES = 5
RAG_DIR = "rag_store"
os.makedirs(RAG_DIR, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# -------------------------------------------------------------------
# Helpers: textual summary of a numeric window
# -------------------------------------------------------------------
def window_to_text_summary(window_df: pd.DataFrame, cols=None):
    """
    Create a short textual summary for a numeric window.
    Keep it short so embeddings remain useful.
    """
    if cols is None:
        cols = ["ret_1", "ret_2", "ret_5", "vol_20", "ma_gap", "rsi_14", "vol_z", "dow"]
    pieces = []
    # last value quick features
    last = window_df.iloc[-1]
    pieces.append(f"end_date:{window_df.index[-1].strftime('%Y-%m-%d')}")
    pieces.append(f"last_ret1:{last['ret_1']:.6f}")
    pieces.append(f"vol20_mean:{window_df['vol_20'].mean():.6f}")
    pieces.append(f"ma_gap_last:{last['ma_gap']:.6f}")
    pieces.append(f"rsi14_last:{last['rsi_14']:.2f}")
    # compressed lists (first 5 returns)
    r1 = ",".join([f"{v:.6f}" for v in window_df["ret_1"].tolist()[-5:]])
    pieces.append(f"recent_ret1:[{r1}]")
    return " | ".join(pieces)

# -------------------------------------------------------------------
# Build sliding-window docs from processed CSV
# -------------------------------------------------------------------
def build_rag_from_csv(csv_path: str, ticker_col: str = "Symbol", ticker_value: str = "GSPC"):
    """
    Reads csv -> constructs input_window_docs, prediction_docs (if predictions exist)
    explain_docs (surrogate), analogue_docs and upserts them to FAISS vectorstore.
    """
    df = pd.read_csv(csv_path, parse_dates=["Price"], infer_datetime_format=True, dayfirst=False)
    # if your csv has a proper date column name, change "Price" above accordingly.
    # We'll create an index with the CSV row order if no explicit date column
    print(df.head())
    if "Price" in df.columns:
        # Price column in your CSV is actually a date string? if not, prefer a 'Date' column
        df.index = pd.to_datetime(df["Price"], errors="coerce", utc=False)
        if df.index.isnull().any():
            # fallback: use default RangeIndex and try to parse a different column
            df.index = pd.RangeIndex(start=0, stop=len(df))
    else:
        df.index = pd.RangeIndex(start=0, stop=len(df))

    # ensure required features present
    feature_cols = ["ret_1", "ret_2", "ret_5", "vol_20", "ma_10", "ma_20", "ma_gap", "rsi_14", "vol_z", "dow"]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")

    # Build list of input_window docs + keep dataset for surrogate training
    window_docs = []
    X_surrogate = []
    y_surrogate = []

    for i in range(WINDOW_SIZE - 1, len(df) - 1):  # -1 because target is next row (target_h1)
        window = df.iloc[i - WINDOW_SIZE + 1 : i + 1]
        window_index_end = df.index[i]  # could be a date or index
        next_row = df.iloc[i + 1]

        # textual summary
        txt = window_to_text_summary(window, cols=feature_cols)
        # metadata
        prediction_id = str(uuid4())
        metadata = {
            "type": "input_window",
            "prediction_id": prediction_id,
            "ticker": ticker_value,
            "window_start": str(df.index[i - WINDOW_SIZE + 1]),
            "window_end": str(window_index_end),
            "created_at": datetime.utcnow().isoformat(),
        }

        # Create Document object (embedding will be computed by FAISS wrapper)
        doc = Document(page_content=txt, metadata=metadata)
        window_docs.append((doc, window, next_row, prediction_id))

        # prepare surrogate training row: use last row's features as simple features
        # (you can expand this to use aggregated features across window)
        X_row = [
            window["ret_1"].mean(),
            window["ret_2"].mean(),
            window["ret_5"].mean(),
            window["vol_20"].mean(),
            window["ma_gap"].iloc[-1],
            window["rsi_14"].iloc[-1],
            window["vol_z"].mean(),
            window["dow"].iloc[-1],
        ]
        # target: next-day log-return if present in CSV as 'target_h1' else next close-return
        if "target_h1" in df.columns:
            y_val = next_row["target_h1"]
        elif "ret_1" in next_row:
            y_val = next_row["ret_1"]
        else:
            y_val = 0.0  # fallback
        X_surrogate.append(X_row)
        y_surrogate.append(y_val)

    # Train surrogate explainer (global linear approximator)
    X_sur = np.array(X_surrogate)
    y_sur = np.array(y_surrogate)
    if len(X_sur) >= 10:
        surrogate = Ridge(alpha=1.0).fit(X_sur, y_sur)
        coef = surrogate.coef_
    else:
        surrogate = None
        coef = np.zeros(X_sur.shape[1]) if X_sur.shape[1:] else np.zeros(8)

    # Create/update FAISS vectorstore for this ticker
    ticker_clean = ticker_value.replace("^", "").replace("=", "")
    save_path = os.path.join(RAG_DIR, f"faiss_{ticker_clean}")
    os.makedirs(save_path, exist_ok=True)

    # prepare documents for upsert (input_window docs)
    documents_to_upsert = []
    for idx, (doc, window, next_row, prediction_id) in enumerate(window_docs):
        # input_window doc already made above
        # create explain_doc: compute local attribution approx = coef * feature_values_last
        last_features = np.array([
            window["ret_1"].mean(),
            window["ret_2"].mean(),
            window["ret_5"].mean(),
            window["vol_20"].mean(),
            window["ma_gap"].iloc[-1],
            window["rsi_14"].iloc[-1],
            window["vol_z"].mean(),
            window["dow"].iloc[-1],
        ])
        attributions = (coef * last_features).tolist() if surrogate is not None else [0.0] * len(last_features)
        explain_text = (
            f"Surrogate linear explanation (approx): features = "
            f"ret1_mean={last_features[0]:.6f}, ret2_mean={last_features[1]:.6f}, ret5_mean={last_features[2]:.6f}, "
            f"vol20_mean={last_features[3]:.6f}, ma_gap={last_features[4]:.6f}, rsi14={last_features[5]:.2f}\n"
            f"Coef × value contributions = {', '.join([f'{x:.6e}' for x in attributions])}"
        )
        explain_meta = {
            "type": "explain_doc",
            "prediction_id": prediction_id,
            "ticker": ticker_value,
            "created_at": datetime.utcnow().isoformat(),
        }
        explain_doc = Document(page_content=explain_text, metadata=explain_meta)

        # prediction_doc: if you have an ESN prediction output, put it here.
        # For now, store realized next-day value (target_h1) as 'realized' for analogues.
        pred_meta = {
            "type": "prediction_doc",
            "prediction_id": prediction_id,
            "ticker": ticker_value,
            "prediction_date": str(df.index[idx + WINDOW_SIZE - 1]),  # end of window
            "created_at": datetime.utcnow().isoformat(),
        }
        pred_text = json.dumps({
            "realized_target_h1": float(next_row.get("target_h1", next_row.get("ret_1", 0.0))),
            "realized_close": float(next_row.get("Close", np.nan)),
        })
        prediction_doc = Document(page_content=pred_text, metadata=pred_meta)

        # analogue_doc placeholder (we will compute analogues after the vectorstore exists)
        analogue_meta = {
            "type": "analogue_doc",
            "prediction_id": prediction_id,
            "ticker": ticker_value,
            "analogue_date": None,
            "distance": None,
            "created_at": datetime.utcnow().isoformat(),
        }
        analogue_doc = Document(page_content="analogue placeholder", metadata=analogue_meta)

        # We'll add input window, explain_doc, prediction_doc now.
        documents_to_upsert.extend([doc, explain_doc, prediction_doc])

    # Upsert into FAISS (create or update)
    if os.path.exists(os.path.join(save_path, "index.faiss")):
        vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(documents_to_upsert)
        print(f"🧩 Updated existing RAG store for {ticker_value} with {len(documents_to_upsert)} docs")
    else:
        vectorstore = FAISS.from_documents(documents_to_upsert, embedding=embeddings)
        print(f"📦 Created new RAG store for {ticker_value} with {len(documents_to_upsert)} docs")

    vectorstore.save_local(save_path)

    # ----------------------------------------------------------------
    # Compute analogues: for each input_window, find nearest historical windows (text-embedding)
    # We'll search the store for each input window summary and store top analogue(s) as separate docs.
    # ----------------------------------------------------------------
    # For analogue lookup we only need the textual summaries. Reconstruct them and query FAISS.
    for (doc, window, next_row, prediction_id) in window_docs:
        query_text = doc.page_content
        # search top_k (we expect input_window docs are present too; filter by type=input_window)
        results = vectorstore.similarity_search(query_text, k=TOP_K_ANALOGUES)
        # find the first different window (non-self) and record its realized next-day return if present
        analogue_entries = []
        for r in results:
            md = r.metadata or {}
            # skip docs that are not prediction docs or that belong to same prediction_id
            if md.get("type") == "prediction_doc":
                try:
                    pdata = json.loads(r.page_content)
                    realized = pdata.get("realized_target_h1", None)
                except Exception:
                    realized = None
                analogue_entries.append({
                    "doc_snippet": r.page_content[:300],
                    "realized": realized,
                    "metadata": md
                })
            # also consider input_window docs as potential analogue candidates
            elif md.get("type") == "input_window" and md.get("prediction_id") != prediction_id:
                analogue_entries.append({
                    "doc_snippet": r.page_content[:300],
                    "realized": None,
                    "metadata": md
                })
            if len(analogue_entries) >= 3:
                break

        # create an analogue_doc summarizing top analogues
        analogue_text = "Top analogues:\n" + "\n".join(
            [f"- meta:{a['metadata']} realized:{a['realized']} snippet:{a['doc_snippet']}" for a in analogue_entries]
        )
        analogue_meta = {
            "type": "analogue_doc",
            "prediction_id": prediction_id,
            "ticker": ticker_value,
            "created_at": datetime.utcnow().isoformat(),
        }
        analogue_doc = Document(page_content=analogue_text, metadata=analogue_meta)
        vectorstore.add_documents([analogue_doc])

    # Save after adding analogues
    vectorstore.save_local(save_path)
    print(f"✅ Completed building RAG store and analogues for {ticker_value} at {save_path}")
    return save_path

# -------------------------------------------------------------------
# Query functions (unchanged except we now support filtering by doc type)
# -------------------------------------------------------------------
def query_rag_store(ticker: str, query: str, k: int = 6):
    ticker_clean = ticker.replace("^", "").replace("=", "")
    load_path = os.path.join(RAG_DIR, f"faiss_{ticker_clean}")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No RAG store found for {ticker} at {load_path}")
    vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=k)
    return results

# -------------------------------------------------------------------
# generate reasoning using Ollama (same approach you had)
# -------------------------------------------------------------------
def generate_reasoning_from_rag(ticker: str, query: str):
    docs = query_rag_store(ticker, query)
    # prioritize: prediction_doc, input_window, explain_doc, analogue_doc, others
    def score_doc(d):
        t = (d.metadata or {}).get("type", "")
        order = {"prediction_doc": 0, "input_window": 1, "explain_doc": 2, "analogue_doc": 3}
        return order.get(t, 10)
    docs = sorted(docs, key=score_doc)[:6]
    context = "\n\n".join([f"[{d.metadata.get('type')}] {d.page_content}" for d in docs])

    try:
        llm = ChatOllama(model="mistral")
    except Exception as e:
        raise ConnectionError(
            "Failed to connect to Ollama. Make sure Ollama is running.\n"
            "Start it with: ollama serve\n"
            f"Original error: {str(e)}"
        )

    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=(
            "You are a market analyst.\n"
            "Using only the recent retrieved docs below, produce:\n"
            "1) Headline (one line: direction + confidence if available)\n"
            "2) One-sentence intuitive rationale\n"
            "3) Top 3 drivers (bullets) referencing evidence\n"
            "4) One historical analogue (date + realized next-day return)\n"
            "5) One-sentence caveat.\n\n"
            "Retrieved Docs:\n{context}\n\n"
            "Question:\n{query}\n\n"
            "Answer:"
        )
    )
    chain = RunnableSequence(first=prompt, last=llm)
    answer = chain.invoke({"context": context, "query": query})
    return answer

# -------------------------------------------------------------------
# Driver: build RAG from CSV and test
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Path to your processed CSV (the data you pasted earlier)
    csv_path = "data/processed/GSPC_features.csv"  # change to your actual file path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Please place your processed CSV at {csv_path}")

    # build rag for ticker GSPC (or change ticker value)
    save_path = build_rag_from_csv(csv_path, ticker_col="Symbol", ticker_value="GSPC")

    # test a query
    q = "Explain the drivers of the most recent input window and give a short analogue."
    result = generate_reasoning_from_rag("GSPC", q)
    print("\n🧠 RAG Explanation:\n", result)
