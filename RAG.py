import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import httpx
from ollama._types import ResponseError

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama  
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# -------------------------------------------------------------------
# 1. Setup
# -------------------------------------------------------------------
load_dotenv()
os.makedirs("rag_store", exist_ok=True)

# -------------------------------------------------------------------
# 2. Fetch news for a ticker
# -------------------------------------------------------------------
def fetch_news_for_ticker(ticker: str, max_articles: int = 50) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    raw_news = tk.news or []
    articles = []
    for item in raw_news[:max_articles]:
        articles.append({
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "link": item.get("link", ""),
            "date": datetime.fromtimestamp(item.get("providerPublishTime", 0))
        })
    df = pd.DataFrame(articles)
    print(f"✅ Fetched {len(df)} news items for {ticker}")
    return df

# -------------------------------------------------------------------
# 3. Build text docs from the news DataFrame
# -------------------------------------------------------------------
def build_documents_from_news(df: pd.DataFrame):
    docs = []
    for _, row in df.iterrows():
        content = f"Title: {row['title']}\nSummary: {row['summary']}\nDate: {row['date'].strftime('%Y-%m-%d')}"
        docs.append(content)
    return docs

# -------------------------------------------------------------------
# 4. Create (or update) the RAG vector store
# -------------------------------------------------------------------
def create_or_update_rag_store(ticker: str, docs):
    ticker_clean = ticker.replace("^", "").replace("=", "")
    save_path = f"rag_store/faiss_{ticker_clean}"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = [Document(page_content=doc, metadata={"ticker": ticker}) for doc in docs]

    if os.path.exists(save_path):
        # incremental update
        vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(documents)
        print(f"🧩 Added {len(documents)} docs to existing store for {ticker}")
    else:
        vectorstore = FAISS.from_documents(documents, embedding=embeddings)
        print(f"📦 Created new store for {ticker} with {len(documents)} docs")

    vectorstore.save_local(save_path)
    return vectorstore

# -------------------------------------------------------------------
# 5. Query the RAG store
# -------------------------------------------------------------------
def query_rag_store(ticker: str, query: str, k: int = 5):
    ticker_clean = ticker.replace("^", "").replace("=", "")
    load_path = f"rag_store/faiss_{ticker_clean}"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No RAG store found for {ticker} at {load_path}")

    vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    results = vectorstore.similarity_search(query, k=k)
    return results

# -------------------------------------------------------------------
# 6. Generate summary / reasoning using the retrieved docs
# -------------------------------------------------------------------
def generate_reasoning_from_rag(ticker: str, query: str):
    docs = query_rag_store(ticker, query)
    context = "\n\n".join([d.page_content for d in docs])

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
            "Using only the recent news below, answer the question clearly and concisely.\n\n"
            "Recent News:\n{context}\n\n"
            "Question:\n{query}\n\n"
            "Answer:"
        )
    )
    chain = RunnableSequence(first=prompt, last=llm)
    
    try:
        answer = chain.invoke({"context": context, "query": query})
    except httpx.ConnectError as e:
        raise ConnectionError(
            "Failed to connect to Ollama. Make sure Ollama is running.\n"
            "Start it with: ollama serve"
        ) from e
    except ResponseError as e:
        if "not found" in str(e).lower() or "404" in str(e):
            raise ValueError(
                f"Model 'mistral' not found in Ollama.\n"
                f"Install it with: ollama pull mistral\n"
                f"Or use a different model that's already installed."
            ) from e
        raise
    
    return answer

# -------------------------------------------------------------------
# 7. Driver: build RAG for multiple tickers & test summary
# -------------------------------------------------------------------
if __name__ == "__main__":
    tickers = ["^GSPC", "SPY", "BTC-USD", "ETH-USD"]  

    for t in tickers:
        df = fetch_news_for_ticker(t)
        if df.empty:
            print(f"No news for {t}")
            continue
        docs = build_documents_from_news(df)
        create_or_update_rag_store(t, docs)

    # Example: generate a reasoning summary for S&P 500
    query = "What factors have recently influenced S&P 500 performance?"
    response = generate_reasoning_from_rag("^GSPC", query)
    print("\n🧠 Explanation:\n", response)
