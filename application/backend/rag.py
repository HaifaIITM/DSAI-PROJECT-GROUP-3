"""
Retrieval-Augmented Generation (RAG) service for explaining predictions.

This module builds lightweight context from stored predictions, features,
and headlines, then queries an Ollama-hosted Finance-Llama model to
generate natural language answers.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from storage import DataStorage

DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")


@dataclass
class RAGDocument:
    """Structured document used for retrieval."""

    title: str
    content: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "content": self.content, "score": self.score}


class RAGService:
    """
    Retrieval-Augmented Generation service that surfaces context from stored
    prediction artifacts and uses an Ollama model for explainability.
    """

    def __init__(
        self,
        storage: DataStorage,
        ollama_url: Optional[str] = None,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout: int = 60,
    ) -> None:
        self.storage = storage
        base_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.ollama_chat_url = f"{base_url.rstrip('/')}/api/chat"
        self.model = model or DEFAULT_OLLAMA_MODEL
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        documents = self._retrieve_documents(question, top_k=top_k)
        context_text = self._format_context(documents)
        answer = self._generate_answer(question=question, context=context_text)

        return {
            "question": question,
            "answer": answer,
            "model": self.model,
            "context": [doc.to_dict() for doc in documents],
        }

    # ------------------------------------------------------------------
    # Context preparation
    # ------------------------------------------------------------------
    def _retrieve_documents(self, query: str, top_k: int = 3) -> List[RAGDocument]:
        documents = self._build_documents()
        if not documents:
            return []

        scored_docs = []
        for doc in documents:
            score = self._simple_score(query, doc.content)
            scored_docs.append(RAGDocument(title=doc.title, content=doc.content, score=score))

        scored_docs.sort(key=lambda d: d.score, reverse=True)

        top_docs = [doc for doc in scored_docs if doc.score > 0][:top_k]
        if not top_docs:
            return scored_docs[:top_k]

        return top_docs

    def _build_documents(self) -> List[RAGDocument]:
        documents: List[RAGDocument] = []

        prediction_doc = self._build_predictions_document()
        if prediction_doc:
            documents.append(prediction_doc)

        feature_doc = self._build_features_document()
        if feature_doc:
            documents.append(feature_doc)

        headline_doc = self._build_headlines_document()
        if headline_doc:
            documents.append(headline_doc)

        metadata_doc = self._build_metadata_document()
        if metadata_doc:
            documents.append(metadata_doc)

        return documents

    def _build_predictions_document(self) -> Optional[RAGDocument]:
        latest_batch = self._load_latest_prediction_batch()
        if not latest_batch:
            return None

        predictions = latest_batch.get("predictions", [])
        metadata = latest_batch.get("metadata", {})

        lines = ["Recent SPY prediction outputs:"]
        for item in predictions:
            lines.append(
                (
                    f"- {item['date']}: h1={item['h1_prediction']:+.6f} ({item['h1_signal']}), "
                    f"h5={item['h5_prediction']:+.6f} ({item['h5_signal']}), "
                    f"h20={item['h20_prediction']:+.6f} ({item['h20_signal']}), "
                    f"close={item['actual_close']:.2f}"
                )
            )

        if metadata:
            horizon_info = metadata.get("model_info", {})
            lines.append("\nModel horizons and Sharpe ratios:")
            for horizon, info in horizon_info.items():
                sharpe = info.get("sharpe")
                lines.append(f"- {horizon.upper()}: Sharpe {sharpe}")

        return RAGDocument(title="Predictions", content="\n".join(lines))

    def _build_features_document(self) -> Optional[RAGDocument]:
        latest_features = self._load_latest_features_frame()
        if latest_features is None or latest_features.empty:
            return None

        summary_lines = ["Key engineered features (latest sampling window):"]
        for feature in ["ret_1", "ret_5", "vol_20", "rsi_14", "ma_gap", "vol_z"]:
            if feature in latest_features.columns:
                values = latest_features[feature].tail(3).tolist()
                formatted = ", ".join(f"{v:+.3f}" for v in values)
                summary_lines.append(f"- {feature}: {formatted}")

        return RAGDocument(title="Feature Signals", content="\n".join(summary_lines))

    def _build_headlines_document(self) -> Optional[RAGDocument]:
        df_news = self.storage.get_headlines(days_back=7)
        if df_news.empty:
            return None

        df_recent = df_news.sort_values("date").tail(10)
        lines = ["Recent news headlines influencing sentiment:"]
        for _, row in df_recent.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d %H:%M") if hasattr(row["date"], "strftime") else str(row["date"])
            title = row.get("title", "").strip()
            publisher = row.get("publisher", "Unknown")
            lines.append(f"- {date_str} | {publisher}: {title}")

        return RAGDocument(title="Market Headlines", content="\n".join(lines))

    def _build_metadata_document(self) -> Optional[RAGDocument]:
        latest_batch = self._load_latest_prediction_batch()
        if not latest_batch:
            return None
        metadata = latest_batch.get("metadata", {})
        if not metadata:
            return None

        lines = ["Prediction metadata:"]
        lines.append(f"- Symbol: {metadata.get('symbol', 'SPY')}")
        lines.append(f"- Samples used: {metadata.get('n_samples', 'N/A')}")
        lines.append(f"- Prediction window: {metadata.get('prediction_days', 'N/A')} days")

        date_range = metadata.get("date_range")
        if date_range:
            lines.append(f"- Date range: {date_range.get('start')} to {date_range.get('end')}")

        lines.append(f"- Normalization: rolling z-score (252d window)")
        lines.append("- Headline embeddings: PCA on Finance news")

        return RAGDocument(title="Metadata", content="\n".join(lines))

    # ------------------------------------------------------------------
    # Ollama interaction
    # ------------------------------------------------------------------
    def _generate_answer(self, question: str, context: str) -> str:
        system_prompt = (
            "You are a financial analysis assistant helping explain SPY predictions. "
            "Use the provided context to justify signals and reference specific data. "
            "If information is missing, state it clearly."
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"User question: {question}\n\n"
            "Provide a concise, well-structured explanation with actionable insights."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        try:
            response = requests.post(
                self.ollama_chat_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid response from Ollama: {exc}") from exc

        message = data.get("message", {})
        answer = message.get("content", "").strip()
        if not answer:
            answer = data.get("response", "").strip()

        return answer or "No answer generated. Please try again later."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _simple_score(self, query: str, text: str) -> float:
        tokens_query = self._tokenize(query)
        tokens_text = self._tokenize(text)

        if not tokens_query or not tokens_text:
            return 0.0

        overlap = len(tokens_query & tokens_text)
        return overlap / float(len(tokens_query))

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    def _format_context(self, documents: List[RAGDocument]) -> str:
        if not documents:
            return "No stored context was available."

        parts: List[str] = []
        for doc in documents:
            parts.append(f"{doc.title}:\n{doc.content}")
        return "\n\n".join(parts)

    def _load_latest_prediction_batch(self) -> Optional[Dict[str, Any]]:
        prediction_files = sorted(
            self.storage.predictions_dir.glob("predictions_batch_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not prediction_files:
            return None

        filepath = prediction_files[0]
        with open(filepath, "r") as f:
            return json.load(f)

    def _load_latest_features_frame(self) -> Optional[pd.DataFrame]:
        csv_files = sorted(
            self.storage.features_dir.glob("features_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not csv_files:
            return None

        latest_csv = csv_files[0]
        try:
            df = pd.read_csv(latest_csv, index_col=0)
            return df.tail(5)
        except Exception:
            return None

