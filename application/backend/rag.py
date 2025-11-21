"""
Retrieval-Augmented Generation (RAG) service for explaining predictions.

This module builds lightweight context from stored predictions, features,
and headlines, then queries OpenAI API to generate natural language answers.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

from storage import DataStorage

DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


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
    prediction artifacts and uses OpenAI API for explainability.
    """

    def __init__(
        self,
        storage: DataStorage,
        api_key: Optional[str] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        timeout: int = 60,
    ) -> None:
        self.storage = storage
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it in your environment or pass it to RAGService."
            )
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model or DEFAULT_OPENAI_MODEL
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
            lines.append("\nModel configuration:")
            for horizon, info in horizon_info.items():
                fold = info.get("fold", "N/A")
                lines.append(f"- {horizon.upper()}: Fold {fold}")

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
    # OpenAI API interaction
    # ------------------------------------------------------------------
    def _generate_answer(self, question: str, context: str) -> str:
        system_prompt = (
            "You are an expert financial analyst specializing in SPY (S&P 500 ETF) predictions. "
            "Your role is to explain model predictions clearly, referencing specific data from the context. "
            "\n\n"
            "**Response Format Guidelines:**\n"
            "1. Start with a brief summary (1-2 sentences) answering the question directly\n"
            "2. Use markdown formatting: headers (###), tables, bullet points, bold/italic\n"
            "3. Structure with clear sections using ### headers\n"
            "4. Include tables for comparisons (use markdown table syntax)\n"
            "5. Reference specific numbers, dates, and signals from the context\n"
            "6. End with actionable insights or recommendations when appropriate\n"
            "\n"
            "**Content Requirements:**\n"
            "- Always cite specific prediction values (e.g., 'h20 prediction: +0.004662')\n"
            "- Reference actual dates and headlines when explaining sentiment\n"
            "- Explain signal classifications (BUY/SELL) with supporting data\n"
            "- Compare different horizons (h1, h5, h20) when relevant\n"
            "- If data is missing, explicitly state what information is unavailable\n"
            "\n"
            "**Tone:** Professional, clear, and data-driven. Avoid speculation beyond the provided context."
        )

        user_prompt = (
            f"**Context Data:**\n{context}\n\n"
            f"**User Question:** {question}\n\n"
            "**Instructions:**\n"
            "Provide a comprehensive, well-structured explanation using markdown formatting. "
            "Include:\n"
            "- A direct answer to the question in the first paragraph\n"
            "- Structured sections with ### headers\n"
            "- Tables for comparing predictions, signals, or features\n"
            "- Specific references to dates, values, and headlines from the context\n"
            "- Clear interpretation of what the data means\n"
            "- Actionable insights or recommendations if applicable\n\n"
            "Format your response as professional markdown that will be rendered in a chat interface."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1500,
            )
            
            answer = response.choices[0].message.content.strip()
            return answer or "No answer generated. Please try again later."
            
        except Exception as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

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
            parts.append(f"=== {doc.title} ===\n{doc.content}")
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

