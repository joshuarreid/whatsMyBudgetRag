from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
from openai import OpenAI
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.db_client import get_collection_identifier, get_vector_search_sql


EmbedFn = Callable[[str], list[float]]


def _get_openai_embedding_dimension(model_name: str) -> int:
    """Return the known vector size for supported OpenAI embedding models."""
    dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    default_size = os.getenv("VECTOR_SIZE") or os.getenv("QDRANT_VECTOR_SIZE", "1536")
    return dimensions.get(model_name, int(default_size))


def get_embedder() -> tuple[EmbedFn, int]:
    """Resolve the configured embedding backend and return both the callable and vector size."""
    provider = os.getenv("EMBEDDINGS_PROVIDER", "sentence_transformers").strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set when EMBEDDINGS_PROVIDER=openai")

        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        client = OpenAI(api_key=api_key)

        def openai_embed(text: str) -> list[float]:
            # The API returns a nested response object; only the first embedding is needed here.
            response = client.embeddings.create(model=embedding_model, input=text)
            return list(response.data[0].embedding)

        return openai_embed, _get_openai_embedding_dimension(embedding_model)

    # Import the local model stack only when that provider is actually selected.
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
    vector_size = model.get_sentence_embedding_dimension()

    def sentence_transformer_embed(text: str) -> list[float]:
        # Normalized vectors work well with cosine similarity and keep score behavior predictable.
        return model.encode(text, normalize_embeddings=True).tolist()

    return sentence_transformer_embed, vector_size


def get_context(query: str, pool: ConnectionPool, embed: EmbedFn) -> tuple[str, list[dict[str, Any]]]:
    """Embed the query, fetch the closest rows, and shape them into LLM-ready context."""
    query_vector = np.array(embed(query), dtype=float)
    order_by_sql, score_sql = get_vector_search_sql()

    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            # The same query vector is used once for sorting and once for the score field.
            cur.execute(
                sql.SQL(
                    """
                    SELECT doc_id, text, metadata, {score_sql} AS score
                    FROM {table_name}
                    ORDER BY {order_by_sql}
                    LIMIT %s
                    """
                ).format(
                    score_sql=sql.SQL(score_sql),
                    table_name=get_collection_identifier(),
                    order_by_sql=sql.SQL(order_by_sql),
                ),
                (query_vector, query_vector, int(os.getenv("RAG_TOP_K", "3"))),
            )
            rows = cur.fetchall()

    sources: list[dict[str, Any]] = []
    context_parts: list[str] = []
    for row in rows:
        # Keep the returned structure simple so the API response is easy to inspect downstream.
        text = row["text"] or ""
        if text:
            context_parts.append(text)
        sources.append(
            {
                "id": row["doc_id"],
                "score": float(row["score"]),
                "text": text,
                "metadata": row["metadata"] or {},
            }
        )

    return "\n\n".join(context_parts), sources


def _generate_answer(query: str, context: str) -> str:
    """Use the retrieved context to generate a final answer, or return context-only output."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # This fallback keeps retrieval testable even when no LLM credentials are configured.
        return f"[No LLM configured] Retrieved context:\n\n{context[:1000]}"

    client = OpenAI(api_key=api_key)
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    # The prompt keeps the model grounded in retrieved content instead of answering from prior knowledge.
    prompt = (
        "Answer the user's question using the provided context. "
        "If the context is insufficient, say so clearly.\n\n"
        f"Context:\n{context or 'No context found.'}\n\n"
        f"Question:\n{query}"
    )
    response = client.responses.create(model=chat_model, input=prompt)
    return response.output_text.strip()


def run_rag(query: str, pool: ConnectionPool, embed: EmbedFn) -> tuple[str, list[dict[str, Any]]]:
    """Top-level RAG flow: retrieve context first, then synthesize an answer."""
    context, sources = get_context(query, pool, embed)
    answer = _generate_answer(query, context)
    return answer, sources
