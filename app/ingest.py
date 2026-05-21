from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.db_client import get_collection_identifier


class IngestDoc(BaseModel):
    """Payload accepted by the ingest endpoint."""
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateDoc(BaseModel):
    """Payload accepted by the update endpoint."""
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


def ingest_document(doc: IngestDoc, pool: ConnectionPool, embed: Callable[[str], list[float]]) -> None:
    """Generate an embedding and upsert the document into PostgreSQL."""
    # pgvector expects a numeric array-like value, so convert the embedder output explicitly.
    vector = np.array(embed(doc.text), dtype=float)
    with pool.connection() as conn:
        conn.execute(
            sql.SQL(
                """
                INSERT INTO {} (doc_id, text, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    -- Re-ingesting the same doc_id refreshes the full stored representation.
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """
            ).format(get_collection_identifier()),
            (doc.doc_id, doc.text, Jsonb(doc.metadata), vector),
        )


def delete_document(doc_id: str, pool: ConnectionPool) -> bool:
    """Delete a document by id and report whether a row was removed."""
    with pool.connection() as conn:
        result = conn.execute(
            sql.SQL("DELETE FROM {} WHERE doc_id = %s").format(get_collection_identifier()),
            (doc_id,),
        )
    return result.rowcount > 0

