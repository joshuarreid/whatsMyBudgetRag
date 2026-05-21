from __future__ import annotations

import os
from urllib.parse import quote

from pgvector.psycopg import register_vector
from psycopg import connect, sql
from psycopg_pool import ConnectionPool


def get_collection_name() -> str:
    """Return the table used to store source text, metadata, and embeddings."""
    return os.getenv("VECTOR_TABLE_NAME") or os.getenv("COLLECTION_NAME", "documents")


def get_collection_parts() -> tuple[str | None, str]:
    """Split an optional schema-qualified collection name into safe SQL identifier parts."""
    collection_name = get_collection_name().strip()
    parts = [part.strip() for part in collection_name.split(".") if part.strip()]
    if not parts:
        raise ValueError("VECTOR_TABLE_NAME must not be empty")
    if len(parts) == 1:
        return None, parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError("VECTOR_TABLE_NAME must be either <table> or <schema>.<table>")


def get_collection_identifier() -> sql.Identifier:
    """Return the configured collection as a psycopg SQL identifier."""
    schema_name, table_name = get_collection_parts()
    if schema_name:
        return sql.Identifier(schema_name, table_name)
    return sql.Identifier(table_name)


def get_database_url() -> str:
    """Build a PostgreSQL connection string from either a full URL or PG* parts."""
    database_url = os.getenv("DATABASE_URL") or os.getenv("VECTORDB_URL")
    if database_url:
        return database_url

    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    sslmode = os.getenv("PGSSLMODE", "require")

    if not all([host, database, user, password]):
        raise ValueError(
            "Set DATABASE_URL or provide PGHOST, PGDATABASE, PGUSER, and PGPASSWORD"
        )

    return (
        f"postgresql://{quote(user)}:{quote(password)}@{host}:{port}/"
        f"{quote(database)}?sslmode={quote(sslmode)}"
    )


def get_db_pool() -> ConnectionPool:
    """Create a small shared pool and register pgvector on every connection."""
    bootstrap_vector_extension()

    pool = ConnectionPool(
        conninfo=get_database_url(),
        min_size=1,
        max_size=int(os.getenv("DB_MAX_POOL_SIZE", "5")),
        open=False,
        configure=register_vector,
    )
    pool.open()
    pool.wait()
    return pool


def bootstrap_vector_extension() -> None:
    """Ensure the pgvector extension exists before pooled connections register its types."""
    with connect(get_database_url(), autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)


def get_vector_distance() -> str:
    """Normalize the configured distance metric and keep a fallback for older env names."""
    return (os.getenv("VECTOR_DISTANCE") or os.getenv("QDRANT_DISTANCE", "cosine")).strip().lower()


def get_vector_search_sql() -> tuple[str, str]:
    """Return the SQL fragments for ordering by similarity and reporting a score."""
    distance_name = get_vector_distance()
    distance_map = {
        "cosine": ("embedding <=> %s", "1 - (embedding <=> %s)"),
        "dot": ("embedding <#> %s", "(embedding <#> %s) * -1"),
        "euclid": ("embedding <-> %s", "(embedding <-> %s) * -1"),
        "manhattan": ("embedding <+> %s", "(embedding <+> %s) * -1"),
    }
    return distance_map.get(distance_name, distance_map["cosine"])


def _get_vector_index_opclass() -> str | None:
    """Map the chosen distance metric to a pgvector HNSW operator class when supported."""
    distance_name = get_vector_distance()
    opclasses = {
        "cosine": "vector_cosine_ops",
        "dot": "vector_ip_ops",
        "euclid": "vector_l2_ops",
    }
    return opclasses.get(distance_name)


def ensure_collection(pool: ConnectionPool, vector_size: int) -> None:
    """Create the pgvector extension, storage table, and search index if missing."""
    if vector_size <= 0:
        raise ValueError("VECTOR_SIZE must be a positive integer")

    schema_name, table_name = get_collection_parts()
    collection_identifier = get_collection_identifier()
    vector_type = sql.SQL("VECTOR({})").format(sql.SQL(str(vector_size)))
    index_name = f"{table_name}_embedding_idx"

    with pool.connection() as conn:
        # The extension provides the VECTOR type and similarity operators used later.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    doc_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding {} NOT NULL
                )
                """
            ).format(collection_identifier, vector_type)
        )

        opclass = _get_vector_index_opclass()
        if opclass:
            # HNSW keeps query latency reasonable as the table grows.
            conn.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding {})"
                ).format(
                    sql.Identifier(index_name),
                    collection_identifier,
                    sql.SQL(opclass),
                )
            )
