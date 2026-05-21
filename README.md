# FastAPI RAG Template

A reusable FastAPI starter for retrieval-augmented generation with DigitalOcean Managed PostgreSQL (pgvector), OpenAI, and sentence-transformers.

## Features

- FastAPI service with `/health`, `/ingest`, `/documents/{doc_id}`, and `/rag` endpoints
- Environment-based configuration for DigitalOcean Managed PostgreSQL with pgvector
- Embeddings via sentence-transformers or OpenAI
- Automatic pgvector extension, table, and index creation on startup
- Dockerfile ready for DigitalOcean App Platform

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── db_client.py
│   ├── ingest.py
│   ├── main.py
│   └── rag.py
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and set the values for your environment.

```env
DATABASE_URL=postgresql://username:password@your-do-postgres-host:25060/database?sslmode=require
# Alternative to DATABASE_URL:
# PGHOST=your-do-postgres-host
# PGPORT=25060
# PGDATABASE=database
# PGUSER=username
# PGPASSWORD=password
# PGSSLMODE=require
VECTOR_TABLE_NAME=documents
EMBEDDINGS_PROVIDER=sentence_transformers
EMBEDDINGS_MODEL=all-MiniLM-L6-v2
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
VECTOR_SIZE=384
VECTOR_DISTANCE=cosine
DB_MAX_POOL_SIZE=5
RAG_TOP_K=3
```

Notes:

- Use `EMBEDDINGS_PROVIDER=openai` to generate embeddings with OpenAI instead of sentence-transformers.
- `VECTOR_TABLE_NAME` may be either a plain table name like `documents` or a schema-qualified name like `appdata.documents`.
- For `sentence_transformers`, keep `VECTOR_SIZE` aligned with the model dimension. `all-MiniLM-L6-v2` uses `384`.
- For `openai`, `text-embedding-3-small` uses `1536` and `text-embedding-3-large` uses `3072`.
- `VECTOR_DISTANCE=cosine` is the safest default for semantic search. `dot`, `euclid`, and `manhattan` are also supported.
- DigitalOcean Managed PostgreSQL requires TLS, so include `sslmode=require` in `DATABASE_URL` or set `PGSSLMODE=require`.

## Local Development

1. Create a virtual environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy the example environment file.

```bash
cp .env.example .env
```

3. Start the API.

```bash
uvicorn app.main:app --reload --port 8080
```

## API Usage

Ingest a document:

```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "doc-1",
    "text": "FastAPI is a modern Python web framework for building APIs.",
    "metadata": {"source": "example"}
  }'
```

Delete a document:

```bash
curl -X DELETE http://localhost:8080/documents/doc-1
```

Update a document:

```bash
curl -X PUT http://localhost:8080/documents/doc-1 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "FastAPI is a Python framework for building APIs quickly.",
    "metadata": {"source": "updated-example"}
  }'
```

Query the RAG endpoint:

```bash
curl "http://localhost:8080/rag?query=What%20is%20FastAPI%3F"
```

## Docker

Build and run locally:

```bash
docker build -t fast-api-rag-template .
docker run --env-file .env -p 8080:8080 fast-api-rag-template
```

## DigitalOcean App Platform

1. Push the repository to GitHub.
2. Provision a Managed PostgreSQL cluster. The app attempts to create the `vector` extension on startup.
3. Create an App Platform app using Docker as the build type.
4. Set environment variables in App Platform instead of committing secrets.
5. Deploy the app. The container listens on port `8080` by default and also respects the `PORT` environment variable.

## Implementation Notes

- The app creates the `vector` extension, document table, and HNSW index on startup if they do not exist.
- If `OPENAI_API_KEY` is not set, `/rag` returns retrieved context with a placeholder answer instead of calling an LLM.
- Documents are stored in PostgreSQL with the original text under `text`, arbitrary metadata under `metadata`, and the embedding in a `vector` column.
# fast-api-rag-template
