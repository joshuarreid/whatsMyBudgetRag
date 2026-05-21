from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request

from app.db_client import ensure_collection, get_db_pool
from app.ingest import IngestDoc, UpdateDoc, delete_document, ingest_document
from app.rag import get_embedder, run_rag

# Load local .env files for development while still allowing real env vars in deployment.
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared dependencies once for the FastAPI process lifetime."""
    pool = get_db_pool()
    embed, vector_size = get_embedder()
    ensure_collection(pool, vector_size)
    # Store long-lived objects on app.state so request handlers can reuse them.
    app.state.db_pool = pool
    app.state.embed = embed
    yield
    pool.close()


app = FastAPI(
    title="FastAPI RAG Template",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Minimal liveness check for container platforms and uptime probes."""
    return {"status": "ok"}


@app.post("/ingest")
def api_ingest(doc: IngestDoc, request: Request) -> dict[str, str]:
    """Accept a source document, embed it, and store it for later retrieval."""
    try:
        ingest_document(doc, request.app.state.db_pool, request.app.state.embed)
        return {"status": "ok", "doc_id": doc.doc_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/documents/{doc_id}")
def api_update_document(doc_id: str, doc: UpdateDoc, request: Request) -> dict[str, str]:
    """Replace a stored document by id."""
    try:
        ingest_document(
            IngestDoc(doc_id=doc_id, text=doc.text, metadata=doc.metadata),
            request.app.state.db_pool,
            request.app.state.embed,
        )
        return {"status": "ok", "doc_id": doc_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/documents/{doc_id}")
def api_delete_document(doc_id: str, request: Request) -> dict[str, str]:
    """Delete a stored document by id."""
    try:
        deleted = delete_document(doc_id, request.app.state.db_pool)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
        return {"status": "ok", "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/rag")
def api_rag(request: Request, query: str = Query(..., min_length=1)) -> dict[str, object]:
    """Retrieve similar documents and optionally ask an LLM to answer from them."""
    try:
        answer, sources = run_rag(query, request.app.state.db_pool, request.app.state.embed)
        return {"answer": answer, "sources": sources}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
