from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.routes import analytics, rag

load_dotenv()

app = FastAPI(title="Finance Intelligence API", version="0.1.0")
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
app.include_router(rag.router, prefix="/rag", tags=["rag"])


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Minimal liveness check for container platforms and uptime probes."""
    return {"status": "ok"}
