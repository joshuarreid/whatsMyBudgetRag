from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request

from app.api.routes import analytics, insights, rag
from app.core.config import get_settings
from app.core.logging import (
    configure_logging,
    reset_request_id,
    reset_transaction_id,
    set_request_id,
    set_transaction_id,
)

load_dotenv()
configure_logging(get_settings().log_level, get_settings().log_format)

logger = logging.getLogger(__name__)

app = FastAPI(title="Finance Intelligence API", version="0.1.0")
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
app.include_router(insights.router, prefix="/insights", tags=["insights"])
app.include_router(rag.router, prefix="/rag", tags=["rag"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex[:12]
    transaction_id = request.headers.get("X-Transaction-ID") or request_id
    request_token = set_request_id(request_id)
    transaction_token = set_transaction_id(transaction_id)
    started_at = perf_counter()
    client_host = request.client.host if request.client else "unknown"

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (perf_counter() - started_at) * 1000
        logger.exception(
            "Request failed method=%s path=%s query=%s client=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            request.url.query or "-",
            client_host,
            duration_ms,
        )
        raise
    else:
        duration_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "Request completed method=%s path=%s query=%s status_code=%s client=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            request.url.query or "-",
            response.status_code,
            client_host,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Transaction-ID"] = transaction_id
        return response
    finally:
        reset_transaction_id(transaction_token)
        reset_request_id(request_token)


@app.on_event("startup")
def log_startup() -> None:
    settings = get_settings()
    logger.info(
        "Application startup complete spring_boot_base_url=%s timeout_seconds=%s log_level=%s log_format=%s openai_enabled=%s",
        settings.spring_boot_base_url,
        settings.request_timeout_seconds,
        settings.log_level,
        settings.log_format,
        bool(settings.openai_api_key),
    )


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Minimal liveness check for container platforms and uptime probes."""
    return {"status": "ok"}
