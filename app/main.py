from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from time import perf_counter
from typing import Any, AsyncIterator, Optional, cast
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import analytics, insights, rag
from app.core.config import Settings, get_settings
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


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        mysql_history_enabled = all(
            [
                resolved_settings.mysql_host,
                resolved_settings.mysql_database,
                resolved_settings.mysql_user,
                resolved_settings.mysql_password,
            ]
        )
        logger.info(
            "Application startup complete spring_boot_base_url=%s timeout_seconds=%s log_level=%s log_format=%s cors_enabled=%s cors_allowed_origin_count=%s openai_enabled=%s mysql_history_enabled=%s mysql_host=%s",
            resolved_settings.spring_boot_base_url,
            resolved_settings.request_timeout_seconds,
            resolved_settings.log_level,
            resolved_settings.log_format,
            resolved_settings.cors_enabled,
            len(resolved_settings.cors_allowed_origins),
            bool(resolved_settings.openai_api_key),
            mysql_history_enabled,
            resolved_settings.mysql_host or "-",
        )
        yield

    application = FastAPI(title="Finance Intelligence API", version="0.1.0", lifespan=lifespan)

    if resolved_settings.cors_enabled:
        application.add_middleware(
            cast(Any, CORSMiddleware),
            allow_origins=list(resolved_settings.cors_allowed_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    application.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
    application.include_router(insights.router, prefix="/insights", tags=["insights"])
    application.include_router(rag.router, prefix="/rag", tags=["rag"])

    @application.middleware("http")
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

    @application.get("/health")
    def healthcheck() -> dict[str, str]:
        """Minimal liveness check for container platforms and uptime probes."""
        return {"status": "ok"}

    return application


app = create_app()

