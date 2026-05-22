from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from contextvars import ContextVar, Token


request_id_context: ContextVar[str] = ContextVar("request_id", default="-")
transaction_id_context: ContextVar[str] = ContextVar("transaction_id", default="-")


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        record.transaction_id = transaction_id_context.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "transaction_id": getattr(record, "transaction_id", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", log_format: str = "text") -> None:
    normalized_level = level.upper()
    normalized_format = log_format.lower()
    formatter_name = "json" if normalized_format == "json" else "standard"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "request_context": {
                    "()": "app.core.logging.RequestContextFilter",
                }
            },
            "formatters": {
                "standard": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] [request_id=%(request_id)s] [transaction_id=%(transaction_id)s] %(message)s",
                },
                "json": {
                    "()": "app.core.logging.JsonFormatter",
                },
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": formatter_name,
                    "filters": ["request_context"],
                }
            },
            "root": {
                "level": normalized_level,
                "handlers": ["default"],
            },
            "loggers": {
                "uvicorn": {
                    "level": normalized_level,
                },
                "uvicorn.error": {
                    "level": normalized_level,
                },
                "uvicorn.access": {
                    "level": normalized_level,
                },
            },
        }
    )


def set_request_id(request_id: str) -> Token[str]:
    return request_id_context.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    request_id_context.reset(token)


def get_request_id() -> str:
    return request_id_context.get()


def set_transaction_id(transaction_id: str) -> Token[str]:
    return transaction_id_context.set(transaction_id)


def reset_transaction_id(token: Token[str]) -> None:
    transaction_id_context.reset(token)


def get_transaction_id() -> str:
    return transaction_id_context.get()