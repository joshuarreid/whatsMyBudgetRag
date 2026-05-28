from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return ()
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    spring_boot_base_url: str
    request_timeout_seconds: float
    log_level: str
    log_format: str
    langgraph_enabled: bool
    cors_enabled: bool
    cors_allowed_origins: tuple[str, ...]
    openai_api_key: Optional[str]
    openai_chat_model: str
    mysql_host: Optional[str]
    mysql_port: int
    mysql_database: Optional[str]
    mysql_user: Optional[str]
    mysql_password: Optional[str]
    mysql_ssl_disabled: bool
    mysql_ssl_ca: Optional[str]
    mysql_sslmode: Optional[str]  # Add sslmode for DigitalOcean
    mysql_connect_timeout_seconds: float
    conversation_default_user: str
    conversation_history_context_limit: int
    insight_high_share_threshold: float
    insight_outlier_amount_threshold: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        spring_boot_base_url=os.getenv("SPRING_BOOT_BASE_URL", "http://springboot-api"),
        request_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "10")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_format=os.getenv("LOG_FORMAT", "text").lower(),
        langgraph_enabled=_env_flag("LANGGRAPH_ENABLED", default=True),
        cors_enabled=_env_flag("CORS_ENABLED", default=False),
        cors_allowed_origins=_env_csv("CORS_ALLOWED_ORIGINS"),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        mysql_host=os.getenv("MYSQL_HOST") or None,
        mysql_port=int(os.getenv("MYSQL_PORT", "25060")),
        mysql_database=os.getenv("MYSQL_DATABASE") or None,
        mysql_user=os.getenv("MYSQL_USER") or None,
        mysql_password=os.getenv("MYSQL_PASSWORD") or None,
        mysql_ssl_disabled=_env_flag("MYSQL_SSL_DISABLED", default=False),
        mysql_ssl_ca=os.getenv("MYSQL_SSL_CA") or None,
        mysql_sslmode=os.getenv("MYSQL_SSLMODE") or None,  # Add sslmode
        mysql_connect_timeout_seconds=float(os.getenv("MYSQL_CONNECT_TIMEOUT_SECONDS", "10")),
        conversation_default_user=os.getenv("CONVERSATION_DEFAULT_USER", "default-user"),
        conversation_history_context_limit=int(os.getenv("CONVERSATION_HISTORY_CONTEXT_LIMIT", "10")),
        insight_high_share_threshold=float(os.getenv("INSIGHT_HIGH_SHARE_THRESHOLD", "45")),
        insight_outlier_amount_threshold=float(
            os.getenv("INSIGHT_OUTLIER_AMOUNT_THRESHOLD", "500")
        ),
    )