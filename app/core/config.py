from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    spring_boot_base_url: str
    request_timeout_seconds: float
    openai_api_key: str | None
    openai_chat_model: str
    default_analytics_period: str | None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        spring_boot_base_url=os.getenv("SPRING_BOOT_BASE_URL", "http://springboot-api"),
        request_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "10")),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        default_analytics_period=os.getenv("DEFAULT_ANALYTICS_PERIOD") or None,
    )