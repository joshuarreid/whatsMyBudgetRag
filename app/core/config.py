from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class Settings:
    spring_boot_base_url: str
    request_timeout_seconds: float
    log_level: str
    log_format: str
    openai_api_key: Optional[str]
    openai_chat_model: str
    default_analytics_period: Optional[str]
    insight_high_share_threshold: float
    insight_outlier_amount_threshold: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        spring_boot_base_url=os.getenv("SPRING_BOOT_BASE_URL", "http://springboot-api"),
        request_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "10")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_format=os.getenv("LOG_FORMAT", "text").lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        default_analytics_period=os.getenv("DEFAULT_ANALYTICS_PERIOD") or None,
        insight_high_share_threshold=float(os.getenv("INSIGHT_HIGH_SHARE_THRESHOLD", "45")),
        insight_outlier_amount_threshold=float(
            os.getenv("INSIGHT_OUTLIER_AMOUNT_THRESHOLD", "500")
        ),
    )