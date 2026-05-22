from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Optional, Union

import requests

from app.core.config import get_settings
from app.core.logging import get_request_id, get_transaction_id


logger = logging.getLogger(__name__)


class SpringBootClient:
    """Thin HTTP client for Spring Boot-owned transaction and analytics endpoints."""

    analytics_base_path = "/api/analytics"

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.spring_boot_base_url.rstrip("/")
        self.timeout_seconds = settings.request_timeout_seconds
        self.session = requests.Session()
        logger.debug(
            "Initialized SpringBootClient base_url=%s timeout_seconds=%s",
            self.base_url,
            self.timeout_seconds,
        )

    def get_periods(self, transaction_id: Optional[str] = None) -> Union[list[str], dict[str, Any]]:
        return self._get("/periods", transaction_id=transaction_id)

    def get_period_overview(
        self,
        period: str,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return self._get(
            f"/periods/{period}/overview",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_category_breakdown(
        self,
        period: str,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            f"/periods/{period}/categories",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_top_categories(
        self,
        period: str,
        limit: int = 10,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            f"/periods/{period}/categories/top",
            params={"limit": str(limit), "paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_account_breakdown(
        self,
        period: str,
        payment_method: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            f"/periods/{period}/accounts",
            params={"paymentMethod": payment_method},
            transaction_id=transaction_id,
        )

    def get_payment_method_breakdown(
        self,
        period: str,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            f"/periods/{period}/payment-methods",
            params={"account": account},
            transaction_id=transaction_id,
        )

    def get_daily_totals(
        self,
        period: str,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            f"/periods/{period}/daily",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_criticality_breakdown(
        self,
        period: str,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            f"/periods/{period}/criticality",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_duplicates(
        self,
        period: str,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            f"/periods/{period}/duplicates",
            transaction_id=transaction_id,
        )

    def get_uncategorized(
        self,
        period: str,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            f"/periods/{period}/uncategorized",
            transaction_id=transaction_id,
        )

    def get_outliers(
        self,
        period: str,
        limit: int = 20,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            f"/periods/{period}/outliers",
            params={"limit": str(limit)},
            transaction_id=transaction_id,
        )

    def _get(
        self,
        path: str,
        params: Optional[dict[str, Optional[str]]] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any], list[str]]:
        url = f"{self.base_url}{self.analytics_base_path}{path}"
        filtered_params = {key: value for key, value in (params or {}).items() if value is not None}
        headers = self._headers(transaction_id)
        started_at = perf_counter()

        try:
            response = self.session.get(
                url,
                params=filtered_params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException:
            duration_ms = (perf_counter() - started_at) * 1000
            logger.exception(
                "Spring Boot request failed before response path=%s params=%s transaction_id=%s duration_ms=%.2f",
                path,
                filtered_params or {},
                headers.get("X-Transaction-ID", "-"),
                duration_ms,
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1000
        if response.ok:
            logger.debug(
                "Spring Boot request completed path=%s params=%s status_code=%s duration_ms=%.2f",
                path,
                filtered_params or {},
                response.status_code,
                duration_ms,
            )
        else:
            logger.error(
                "Spring Boot request returned error path=%s params=%s status_code=%s duration_ms=%.2f body=%s",
                path,
                filtered_params or {},
                response.status_code,
                duration_ms,
                response.text[:500],
            )

        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, (list, dict)):
            return payload
        raise ValueError(f"Unsupported response payload from Spring Boot endpoint: {path}")

    @staticmethod
    def _headers(transaction_id: Optional[str]) -> dict[str, str]:
        resolved_transaction_id = transaction_id or get_transaction_id()
        resolved_request_id = get_request_id()
        headers: dict[str, str] = {}
        if resolved_transaction_id and resolved_transaction_id != "-":
            headers["X-Transaction-ID"] = resolved_transaction_id
        if resolved_request_id and resolved_request_id != "-":
            headers["X-Request-ID"] = resolved_request_id
        return headers