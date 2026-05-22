from __future__ import annotations

from typing import Any

import requests

from app.core.config import get_settings


class SpringBootClient:
    """Thin HTTP client for Spring Boot-owned transaction and analytics endpoints."""

    analytics_base_path = "/api/analytics"

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.spring_boot_base_url.rstrip("/")
        self.timeout_seconds = settings.request_timeout_seconds
        self.session = requests.Session()

    def get_periods(self, transaction_id: str | None = None) -> list[str] | dict[str, Any]:
        return self._get("/periods", transaction_id=transaction_id)

    def get_period_overview(
        self,
        period: str,
        payment_method: str | None = None,
        account: str | None = None,
        transaction_id: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            f"/periods/{period}/overview",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_category_breakdown(
        self,
        period: str,
        payment_method: str | None = None,
        account: str | None = None,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        return self._get(
            f"/periods/{period}/categories",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_top_categories(
        self,
        period: str,
        limit: int = 10,
        payment_method: str | None = None,
        account: str | None = None,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        return self._get(
            f"/periods/{period}/categories/top",
            params={"limit": str(limit), "paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_account_breakdown(
        self,
        period: str,
        payment_method: str | None = None,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        return self._get(
            f"/periods/{period}/accounts",
            params={"paymentMethod": payment_method},
            transaction_id=transaction_id,
        )

    def get_payment_method_breakdown(
        self,
        period: str,
        account: str | None = None,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        return self._get(
            f"/periods/{period}/payment-methods",
            params={"account": account},
            transaction_id=transaction_id,
        )

    def get_daily_totals(
        self,
        period: str,
        payment_method: str | None = None,
        account: str | None = None,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        return self._get(
            f"/periods/{period}/daily",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_criticality_breakdown(
        self,
        period: str,
        payment_method: str | None = None,
        account: str | None = None,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        return self._get(
            f"/periods/{period}/criticality",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_duplicates(
        self,
        period: str,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        return self._get(
            f"/periods/{period}/duplicates",
            transaction_id=transaction_id,
        )

    def get_uncategorized(
        self,
        period: str,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        return self._get(
            f"/periods/{period}/uncategorized",
            transaction_id=transaction_id,
        )

    def get_outliers(
        self,
        period: str,
        limit: int = 20,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        return self._get(
            f"/periods/{period}/outliers",
            params={"limit": str(limit)},
            transaction_id=transaction_id,
        )

    def _get(
        self,
        path: str,
        params: dict[str, str | None] | None = None,
        transaction_id: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any] | list[str]:
        response = self.session.get(
            f"{self.base_url}{self.analytics_base_path}{path}",
            params={key: value for key, value in (params or {}).items() if value is not None},
            headers=self._headers(transaction_id),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, (list, dict)):
            return payload
        raise ValueError(f"Unsupported response payload from Spring Boot endpoint: {path}")

    @staticmethod
    def _headers(transaction_id: str | None) -> dict[str, str]:
        if not transaction_id:
            return {}
        return {"X-Transaction-ID": transaction_id}