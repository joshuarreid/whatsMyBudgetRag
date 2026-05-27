from __future__ import annotations

from datetime import date
import logging
from time import perf_counter
from typing import Any, Optional, Union, cast

import requests

from app.core.config import get_settings
from app.core.logging import get_request_id, get_transaction_id
from app.models.schemas import RagTimeScope


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
        return self._get_periods_payload("/periods", transaction_id=transaction_id)

    def get_period_overview(
        self,
        period: str,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return self._get_object(
            f"/periods/{period}/overview",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_range_overview(
        self,
        start_date: date,
        end_date: date,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return self._get_object(
            "/range/overview",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "paymentMethod": payment_method,
                "account": account,
            },
            transaction_id=transaction_id,
        )

    def get_category_breakdown(
        self,
        period: str,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            f"/periods/{period}/categories",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_range_category_breakdown(
        self,
        start_date: date,
        end_date: date,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            "/range/categories",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "paymentMethod": payment_method,
                "account": account,
            },
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
        return self._get_collection(
            f"/periods/{period}/categories/top",
            params={"limit": str(limit), "paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_range_top_categories(
        self,
        start_date: date,
        end_date: date,
        limit: int = 10,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            "/range/categories/top",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "limit": str(limit),
                "paymentMethod": payment_method,
                "account": account,
            },
            transaction_id=transaction_id,
        )

    def get_account_breakdown(
        self,
        period: str,
        payment_method: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            f"/periods/{period}/accounts",
            params={"paymentMethod": payment_method},
            transaction_id=transaction_id,
        )

    def get_range_account_breakdown(
        self,
        start_date: date,
        end_date: date,
        payment_method: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            "/range/accounts",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "paymentMethod": payment_method,
            },
            transaction_id=transaction_id,
        )

    def get_payment_method_breakdown(
        self,
        period: str,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            f"/periods/{period}/payment-methods",
            params={"account": account},
            transaction_id=transaction_id,
        )

    def get_range_payment_method_breakdown(
        self,
        start_date: date,
        end_date: date,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            "/range/payment-methods",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "account": account,
            },
            transaction_id=transaction_id,
        )

    def get_daily_totals(
        self,
        period: str,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            f"/periods/{period}/daily",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_range_daily_totals(
        self,
        start_date: date,
        end_date: date,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            "/range/daily",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "paymentMethod": payment_method,
                "account": account,
            },
            transaction_id=transaction_id,
        )

    def get_criticality_breakdown(
        self,
        period: str,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            f"/periods/{period}/criticality",
            params={"paymentMethod": payment_method, "account": account},
            transaction_id=transaction_id,
        )

    def get_range_criticality_breakdown(
        self,
        start_date: date,
        end_date: date,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            "/range/criticality",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "paymentMethod": payment_method,
                "account": account,
            },
            transaction_id=transaction_id,
        )

    def get_duplicates(
        self,
        period: str,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            f"/periods/{period}/duplicates",
            transaction_id=transaction_id,
        )

    def get_range_duplicates(
        self,
        start_date: date,
        end_date: date,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            "/range/duplicates",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
            },
            transaction_id=transaction_id,
        )

    def get_uncategorized(
        self,
        period: str,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            f"/periods/{period}/uncategorized",
            transaction_id=transaction_id,
        )

    def get_range_uncategorized(
        self,
        start_date: date,
        end_date: date,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            "/range/uncategorized",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
            },
            transaction_id=transaction_id,
        )

    def get_outliers(
        self,
        period: str,
        limit: int = 20,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            f"/periods/{period}/outliers",
            params={"limit": str(limit)},
            transaction_id=transaction_id,
        )

    def get_range_outliers(
        self,
        start_date: date,
        end_date: date,
        limit: int = 20,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            "/range/outliers",
            params={
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "limit": str(limit),
            },
            transaction_id=transaction_id,
        )

    def get_statement_period_summary(
        self,
        period: str,
        transaction_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return self._get_object(
            f"/summaries/{period}",
            transaction_id=transaction_id,
        )

    def get_statement_period_summaries(
        self,
        start_period: str,
        end_period: str,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        return self._get_collection(
            "/summaries",
            params={
                "startPeriod": start_period,
                "endPeriod": end_period,
            },
            transaction_id=transaction_id,
        )

    def get_statement_period_summary_for_time_scope(
        self,
        time_scope: RagTimeScope,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        if time_scope.scope_type == "statement_period":
            return self.get_statement_period_summary(
                period=self._require_statement_period(time_scope),
                transaction_id=transaction_id,
            )
        if time_scope.scope_type == "statement_period_range":
            start_period, end_period = self._require_statement_period_range(time_scope)
            return self.get_statement_period_summaries(
                start_period=start_period,
                end_period=end_period,
                transaction_id=transaction_id,
            )
        raise ValueError(f"Unsupported time scope for statement-period summary endpoint: {time_scope.scope_type}")

    def get_overview_for_time_scope(
        self,
        time_scope: RagTimeScope,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if time_scope.scope_type == "statement_period":
            period = self._require_statement_period(time_scope)
            return self.get_period_overview(
                period=period,
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
        start_date, end_date = self._require_date_range(time_scope)
        return self.get_range_overview(
            start_date=start_date,
            end_date=end_date,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )

    def get_category_breakdown_for_time_scope(
        self,
        time_scope: RagTimeScope,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        if time_scope.scope_type == "statement_period":
            return self.get_category_breakdown(
                period=self._require_statement_period(time_scope),
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
        start_date, end_date = self._require_date_range(time_scope)
        return self.get_range_category_breakdown(
            start_date=start_date,
            end_date=end_date,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )

    def get_top_categories_for_time_scope(
        self,
        time_scope: RagTimeScope,
        limit: int = 10,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        if time_scope.scope_type == "statement_period":
            return self.get_top_categories(
                period=self._require_statement_period(time_scope),
                limit=limit,
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
        start_date, end_date = self._require_date_range(time_scope)
        return self.get_range_top_categories(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )

    def get_account_breakdown_for_time_scope(
        self,
        time_scope: RagTimeScope,
        payment_method: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        if time_scope.scope_type == "statement_period":
            return self.get_account_breakdown(
                period=self._require_statement_period(time_scope),
                payment_method=payment_method,
                transaction_id=transaction_id,
            )
        start_date, end_date = self._require_date_range(time_scope)
        return self.get_range_account_breakdown(
            start_date=start_date,
            end_date=end_date,
            payment_method=payment_method,
            transaction_id=transaction_id,
        )

    def get_payment_method_breakdown_for_time_scope(
        self,
        time_scope: RagTimeScope,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        if time_scope.scope_type == "statement_period":
            return self.get_payment_method_breakdown(
                period=self._require_statement_period(time_scope),
                account=account,
                transaction_id=transaction_id,
            )
        start_date, end_date = self._require_date_range(time_scope)
        return self.get_range_payment_method_breakdown(
            start_date=start_date,
            end_date=end_date,
            account=account,
            transaction_id=transaction_id,
        )

    def get_daily_totals_for_time_scope(
        self,
        time_scope: RagTimeScope,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        if time_scope.scope_type == "statement_period":
            return self.get_daily_totals(
                period=self._require_statement_period(time_scope),
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
        start_date, end_date = self._require_date_range(time_scope)
        return self.get_range_daily_totals(
            start_date=start_date,
            end_date=end_date,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )

    def get_criticality_breakdown_for_time_scope(
        self,
        time_scope: RagTimeScope,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        if time_scope.scope_type == "statement_period":
            return self.get_criticality_breakdown(
                period=self._require_statement_period(time_scope),
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
        start_date, end_date = self._require_date_range(time_scope)
        return self.get_range_criticality_breakdown(
            start_date=start_date,
            end_date=end_date,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )

    def get_duplicates_for_time_scope(
        self,
        time_scope: RagTimeScope,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        if time_scope.scope_type == "statement_period":
            return self.get_duplicates(
                period=self._require_statement_period(time_scope),
                transaction_id=transaction_id,
            )
        start_date, end_date = self._require_date_range(time_scope)
        return self.get_range_duplicates(
            start_date=start_date,
            end_date=end_date,
            transaction_id=transaction_id,
        )

    def get_uncategorized_for_time_scope(
        self,
        time_scope: RagTimeScope,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        if time_scope.scope_type == "statement_period":
            return self.get_uncategorized(
                period=self._require_statement_period(time_scope),
                transaction_id=transaction_id,
            )
        start_date, end_date = self._require_date_range(time_scope)
        return self.get_range_uncategorized(
            start_date=start_date,
            end_date=end_date,
            transaction_id=transaction_id,
        )

    def get_outliers_for_time_scope(
        self,
        time_scope: RagTimeScope,
        limit: int = 20,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        if time_scope.scope_type == "statement_period":
            return self.get_outliers(
                period=self._require_statement_period(time_scope),
                limit=limit,
                transaction_id=transaction_id,
            )
        start_date, end_date = self._require_date_range(time_scope)
        return self.get_range_outliers(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            transaction_id=transaction_id,
        )

    @staticmethod
    def _require_statement_period(time_scope: RagTimeScope) -> str:
        if time_scope.scope_type != "statement_period" or not time_scope.statement_period:
            raise ValueError(f"Unsupported time scope for statement-period endpoint: {time_scope.scope_type}")
        return time_scope.statement_period

    @staticmethod
    def _require_statement_period_range(time_scope: RagTimeScope) -> tuple[str, str]:
        if time_scope.scope_type != "statement_period_range" or not time_scope.start_period or not time_scope.end_period:
            raise ValueError(f"Unsupported time scope for statement-period range endpoint: {time_scope.scope_type}")
        return time_scope.start_period, time_scope.end_period

    @staticmethod
    def _require_date_range(time_scope: RagTimeScope) -> tuple[date, date]:
        if time_scope.scope_type != "date_range" or time_scope.start_date is None or time_scope.end_date is None:
            raise ValueError(f"Unsupported time scope for date-range endpoint: {time_scope.scope_type}")
        return time_scope.start_date, time_scope.end_date

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

    def _get_object(
        self,
        path: str,
        params: Optional[dict[str, Optional[str]]] = None,
        transaction_id: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = self._get(path, params=params, transaction_id=transaction_id)
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        raise ValueError(f"Expected object response payload from Spring Boot endpoint: {path}")

    def _get_collection(
        self,
        path: str,
        params: Optional[dict[str, Optional[str]]] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[dict[str, Any]], dict[str, Any]]:
        payload = self._get(path, params=params, transaction_id=transaction_id)
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
            return cast(list[dict[str, Any]], payload)
        raise ValueError(f"Expected collection response payload from Spring Boot endpoint: {path}")

    def _get_periods_payload(
        self,
        path: str,
        params: Optional[dict[str, Optional[str]]] = None,
        transaction_id: Optional[str] = None,
    ) -> Union[list[str], dict[str, Any]]:
        payload = self._get(path, params=params, transaction_id=transaction_id)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
            return cast(list[str], payload)
        raise ValueError(f"Expected periods response payload from Spring Boot endpoint: {path}")

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