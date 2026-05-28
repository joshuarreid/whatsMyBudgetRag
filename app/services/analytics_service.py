from __future__ import annotations

from typing import Optional

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsPeriodOverviewResponse, RagTimeScope


class AnalyticsService:
    """Helpers for extracting stable values from Spring Boot analytics payloads."""

    def __init__(self, client: SpringBootClient) -> None:
        self.client = client

    def period_overview(
        self,
        period: Optional[str] = None,
        *,
        time_scope: Optional[RagTimeScope] = None,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> AnalyticsPeriodOverviewResponse:
        resolved_time_scope = time_scope or self._resolve_time_scope(period)
        payload = self.client.get_overview_for_time_scope(
            time_scope=resolved_time_scope,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )
        return AnalyticsPeriodOverviewResponse.model_validate(payload)

    @staticmethod
    def _resolve_time_scope(period: Optional[str]) -> RagTimeScope:
        if not period:
            raise ValueError("Analytics overview requires either a statement period or a date range time_scope")
        return RagTimeScope.from_period(period)
