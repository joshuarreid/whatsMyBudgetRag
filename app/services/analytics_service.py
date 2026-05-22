from __future__ import annotations

from typing import Optional

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsPeriodOverviewResponse


class AnalyticsService:
    """Helpers for extracting stable values from Spring Boot analytics payloads."""

    def __init__(self, client: SpringBootClient) -> None:
        self.client = client

    def period_overview(
        self,
        period: str,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> AnalyticsPeriodOverviewResponse:
        payload = self.client.get_period_overview(
            period=period,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )
        return AnalyticsPeriodOverviewResponse.model_validate(payload)