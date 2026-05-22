from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, Depends, Header, Query

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import (
    InsightAveragesResponse,
    InsightBehaviorSummaryResponse,
    InsightMonthOverMonthResponse,
    InsightPeriodSummaryResponse,
)
from app.services.analytics_service import AnalyticsService
from app.services.insight_service import InsightService

router = APIRouter()


@lru_cache(maxsize=1)
def get_insight_service() -> InsightService:
    client = SpringBootClient()
    analytics = AnalyticsService(client)
    return InsightService(client, analytics)


@router.get("/periods/{period}/summary", response_model=InsightPeriodSummaryResponse)
def period_summary(
    period: str,
    payment_method: Optional[str] = Query(default=None),
    account: Optional[str] = Query(default=None),
    transaction_id: Optional[str] = Header(default=None, alias="X-Transaction-ID"),
    insight_service: InsightService = Depends(get_insight_service),
) -> InsightPeriodSummaryResponse:
    return insight_service.period_summary(
        period=period,
        payment_method=payment_method,
        account=account,
        transaction_id=transaction_id,
    )


@router.get("/periods/{period}/behavior", response_model=InsightBehaviorSummaryResponse)
def behavior_summary(
    period: str,
    payment_method: Optional[str] = Query(default=None),
    account: Optional[str] = Query(default=None),
    transaction_id: Optional[str] = Header(default=None, alias="X-Transaction-ID"),
    insight_service: InsightService = Depends(get_insight_service),
) -> InsightBehaviorSummaryResponse:
    return insight_service.behavior_summary(
        period=period,
        payment_method=payment_method,
        account=account,
        transaction_id=transaction_id,
    )


@router.get("/periods/{period}/averages", response_model=InsightAveragesResponse)
def averages(
    period: str,
    payment_method: Optional[str] = Query(default=None),
    account: Optional[str] = Query(default=None),
    transaction_id: Optional[str] = Header(default=None, alias="X-Transaction-ID"),
    insight_service: InsightService = Depends(get_insight_service),
) -> InsightAveragesResponse:
    return insight_service.averages(
        period=period,
        payment_method=payment_method,
        account=account,
        transaction_id=transaction_id,
    )


@router.get("/periods/{period}/month-over-month", response_model=InsightMonthOverMonthResponse)
def month_over_month(
    period: str,
    compare_to: Optional[str] = Query(default=None),
    payment_method: Optional[str] = Query(default=None),
    account: Optional[str] = Query(default=None),
    transaction_id: Optional[str] = Header(default=None, alias="X-Transaction-ID"),
    insight_service: InsightService = Depends(get_insight_service),
) -> InsightMonthOverMonthResponse:
    return insight_service.month_over_month(
        period=period,
        compare_to=compare_to,
        payment_method=payment_method,
        account=account,
        transaction_id=transaction_id,
    )