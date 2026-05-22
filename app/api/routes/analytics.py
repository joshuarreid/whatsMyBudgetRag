from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, Header, Query

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import (
    AnalyticsAccountBreakdownResponse,
    AnalyticsCategoryBreakdownResponse,
    AnalyticsCriticalityBreakdownResponse,
    AnalyticsDailyTotalResponse,
    AnalyticsDuplicateResponse,
    AnalyticsPaymentMethodBreakdownResponse,
    AnalyticsPeriodOverviewResponse,
    AnalyticsPeriodsResponse,
    BudgetTransactionResponse,
)
from app.services.analytics_service import AnalyticsService
from app.services.rag_service import normalize_list_response, normalize_periods_response

router = APIRouter()


@lru_cache(maxsize=1)
def get_spring_boot_client() -> SpringBootClient:
    return SpringBootClient()


def get_analytics_service(
    client: SpringBootClient = Depends(get_spring_boot_client),
) -> AnalyticsService:
    return AnalyticsService(client)


@router.get("/periods", response_model=AnalyticsPeriodsResponse)
def periods(
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
) -> AnalyticsPeriodsResponse:
    payload = get_spring_boot_client().get_periods(transaction_id=transaction_id)
    return normalize_periods_response(payload)


@router.get("/periods/{period}/overview", response_model=AnalyticsPeriodOverviewResponse)
def period_overview(
    period: str,
    payment_method: str | None = Query(default=None),
    account: str | None = Query(default=None),
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
    service: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsPeriodOverviewResponse:
    return service.period_overview(
        period=period,
        payment_method=payment_method,
        account=account,
        transaction_id=transaction_id,
    )


@router.get("/periods/{period}/categories", response_model=list[AnalyticsCategoryBreakdownResponse])
def category_breakdown(
    period: str,
    payment_method: str | None = Query(default=None),
    account: str | None = Query(default=None),
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
    client: SpringBootClient = Depends(get_spring_boot_client),
) -> list[AnalyticsCategoryBreakdownResponse]:
    payload = client.get_category_breakdown(
        period=period,
        payment_method=payment_method,
        account=account,
        transaction_id=transaction_id,
    )
    return normalize_list_response(payload, AnalyticsCategoryBreakdownResponse)


@router.get("/periods/{period}/categories/top", response_model=list[AnalyticsCategoryBreakdownResponse])
def top_categories(
    period: str,
    limit: int = Query(default=10, ge=0, le=100),
    payment_method: str | None = Query(default=None),
    account: str | None = Query(default=None),
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
    client: SpringBootClient = Depends(get_spring_boot_client),
) -> list[AnalyticsCategoryBreakdownResponse]:
    payload = client.get_top_categories(
        period=period,
        limit=limit,
        payment_method=payment_method,
        account=account,
        transaction_id=transaction_id,
    )
    return normalize_list_response(payload, AnalyticsCategoryBreakdownResponse)


@router.get("/periods/{period}/accounts", response_model=list[AnalyticsAccountBreakdownResponse])
def account_breakdown(
    period: str,
    payment_method: str | None = Query(default=None),
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
    client: SpringBootClient = Depends(get_spring_boot_client),
) -> list[AnalyticsAccountBreakdownResponse]:
    payload = client.get_account_breakdown(
        period=period,
        payment_method=payment_method,
        transaction_id=transaction_id,
    )
    return normalize_list_response(payload, AnalyticsAccountBreakdownResponse)


@router.get("/periods/{period}/payment-methods", response_model=list[AnalyticsPaymentMethodBreakdownResponse])
def payment_method_breakdown(
    period: str,
    account: str | None = Query(default=None),
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
    client: SpringBootClient = Depends(get_spring_boot_client),
) -> list[AnalyticsPaymentMethodBreakdownResponse]:
    payload = client.get_payment_method_breakdown(
        period=period,
        account=account,
        transaction_id=transaction_id,
    )
    return normalize_list_response(payload, AnalyticsPaymentMethodBreakdownResponse)


@router.get("/periods/{period}/daily", response_model=list[AnalyticsDailyTotalResponse])
def daily_totals(
    period: str,
    payment_method: str | None = Query(default=None),
    account: str | None = Query(default=None),
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
    client: SpringBootClient = Depends(get_spring_boot_client),
) -> list[AnalyticsDailyTotalResponse]:
    payload = client.get_daily_totals(
        period=period,
        payment_method=payment_method,
        account=account,
        transaction_id=transaction_id,
    )
    return normalize_list_response(payload, AnalyticsDailyTotalResponse)


@router.get("/periods/{period}/criticality", response_model=list[AnalyticsCriticalityBreakdownResponse])
def criticality_breakdown(
    period: str,
    payment_method: str | None = Query(default=None),
    account: str | None = Query(default=None),
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
    client: SpringBootClient = Depends(get_spring_boot_client),
) -> list[AnalyticsCriticalityBreakdownResponse]:
    payload = client.get_criticality_breakdown(
        period=period,
        payment_method=payment_method,
        account=account,
        transaction_id=transaction_id,
    )
    return normalize_list_response(payload, AnalyticsCriticalityBreakdownResponse)


@router.get("/periods/{period}/duplicates", response_model=list[AnalyticsDuplicateResponse])
def duplicates(
    period: str,
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
    client: SpringBootClient = Depends(get_spring_boot_client),
) -> list[AnalyticsDuplicateResponse]:
    payload = client.get_duplicates(period=period, transaction_id=transaction_id)
    return normalize_list_response(payload, AnalyticsDuplicateResponse)


@router.get("/periods/{period}/uncategorized", response_model=list[BudgetTransactionResponse])
def uncategorized(
    period: str,
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
    client: SpringBootClient = Depends(get_spring_boot_client),
) -> list[BudgetTransactionResponse]:
    payload = client.get_uncategorized(period=period, transaction_id=transaction_id)
    return normalize_list_response(payload, BudgetTransactionResponse)


@router.get("/periods/{period}/outliers", response_model=list[BudgetTransactionResponse])
def outliers(
    period: str,
    limit: int = Query(default=20, ge=0, le=200),
    transaction_id: str | None = Header(default=None, alias="X-Transaction-ID"),
    client: SpringBootClient = Depends(get_spring_boot_client),
) -> list[BudgetTransactionResponse]:
    payload = client.get_outliers(period=period, limit=limit, transaction_id=transaction_id)
    return normalize_list_response(payload, BudgetTransactionResponse)