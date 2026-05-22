from __future__ import annotations

from typing import Any, Type, Union

from app.models.schemas import (
    AnalyticsAccountBreakdownResponse,
    AnalyticsCategoryBreakdownResponse,
    AnalyticsCriticalityBreakdownResponse,
    AnalyticsDailyTotalResponse,
    AnalyticsDuplicateResponse,
    AnalyticsPaymentMethodBreakdownResponse,
    AnalyticsPeriodsResponse,
    BudgetTransactionResponse,
)


def normalize_periods_response(
    payload: Union[list[str], list[dict[str, Any]], dict[str, Any]],
) -> AnalyticsPeriodsResponse:
    if isinstance(payload, dict):
        return AnalyticsPeriodsResponse.model_validate(payload)

    periods = extract_periods(payload)
    return AnalyticsPeriodsResponse(periods=periods, count=len(periods))


def normalize_list_response(
    payload: Union[list[dict[str, Any]], dict[str, Any]],
    item_model: Type[
        Union[
            AnalyticsAccountBreakdownResponse,
            AnalyticsCategoryBreakdownResponse,
            AnalyticsPaymentMethodBreakdownResponse,
            AnalyticsDailyTotalResponse,
            AnalyticsCriticalityBreakdownResponse,
            AnalyticsDuplicateResponse,
            BudgetTransactionResponse,
        ]
    ],
) -> list[
    Union[
        AnalyticsAccountBreakdownResponse,
        AnalyticsCategoryBreakdownResponse,
        AnalyticsPaymentMethodBreakdownResponse,
        AnalyticsDailyTotalResponse,
        AnalyticsCriticalityBreakdownResponse,
        AnalyticsDuplicateResponse,
        BudgetTransactionResponse,
    ]
]:
    if isinstance(payload, list):
        items = payload
    else:
        candidate_items = payload.get("items") or payload.get("results") or payload.get("data") or []
        items = candidate_items if isinstance(candidate_items, list) else []
    return [item_model.model_validate(item) for item in items if isinstance(item, dict)]


def extract_periods(payload: Union[list[str], list[dict[str, Any]], dict[str, Any]]) -> list[str]:
    if isinstance(payload, list):
        periods: list[str] = []
        for item in payload:
            if isinstance(item, str) and item.strip():
                periods.append(item.strip())
            elif isinstance(item, dict):
                for key in ["period", "name", "value"]:
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        periods.append(value.strip())
                        break
        return periods
    for key in ["periods", "items", "results"]:
        value = payload.get(key)
        if isinstance(value, list):
            return extract_periods(value)
    return []