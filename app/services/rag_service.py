from __future__ import annotations

from typing import Any

from app.clients.spring_boot_client import SpringBootClient
from app.core.config import get_settings
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
    RagAnswerResponse,
)
from app.services.analytics_service import AnalyticsService
from app.services.llm_service import LLMService


def normalize_periods_response(
    payload: list[str] | list[dict[str, Any]] | dict[str, Any],
) -> AnalyticsPeriodsResponse:
    if isinstance(payload, dict):
        return AnalyticsPeriodsResponse.model_validate(payload)

    periods = RAGService._extract_periods(payload)
    return AnalyticsPeriodsResponse(periods=periods, count=len(periods))


def normalize_list_response(
    payload: list[dict[str, Any]] | dict[str, Any],
    item_model: type[
        AnalyticsAccountBreakdownResponse
        | AnalyticsCategoryBreakdownResponse
        | AnalyticsPaymentMethodBreakdownResponse
        | AnalyticsDailyTotalResponse
        | AnalyticsCriticalityBreakdownResponse
        | AnalyticsDuplicateResponse
        | BudgetTransactionResponse
    ],
) -> list[
    AnalyticsAccountBreakdownResponse
    | AnalyticsCategoryBreakdownResponse
    | AnalyticsPaymentMethodBreakdownResponse
    | AnalyticsDailyTotalResponse
    | AnalyticsCriticalityBreakdownResponse
    | AnalyticsDuplicateResponse
    | BudgetTransactionResponse
]:
    if isinstance(payload, list):
        items = payload
    else:
        candidate_items = payload.get("items") or payload.get("results") or payload.get("data") or []
        items = candidate_items if isinstance(candidate_items, list) else []
    return [item_model.model_validate(item) for item in items if isinstance(item, dict)]


class RAGService:
    """Tool-style orchestration over Spring Boot APIs without vector retrieval."""

    def __init__(
        self,
        spring_client: SpringBootClient,
        analytics_service: AnalyticsService,
        llm_service: LLMService | None = None,
    ) -> None:
        self.spring = spring_client
        self.analytics = analytics_service
        self.llm = llm_service
        settings = get_settings()
        self.default_period = settings.default_analytics_period

    def answer(
        self,
        question: str,
        period: str | None = None,
        payment_method: str | None = None,
        account: str | None = None,
        transaction_id: str | None = None,
    ) -> RagAnswerResponse:
        selected_period = self._resolve_period(period=period, transaction_id=transaction_id)
        plan = self._select_tools(question)
        context = self._build_context(
            question=question,
            period=selected_period,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
            plan=plan,
        )
        llm_answer = self.llm.generate_answer(question, context) if self.llm else None
        answer = llm_answer or self._fallback_answer(context)
        return RagAnswerResponse(
            question=question,
            period=selected_period,
            plan=plan,
            context=context,
            answer=answer,
        )

    def _resolve_period(self, period: str | None, transaction_id: str | None) -> str:
        if period:
            return period
        if self.default_period:
            return self.default_period

        payload = self.spring.get_periods(transaction_id=transaction_id)
        periods_response = normalize_periods_response(payload)
        if not periods_response.periods:
            raise ValueError("No analytics periods were returned by Spring Boot")
        return sorted(periods_response.periods)[-1]

    def _select_tools(self, question: str) -> list[str]:
        lowered = question.lower()
        selected_tools: list[str] = []
        if any(keyword in lowered for keyword in ["overview", "summary", "spend", "total"]):
            selected_tools.append("overview")
        if any(keyword in lowered for keyword in ["category", "categories"]):
            selected_tools.append("categories")
        if any(keyword in lowered for keyword in ["top category", "top categories"]):
            selected_tools.append("top_categories")
        if any(keyword in lowered for keyword in ["account", "breakdown"]):
            selected_tools.append("account_breakdown")
        if any(keyword in lowered for keyword in ["payment method", "card", "cash"]):
            selected_tools.append("payment_methods")
        if any(keyword in lowered for keyword in ["daily", "trend", "time series"]):
            selected_tools.append("daily")
        if any(keyword in lowered for keyword in ["criticality", "essential", "non-essential"]):
            selected_tools.append("criticality")
        if "duplicate" in lowered:
            selected_tools.append("duplicates")
        if "uncategorized" in lowered:
            selected_tools.append("uncategorized")
        if any(keyword in lowered for keyword in ["outlier", "largest"]):
            selected_tools.append("outliers")
        if not selected_tools:
            selected_tools.append("overview")
        return list(dict.fromkeys(selected_tools))

    def _build_context(
        self,
        question: str,
        period: str,
        payment_method: str | None,
        account: str | None,
        transaction_id: str | None,
        plan: list[str],
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "question": question,
            "period": period,
            "filters": {
                "payment_method": payment_method,
                "account": account,
            },
        }

        if "overview" in plan:
            overview = self.analytics.period_overview(
                period=period,
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
            context["overview"] = overview.model_dump(mode="json")

        if "categories" in plan:
            categories = self.spring.get_category_breakdown(
                period=period,
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
            context["categories"] = [
                item.model_dump(mode="json")
                for item in normalize_list_response(categories, AnalyticsCategoryBreakdownResponse)
            ]

        if "top_categories" in plan:
            top_categories = self.spring.get_top_categories(
                period=period,
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
            context["top_categories"] = [
                item.model_dump(mode="json")
                for item in normalize_list_response(top_categories, AnalyticsCategoryBreakdownResponse)
            ]

        if "account_breakdown" in plan:
            breakdown = self.spring.get_account_breakdown(
                period=period,
                payment_method=payment_method,
                transaction_id=transaction_id,
            )
            context["account_breakdown"] = [
                item.model_dump(mode="json")
                for item in normalize_list_response(breakdown, AnalyticsAccountBreakdownResponse)
            ]

        if "payment_methods" in plan:
            payment_methods = self.spring.get_payment_method_breakdown(
                period=period,
                account=account,
                transaction_id=transaction_id,
            )
            context["payment_methods"] = [
                item.model_dump(mode="json")
                for item in normalize_list_response(payment_methods, AnalyticsPaymentMethodBreakdownResponse)
            ]

        if "daily" in plan:
            daily_totals = self.spring.get_daily_totals(
                period=period,
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
            context["daily_totals"] = [
                item.model_dump(mode="json")
                for item in normalize_list_response(daily_totals, AnalyticsDailyTotalResponse)
            ]

        if "criticality" in plan:
            criticality = self.spring.get_criticality_breakdown(
                period=period,
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
            context["criticality"] = [
                item.model_dump(mode="json")
                for item in normalize_list_response(criticality, AnalyticsCriticalityBreakdownResponse)
            ]

        if "duplicates" in plan:
            duplicates = self.spring.get_duplicates(
                period=period,
                transaction_id=transaction_id,
            )
            context["duplicates"] = [
                item.model_dump(mode="json")
                for item in normalize_list_response(duplicates, AnalyticsDuplicateResponse)
            ]

        if "uncategorized" in plan:
            uncategorized = self.spring.get_uncategorized(
                period=period,
                transaction_id=transaction_id,
            )
            context["uncategorized"] = [
                item.model_dump(mode="json")
                for item in normalize_list_response(uncategorized, BudgetTransactionResponse)
            ]

        if "outliers" in plan:
            outliers = self.spring.get_outliers(
                period=period,
                transaction_id=transaction_id,
            )
            context["outliers"] = [
                item.model_dump(mode="json")
                for item in normalize_list_response(outliers, BudgetTransactionResponse)
            ]

        return context

    def _fallback_answer(self, context: dict[str, Any]) -> str:
        fragments: list[str] = []
        overview = context.get("overview")
        if isinstance(overview, dict):
            total_spend = overview.get("total_spend")
            transaction_count = overview.get("transaction_count")
            if total_spend is not None:
                fragments.append(f"Total spend for period {context.get('period')} is {total_spend}.")
            if transaction_count is not None:
                fragments.append(f"Transaction count is {transaction_count}.")

        if "categories" in context:
            fragments.append("Category breakdown data was included from Spring Boot analytics.")

        if "top_categories" in context:
            fragments.append("Top category spend data was included from Spring Boot analytics.")

        if "account_breakdown" in context:
            fragments.append("Account breakdown data was included for the selected period.")

        if "payment_methods" in context:
            fragments.append("Payment method breakdown data was included for the selected period.")

        if "daily_totals" in context:
            fragments.append("Daily totals were included for the selected period.")

        if "criticality" in context:
            fragments.append("Criticality breakdown data was included for the selected period.")

        if "duplicates" in context:
            fragments.append("Duplicate transaction candidates were included for the selected period.")

        if "uncategorized" in context:
            fragments.append("Uncategorized transactions were included for the selected period.")

        if "outliers" in context:
            fragments.append("Outlier transactions were included for the selected period.")

        if not fragments:
            return "No matching finance context was available for this question."
        return " ".join(fragments)

    @staticmethod
    def _extract_periods(payload: list[str] | list[dict[str, Any]] | dict[str, Any]) -> list[str]:
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
                return RAGService._extract_periods(value)
        return []