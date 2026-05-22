from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import date
from typing import Any, Optional

import requests

from app.clients.spring_boot_client import SpringBootClient
from app.core.config import get_settings
from app.models.schemas import (
    AnalyticsAccountBreakdownResponse,
    AnalyticsCategoryBreakdownResponse,
    AnalyticsCriticalityBreakdownResponse,
    AnalyticsDailyTotalResponse,
    AnalyticsDuplicateResponse,
    AnalyticsPaymentMethodBreakdownResponse,
    BudgetTransactionResponse,
    RagAnswerResponse,
)
from app.services.analytics_service import AnalyticsService
from app.services.insight_service import InsightService
from app.services.llm_service import LLMService
from app.services.normalizers import normalize_list_response, normalize_periods_response


logger = logging.getLogger(__name__)

STATEMENT_PERIOD_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
STATEMENT_PERIOD_MONTH_PATTERN = "|".join(STATEMENT_PERIOD_MONTHS)
STATEMENT_PERIOD_MONTH_LOOKUP: dict[str, int] = {
    month.lower(): index for index, month in enumerate(STATEMENT_PERIOD_MONTHS, start=1)
}
EXPLICIT_STATEMENT_PERIOD_PATTERN = re.compile(
    rf"\b(?P<month>{STATEMENT_PERIOD_MONTH_PATTERN})\s*(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
CONTEXTUAL_MONTH_PATTERN = re.compile(
    rf"\b(?:in|for|during|on|from)\s+(?:the\s+month\s+of\s+)?(?P<month>{STATEMENT_PERIOD_MONTH_PATTERN})\b",
    re.IGNORECASE,
)
MONTH_OF_PATTERN = re.compile(
    rf"\bmonth\s+of\s+(?P<month>{STATEMENT_PERIOD_MONTH_PATTERN})\b",
    re.IGNORECASE,
)


class RAGService:
    """Tool-style orchestration over Spring Boot APIs without vector retrieval."""

    def __init__(
        self,
        spring_client: SpringBootClient,
        analytics_service: AnalyticsService,
        insight_service: InsightService,
        llm_service: Optional[LLMService] = None,
    ) -> None:
        self.spring = spring_client
        self.analytics = analytics_service
        self.insights = insight_service
        self.llm = llm_service
        settings = get_settings()
        self.default_period = settings.default_analytics_period

    def answer(
        self,
        question: str,
        period: Optional[str] = None,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> RagAnswerResponse:
        logger.info(
            "RAG answer requested requested_period=%s payment_method=%s account=%s transaction_id=%s question_length=%s",
            period or "-",
            payment_method or "-",
            account or "-",
            transaction_id or "-",
            len(question),
        )
        try:
            selected_period, period_interpretation = self._resolve_period(
                question=question,
                period=period,
                transaction_id=transaction_id,
            )
            plan = self._select_tools(question)
            logger.info(
                "RAG execution plan selected period=%s plan=%s period_source=%s",
                selected_period,
                plan,
                period_interpretation.get("source", "unknown"),
            )
            context = self._build_context(
                question=question,
                period=selected_period,
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
                plan=plan,
                period_interpretation=period_interpretation,
            )
            llm_answer = self.llm.generate_answer(question, context) if self.llm else None
            answer = llm_answer or self._fallback_answer(context)
            logger.info(
                "RAG answer completed period=%s context_keys=%s used_llm=%s",
                selected_period,
                sorted(context.keys()),
                bool(llm_answer),
            )
            return RagAnswerResponse(
                question=question,
                period=selected_period,
                plan=plan,
                context=context,
                answer=answer,
            )
        except Exception:
            logger.exception("RAG answer failed period=%s payment_method=%s account=%s", period, payment_method, account)
            raise

    def _resolve_period(
        self,
        question: str,
        period: Optional[str],
        transaction_id: Optional[str],
        *,
        today: Optional[date] = None,
    ) -> tuple[str, dict[str, str]]:
        reference_date = today or date.today()
        if period:
            logger.debug("Using requested analytics period=%s", period)
            return period, {
                "source": "request_parameter",
                "matched_text": period,
                "resolved_period": period,
            }

        inferred_period = self._infer_period_from_question(question=question, today=reference_date)
        if inferred_period is not None:
            logger.debug(
                "Resolved analytics period from question matched_text=%s resolved_period=%s",
                inferred_period.get("matched_text"),
                inferred_period["resolved_period"],
            )
            return inferred_period["resolved_period"], inferred_period

        if self.default_period:
            logger.debug("Using default analytics period=%s", self.default_period)
            return self.default_period, {
                "source": "default_setting",
                "matched_text": self.default_period,
                "resolved_period": self.default_period,
            }

        payload = self.spring.get_periods(transaction_id=transaction_id)
        periods_response = normalize_periods_response(payload)
        if not periods_response.periods:
            raise ValueError("No analytics periods were returned by Spring Boot")
        resolved_period = sorted(periods_response.periods)[-1]
        logger.debug("Resolved latest analytics period=%s", resolved_period)
        return resolved_period, {
            "source": "latest_available_period",
            "matched_text": resolved_period,
            "resolved_period": resolved_period,
        }

    def _infer_period_from_question(self, question: str, today: date) -> Optional[dict[str, str]]:
        lowered = question.lower()
        current_statement_period = self._format_statement_period(today)

        explicit_period_match = EXPLICIT_STATEMENT_PERIOD_PATTERN.search(question)
        if explicit_period_match:
            resolved_period = self._format_statement_period(
                date(
                    int(explicit_period_match.group("year")),
                    self._month_number(explicit_period_match.group("month")),
                    1,
                )
            )
            return {
                "source": "question_explicit_period",
                "matched_text": explicit_period_match.group(0),
                "resolved_period": resolved_period,
                "resolution_rule": "explicit MonthYear or Month YYYY reference from the question",
            }

        if "this period" in lowered or "current period" in lowered:
            return {
                "source": "question_current_period",
                "matched_text": "this period" if "this period" in lowered else "current period",
                "resolved_period": current_statement_period,
                "resolution_rule": "current statement period implied by the question",
            }

        if "this month" in lowered or "current month" in lowered:
            return {
                "source": "question_current_month",
                "matched_text": "this month" if "this month" in lowered else "current month",
                "resolved_period": current_statement_period,
                "resolution_rule": "current month resolves to the current statement period",
            }

        if "last month" in lowered or "previous month" in lowered:
            matched_text = "last month" if "last month" in lowered else "previous month"
            return {
                "source": "question_relative_month",
                "matched_text": matched_text,
                "resolved_period": self._format_statement_period(self._shift_month(today, offset=-1)),
                "resolution_rule": "relative month resolves from the current statement period",
            }

        contextual_month_match = CONTEXTUAL_MONTH_PATTERN.search(question) or MONTH_OF_PATTERN.search(question)
        if contextual_month_match:
            matched_month = contextual_month_match.group("month")
            return {
                "source": "question_bare_month",
                "matched_text": matched_month,
                "resolved_period": self._resolve_recent_month_reference(matched_month, today),
                "resolution_rule": "bare month names resolve to the most recent matching statement period not in the future",
            }

        return None

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
        if any(keyword in lowered for keyword in ["average", "avg", "mean"]):
            selected_tools.append("averages")
        if any(keyword in lowered for keyword in ["month over month", "mom", "versus last month", "compare last month"]):
            selected_tools.append("month_over_month")
        if any(keyword in lowered for keyword in ["summary", "summarize", "snapshot"]):
            selected_tools.append("period_summary")
        if any(keyword in lowered for keyword in ["behavior", "habit", "pattern"]):
            selected_tools.append("behavior_summary")
        if any(keyword in lowered for keyword in ["anomaly", "anomalies", "flag", "flags"]):
            selected_tools.append("period_summary")
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
        payment_method: Optional[str],
        account: Optional[str],
        transaction_id: Optional[str],
        plan: list[str],
        period_interpretation: dict[str, str],
    ) -> dict[str, Any]:
        timeline_context = self._build_timeline_context(period=period, period_interpretation=period_interpretation)
        context: dict[str, Any] = {
            "question": question,
            "period": period,
            "filters": {
                "payment_method": payment_method,
                "account": account,
            },
            "timeline_context": timeline_context,
            "period_interpretation": period_interpretation,
        }
        unavailable_tools: list[dict[str, Any]] = []

        if "overview" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="overview",
                context_key="overview",
                builder=lambda: self.analytics.period_overview(
                    period=period,
                    payment_method=payment_method,
                    account=account,
                    transaction_id=transaction_id,
                ).model_dump(mode="json"),
                required=True,
            )

        if "period_summary" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="period_summary",
                context_key="period_summary",
                builder=lambda: self.insights.period_summary(
                    period=period,
                    payment_method=payment_method,
                    account=account,
                    transaction_id=transaction_id,
                ).model_dump(mode="json"),
            )

        if "behavior_summary" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="behavior_summary",
                context_key="behavior_summary",
                builder=lambda: self.insights.behavior_summary(
                    period=period,
                    payment_method=payment_method,
                    account=account,
                    transaction_id=transaction_id,
                ).model_dump(mode="json"),
            )

        if "averages" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="averages",
                context_key="averages",
                builder=lambda: self.insights.averages(
                    period=period,
                    payment_method=payment_method,
                    account=account,
                    transaction_id=transaction_id,
                ).model_dump(mode="json"),
            )

        if "month_over_month" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="month_over_month",
                context_key="month_over_month",
                builder=lambda: self.insights.month_over_month(
                    period=period,
                    payment_method=payment_method,
                    account=account,
                    transaction_id=transaction_id,
                ).model_dump(mode="json"),
            )

        if "categories" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="categories",
                context_key="categories",
                builder=lambda: [
                    item.model_dump(mode="json")
                    for item in normalize_list_response(
                        self.spring.get_category_breakdown(
                            period=period,
                            payment_method=payment_method,
                            account=account,
                            transaction_id=transaction_id,
                        ),
                        AnalyticsCategoryBreakdownResponse,
                    )
                ],
            )

        if "top_categories" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="top_categories",
                context_key="top_categories",
                builder=lambda: [
                    item.model_dump(mode="json")
                    for item in normalize_list_response(
                        self.spring.get_top_categories(
                            period=period,
                            payment_method=payment_method,
                            account=account,
                            transaction_id=transaction_id,
                        ),
                        AnalyticsCategoryBreakdownResponse,
                    )
                ],
            )

        if "account_breakdown" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="account_breakdown",
                context_key="account_breakdown",
                builder=lambda: [
                    item.model_dump(mode="json")
                    for item in normalize_list_response(
                        self.spring.get_account_breakdown(
                            period=period,
                            payment_method=payment_method,
                            transaction_id=transaction_id,
                        ),
                        AnalyticsAccountBreakdownResponse,
                    )
                ],
            )

        if "payment_methods" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="payment_methods",
                context_key="payment_methods",
                builder=lambda: [
                    item.model_dump(mode="json")
                    for item in normalize_list_response(
                        self.spring.get_payment_method_breakdown(
                            period=period,
                            account=account,
                            transaction_id=transaction_id,
                        ),
                        AnalyticsPaymentMethodBreakdownResponse,
                    )
                ],
            )

        if "daily" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="daily",
                context_key="daily_totals",
                builder=lambda: [
                    item.model_dump(mode="json")
                    for item in normalize_list_response(
                        self.spring.get_daily_totals(
                            period=period,
                            payment_method=payment_method,
                            account=account,
                            transaction_id=transaction_id,
                        ),
                        AnalyticsDailyTotalResponse,
                    )
                ],
            )

        if "criticality" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="criticality",
                context_key="criticality",
                builder=lambda: [
                    item.model_dump(mode="json")
                    for item in normalize_list_response(
                        self.spring.get_criticality_breakdown(
                            period=period,
                            payment_method=payment_method,
                            account=account,
                            transaction_id=transaction_id,
                        ),
                        AnalyticsCriticalityBreakdownResponse,
                    )
                ],
            )

        if "duplicates" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="duplicates",
                context_key="duplicates",
                builder=lambda: [
                    item.model_dump(mode="json")
                    for item in normalize_list_response(
                        self.spring.get_duplicates(
                            period=period,
                            transaction_id=transaction_id,
                        ),
                        AnalyticsDuplicateResponse,
                    )
                ],
            )

        if "uncategorized" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="uncategorized",
                context_key="uncategorized",
                builder=lambda: [
                    item.model_dump(mode="json")
                    for item in normalize_list_response(
                        self.spring.get_uncategorized(
                            period=period,
                            transaction_id=transaction_id,
                        ),
                        BudgetTransactionResponse,
                    )
                ],
            )

        if "outliers" in plan:
            self._populate_context_section(
                context=context,
                unavailable_tools=unavailable_tools,
                tool_name="outliers",
                context_key="outliers",
                builder=lambda: [
                    item.model_dump(mode="json")
                    for item in normalize_list_response(
                        self.spring.get_outliers(
                            period=period,
                            transaction_id=transaction_id,
                        ),
                        BudgetTransactionResponse,
                    )
                ],
            )

        if unavailable_tools:
            context["unavailable_tools"] = unavailable_tools
            context["degraded"] = True

        return context

    def _build_timeline_context(
        self,
        period: str,
        period_interpretation: dict[str, str],
        *,
        today: Optional[date] = None,
    ) -> dict[str, Any]:
        reference_date = today or date.today()
        current_statement_period = self._format_statement_period(reference_date)
        return {
            "current_date": reference_date.isoformat(),
            "current_month": current_statement_period,
            "current_statement_period": current_statement_period,
            "selected_statement_period": period,
            "statement_period_format": "MonthYear",
            "statement_period_examples": ["October2025", "May2026"],
            "period_resolution_rules": {
                "bare_month_name": "Resolve to the most recent matching month not in the future relative to current_statement_period.",
                "current_month": "Resolve to current_statement_period.",
                "last_month": "Resolve to the statement period immediately before current_statement_period.",
            },
            "selection_source": period_interpretation.get("source", "unknown"),
        }

    @staticmethod
    def _month_number(month_text: str) -> int:
        month_number = STATEMENT_PERIOD_MONTH_LOOKUP.get(month_text.lower())
        if month_number is None:
            raise ValueError(f"Unsupported statement period month: {month_text}")
        return month_number

    @staticmethod
    def _format_statement_period(reference_date: date) -> str:
        return reference_date.strftime("%B%Y")

    @staticmethod
    def _shift_month(reference_date: date, offset: int) -> date:
        target_index = reference_date.month - 1 + offset
        target_year = reference_date.year + (target_index // 12)
        target_month = (target_index % 12) + 1
        return date(target_year, target_month, 1)

    def _resolve_recent_month_reference(self, month_text: str, today: date) -> str:
        target_month = self._month_number(month_text)
        target_year = today.year if target_month <= today.month else today.year - 1
        return self._format_statement_period(date(target_year, target_month, 1))

    def _populate_context_section(
        self,
        context: dict[str, Any],
        unavailable_tools: list[dict[str, Any]],
        tool_name: str,
        context_key: str,
        builder: Callable[[], Any],
        required: bool = False,
    ) -> None:
        try:
            context[context_key] = builder()
        except requests.RequestException as exc:
            if required:
                raise
            unavailable_tools.append(self._serialize_tool_error(tool_name, exc))
            logger.warning(
                "Skipping unavailable RAG tool tool=%s context_key=%s error=%s",
                tool_name,
                context_key,
                unavailable_tools[-1],
            )
        except ValueError as exc:
            if required:
                raise
            unavailable_tools.append(
                {
                    "tool": tool_name,
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                }
            )
            logger.warning(
                "Skipping invalid RAG tool payload tool=%s context_key=%s detail=%s",
                tool_name,
                context_key,
                exc,
            )

    @staticmethod
    def _serialize_tool_error(tool_name: str, exc: requests.RequestException) -> dict[str, Any]:
        error: dict[str, Any] = {
            "tool": tool_name,
            "error_type": type(exc).__name__,
            "detail": str(exc),
        }
        response = getattr(exc, "response", None)
        if response is not None:
            error["status_code"] = response.status_code
            response_text = response.text.strip()
            if response_text:
                error["response_body"] = response_text[:300]
        return error

    def _fallback_answer(self, context: dict[str, Any]) -> str:
        fragments: list[str] = []
        overview = context.get("overview")
        if isinstance(overview, dict):
            total_spend = overview.get("total_amount")
            transaction_count = overview.get("transaction_count")
            if total_spend is not None:
                fragments.append(f"Total spend for period {context.get('period')} is {total_spend}.")
            if transaction_count is not None:
                fragments.append(f"Transaction count is {transaction_count}.")

        period_summary = context.get("period_summary")
        if isinstance(period_summary, dict):
            flags = period_summary.get("flags")
            if isinstance(flags, list) and flags:
                fragments.append(f"There are {len(flags)} derived anomaly or concentration flags for this period.")

        behavior_summary = context.get("behavior_summary")
        if isinstance(behavior_summary, dict):
            behavior_lines = behavior_summary.get("behavior_summary")
            if isinstance(behavior_lines, list) and behavior_lines:
                fragments.append(str(behavior_lines[0]))

        averages = context.get("averages")
        if isinstance(averages, dict):
            average_transaction_amount = averages.get("average_transaction_amount")
            if average_transaction_amount is not None:
                fragments.append(f"Average transaction amount is {average_transaction_amount}.")

        month_over_month = context.get("month_over_month")
        if isinstance(month_over_month, dict):
            highlights = month_over_month.get("highlights")
            if isinstance(highlights, list) and highlights:
                fragments.append(str(highlights[0]))

        if "categories" in context:
            fragments.append("Category breakdown data was included from Spring Boot analytics.")

        if "top_categories" in context:
            fragments.append("Top category spend data was included from Spring Boot analytics.")

        if "account_breakdown" in context:
            fragments.append("Account breakdown data was included for the selected period.")

        unavailable_tools = context.get("unavailable_tools")
        if isinstance(unavailable_tools, list) and unavailable_tools:
            missing_tools = ", ".join(
                item.get("tool", "unknown")
                for item in unavailable_tools
                if isinstance(item, dict)
            )
            if missing_tools:
                fragments.append(f"Some analytics sources were unavailable: {missing_tools}.")
            if any(
                isinstance(item, dict) and item.get("tool") == "account_breakdown"
                for item in unavailable_tools
            ):
                fragments.append(
                    "I could not determine which accounts drove the most spending because the account breakdown endpoint was unavailable."
                )

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
