from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Optional

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import RagAnswerResponse, RagIntentResponse
from app.skills.base import Skill, SkillRequest
from app.skills.registry import SkillRegistry

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
    """Skill-oriented orchestration over Spring Boot APIs without vector retrieval."""

    def __init__(
        self,
        spring_client: SpringBootClient,
        skill_registry: SkillRegistry,
        llm_service: Optional[Any] = None,
    ) -> None:
        self.spring = spring_client
        self.skill_registry = skill_registry
        self.llm = llm_service

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
            selected_skills, routing_metadata = self._select_skills(question)
            plan = [skill.skill_id for skill in selected_skills]
            logger.info(
                "RAG execution plan selected period=%s plan=%s period_source=%s routing_source=%s",
                selected_period,
                plan,
                period_interpretation.get("source", "unknown"),
                routing_metadata.get("source", "unknown"),
            )
            skill_request = SkillRequest(
                question=question,
                period=selected_period,
                payment_method=payment_method,
                account=account,
                transaction_id=transaction_id,
            )
            context = self._build_context(
                skill_request=skill_request,
                plan=plan,
                period_interpretation=period_interpretation,
                skills=selected_skills,
                routing_metadata=routing_metadata,
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
                tool_selection=routing_metadata.get("tool_selection", {}),
                context=context,
                answer=answer,
            )
        except Exception:
            logger.exception(
                "RAG answer failed period=%s payment_method=%s account=%s",
                period,
                payment_method,
                account,
            )
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

        resolved_period = self._format_statement_period(reference_date)
        logger.debug("Falling back to current statement period=%s", resolved_period)
        return resolved_period, {
            "source": "current_statement_period_fallback",
            "matched_text": resolved_period,
            "resolved_period": resolved_period,
            "resolution_rule": "questions without a time reference fall back to the current statement period",
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

    def _build_context(
        self,
        skill_request: SkillRequest,
        plan: list[str],
        period_interpretation: dict[str, str],
        skills: list[Skill],
        routing_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timeline_context = self._build_timeline_context(
            period=skill_request.period,
            period_interpretation=period_interpretation,
        )
        context: dict[str, Any] = {
            "question": skill_request.question,
            "period": skill_request.period,
            "filters": {
                "payment_method": skill_request.payment_method,
                "account": skill_request.account,
            },
            "timeline_context": timeline_context,
            "period_interpretation": period_interpretation,
            "skills": {
                "selected": plan,
                "count": len(plan),
            },
            "routing": routing_metadata,
        }
        skill_context, unavailable_skills = self.skill_registry.execute(skills, skill_request)
        context.update(skill_context)

        if unavailable_skills:
            context["unavailable_tools"] = unavailable_skills
            context["degraded"] = True

        return context

    def _select_skills(self, question: str) -> tuple[list[Skill], dict[str, Any]]:
        llm_intent = self._classify_intent(question)
        llm_selected_skills = self.skill_registry.resolve(llm_intent.skill_ids) if llm_intent is not None else []
        deterministic_skills = self.skill_registry.select(question)
        tool_selection = {
            "llm_suggested_tools": [skill.skill_id for skill in llm_selected_skills],
            "deterministic_tools": [skill.skill_id for skill in deterministic_skills],
            "union_tools": self._union_skill_ids(llm_selected_skills, deterministic_skills),
        }

        if llm_intent is not None:
            if llm_selected_skills:
                return llm_selected_skills, {
                    "source": "llm_intent",
                    "llm_intent": llm_intent.model_dump(mode="json", exclude_none=True),
                    "resolved_skill_ids": [skill.skill_id for skill in llm_selected_skills],
                    "llm_raw_suggested_tools": llm_intent.skill_ids,
                    "tool_selection": tool_selection,
                }

        return deterministic_skills, {
            "source": "keyword_match" if llm_intent is None else "keyword_fallback",
            "llm_intent": (
                llm_intent.model_dump(mode="json", exclude_none=True) if llm_intent is not None else None
            ),
            "resolved_skill_ids": [skill.skill_id for skill in deterministic_skills],
            "llm_raw_suggested_tools": llm_intent.skill_ids if llm_intent is not None else [],
            "tool_selection": tool_selection,
        }

    def _classify_intent(self, question: str) -> Optional[RagIntentResponse]:
        if self.llm is None or not hasattr(self.llm, "classify_intent"):
            return None

        try:
            return self.llm.classify_intent(
                question=question,
                available_skills=self.skill_registry.available_skills(),
            )
        except Exception:
            logger.exception("Intent classification raised unexpectedly; continuing with deterministic routing")
            return None

    @staticmethod
    def _union_skill_ids(*skill_groups: list[Skill]) -> list[str]:
        union_skill_ids: list[str] = []
        seen_skill_ids: set[str] = set()
        for skill_group in skill_groups:
            for skill in skill_group:
                if skill.skill_id in seen_skill_ids:
                    continue
                union_skill_ids.append(skill.skill_id)
                seen_skill_ids.add(skill.skill_id)
        return union_skill_ids

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
