from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date
from typing import Any, Optional

from app.clients.spring_boot_client import SpringBootClient
from app.core.config import get_settings
from app.core.logging import get_request_id
from app.models.schemas import (
    RagAnswerResponse,
    RagCacheMetadataResponse,
    RagCitationResponse,
    RagConversationMessageResponse,
    RagConversationResponse,
    RagIntentResponse,
    RagToolTraceResponse,
)
from app.repositories import (
    ConversationHistoryDisabledError,
    ConversationHistoryRepository,
    ConversationMessageRecord,
    ConversationNotFoundError,
    NullConversationHistoryRepository,
)
from app.skills.base import Skill, SkillRequest
from app.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

TOOL_CACHE_TTL_SECONDS = 900
MAX_RESPONSE_CITATIONS = 5

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
        conversation_history: Optional[ConversationHistoryRepository] = None,
    ) -> None:
        self.spring = spring_client
        self.skill_registry = skill_registry
        self.llm = llm_service
        settings = get_settings()
        self.conversation_history = conversation_history or NullConversationHistoryRepository()
        self.conversation_history_context_limit = settings.conversation_history_context_limit

    def answer(
        self,
        question: str,
        conversation_id: Optional[str] = None,
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
            resolved_conversation_id, prior_messages = self._prepare_conversation(
                conversation_id=conversation_id,
                question=question,
            )
            self._persist_message(
                resolved_conversation_id,
                role="user",
                content=question,
                transaction_id=transaction_id,
            )
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
            context, tool_traces, response_citations, cache_metadata = self._build_context(
                skill_request=skill_request,
                plan=plan,
                period_interpretation=period_interpretation,
                skills=selected_skills,
                routing_metadata=routing_metadata,
                conversation_id=resolved_conversation_id,
                conversation_history=prior_messages,
            )
            llm_answer = self.llm.generate_answer(question, context) if self.llm else None
            answer = llm_answer or self._fallback_answer(context)
            assistant_message = self._persist_message(
                resolved_conversation_id,
                role="assistant",
                content=answer,
                period=selected_period,
                period_source=period_interpretation.get("source"),
                transaction_id=transaction_id,
                model_name=getattr(self.llm, "model", None),
                tool_plan=routing_metadata.get("tool_selection"),
                context_json=context,
                answer_json={
                    "question": question,
                    "conversation_id": resolved_conversation_id,
                    "period": selected_period,
                    "plan": plan,
                    "tool_selection": routing_metadata.get("tool_selection", {}),
                    "citations": response_citations,
                    "tool_traces": self._response_tool_traces(tool_traces),
                    "cache": cache_metadata,
                    "answer": answer,
                },
            )
            self._persist_tooling_metadata(
                assistant_message,
                conversation_id=resolved_conversation_id,
                tool_traces=tool_traces,
                citations=response_citations,
                cache_metadata=cache_metadata,
                period=selected_period,
            )
            logger.info(
                "RAG answer completed period=%s context_keys=%s used_llm=%s",
                selected_period,
                sorted(context.keys()),
                bool(llm_answer),
            )
            return RagAnswerResponse(
                question=question,
                conversation_id=resolved_conversation_id,
                period=selected_period,
                plan=plan,
                tool_selection=routing_metadata.get("tool_selection", {}),
                context=context,
                citations=[RagCitationResponse.model_validate(citation) for citation in response_citations],
                tool_traces=self._response_tool_traces(tool_traces),
                cache=RagCacheMetadataResponse.model_validate(cache_metadata),
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

    def get_conversation_history(self, conversation_id: str, *, limit: int = 50) -> RagConversationResponse:
        self._require_history_enabled()
        conversation = self.conversation_history.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(f"Conversation {conversation_id} was not found")

        messages = self.conversation_history.list_messages(conversation_id, limit=limit)
        return RagConversationResponse(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            last_message_at=conversation.last_message_at,
            messages=[self._message_response(message) for message in messages],
        )

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
        *,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[list[ConversationMessageRecord]] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
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
        if conversation_id is not None:
            context["conversation"] = {
                "conversation_id": conversation_id,
                "history_message_count": len(conversation_history or []),
            }
        if conversation_history:
            context["conversation_history"] = [
                {
                    "message_id": message.message_id,
                    "role": message.role,
                    "content": message.content,
                    "period": message.period,
                    "period_source": message.period_source,
                    "created_at": message.created_at.isoformat(),
                }
                for message in conversation_history
            ]
        skill_context, unavailable_skills, tool_traces = self.skill_registry.execute(
            skills,
            skill_request,
            cache_lookup=(self._cache_lookup(conversation_id) if conversation_id is not None else None),
        )
        context.update(skill_context)

        if unavailable_skills:
            context["unavailable_tools"] = unavailable_skills
            context["degraded"] = True

        response_citations = self._response_citations(tool_traces)
        cache_metadata = self._cache_metadata(conversation_id, tool_traces)
        context["supporting_sources"] = response_citations
        context["cache"] = cache_metadata
        context["tool_trace_summaries"] = [trace.model_dump(mode="json") for trace in self._response_tool_traces(tool_traces)]

        return context, tool_traces, response_citations, cache_metadata

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

    def _prepare_conversation(
        self,
        *,
        conversation_id: Optional[str],
        question: str,
    ) -> tuple[Optional[str], list[ConversationMessageRecord]]:
        if not self.conversation_history.is_enabled():
            if conversation_id is not None:
                raise ConversationHistoryDisabledError("Conversation history is not configured for this deployment")
            return None, []

        if conversation_id is None:
            conversation = self.conversation_history.create_conversation(title=self._conversation_title(question))
            return conversation.conversation_id, []

        conversation = self.conversation_history.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(f"Conversation {conversation_id} was not found")

        messages = self.conversation_history.list_messages(
            conversation_id,
            limit=self.conversation_history_context_limit,
        )
        return conversation_id, messages

    def _persist_message(
        self,
        conversation_id: Optional[str],
        *,
        role: str,
        content: str,
        period: Optional[str] = None,
        period_source: Optional[str] = None,
        transaction_id: Optional[str] = None,
        model_name: Optional[str] = None,
        tool_plan: Optional[dict[str, Any]] = None,
        context_json: Optional[dict[str, Any]] = None,
        answer_json: Optional[dict[str, Any]] = None,
    ) -> Optional[ConversationMessageRecord]:
        if conversation_id is None:
            return None

        return self.conversation_history.append_message(
            conversation_id,
            role=role,
            content=content,
            period=period,
            period_source=period_source,
            transaction_id=transaction_id,
            request_id=get_request_id(),
            model_name=model_name,
            tool_plan=tool_plan,
            context_json=context_json,
            answer_json=answer_json,
        )

    def _persist_tooling_metadata(
        self,
        assistant_message: Optional[ConversationMessageRecord],
        *,
        conversation_id: Optional[str],
        tool_traces: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        cache_metadata: dict[str, Any],
        period: str,
    ) -> None:
        if assistant_message is None or conversation_id is None:
            return

        persisted_tool_calls = [
            {
                "tool_name": trace.get("tool_name"),
                "arguments": trace.get("arguments", {}),
                "result": {
                    "context_key": trace.get("context_key"),
                    "result_summary": trace.get("result_summary", {}),
                    "cache_hit": trace.get("cache_hit", False),
                    "citation": trace.get("citation"),
                },
                "status": trace.get("status", "ok"),
                "error_text": trace.get("error_text"),
                "duration_ms": trace.get("duration_ms"),
            }
            for trace in tool_traces
        ]
        self.conversation_history.append_message_tool_calls(
            assistant_message.message_id,
            tool_calls=persisted_tool_calls,
        )
        self.conversation_history.append_message_citations(
            assistant_message.message_id,
            citations=citations,
        )

        writes = 0
        for trace in tool_traces:
            if trace.get("status") != "ok" or trace.get("cache_hit") or not trace.get("cacheable"):
                continue
            tool_name = trace.get("tool_name")
            arguments = trace.get("arguments", {})
            if not isinstance(tool_name, str):
                continue
            cache_key = self._tool_cache_key(tool_name, arguments)
            self.conversation_history.upsert_tool_cache(
                conversation_id,
                tool_name=tool_name,
                cache_key=cache_key,
                period=period,
                params_json=self._cacheable_arguments(arguments),
                response_json={
                    "context_key": trace.get("context_key"),
                    "payload": trace.get("result"),
                    "metadata": {
                        "citation": trace.get("citation"),
                        "description": trace.get("description"),
                    },
                },
                source_message_db_id=assistant_message.db_id,
                ttl_seconds=TOOL_CACHE_TTL_SECONDS,
            )
            writes += 1
        cache_metadata["writes"] = writes

    def _cache_lookup(
        self,
        conversation_id: str,
    ):
        def lookup(skill: Skill, arguments: dict[str, Any]) -> Optional[dict[str, Any]]:
            cache_key = self._tool_cache_key(skill.skill_id, arguments)
            cached_entry = self.conversation_history.get_tool_cache(
                conversation_id,
                tool_name=skill.skill_id,
                cache_key=cache_key,
            )
            if cached_entry is None:
                return None
            response_json = cached_entry.get("response_json")
            if not isinstance(response_json, dict):
                return None
            return {
                "context_key": response_json.get("context_key", skill.context_key),
                "payload": response_json.get("payload"),
                "metadata": response_json.get("metadata", {}),
                "created_at": cached_entry.get("created_at"),
                "expires_at": cached_entry.get("expires_at"),
            }

        return lookup

    @staticmethod
    def _cacheable_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in arguments.items()
            if value is not None and key not in {"transaction_id"}
        }

    def _tool_cache_key(self, tool_name: str, arguments: dict[str, Any]) -> str:
        cache_input = {
            "tool_name": tool_name,
            "arguments": self._cacheable_arguments(arguments),
        }
        return hashlib.sha256(json.dumps(cache_input, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _response_citations(self, tool_traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen_source_refs: set[str] = set()
        for trace in tool_traces:
            citation = trace.get("citation")
            if trace.get("status") != "ok" or not isinstance(citation, dict):
                continue
            source_ref = citation.get("source_ref")
            if not isinstance(source_ref, str) or source_ref in seen_source_refs:
                continue
            seen_source_refs.add(source_ref)
            citations.append(citation)
            if len(citations) >= MAX_RESPONSE_CITATIONS:
                break
        return citations

    def _response_tool_traces(self, tool_traces: list[dict[str, Any]]) -> list[RagToolTraceResponse]:
        return [
            RagToolTraceResponse(
                tool_name=str(trace.get("tool_name", "unknown")),
                context_key=str(trace.get("context_key", "unknown")),
                category=str(trace.get("category", "unknown")),
                status=str(trace.get("status", "ok")),
                duration_ms=trace.get("duration_ms"),
                cache_hit=bool(trace.get("cache_hit", False)),
                arguments=trace.get("arguments", {}) if isinstance(trace.get("arguments"), dict) else {},
                result_summary=trace.get("result_summary", {}) if isinstance(trace.get("result_summary"), dict) else {},
                error_text=trace.get("error_text"),
            )
            for trace in tool_traces
        ]

    def _cache_metadata(self, conversation_id: Optional[str], tool_traces: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "enabled": conversation_id is not None and self.conversation_history.is_enabled(),
            "hits": sum(1 for trace in tool_traces if trace.get("cache_hit")),
            "misses": sum(
                1
                for trace in tool_traces
                if trace.get("status") == "ok" and trace.get("cacheable") and not trace.get("cache_hit")
            ),
            "writes": 0,
        }

    def _require_history_enabled(self) -> None:
        if not self.conversation_history.is_enabled():
            raise ConversationHistoryDisabledError("Conversation history is not configured for this deployment")

    @staticmethod
    def _conversation_title(question: str) -> str:
        normalized = " ".join(question.split())
        return normalized[:77] + "..." if len(normalized) > 80 else normalized

    @staticmethod
    def _message_response(message: ConversationMessageRecord) -> RagConversationMessageResponse:
        return RagConversationMessageResponse(
            message_id=message.message_id,
            role=message.role,
            content=message.content,
            period=message.period,
            period_source=message.period_source,
            created_at=message.created_at,
        )


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
