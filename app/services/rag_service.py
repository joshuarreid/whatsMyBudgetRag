from __future__ import annotations

import hashlib
import json
import logging
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
    RagTimeScope,
    RagToolTraceResponse,
)
from app.repositories import (
    ConversationHistoryDisabledError,
    ConversationHistoryRepository,
    ConversationMessageRecord,
    ConversationNotFoundError,
    NullConversationHistoryRepository,
)
from app.services.intent_service import IntentService
from app.skills.base import Skill, SkillRequest
from app.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

TOOL_CACHE_TTL_SECONDS = 900
MAX_RESPONSE_CITATIONS = 5


class RAGService:
    """Skill-oriented orchestration over Spring Boot APIs without vector retrieval."""

    def __init__(
        self,
        spring_client: SpringBootClient,
        skill_registry: SkillRegistry,
        llm_service: Optional[Any] = None,
        conversation_history: Optional[ConversationHistoryRepository] = None,
        intent_service: Optional[Any] = None,
    ) -> None:
        self.spring = spring_client
        self.skill_registry = skill_registry
        self.llm = llm_service
        self.intent_service = intent_service
        self.intent_parser = intent_service or IntentService(enable_llm=False)
        settings = get_settings()
        self.conversation_history = conversation_history or NullConversationHistoryRepository()
        self.conversation_history_context_limit = settings.conversation_history_context_limit

    def answer(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        time_scope: Optional[RagTimeScope] = None,
        period: Optional[str] = None,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> RagAnswerResponse:
        requested_time_scope = self._normalize_time_scope_input(time_scope=time_scope, period=period)
        logger.info(
            "RAG answer requested requested_period=%s requested_time_scope=%s payment_method=%s account=%s transaction_id=%s question_length=%s",
            period or "-",
            requested_time_scope.model_dump(mode="json", exclude_none=True) if requested_time_scope is not None else {},
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
                period=self._derive_period_from_time_scope(requested_time_scope),
                transaction_id=transaction_id,
            )
            llm_intent = self._classify_intent(question)
            selected_time_scope, period_interpretation = self._resolve_time_scope(
                question=question,
                time_scope=requested_time_scope,
                period=period,
                transaction_id=transaction_id,
                conversation_history=prior_messages,
                llm_intent=llm_intent,
            )
            selected_period = self._derive_period_from_time_scope(selected_time_scope)
            selected_account, account_interpretation = self._resolve_account(
                question=question,
                account=account,
                conversation_history=prior_messages,
                llm_intent=llm_intent,
            )
            selected_skills, routing_metadata = self._select_skills(question, llm_intent=llm_intent)
            plan = [skill.skill_id for skill in selected_skills]
            logger.info(
                "RAG execution plan selected period=%s time_scope=%s account=%s plan=%s period_source=%s account_source=%s routing_source=%s",
                selected_period or "-",
                selected_time_scope.model_dump(mode="json", exclude_none=True),
                selected_account or "-",
                plan,
                period_interpretation.get("source", "unknown"),
                account_interpretation.get("source", "unknown"),
                routing_metadata.get("source", "unknown"),
            )
            skill_request = SkillRequest(
                question=question,
                time_scope=selected_time_scope,
                period=selected_period,
                payment_method=payment_method,
                account=selected_account,
                transaction_id=transaction_id,
            )
            context, tool_traces, response_citations, cache_metadata = self._build_context(
                skill_request=skill_request,
                plan=plan,
                period_interpretation=period_interpretation,
                account_interpretation=account_interpretation,
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
                    "time_scope": selected_time_scope.model_dump(mode="json", exclude_none=True),
                    "period": selected_period,
                    "period_source": period_interpretation.get("source"),
                    "resolved_filters": {
                        "payment_method": payment_method,
                        "account": selected_account,
                    },
                    "filter_interpretation": {
                        "account": account_interpretation,
                    },
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
                "RAG answer completed period=%s time_scope=%s context_keys=%s used_llm=%s",
                selected_period or "-",
                selected_time_scope.model_dump(mode="json", exclude_none=True),
                sorted(context.keys()),
                bool(llm_answer),
            )
            return RagAnswerResponse(
                question=question,
                conversation_id=resolved_conversation_id,
                time_scope=selected_time_scope,
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

    def _resolve_time_scope(
        self,
        question: str,
        time_scope: Optional[RagTimeScope],
        period: Optional[str],
        transaction_id: Optional[str],
        *,
        conversation_history: Optional[list[ConversationMessageRecord]] = None,
        llm_intent: Optional[RagIntentResponse] = None,
        today: Optional[date] = None,
    ) -> tuple[RagTimeScope, dict[str, Any]]:
        reference_date = today or date.today()
        inferred_time_scope = self.intent_parser.infer_time_scope(
            question=question,
            today=reference_date,
            llm_intent=llm_intent,
        )
        if inferred_time_scope is not None:
            resolved_time_scope = RagTimeScope.model_validate(inferred_time_scope["time_scope"])
            logger.debug(
                "Resolved analytics time scope from question matched_text=%s time_scope=%s",
                inferred_time_scope.get("matched_text"),
                resolved_time_scope.model_dump(mode="json", exclude_none=True),
            )
            return resolved_time_scope, inferred_time_scope

        prior_time_scope = self._last_resolved_time_scope(conversation_history)
        if prior_time_scope is not None:
            logger.debug(
                "Reusing prior conversation analytics time_scope=%s",
                prior_time_scope.model_dump(mode="json", exclude_none=True),
            )
            return prior_time_scope, self._build_time_scope_interpretation(
                source="conversation_history_time_scope",
                matched_text=self._time_scope_label(prior_time_scope),
                time_scope=prior_time_scope,
                resolution_rule="follow-up questions without a new time reference reuse the last resolved time scope from the conversation",
            )

        if time_scope is not None:
            logger.debug(
                "Using requested analytics time_scope=%s",
                time_scope.model_dump(mode="json", exclude_none=True),
            )
            return time_scope, self._build_time_scope_interpretation(
                source="request_time_scope" if period is None else "request_parameter",
                matched_text=self._time_scope_label(time_scope),
                time_scope=time_scope,
                resolution_rule="request time_scope overrides the default fallback when the question and conversation do not introduce a new time reference",
            )

        resolved_time_scope = RagTimeScope(
            scope_type="statement_period",
            statement_period=self.intent_parser.format_statement_period(reference_date),
        )
        logger.debug(
            "Falling back to current statement period=%s",
            resolved_time_scope.statement_period,
        )
        return resolved_time_scope, self._build_time_scope_interpretation(
            source="current_statement_period_fallback",
            matched_text=resolved_time_scope.statement_period or "",
            time_scope=resolved_time_scope,
            resolution_rule="questions without a time reference fall back to the current statement period",
        )

    def _resolve_period(
        self,
        question: str,
        period: Optional[str],
        transaction_id: Optional[str],
        *,
        conversation_history: Optional[list[ConversationMessageRecord]] = None,
        llm_intent: Optional[RagIntentResponse] = None,
        today: Optional[date] = None,
    ) -> tuple[Optional[str], dict[str, Any]]:
        resolved_time_scope, interpretation = self._resolve_time_scope(
            question=question,
            time_scope=None,
            period=period,
            transaction_id=transaction_id,
            conversation_history=conversation_history,
            llm_intent=llm_intent,
            today=today,
        )
        return self._derive_period_from_time_scope(resolved_time_scope), interpretation

    def _resolve_account(
        self,
        question: str,
        account: Optional[str],
        *,
        conversation_history: Optional[list[ConversationMessageRecord]] = None,
        llm_intent: Optional[RagIntentResponse] = None,
    ) -> tuple[Optional[str], dict[str, Optional[str]]]:
        inferred_account = self.intent_parser.infer_account(question=question, llm_intent=llm_intent)
        prior_account = self._last_resolved_filter(conversation_history, "account")

        if inferred_account is not None and inferred_account.get("resolved_account"):
            resolved_account = inferred_account["resolved_account"]
            logger.debug("Resolved analytics account from question matched_text=%s resolved_account=%s", inferred_account.get("matched_text"), resolved_account)
            return resolved_account, inferred_account

        if inferred_account is not None and inferred_account.get("source") == "question_contextual_account_reference":
            if prior_account:
                return prior_account, {
                    "source": "question_contextual_account_reference",
                    "matched_text": inferred_account.get("matched_text"),
                    "resolved_account": prior_account,
                    "resolution_rule": "contextual references like this account reuse the last resolved account from the conversation",
                }
            if account:
                return account, {
                    "source": "question_contextual_request_account",
                    "matched_text": inferred_account.get("matched_text"),
                    "resolved_account": account,
                    "resolution_rule": "contextual references like this account reuse the request account when the conversation has not resolved one yet",
                }

        if prior_account:
            logger.debug("Reusing prior conversation analytics account=%s", prior_account)
            return prior_account, {
                "source": "conversation_history_account",
                "matched_text": prior_account,
                "resolved_account": prior_account,
                "resolution_rule": "follow-up questions without a new account reference reuse the last resolved account from the conversation",
            }

        if account:
            logger.debug("Using requested analytics account=%s", account)
            return account, {
                "source": "request_parameter",
                "matched_text": account,
                "resolved_account": account,
            }

        return None, {
            "source": "no_account_filter",
            "matched_text": None,
            "resolved_account": None,
            "resolution_rule": "questions without an account reference or prior account context do not apply an account filter",
        }

    def _build_context(
        self,
        skill_request: SkillRequest,
        plan: list[str],
        period_interpretation: dict[str, Any],
        account_interpretation: dict[str, Optional[str]],
        skills: list[Skill],
        routing_metadata: dict[str, Any],
        *,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[list[ConversationMessageRecord]] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        timeline_context = self._build_timeline_context(
            time_scope=skill_request.time_scope,
            period_interpretation=period_interpretation,
        )
        context: dict[str, Any] = {
            "question": skill_request.question,
            "time_scope": skill_request.time_scope.model_dump(mode="json", exclude_none=True),
            "period": skill_request.period,
            "filters": {
                "payment_method": skill_request.payment_method,
                "account": skill_request.account,
            },
            "timeline_context": timeline_context,
            "period_interpretation": period_interpretation,
            "filter_interpretation": {
                "account": account_interpretation,
            },
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
                    "time_scope": self._message_time_scope(message).model_dump(mode="json", exclude_none=True)
                    if self._message_time_scope(message) is not None
                    else None,
                    "period": message.period,
                    "period_source": message.period_source,
                    "filters": self._message_resolved_filters(message),
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

    def _select_skills(
        self,
        question: str,
        *,
        llm_intent: Optional[RagIntentResponse] = None,
    ) -> tuple[list[Skill], dict[str, Any]]:
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
        service = self.intent_service
        if service is None and self.llm is not None and hasattr(self.llm, "classify_intent"):
            service = self.llm

        if service is None or not hasattr(service, "classify_intent"):
            return None

        try:
            return service.classify_intent(
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
        time_scope: RagTimeScope,
        period_interpretation: dict[str, Any],
        *,
        today: Optional[date] = None,
    ) -> dict[str, Any]:
        reference_date = today or date.today()
        current_statement_period = self.intent_parser.format_statement_period(reference_date)
        derived_period = self._derive_period_from_time_scope(time_scope)
        return {
            "current_date": reference_date.isoformat(),
            "current_month": current_statement_period,
            "current_statement_period": current_statement_period,
            "selected_time_scope": time_scope.model_dump(mode="json", exclude_none=True),
            "selected_statement_period": derived_period,
            "selected_statement_period_range": {
                "start_period": time_scope.start_period,
                "end_period": time_scope.end_period,
            }
            if time_scope.scope_type == "statement_period_range"
            else None,
            "selected_date_range": {
                "start_date": time_scope.start_date.isoformat() if time_scope.start_date is not None else None,
                "end_date": time_scope.end_date.isoformat() if time_scope.end_date is not None else None,
            }
            if time_scope.scope_type == "date_range"
            else None,
            "statement_period_format": "MonthYear",
            "statement_period_examples": ["October2025", "May2026"],
            "period_resolution_rules": {
                "bare_month_name": "Resolve to the most recent matching month not in the future relative to current_statement_period.",
                "current_month": "Resolve to current_statement_period.",
                "last_month": "Resolve to the statement period immediately before current_statement_period.",
                "month_range": "Resolve to an inclusive statement period range between the referenced months.",
                "date_range": "Resolve to inclusive ISO date boundaries when the question uses week, day, or partial-month phrasing.",
            },
            "selection_source": period_interpretation.get("source", "unknown"),
        }

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

    @staticmethod
    def _normalize_time_scope_input(
        *,
        time_scope: Optional[Any],
        period: Optional[str],
    ) -> Optional[RagTimeScope]:
        if isinstance(time_scope, RagTimeScope):
            return time_scope
        if isinstance(time_scope, dict):
            return RagTimeScope.model_validate(time_scope)
        if isinstance(period, str) and period.strip():
            return RagTimeScope(scope_type="statement_period", statement_period=period.strip())
        return None

    @staticmethod
    def _derive_period_from_time_scope(time_scope: Optional[RagTimeScope]) -> Optional[str]:
        if time_scope is None:
            return None
        if time_scope.scope_type == "statement_period":
            return time_scope.statement_period
        return None

    def _build_time_scope_interpretation(
        self,
        *,
        source: str,
        matched_text: str,
        time_scope: RagTimeScope,
        resolution_rule: Optional[str] = None,
    ) -> dict[str, Any]:
        interpretation: dict[str, Any] = {
            "source": source,
            "matched_text": matched_text,
            "scope_type": time_scope.scope_type,
            "time_scope": time_scope.model_dump(mode="json", exclude_none=True),
        }
        derived_period = self._derive_period_from_time_scope(time_scope)
        if derived_period is not None:
            interpretation["resolved_period"] = derived_period
        if time_scope.start_period is not None:
            interpretation["resolved_start_period"] = time_scope.start_period
        if time_scope.end_period is not None:
            interpretation["resolved_end_period"] = time_scope.end_period
        if time_scope.start_date is not None:
            interpretation["resolved_start_date"] = time_scope.start_date.isoformat()
        if time_scope.end_date is not None:
            interpretation["resolved_end_date"] = time_scope.end_date.isoformat()
        if resolution_rule is not None:
            interpretation["resolution_rule"] = resolution_rule
        return interpretation

    @staticmethod
    def _time_scope_label(time_scope: RagTimeScope) -> str:
        if time_scope.scope_type == "statement_period":
            return time_scope.statement_period or "statement period"
        if time_scope.scope_type == "statement_period_range":
            return f"{time_scope.start_period or '?'} through {time_scope.end_period or '?'}"
        if time_scope.scope_type == "date_range":
            return f"{time_scope.start_date.isoformat() if time_scope.start_date else '?'} through {time_scope.end_date.isoformat() if time_scope.end_date else '?'}"
        return time_scope.scope_type

    def _last_resolved_time_scope(
        self,
        conversation_history: Optional[list[ConversationMessageRecord]],
    ) -> Optional[RagTimeScope]:
        if not conversation_history:
            return None
        for message in reversed(conversation_history):
            if message.role != "assistant":
                continue
            resolved_time_scope = self._message_time_scope(message)
            if resolved_time_scope is not None:
                return resolved_time_scope
        return None

    def _message_time_scope(self, message: ConversationMessageRecord) -> Optional[RagTimeScope]:
        answer_json = message.answer_json if isinstance(message.answer_json, dict) else {}
        for container in (answer_json, message.context_json if isinstance(message.context_json, dict) else {}):
            time_scope_payload = container.get("time_scope") if isinstance(container, dict) else None
            if isinstance(time_scope_payload, dict):
                try:
                    return RagTimeScope.model_validate(time_scope_payload)
                except Exception:
                    logger.debug("Ignoring invalid persisted time_scope payload for message=%s", message.message_id)
        if isinstance(message.period, str) and message.period.strip():
            return RagTimeScope(scope_type="statement_period", statement_period=message.period.strip())
        return None

    @staticmethod
    def _last_resolved_period(
        conversation_history: Optional[list[ConversationMessageRecord]],
    ) -> Optional[str]:
        if not conversation_history:
            return None
        for message in reversed(conversation_history):
            if message.role != "assistant":
                continue
            answer_json = message.answer_json if isinstance(message.answer_json, dict) else {}
            time_scope_payload = answer_json.get("time_scope")
            if isinstance(time_scope_payload, dict):
                statement_period = time_scope_payload.get("statement_period")
                if isinstance(statement_period, str) and statement_period.strip():
                    return statement_period.strip()
            if isinstance(message.period, str) and message.period.strip():
                return message.period.strip()
        return None

    def _last_resolved_filter(
        self,
        conversation_history: Optional[list[ConversationMessageRecord]],
        filter_name: str,
    ) -> Optional[str]:
        if not conversation_history:
            return None
        for message in reversed(conversation_history):
            if message.role != "assistant":
                continue
            resolved_filters = self._message_resolved_filters(message)
            filter_value = resolved_filters.get(filter_name)
            if isinstance(filter_value, str) and filter_value.strip():
                return filter_value.strip()
        return None

    @staticmethod
    def _message_resolved_filters(message: ConversationMessageRecord) -> dict[str, Any]:
        answer_json = message.answer_json if isinstance(message.answer_json, dict) else {}
        resolved_filters = answer_json.get("resolved_filters")
        if isinstance(resolved_filters, dict):
            return resolved_filters

        context_json = message.context_json if isinstance(message.context_json, dict) else {}
        context_filters = context_json.get("filters")
        if isinstance(context_filters, dict):
            return context_filters

        fallback_filters = answer_json.get("filters")
        return fallback_filters if isinstance(fallback_filters, dict) else {}

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
        period: Optional[str],
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
        time_scope_payload = None
        answer_json = message.answer_json if isinstance(message.answer_json, dict) else {}
        if isinstance(answer_json.get("time_scope"), dict):
            time_scope_payload = answer_json.get("time_scope")
        elif isinstance(message.context_json, dict) and isinstance(message.context_json.get("time_scope"), dict):
            time_scope_payload = message.context_json.get("time_scope")
        elif isinstance(message.period, str) and message.period.strip():
            time_scope_payload = {
                "scope_type": "statement_period",
                "statement_period": message.period.strip(),
            }
        return RagConversationMessageResponse(
            message_id=message.message_id,
            role=message.role,
            content=message.content,
            time_scope=RagTimeScope.model_validate(time_scope_payload) if isinstance(time_scope_payload, dict) else None,
            period=message.period,
            period_source=message.period_source,
            created_at=message.created_at,
        )


    def _fallback_answer(self, context: dict[str, Any]) -> str:
        fragments: list[str] = []
        selected_scope_label = context.get("period") or self._time_scope_label(
            RagTimeScope.model_validate(context.get("time_scope", {"scope_type": "statement_period"}))
        )
        overview = context.get("overview")
        if isinstance(overview, dict):
            total_spend = overview.get("total_amount")
            transaction_count = overview.get("transaction_count")
            if total_spend is not None:
                if context.get("period"):
                    fragments.append(f"Total spend for period {context.get('period')} is {total_spend}.")
                else:
                    fragments.append(f"Total spend for {selected_scope_label} is {total_spend}.")
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
