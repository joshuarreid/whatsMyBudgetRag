from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from time import perf_counter
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
    RagExecutionPlan,
    RagIntentResponse,
    RagPlanStep,
    RagTimingMetadataResponse,
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
from app.services.langgraph_reasoning_service import LangGraphReasoningService
from app.services.planner_service import PlannerService
from app.skills.base import Skill, SkillRequest
from app.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

TOOL_CACHE_TTL_SECONDS = 900
MAX_RESPONSE_CITATIONS = 5
TREND_ROUTING_KEYWORDS = ("daily", "trend", "over time", "time series")
SUMMARY_ROUTING_KEYWORDS = ("overview", "summary", "summarize", "total", "how much")
ANSWER_CONTEXT_EXCLUDED_KEYS = {
    "cache",
    "conversation",
    "conversation_history",
    "execution_plan",
    "routing",
    "supporting_sources",
    "tool_trace_summaries",
}
DAILY_CONTEXT_KEY_PREFIX = "daily_totals"
DECIMAL_CENTS = Decimal("0.01")


class RAGService:
    """Skill-oriented orchestration over Spring Boot APIs without vector retrieval."""

    def __init__(
        self,
        spring_client: SpringBootClient,
        skill_registry: SkillRegistry,
        llm_service: Optional[Any] = None,
        conversation_history: Optional[ConversationHistoryRepository] = None,
        intent_service: Optional[Any] = None,
        langgraph_service: Optional[LangGraphReasoningService] = None,
    ) -> None:
        self.spring = spring_client
        self.skill_registry = skill_registry
        self.llm = llm_service
        self.intent_service = intent_service
        self.intent_parser = intent_service or IntentService(enable_llm=False)
        self.planner = PlannerService(intent_service=self.intent_parser)
        self.langgraph_service = langgraph_service
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
        reference_date = date.today()
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
            intent_classification = self._intent_classification_policy(question, today=reference_date)
            classifier_started_at = perf_counter()
            llm_intent = None if intent_classification["skip"] else self._classify_intent(question)
            classifier_latency_ms = int((perf_counter() - classifier_started_at) * 1000)
            selected_time_scope, period_interpretation = self._resolve_time_scope(
                question=question,
                time_scope=requested_time_scope,
                period=period,
                transaction_id=transaction_id,
                conversation_history=prior_messages,
                llm_intent=llm_intent,
                today=reference_date,
            )
            selected_period = self._derive_period_from_time_scope(selected_time_scope)
            selected_account, account_interpretation = self._resolve_account(
                question=question,
                account=account,
                conversation_history=prior_messages,
                llm_intent=llm_intent,
            )
            selected_skills, routing_metadata = self._select_skills(
                question,
                time_scope=selected_time_scope,
                llm_intent=llm_intent,
            )
            routing_metadata["intent_classification"] = {
                "attempted": not intent_classification["skip"],
                "skipped": intent_classification["skip"],
                "reason": intent_classification["reason"],
                "direct_matches": intent_classification["direct_matches"],
                "inferred_time_scope": intent_classification.get("inferred_time_scope"),
            }
            execution_plan = self.planner.build_plan(
                question=question,
                skills=selected_skills,
                time_scope=selected_time_scope,
                period=selected_period,
                payment_method=payment_method,
                account=selected_account,
            )
            plan = [step.skill_id for step in execution_plan.steps]
            logger.info(
                "RAG execution plan selected period=%s time_scope=%s account=%s plan=%s strategy=%s period_source=%s account_source=%s routing_source=%s",
                selected_period or "-",
                selected_time_scope.model_dump(mode="json", exclude_none=True),
                selected_account or "-",
                plan,
                execution_plan.strategy,
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
            plan_execution_started_at = perf_counter()
            context, tool_traces, response_citations, cache_metadata = self._build_context(
                skill_request=skill_request,
                plan=plan,
                period_interpretation=period_interpretation,
                account_interpretation=account_interpretation,
                skills=selected_skills,
                execution_plan=execution_plan,
                routing_metadata=routing_metadata,
                conversation_id=resolved_conversation_id,
                conversation_history=prior_messages,
            )
            plan_execution_latency_ms = int((perf_counter() - plan_execution_started_at) * 1000)
            answer_generation_started_at = perf_counter()
            deterministic_answer = self._deterministic_answer(context)
            llm_answer = None
            if deterministic_answer is None:
                answer_context = self._build_answer_context(context)
                llm_answer = self.llm.generate_answer(question, answer_context) if self.llm else None
            answer_generation_latency_ms = int((perf_counter() - answer_generation_started_at) * 1000)
            timing_metadata = self._build_timing_metadata(
                tool_traces=tool_traces,
                classifier_latency_ms=classifier_latency_ms,
                plan_execution_latency_ms=plan_execution_latency_ms,
                answer_generation_latency_ms=answer_generation_latency_ms,
            )
            answer = deterministic_answer or llm_answer or self._fallback_answer(context)
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
                    "execution_plan": execution_plan.model_dump(mode="json", exclude_none=True),
                    "tool_selection": routing_metadata.get("tool_selection", {}),
                    "citations": response_citations,
                    "tool_traces": self._response_tool_traces(tool_traces),
                    "cache": cache_metadata,
                    "timing": timing_metadata,
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
                timing=RagTimingMetadataResponse.model_validate(timing_metadata),
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
        execution_plan: RagExecutionPlan,
        routing_metadata: dict[str, Any],
        *,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[list[ConversationMessageRecord]] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        if skill_request.time_scope is None:
            raise ValueError("RAG context building requires a resolved time_scope")
        resolved_time_scope = skill_request.time_scope
        cache_enabled = conversation_id is not None and self.conversation_history.is_enabled()
        cache_policy = self._cache_policy(resolved_time_scope, enabled=cache_enabled)
        cache_allowed = cache_policy["eligible"]
        timeline_context = self._build_timeline_context(
            time_scope=resolved_time_scope,
            period_interpretation=period_interpretation,
        )
        context: dict[str, Any] = {
            "question": skill_request.question,
            "time_scope": resolved_time_scope.model_dump(mode="json", exclude_none=True),
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
            "execution_plan": execution_plan.model_dump(mode="json", exclude_none=True),
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
        cache_lookup = self._cache_lookup(conversation_id) if conversation_id is not None and cache_allowed else None
        if execution_plan.steps:
            skill_context, unavailable_skills, tool_traces = self._execute_plan(
                execution_plan=execution_plan,
                skills=skills,
                question=skill_request.question,
                transaction_id=skill_request.transaction_id,
                cache_lookup=cache_lookup,
            )
        else:
            skill_context, unavailable_skills, tool_traces = self.skill_registry.execute(
                skills,
                skill_request,
                cache_lookup=cache_lookup,
            )
        self._apply_cache_policy(tool_traces, cache_allowed=cache_allowed)
        context.update(skill_context)

        if unavailable_skills:
            context["unavailable_tools"] = unavailable_skills
            context["degraded"] = True

        response_citations = self._response_citations(tool_traces)
        cache_metadata = self._cache_metadata(tool_traces, cache_policy=cache_policy)
        context["supporting_sources"] = response_citations
        context["cache"] = cache_metadata
        context["tool_trace_summaries"] = [trace.model_dump(mode="json") for trace in self._response_tool_traces(tool_traces)]

        return context, tool_traces, response_citations, cache_metadata

    def _execute_plan(
        self,
        *,
        execution_plan: RagExecutionPlan,
        skills: list[Skill],
        question: str,
        transaction_id: Optional[str],
        cache_lookup: Optional[Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        context_sections: dict[str, Any] = {}
        unavailable_skills: list[dict[str, Any]] = []
        tool_traces: list[dict[str, Any]] = []
        resolved_skills = {skill.skill_id: skill for skill in skills}

        indexed_results: list[tuple[int, RagPlanStep, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]]
        if len(execution_plan.steps) <= 1:
            indexed_results = [
                self._execute_plan_step(
                    step_index=step_index,
                    step=step,
                    resolved_skills=resolved_skills,
                    question=question,
                    transaction_id=transaction_id,
                    cache_lookup=cache_lookup,
                )
                for step_index, step in enumerate(execution_plan.steps)
            ]
        else:
            max_workers = min(len(execution_plan.steps), 8)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._execute_plan_step,
                        step_index=step_index,
                        step=step,
                        resolved_skills=resolved_skills,
                        question=question,
                        transaction_id=transaction_id,
                        cache_lookup=cache_lookup,
                    )
                    for step_index, step in enumerate(execution_plan.steps)
                ]
                indexed_results = [future.result() for future in as_completed(futures)]
            indexed_results.sort(key=lambda item: item[0])

        for _, step, step_context, step_unavailable, step_traces in indexed_results:
            if step_context:
                payload = step_context.get(step.context_key)
                if payload is None:
                    payload = next(iter(step_context.values()))
                context_sections[step.output_key] = payload

            unavailable_skills.extend(
                {
                    **item,
                    "step_id": step.step_id,
                    "output_key": step.output_key,
                    "label": step.label,
                }
                for item in step_unavailable
            )

            for trace in step_traces:
                trace["plan_step_id"] = step.step_id
                trace["plan_step_label"] = step.label
                trace["cache_context_key"] = trace.get("context_key")
                trace["context_key"] = step.output_key
                trace["output_key"] = step.output_key
                tool_traces.append(trace)

        return context_sections, unavailable_skills, tool_traces

    def _execute_plan_step(
        self,
        *,
        step_index: int,
        step: RagPlanStep,
        resolved_skills: dict[str, Skill],
        question: str,
        transaction_id: Optional[str],
        cache_lookup: Optional[Any],
    ) -> tuple[int, RagPlanStep, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        skill = resolved_skills.get(step.skill_id)
        if skill is None:
            return step_index, step, {}, [
                {
                    "tool": step.skill_id,
                    "error_type": "UnknownSkill",
                    "detail": f"Planned skill {step.skill_id} is not registered",
                }
            ], []

        step_request = SkillRequest(
            question=question,
            time_scope=step.time_scope,
            period=step.period,
            payment_method=step.payment_method,
            account=step.account,
            transaction_id=transaction_id,
        )
        step_context, step_unavailable, step_traces = self.skill_registry.execute(
            [skill],
            step_request,
            cache_lookup=cache_lookup,
        )
        return step_index, step, step_context, step_unavailable, step_traces

    def _apply_cache_policy(self, tool_traces: list[dict[str, Any]], *, cache_allowed: bool) -> None:
        if cache_allowed:
            return
        for trace in tool_traces:
            trace["cache_hit"] = False
            trace["cacheable"] = False

    def _select_skills(
        self,
        question: str,
        *,
        time_scope: RagTimeScope,
        llm_intent: Optional[RagIntentResponse] = None,
    ) -> tuple[list[Skill], dict[str, Any]]:
        llm_selected_skills = self._optimize_skill_selection(
            question,
            self.skill_registry.resolve(llm_intent.skill_ids) if llm_intent is not None else [],
        )
        deterministic_skills = self._optimize_skill_selection(question, self.skill_registry.select(question))
        tool_selection = {
            "llm_suggested_tools": [skill.skill_id for skill in llm_selected_skills],
            "deterministic_tools": [skill.skill_id for skill in deterministic_skills],
            "union_tools": self._union_skill_ids(llm_selected_skills, deterministic_skills),
        }
        registry_skills = getattr(self.skill_registry, "skills", None)
        if self.langgraph_service is not None and isinstance(registry_skills, list):
            seed_skills = self._optimize_skill_selection(
                question,
                [*llm_selected_skills, *deterministic_skills],
            )
            graph_selected_skills, graph_metadata = self.langgraph_service.plan(
                question=question,
                time_scope=time_scope,
                available_skills=registry_skills,
                seed_skills=seed_skills,
            )
            if graph_metadata.get("applied"):
                return self._optimize_skill_selection(question, graph_selected_skills), {
                    "source": "langgraph_reasoning",
                    "llm_intent": (
                        llm_intent.model_dump(mode="json", exclude_none=True) if llm_intent is not None else None
                    ),
                    "resolved_skill_ids": [skill.skill_id for skill in graph_selected_skills],
                    "llm_raw_suggested_tools": llm_intent.skill_ids if llm_intent is not None else [],
                    "tool_selection": tool_selection,
                    "reasoning_graph": graph_metadata,
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

    def _optimize_skill_selection(self, question: str, skills: list[Skill]) -> list[Skill]:
        optimized: list[Skill] = []
        seen_skill_ids: set[str] = set()
        for skill in skills:
            if skill.skill_id in seen_skill_ids:
                continue
            optimized.append(skill)
            seen_skill_ids.add(skill.skill_id)

        if self._is_trend_only_question(question) and any(skill.skill_id != "overview" for skill in optimized):
            optimized = [skill for skill in optimized if skill.skill_id != "overview"]
        return optimized

    @staticmethod
    def _is_trend_only_question(question: str) -> bool:
        lowered = question.lower()
        has_trend_signal = any(keyword in lowered for keyword in TREND_ROUTING_KEYWORDS)
        has_summary_signal = any(keyword in lowered for keyword in SUMMARY_ROUTING_KEYWORDS)
        return has_trend_signal and not has_summary_signal

    def _intent_classification_policy(
        self,
        question: str,
        *,
        today: Optional[date] = None,
    ) -> dict[str, Any]:
        registry_skills = getattr(self.skill_registry, "skills", None)
        direct_matches = self._optimize_skill_selection(
            question,
            [skill for skill in registry_skills if skill.matches(question)] if isinstance(registry_skills, list) else [],
        )
        direct_match_ids = [skill.skill_id for skill in direct_matches]
        classifier = self._intent_classifier()
        if classifier is None:
            return {
                "skip": True,
                "reason": "intent_service_unavailable",
                "direct_matches": direct_match_ids,
            }
        if not direct_matches:
            return {"skip": False, "reason": None, "direct_matches": direct_match_ids}
        if len(direct_matches) > 2:
            return {
                "skip": False,
                "reason": "too_many_deterministic_matches",
                "direct_matches": direct_match_ids,
            }
        lowered = question.lower()
        if any(marker in lowered for marker in (" and then ", ";", " plus ", " also ")):
            return {
                "skip": False,
                "reason": "compound_question",
                "direct_matches": direct_match_ids,
            }
        if len(direct_matches) == 1:
            inferred_time_scope = self.intent_parser.infer_time_scope(
                question=question,
                today=today or date.today(),
                llm_intent=None,
            )
            return {
                "skip": True,
                "reason": "deterministic_routing_confident",
                "direct_matches": direct_match_ids,
                "inferred_time_scope": inferred_time_scope.get("time_scope") if inferred_time_scope is not None else None,
            }
        if not self._is_trend_only_question(question):
            return {
                "skip": False,
                "reason": "requires_llm_disambiguation",
                "direct_matches": direct_match_ids,
            }
        inferred_time_scope = self.intent_parser.infer_time_scope(
            question=question,
            today=today or date.today(),
            llm_intent=None,
        )
        return {
            "skip": True,
            "reason": "deterministic_routing_confident",
            "direct_matches": direct_match_ids,
            "inferred_time_scope": inferred_time_scope.get("time_scope") if inferred_time_scope is not None else None,
        }

    def _intent_classifier(self) -> Optional[Any]:
        service = self.intent_service
        if service is None and self.llm is not None and hasattr(self.llm, "classify_intent"):
            service = self.llm
        if service is None or not hasattr(service, "classify_intent"):
            return None
        return service

    def _classify_intent(self, question: str) -> Optional[RagIntentResponse]:
        service = self._intent_classifier()
        if service is None:
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

    def _build_answer_context(self, context: dict[str, Any]) -> dict[str, Any]:
        answer_context: dict[str, Any] = {}
        daily_compact_sections: dict[str, Any] = {}
        for key, value in context.items():
            if key in ANSWER_CONTEXT_EXCLUDED_KEYS:
                continue
            if key == "unavailable_tools" and isinstance(value, list):
                answer_context[key] = [
                    {
                        "tool": item.get("tool"),
                        "detail": item.get("detail"),
                        "label": item.get("label"),
                    }
                    for item in value
                    if isinstance(item, dict)
                ]
                continue
            if self._is_daily_context_key(key) and isinstance(value, list):
                daily_compact_sections[key] = self._compact_daily_series(
                    context_key=key,
                    rows=value,
                    label=self._daily_context_label(context, key),
                )
                continue
            answer_context[key] = value
        if daily_compact_sections:
            answer_context["daily_trend_summary"] = self._compact_daily_trend_summary(context, daily_compact_sections)
        return answer_context

    @classmethod
    def _is_daily_context_key(cls, key: str) -> bool:
        return key == DAILY_CONTEXT_KEY_PREFIX or key.startswith(f"{DAILY_CONTEXT_KEY_PREFIX}_")

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        return format(value.quantize(DECIMAL_CENTS), "f")

    @staticmethod
    def _to_decimal(value: Any) -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return Decimal("0")

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _parse_iso_date(value: Any) -> Optional[date]:
        if isinstance(value, date):
            return value
        if not isinstance(value, str) or not value:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_period_label(period: Optional[str]) -> Optional[date]:
        if not isinstance(period, str) or not period.strip():
            return None
        try:
            return datetime.strptime(period.strip(), "%B%Y").date()
        except ValueError:
            return None

    @classmethod
    def _sort_period_labels(cls, periods: list[str]) -> list[str]:
        return sorted(
            periods,
            key=lambda period: cls._parse_period_label(period) or date.max,
        )

    def _daily_context_label(self, context: dict[str, Any], context_key: str) -> str:
        execution_plan = context.get("execution_plan")
        if isinstance(execution_plan, dict):
            steps = execution_plan.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    if step.get("output_key") == context_key and isinstance(step.get("label"), str):
                        return str(step["label"])
        if context_key == DAILY_CONTEXT_KEY_PREFIX:
            return context.get("period") or "selected period"
        suffix = context_key[len(f"{DAILY_CONTEXT_KEY_PREFIX}_") :] if context_key.startswith(f"{DAILY_CONTEXT_KEY_PREFIX}_") else context_key
        if suffix:
            parts = [part.capitalize() for part in suffix.split("_") if part]
            if parts:
                return "".join(parts)
        return context_key

    def _normalize_daily_rows(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_date = self._parse_iso_date(row.get("date"))
            if row_date is None:
                continue
            normalized_row = {
                "date": row_date.isoformat(),
                "date_value": row_date,
                "total_amount": self._to_decimal(row.get("total_amount")),
                "transaction_count": self._to_int(row.get("transaction_count")),
            }
            grouped_rows.setdefault(row_date.isoformat(), []).append(normalized_row)

        normalized_rows: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        for row_date in sorted(grouped_rows.keys()):
            entries = grouped_rows[row_date]
            unique_entries: list[tuple[Decimal, int]] = []
            for entry in entries:
                signature = (entry["total_amount"], entry["transaction_count"])
                if signature not in unique_entries:
                    unique_entries.append(signature)
            chosen_entry = max(entries, key=lambda entry: (entry["total_amount"], entry["transaction_count"]))
            normalized_rows.append(chosen_entry)
            if len(unique_entries) > 1:
                conflicts.append(
                    {
                        "date": row_date,
                        "reported_values": [
                            {
                                "total_amount": self._format_decimal(total_amount),
                                "transaction_count": transaction_count,
                            }
                            for total_amount, transaction_count in unique_entries
                        ],
                        "selected_value": {
                            "total_amount": self._format_decimal(chosen_entry["total_amount"]),
                            "transaction_count": chosen_entry["transaction_count"],
                        },
                    }
                )
        return normalized_rows, conflicts

    def _compact_daily_series(self, *, context_key: str, rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        normalized_rows, conflicts = self._normalize_daily_rows(rows)
        total_spend = sum((row["total_amount"] for row in normalized_rows), start=Decimal("0"))
        total_transactions = sum(row["transaction_count"] for row in normalized_rows)
        peak_day = max(normalized_rows, key=lambda row: row["total_amount"], default=None)
        return {
            "context_key": context_key,
            "label": label,
            "active_days": len(normalized_rows),
            "total_spend": self._format_decimal(total_spend),
            "transaction_count": total_transactions,
            "average_daily_spend": self._format_decimal(total_spend / len(normalized_rows)) if normalized_rows else "0.00",
            "date_range": {
                "start_date": normalized_rows[0]["date"] if normalized_rows else None,
                "end_date": normalized_rows[-1]["date"] if normalized_rows else None,
            },
            "peak_day": (
                {
                    "date": peak_day["date"],
                    "total_amount": self._format_decimal(peak_day["total_amount"]),
                    "transaction_count": peak_day["transaction_count"],
                }
                if peak_day is not None
                else None
            ),
            "conflicts": conflicts,
        }

    def _compact_daily_trend_summary(
        self,
        context: dict[str, Any],
        daily_compact_sections: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        time_scope = self._context_time_scope(context)
        ordered_sections = [
            daily_compact_sections[key]
            for key in sorted(
                daily_compact_sections.keys(),
                key=lambda item: self._parse_period_label(self._daily_context_label(context, item)) or date.max,
            )
        ]
        highest_peak = max(
            (section["peak_day"] for section in ordered_sections if isinstance(section.get("peak_day"), dict)),
            key=lambda peak: self._to_decimal(peak.get("total_amount")),
            default=None,
        )
        return {
            "range_label": self._time_scope_label(time_scope) if time_scope is not None else "selected range",
            "series": ordered_sections,
            "highest_peak_day": highest_peak,
            "conflicts": [
                {"label": section["label"], "items": section["conflicts"]}
                for section in ordered_sections
                if section.get("conflicts")
            ],
        }

    def _deterministic_answer(self, context: dict[str, Any]) -> Optional[str]:
        if self._is_deterministic_daily_range_context(context):
            return self._deterministic_daily_range_answer(context)
        if self._is_deterministic_single_scope_daily_context(context):
            return self._deterministic_single_scope_daily_answer(context)
        return None

    def _daily_sections(self, context: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
        daily_sections: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
        for key, value in context.items():
            if not self._is_daily_context_key(key) or not isinstance(value, list):
                continue
            normalized_rows, conflicts = self._normalize_daily_rows(value)
            daily_sections.append((self._daily_context_label(context, key), normalized_rows, conflicts))
        daily_sections.sort(key=lambda item: self._parse_period_label(item[0]) or date.max)
        return daily_sections

    @staticmethod
    def _plan_skill_ids(context: dict[str, Any]) -> list[str]:
        execution_plan = context.get("execution_plan")
        if not isinstance(execution_plan, dict):
            return []
        steps = execution_plan.get("steps")
        if not isinstance(steps, list):
            return []
        return [str(step.get("skill_id")) for step in steps if isinstance(step, dict) and step.get("skill_id")]

    def _is_deterministic_daily_range_context(self, context: dict[str, Any]) -> bool:
        time_scope = self._context_time_scope(context)
        if time_scope is None:
            return False
        if time_scope.scope_type != "statement_period_range":
            return False
        plan_skill_ids = self._plan_skill_ids(context)
        return bool(plan_skill_ids) and all(skill_id == "daily" for skill_id in plan_skill_ids)

    def _is_deterministic_single_scope_daily_context(self, context: dict[str, Any]) -> bool:
        time_scope = self._context_time_scope(context)
        if time_scope is None or time_scope.scope_type != "date_range":
            return False
        daily_sections = self._daily_sections(context)
        if len(daily_sections) != 1:
            return False
        plan_skill_ids = set(self._plan_skill_ids(context))
        return bool(plan_skill_ids) and plan_skill_ids.issubset({"overview", "daily"})

    @staticmethod
    def _trend_direction(current_value: Decimal, next_value: Decimal) -> str:
        if next_value > current_value:
            return "increased"
        if next_value < current_value:
            return "decreased"
        return "stayed flat"

    def _deterministic_daily_range_answer(self, context: dict[str, Any]) -> str:
        daily_sections = self._daily_sections(context)
        if not daily_sections:
            return self._fallback_answer(context)
        scope = self._context_time_scope(context)
        resolved_label = self._time_scope_label(scope) if scope is not None else "selected range"
        lines = [f"Period resolved: {resolved_label}.", "", "## Daily spending trend"]

        monthly_totals: list[tuple[str, Decimal]] = []
        overall_peak: Optional[tuple[str, dict[str, Any]]] = None
        conflict_lines: list[str] = []

        for label, rows, conflicts in daily_sections:
            total_spend = sum((row["total_amount"] for row in rows), start=Decimal("0"))
            average_daily_spend = total_spend / len(rows) if rows else Decimal("0")
            peak_day = max(rows, key=lambda row: row["total_amount"], default=None)
            monthly_totals.append((label, total_spend))
            if peak_day is not None:
                if overall_peak is None or peak_day["total_amount"] > overall_peak[1]["total_amount"]:
                    overall_peak = (label, peak_day)
            summary_line = (
                f"- {label}: {len(rows)} active days, ${self._format_decimal(total_spend)} total spend, "
                f"${self._format_decimal(average_daily_spend)} average spend on active days"
            )
            if peak_day is not None:
                summary_line += (
                    f", peak day {peak_day['date']} at ${self._format_decimal(peak_day['total_amount'])} "
                    f"across {peak_day['transaction_count']} transactions"
                )
            summary_line += "."
            lines.append(summary_line)
            for conflict in conflicts:
                reported = ", ".join(
                    f"${item['total_amount']} ({item['transaction_count']} txns)"
                    for item in conflict["reported_values"]
                )
                selected = conflict["selected_value"]
                conflict_lines.append(
                    f"- {label} {conflict['date']}: reported values {reported}; used ${selected['total_amount']} ({selected['transaction_count']} txns)."
                )

        trend_lines: list[str] = []
        for index in range(len(monthly_totals) - 1):
            current_label, current_total = monthly_totals[index]
            next_label, next_total = monthly_totals[index + 1]
            trend_lines.append(
                f"- Total spend {self._trend_direction(current_total, next_total)} from {current_label} (${self._format_decimal(current_total)}) to {next_label} (${self._format_decimal(next_total)})."
            )
        if overall_peak is not None:
            peak_label, peak_day = overall_peak
            trend_lines.append(
                f"- Highest daily spend in the range was {peak_day['date']} in {peak_label} at ${self._format_decimal(peak_day['total_amount'])}."
            )
        if trend_lines:
            lines.extend(["", "## Overall observations", *trend_lines])
        if conflict_lines:
            lines.extend(["", "## Data notes", *conflict_lines])

        lines.append("")
        lines.append("## Daily totals")
        for label, rows, _ in daily_sections:
            lines.extend(["", f"### {label}"])
            if not rows:
                lines.append("- No in-scope daily totals were returned for this period.")
                continue
            for row in rows:
                lines.append(
                    f"- {row['date']}: ${self._format_decimal(row['total_amount'])} across {row['transaction_count']} transactions"
                )

        return "\n".join(lines)

    def _deterministic_single_scope_daily_answer(self, context: dict[str, Any]) -> str:
        daily_sections = self._daily_sections(context)
        if len(daily_sections) != 1:
            return self._fallback_answer(context)

        _, rows, conflicts = daily_sections[0]
        if not rows:
            return self._fallback_answer(context)

        time_scope = self._context_time_scope(context)
        resolved_label = self._time_scope_label(time_scope) if time_scope is not None else "the selected range"
        total_spend = sum((row["total_amount"] for row in rows), start=Decimal("0"))
        transaction_count = sum(row["transaction_count"] for row in rows)
        average_daily_spend = total_spend / len(rows)
        peak_day = max(rows, key=lambda row: row["total_amount"], default=None)

        lines = [f"Here's what I found for {resolved_label}:", ""]
        lines.append(f"- Date range: {rows[0]['date']} -> {rows[-1]['date']}")
        lines.append(f"- Total spend: ${self._format_decimal(total_spend)}")
        lines.append(f"- Transaction count: {transaction_count}")
        lines.append(f"- Average daily spend: ${self._format_decimal(average_daily_spend)}")
        if peak_day is not None:
            lines.append(
                f"- Peak day: {peak_day['date']} - ${self._format_decimal(peak_day['total_amount'])} ({peak_day['transaction_count']} transactions)"
            )

        if conflicts:
            lines.extend(["", "## Data notes"])
            for conflict in conflicts:
                reported = ", ".join(
                    f"${item['total_amount']} ({item['transaction_count']} txns)"
                    for item in conflict["reported_values"]
                )
                selected = conflict["selected_value"]
                lines.append(
                    f"- {conflict['date']}: reported values {reported}; used ${selected['total_amount']} ({selected['transaction_count']} txns)."
                )

        lines.extend(["", "## Daily totals"])
        for row in rows:
            lines.append(
                f"- {row['date']}: ${self._format_decimal(row['total_amount'])} across {row['transaction_count']} transactions"
            )

        return "\n".join(lines)

    @staticmethod
    def _build_timing_metadata(
        *,
        tool_traces: list[dict[str, Any]],
        classifier_latency_ms: int,
        plan_execution_latency_ms: int,
        answer_generation_latency_ms: int,
    ) -> dict[str, int]:
        return {
            "classifier_latency_ms": classifier_latency_ms,
            "plan_execution_latency_ms": plan_execution_latency_ms,
            "cache_lookup_latency_ms": sum(
                int(trace.get("cache_lookup_duration_ms", 0) or 0)
                for trace in tool_traces
            ),
            "tool_execution_latency_ms": sum(
                int(trace.get("tool_execution_duration_ms", 0) or 0)
                for trace in tool_traces
            ),
            "answer_generation_latency_ms": answer_generation_latency_ms,
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
            return RagTimeScope.from_period(period.strip())
        return None

    @staticmethod
    def _derive_period_from_time_scope(time_scope: Optional[RagTimeScope]) -> Optional[str]:
        return time_scope.derived_period if time_scope is not None else None

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
        derived_period = time_scope.derived_period
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
        return time_scope.label

    @staticmethod
    def _context_time_scope(context: dict[str, Any]) -> Optional[RagTimeScope]:
        time_scope_payload = context.get("time_scope")
        if not isinstance(time_scope_payload, dict):
            return None
        try:
            return RagTimeScope.model_validate(time_scope_payload)
        except Exception:
            return None

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
            return RagTimeScope.from_period(message.period.strip())
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
            cacheable_arguments = self._cacheable_arguments(arguments)
            cache_key = self._tool_cache_key(tool_name, arguments)
            response_payload = {
                "context_key": trace.get("cache_context_key", trace.get("context_key")),
                "payload": trace.get("result"),
                "metadata": {
                    "citation": trace.get("citation"),
                    "description": trace.get("description"),
                },
            }
            self.conversation_history.upsert_tool_cache(
                conversation_id,
                tool_name=tool_name,
                cache_key=cache_key,
                period=period,
                params_json=cacheable_arguments,
                response_json=response_payload,
                source_message_db_id=assistant_message.db_id,
                ttl_seconds=TOOL_CACHE_TTL_SECONDS,
            )
            self.conversation_history.upsert_shared_tool_cache(
                tool_name=tool_name,
                cache_key=cache_key,
                params_json=cacheable_arguments,
                response_json=response_payload,
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
                cached_entry = self.conversation_history.get_shared_tool_cache(
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

    def _is_cache_allowed_for_time_scope(
        self,
        time_scope: Optional[RagTimeScope],
        *,
        today: Optional[date] = None,
    ) -> bool:
        if time_scope is None:
            return False

        reference_date = today or date.today()
        current_period = self.intent_parser.format_statement_period(reference_date)
        current_month_start = reference_date.replace(day=1)

        if time_scope.scope_type == "statement_period":
            return bool(time_scope.derived_period and time_scope.derived_period != current_period)

        if time_scope.scope_type == "statement_period_range":
            assert time_scope.start_period is not None
            assert time_scope.end_period is not None
            start_period_date = RagTimeScope.parse_statement_period(time_scope.start_period)
            end_period_date = RagTimeScope.parse_statement_period(time_scope.end_period)
            current_period_date = current_month_start
            return not (start_period_date <= current_period_date <= end_period_date)

        if time_scope.scope_type == "date_range":
            assert time_scope.start_date is not None
            assert time_scope.end_date is not None
            return time_scope.end_date < current_month_start

        return False

    def _cache_policy(
        self,
        time_scope: Optional[RagTimeScope],
        *,
        enabled: bool,
        today: Optional[date] = None,
    ) -> dict[str, Any]:
        if not enabled:
            return {
                "enabled": False,
                "eligible": False,
                "reason": "cache_unavailable",
            }

        if time_scope is None:
            return {
                "enabled": True,
                "eligible": False,
                "reason": "missing_time_scope",
            }

        reference_date = today or date.today()
        current_period = self.intent_parser.format_statement_period(reference_date)
        current_month_start = reference_date.replace(day=1)

        if time_scope.scope_type == "statement_period":
            if time_scope.derived_period == current_period:
                return {"enabled": True, "eligible": False, "reason": "current_month_not_cacheable"}
            return {"enabled": True, "eligible": True, "reason": None}

        if time_scope.scope_type == "statement_period_range":
            assert time_scope.start_period is not None
            assert time_scope.end_period is not None
            start_period_date = RagTimeScope.parse_statement_period(time_scope.start_period)
            end_period_date = RagTimeScope.parse_statement_period(time_scope.end_period)
            if start_period_date <= current_month_start <= end_period_date:
                return {"enabled": True, "eligible": False, "reason": "includes_current_month_not_cacheable"}
            return {"enabled": True, "eligible": True, "reason": None}

        if time_scope.scope_type == "date_range":
            assert time_scope.start_date is not None
            assert time_scope.end_date is not None
            if time_scope.end_date >= current_month_start:
                return {"enabled": True, "eligible": False, "reason": "includes_current_month_not_cacheable"}
            return {"enabled": True, "eligible": True, "reason": None}

        return {"enabled": True, "eligible": False, "reason": f"unsupported_time_scope:{time_scope.scope_type}"}

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

    def _cache_metadata(self, tool_traces: list[dict[str, Any]], *, cache_policy: dict[str, Any]) -> dict[str, Any]:
        return {
            "enabled": bool(cache_policy.get("enabled", False)),
            "eligible": bool(cache_policy.get("eligible", False)),
            "reason": cache_policy.get("reason"),
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
        selected_scope = self._context_time_scope(context)
        selected_scope_label = context.get("period") or (
            self._time_scope_label(selected_scope) if selected_scope is not None else "the selected scope"
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

        available_periods = context.get("available_periods")
        if isinstance(available_periods, dict):
            periods = available_periods.get("periods")
            count = available_periods.get("count")
            if isinstance(periods, list) and periods:
                preview = ", ".join(str(period) for period in periods[:5])
                if count is not None:
                    fragments.append(f"There are {count} available statement periods. Examples: {preview}.")
                else:
                    fragments.append(f"Available statement periods include: {preview}.")

        statement_period_summary = context.get("statement_period_summary")
        if isinstance(statement_period_summary, dict):
            summary_period = statement_period_summary.get("statement_period")
            total_amount = statement_period_summary.get("total_amount")
            transaction_count = statement_period_summary.get("transaction_count")
            if summary_period and total_amount is not None:
                fragments.append(f"Statement period summary for {summary_period} shows total spend of {total_amount}.")
            if transaction_count is not None:
                fragments.append(f"That summary includes {transaction_count} transactions.")

        statement_period_summary_range = context.get("statement_period_summary_range")
        if isinstance(statement_period_summary_range, list):
            fragments.append(
                f"Statement period summaries were included for {len(statement_period_summary_range)} periods in the selected range."
            )

        execution_plan = context.get("execution_plan")
        if isinstance(execution_plan, dict):
            steps = execution_plan.get("steps")
            if execution_plan.get("strategy") == "multi_scope" and isinstance(steps, list) and len(steps) > 1:
                fragments.append(f"Executed {len(steps)} planned analytics steps for this question.")

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
