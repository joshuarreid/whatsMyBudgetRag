from __future__ import annotations

import logging
from datetime import date
from time import perf_counter
from typing import Any, Optional

from app.clients.spring_boot_client import SpringBootClient
from app.core.config import get_settings
from app.models.schemas import (
    RagAnswerResponse,
    RagCacheMetadataResponse,
    RagCitationResponse,
    RagTimingMetadataResponse,
    RagTimeScope,
)
from app.repositories import ConversationHistoryRepository, NullConversationHistoryRepository
from app.services.intent_service import IntentService
from app.services.langgraph_reasoning_service import LangGraphReasoningService
from app.services.planner_service import PlannerService
from app.services.rag import (
    RAGAnswerMixin,
    RAGCacheMixin,
    RAGExecutionMixin,
    RAGHistoryMixin,
    RAGRoutingMixin,
)
from app.skills.base import SkillRequest
from app.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class RAGService(
    RAGRoutingMixin,
    RAGExecutionMixin,
    RAGCacheMixin,
    RAGHistoryMixin,
    RAGAnswerMixin,
):
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

