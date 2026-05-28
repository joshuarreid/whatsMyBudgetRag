from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any, Optional

from app.models.schemas import RagExecutionPlan, RagPlanStep, RagTimeScope
from app.repositories import ConversationMessageRecord
from app.skills.base import Skill, SkillRequest


class RAGExecutionMixin:
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


