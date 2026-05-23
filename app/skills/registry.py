from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Callable, Optional

import requests

from app.skills.base import Skill, SkillRequest

logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self, skills: list[Skill]) -> None:
        self.skills = skills
        self._skills_by_id = {skill.skill_id: skill for skill in skills}

    def available_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "skill_id": skill.skill_id,
                "category": skill.category,
                "context_key": skill.context_key,
                "description": skill.definition.description,
                "keywords": list(skill.definition.keywords),
                "required": skill.required,
            }
            for skill in self.skills
        ]

    def resolve(self, skill_ids: list[str]) -> list[Skill]:
        resolved: list[Skill] = []
        seen_skill_ids: set[str] = set()
        for skill_id in skill_ids:
            if skill_id in seen_skill_ids:
                continue
            skill = self._skills_by_id.get(skill_id)
            if skill is None:
                continue
            resolved.append(skill)
            seen_skill_ids.add(skill_id)
        return resolved

    def select(self, question: str) -> list[Skill]:
        selected_skills = [skill for skill in self.skills if skill.matches(question)]
        if selected_skills:
            return selected_skills
        default_skill = self._skills_by_id.get("overview")
        return [default_skill] if default_skill is not None else []

    def execute(
        self,
        skills: list[Skill],
        request: SkillRequest,
        *,
        cache_lookup: Optional[Callable[[Skill, dict[str, Any]], Optional[dict[str, Any]]]] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        context_sections: dict[str, Any] = {}
        unavailable_skills: list[dict[str, Any]] = []
        tool_traces: list[dict[str, Any]] = []

        for skill in skills:
            arguments = self._build_tool_arguments(request)
            started_at = perf_counter()
            try:
                cached_result = cache_lookup(skill, arguments) if cache_lookup is not None and skill.definition.cacheable else None
                cache_hit = cached_result is not None

                if cached_result is not None:
                    payload = cached_result.get("payload")
                    metadata = self._normalize_result_metadata(
                        skill,
                        arguments,
                        payload,
                        metadata=cached_result.get("metadata"),
                    )
                    context_key = str(cached_result.get("context_key") or skill.context_key)
                else:
                    result = skill.execute(request)
                    payload = result.payload
                    metadata = self._normalize_result_metadata(
                        skill,
                        arguments,
                        payload,
                        metadata=result.metadata,
                    )
                    context_key = result.context_key

                duration_ms = int((perf_counter() - started_at) * 1000)
                trace = {
                    "tool_name": skill.skill_id,
                    "context_key": context_key,
                    "category": skill.category,
                    "status": "ok",
                    "duration_ms": duration_ms,
                    "cache_hit": cache_hit,
                    "cacheable": skill.definition.cacheable,
                    "arguments": arguments,
                    "result": payload,
                    "result_summary": self._summarize_payload(payload),
                    "citation": metadata.get("citation"),
                    "description": skill.definition.description,
                }
                if cached_result is not None:
                    trace["cache_entry"] = {
                        "created_at": cached_result.get("created_at"),
                        "expires_at": cached_result.get("expires_at"),
                    }

                context_sections[context_key] = payload
                tool_traces.append(trace)
            except requests.RequestException as exc:
                if skill.required:
                    raise
                duration_ms = int((perf_counter() - started_at) * 1000)
                unavailable_skills.append(self._serialize_request_error(skill.skill_id, exc))
                tool_traces.append(
                    {
                        "tool_name": skill.skill_id,
                        "context_key": skill.context_key,
                        "category": skill.category,
                        "status": "error",
                        "duration_ms": duration_ms,
                        "cache_hit": False,
                        "cacheable": skill.definition.cacheable,
                        "arguments": arguments,
                        "result": None,
                        "result_summary": {},
                        "citation": None,
                        "error_text": str(exc),
                    }
                )
                logger.warning(
                    "Skipping unavailable RAG skill skill=%s context_key=%s error=%s",
                    skill.skill_id,
                    skill.context_key,
                    unavailable_skills[-1],
                )
            except ValueError as exc:
                if skill.required:
                    raise
                duration_ms = int((perf_counter() - started_at) * 1000)
                unavailable_skills.append(
                    {
                        "tool": skill.skill_id,
                        "error_type": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
                tool_traces.append(
                    {
                        "tool_name": skill.skill_id,
                        "context_key": skill.context_key,
                        "category": skill.category,
                        "status": "error",
                        "duration_ms": duration_ms,
                        "cache_hit": False,
                        "cacheable": skill.definition.cacheable,
                        "arguments": arguments,
                        "result": None,
                        "result_summary": {},
                        "citation": None,
                        "error_text": str(exc),
                    }
                )
                logger.warning(
                    "Skipping invalid RAG skill payload skill=%s context_key=%s detail=%s",
                    skill.skill_id,
                    skill.context_key,
                    exc,
                )

        return context_sections, unavailable_skills, tool_traces

    @staticmethod
    def _build_tool_arguments(request: SkillRequest) -> dict[str, Any]:
        return {
            "period": request.period,
            "payment_method": request.payment_method,
            "account": request.account,
            "transaction_id": request.transaction_id,
        }

    def _normalize_result_metadata(
        self,
        skill: Skill,
        arguments: dict[str, Any],
        payload: Any,
        *,
        metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized_metadata = dict(metadata or {})
        normalized_metadata.setdefault("tool_name", skill.skill_id)
        normalized_metadata.setdefault("category", skill.category)
        normalized_metadata.setdefault("context_key", skill.context_key)
        normalized_metadata.setdefault("description", skill.definition.description)
        normalized_metadata.setdefault("cacheable", skill.definition.cacheable)
        normalized_metadata.setdefault("arguments", arguments)
        normalized_metadata.setdefault("citation", self._build_citation(skill, arguments, payload))
        return normalized_metadata

    def _build_citation(self, skill: Skill, arguments: dict[str, Any], payload: Any) -> dict[str, Any]:
        period = arguments.get("period")
        payment_method = arguments.get("payment_method")
        account = arguments.get("account")
        source_ref = f"api://{skill.skill_id}?period={period or '-'}"
        if payment_method:
            source_ref += f"&payment_method={payment_method}"
        if account:
            source_ref += f"&account={account}"
        return {
            "source_type": "api",
            "source_ref": source_ref,
            "source_title": skill.definition.description,
            "snippet": self._build_snippet(skill.skill_id, payload),
            "score": 1.0,
        }

    def _build_snippet(self, skill_id: str, payload: Any) -> str:
        summary = self._summarize_payload(payload)
        if not summary:
            return f"{skill_id} returned no summarized fields."
        summary_parts = ", ".join(f"{key}={value}" for key, value in summary.items())
        return f"{skill_id} -> {summary_parts}"

    @staticmethod
    def _summarize_payload(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            summary: dict[str, Any] = {}
            for key in ("total_amount", "transaction_count", "average_transaction_amount", "period", "previous_period"):
                if key in payload:
                    summary[key] = payload[key]
            if not summary:
                summary["keys"] = list(payload.keys())[:5]
            return summary
        if isinstance(payload, list):
            summary: dict[str, Any] = {"item_count": len(payload)}
            if payload and isinstance(payload[0], dict):
                summary["sample_keys"] = list(payload[0].keys())[:5]
            return summary
        if payload is None:
            return {}
        return {"value": payload}

    @staticmethod
    def _serialize_request_error(skill_id: str, exc: requests.RequestException) -> dict[str, Any]:
        error: dict[str, Any] = {
            "tool": skill_id,
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

