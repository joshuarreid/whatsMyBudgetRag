from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Optional

from app.models.schemas import RagTimeScope, RagToolTraceResponse
from app.repositories import ConversationMessageRecord
from app.services.rag.constants import MAX_RESPONSE_CITATIONS, TOOL_CACHE_TTL_SECONDS
from app.skills.base import Skill


class RAGCacheMixin:
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

