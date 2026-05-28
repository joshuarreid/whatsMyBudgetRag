from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.logging import get_request_id
from app.models.schemas import RagConversationMessageResponse, RagConversationResponse, RagTimeScope
from app.repositories import ConversationHistoryDisabledError, ConversationMessageRecord, ConversationNotFoundError

logger = logging.getLogger(__name__)


class RAGHistoryMixin:
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

