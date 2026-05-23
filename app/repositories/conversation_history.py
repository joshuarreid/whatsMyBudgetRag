from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol
from uuid import uuid4

from app.core.config import Settings

try:
    import mysql.connector
except ImportError:  # pragma: no cover - exercised only when dependency is missing locally.
    mysql = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ConversationHistoryDisabledError(RuntimeError):
    """Raised when conversation history is requested but MySQL persistence is not configured."""


class ConversationNotFoundError(LookupError):
    """Raised when a requested conversation does not exist."""


@dataclass(frozen=True)
class ConversationRecord:
    conversation_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime]


@dataclass(frozen=True)
class ConversationMessageRecord:
    message_id: str
    role: str
    content: str
    period: Optional[str]
    period_source: Optional[str]
    created_at: datetime
    db_id: Optional[int] = None


class ConversationHistoryRepository(Protocol):
    def is_enabled(self) -> bool: ...

    def create_conversation(
        self,
        *,
        title: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationRecord: ...

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]: ...

    def list_messages(self, conversation_id: str, *, limit: int = 50) -> list[ConversationMessageRecord]: ...

    def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        period: Optional[str] = None,
        period_source: Optional[str] = None,
        transaction_id: Optional[str] = None,
        request_id: Optional[str] = None,
        model_name: Optional[str] = None,
        tool_plan: Optional[dict[str, Any]] = None,
        context_json: Optional[dict[str, Any]] = None,
        answer_json: Optional[dict[str, Any]] = None,
    ) -> ConversationMessageRecord: ...

    def append_message_tool_calls(
        self,
        message_id: str,
        *,
        tool_calls: list[dict[str, Any]],
    ) -> None: ...

    def append_message_citations(
        self,
        message_id: str,
        *,
        citations: list[dict[str, Any]],
    ) -> None: ...

    def get_tool_cache(
        self,
        conversation_id: str,
        *,
        tool_name: str,
        cache_key: str,
    ) -> Optional[dict[str, Any]]: ...

    def upsert_tool_cache(
        self,
        conversation_id: str,
        *,
        tool_name: str,
        cache_key: str,
        period: Optional[str],
        params_json: dict[str, Any],
        response_json: dict[str, Any],
        source_message_db_id: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None: ...


class NullConversationHistoryRepository:
    def is_enabled(self) -> bool:
        return False

    def create_conversation(
        self,
        *,
        title: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationRecord:
        raise ConversationHistoryDisabledError("Conversation history persistence is not configured")

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        raise ConversationHistoryDisabledError("Conversation history persistence is not configured")

    def list_messages(self, conversation_id: str, *, limit: int = 50) -> list[ConversationMessageRecord]:
        raise ConversationHistoryDisabledError("Conversation history persistence is not configured")

    def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        period: Optional[str] = None,
        period_source: Optional[str] = None,
        transaction_id: Optional[str] = None,
        request_id: Optional[str] = None,
        model_name: Optional[str] = None,
        tool_plan: Optional[dict[str, Any]] = None,
        context_json: Optional[dict[str, Any]] = None,
        answer_json: Optional[dict[str, Any]] = None,
    ) -> ConversationMessageRecord:
        raise ConversationHistoryDisabledError("Conversation history persistence is not configured")

    def append_message_tool_calls(
        self,
        message_id: str,
        *,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        raise ConversationHistoryDisabledError("Conversation history persistence is not configured")

    def append_message_citations(
        self,
        message_id: str,
        *,
        citations: list[dict[str, Any]],
    ) -> None:
        raise ConversationHistoryDisabledError("Conversation history persistence is not configured")

    def get_tool_cache(
        self,
        conversation_id: str,
        *,
        tool_name: str,
        cache_key: str,
    ) -> Optional[dict[str, Any]]:
        raise ConversationHistoryDisabledError("Conversation history persistence is not configured")

    def upsert_tool_cache(
        self,
        conversation_id: str,
        *,
        tool_name: str,
        cache_key: str,
        period: Optional[str],
        params_json: dict[str, Any],
        response_json: dict[str, Any],
        source_message_db_id: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        raise ConversationHistoryDisabledError("Conversation history persistence is not configured")


class MySQLConversationHistoryRepository:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._enabled = all(
            [
                settings.mysql_host,
                settings.mysql_database,
                settings.mysql_user,
                settings.mysql_password,
            ]
        )
        self._default_user = settings.conversation_default_user
        logger.info(
            "Initialized MySQLConversationHistoryRepository enabled=%s host=%s database=%s default_user=%s",
            self._enabled,
            settings.mysql_host or "-",
            settings.mysql_database or "-",
            self._default_user,
        )

    def is_enabled(self) -> bool:
        return self._enabled

    def create_conversation(
        self,
        *,
        title: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationRecord:
        self._ensure_enabled()
        conversation_id = str(uuid4())
        created_at = self._utcnow()

        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(
                    """
                    INSERT INTO conversations (conversation_uuid, user_id, title, metadata)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        conversation_id,
                        self._default_user,
                        title,
                        self._json_or_none(metadata),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                logger.exception("Failed to create conversation conversation_id=%s", conversation_id)
                raise
            finally:
                cursor.close()
        finally:
            connection.close()

        return ConversationRecord(
            conversation_id=conversation_id,
            title=title,
            created_at=created_at,
            updated_at=created_at,
            last_message_at=None,
        )

    def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        self._ensure_enabled()

        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(
                    """
                    SELECT conversation_uuid, title, created_at, updated_at, last_message_at
                    FROM conversations
                    WHERE conversation_uuid = %s AND status <> 'deleted'
                    """,
                    (conversation_id,),
                )
                row = cursor.fetchone()
            finally:
                cursor.close()
        finally:
            connection.close()

        # Ensure row is a dict (should be with dictionary=True, but be defensive)
        if row is not None and not isinstance(row, dict):
            # Convert tuple to dict using cursor description if needed
            raise TypeError("Row returned from fetchone is not a dict. Check cursor configuration.")
        return self._conversation_from_row(row) if row is not None else None

    def list_messages(self, conversation_id: str, *, limit: int = 50) -> list[ConversationMessageRecord]:
        self._ensure_enabled()
        bounded_limit = max(1, limit)

        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(
                    """
                    SELECT m.message_uuid, m.role, m.content, m.period, m.period_source, m.created_at
                    FROM messages m
                    INNER JOIN conversations c ON c.id = m.conversation_id
                    WHERE c.conversation_uuid = %s AND c.status <> 'deleted' AND m.is_deleted = 0
                    ORDER BY m.sequence_no DESC
                    LIMIT %s
                    """,
                    (conversation_id, bounded_limit),
                )
                rows = cursor.fetchall()
            finally:
                cursor.close()
        finally:
            connection.close()

        # Ensure each row is a dict (should be with dictionary=True, but be defensive)
        result: list[ConversationMessageRecord] = []
        for row in reversed(rows):
            if not isinstance(row, dict):
                raise TypeError("Row returned from fetchall is not a dict. Check cursor configuration.")
            result.append(self._message_from_row(row))
        return result

    def append_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        period: Optional[str] = None,
        period_source: Optional[str] = None,
        transaction_id: Optional[str] = None,
        request_id: Optional[str] = None,
        model_name: Optional[str] = None,
        tool_plan: Optional[dict[str, Any]] = None,
        context_json: Optional[dict[str, Any]] = None,
        answer_json: Optional[dict[str, Any]] = None,
    ) -> ConversationMessageRecord:
        self._ensure_enabled()
        message_id = str(uuid4())
        created_at = self._utcnow()

        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(
                    """
                    SELECT id
                    FROM conversations
                    WHERE conversation_uuid = %s AND status <> 'deleted'
                    FOR UPDATE
                    """,
                    (conversation_id,),
                )
                conversation_row = cursor.fetchone()
                if conversation_row is None:
                    raise ConversationNotFoundError(f"Conversation {conversation_id} was not found")

                # Defensive: handle Decimal, float, int, str, etc.
                raw_id = conversation_row["id"]
                internal_conversation_id = self._coerce_int(raw_id, field_name="conversation id")
                cursor.execute(
                    """
                    SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_sequence_no
                    FROM messages
                    WHERE conversation_id = %s
                    FOR UPDATE
                    """,
                    (internal_conversation_id,),
                )
                raw_seq = cursor.fetchone()["next_sequence_no"]
                next_sequence_no = self._coerce_int(raw_seq, field_name="next_sequence_no")

                cursor.execute(
                    """
                    INSERT INTO messages (
                        message_uuid,
                        conversation_id,
                        sequence_no,
                        role,
                        content,
                        transaction_id,
                        request_id,
                        model_name,
                        period,
                        period_source,
                        tool_plan,
                        context_json,
                        answer_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(message_id),
                        int(internal_conversation_id),
                        int(next_sequence_no),
                        str(role),
                        str(content),
                        transaction_id if transaction_id is None or isinstance(transaction_id, str) else str(transaction_id),
                        request_id if request_id is None or isinstance(request_id, str) else str(request_id),
                        model_name if model_name is None or isinstance(model_name, str) else str(model_name),
                        period if period is None or isinstance(period, str) else str(period),
                        period_source if period_source is None or isinstance(period_source, str) else str(period_source),
                        self._json_or_none(tool_plan),
                        self._json_or_none(context_json),
                        self._json_or_none(answer_json),
                    ),
                )
                inserted_message_row_id = cursor.lastrowid
                cursor.execute(
                    """
                    UPDATE conversations
                    SET last_message_at = CURRENT_TIMESTAMP(6)
                    WHERE id = %s
                    """,
                    (int(internal_conversation_id),),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                logger.exception(
                    "Failed to append conversation message conversation_id=%s role=%s",
                    conversation_id,
                    role,
                )
                raise
            finally:
                cursor.close()
        finally:
            connection.close()

        return ConversationMessageRecord(
            db_id=self._coerce_int(inserted_message_row_id, field_name="inserted message id"),
            message_id=message_id,
            role=role,
            content=content,
            period=period,
            period_source=period_source,
            created_at=created_at,
        )

    def append_message_tool_calls(
        self,
        message_id: str,
        *,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        self._ensure_enabled()
        if not tool_calls:
            return

        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                message_row_id = self._get_message_row_id(cursor, message_id)
                for index, tool_call in enumerate(tool_calls, start=1):
                    cursor.execute(
                        """
                        INSERT INTO message_tool_calls (
                            message_id,
                            tool_name,
                            call_order,
                            arguments_json,
                            result_json,
                            status,
                            error_text,
                            duration_ms
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            message_row_id,
                            tool_call.get("tool_name", "unknown"),
                            index,
                            self._json_or_none(tool_call.get("arguments")),
                            self._json_or_none(tool_call.get("result")),
                            tool_call.get("status", "ok"),
                            tool_call.get("error_text"),
                            tool_call.get("duration_ms"),
                        ),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                logger.exception("Failed to append message tool calls message_id=%s", message_id)
                raise
            finally:
                cursor.close()
        finally:
            connection.close()

    def append_message_citations(
        self,
        message_id: str,
        *,
        citations: list[dict[str, Any]],
    ) -> None:
        self._ensure_enabled()
        if not citations:
            return

        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                message_row_id = self._get_message_row_id(cursor, message_id)
                for index, citation in enumerate(citations, start=1):
                    cursor.execute(
                        """
                        INSERT INTO message_citations (
                            message_id,
                            citation_order,
                            source_type,
                            source_ref,
                            source_title,
                            snippet,
                            score
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            message_row_id,
                            index,
                            citation.get("source_type", "api"),
                            citation.get("source_ref", "unknown"),
                            citation.get("source_title"),
                            citation.get("snippet"),
                            citation.get("score"),
                        ),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                logger.exception("Failed to append message citations message_id=%s", message_id)
                raise
            finally:
                cursor.close()
        finally:
            connection.close()

    def get_tool_cache(
        self,
        conversation_id: str,
        *,
        tool_name: str,
        cache_key: str,
    ) -> Optional[dict[str, Any]]:
        self._ensure_enabled()

        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(
                    """
                    SELECT ctc.id, ctc.response_json, ctc.created_at, ctc.expires_at
                    FROM conversation_tool_cache ctc
                    INNER JOIN conversations c ON c.id = ctc.conversation_id
                    WHERE c.conversation_uuid = %s
                      AND c.status <> 'deleted'
                      AND ctc.tool_name = %s
                      AND ctc.cache_key = %s
                      AND ctc.invalidated_at IS NULL
                      AND (ctc.expires_at IS NULL OR ctc.expires_at >= UTC_TIMESTAMP(6))
                    LIMIT 1
                    """,
                    (conversation_id, tool_name, cache_key),
                )
                row = cursor.fetchone()
                if row is None:
                    return None

                cursor.execute(
                    """
                    UPDATE conversation_tool_cache
                    SET hit_count = hit_count + 1,
                        last_hit_at = CURRENT_TIMESTAMP(6)
                    WHERE id = %s
                    """,
                    (self._coerce_int(row["id"], field_name="cache id"),),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                logger.exception(
                    "Failed to read conversation tool cache conversation_id=%s tool_name=%s",
                    conversation_id,
                    tool_name,
                )
                raise
            finally:
                cursor.close()
        finally:
            connection.close()

        response_json = self._json_from_db_value(row.get("response_json"))
        if not isinstance(response_json, dict):
            return None
        return {
            "response_json": response_json,
            "created_at": self._datetime_to_isoformat(row.get("created_at")),
            "expires_at": self._datetime_to_isoformat(row.get("expires_at")),
        }

    def upsert_tool_cache(
        self,
        conversation_id: str,
        *,
        tool_name: str,
        cache_key: str,
        period: Optional[str],
        params_json: dict[str, Any],
        response_json: dict[str, Any],
        source_message_db_id: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self._ensure_enabled()
        expires_at = self._utcnow() + timedelta(seconds=ttl_seconds) if ttl_seconds else None

        connection = self._connect()
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                conversation_row_id = self._get_conversation_row_id(cursor, conversation_id)
                cursor.execute(
                    """
                    INSERT INTO conversation_tool_cache (
                        conversation_id,
                        tool_name,
                        cache_key,
                        period,
                        params_json,
                        response_json,
                        source_message_id,
                        expires_at,
                        invalidated_at,
                        hit_count,
                        last_hit_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, 0, NULL)
                    ON DUPLICATE KEY UPDATE
                        period = VALUES(period),
                        params_json = VALUES(params_json),
                        response_json = VALUES(response_json),
                        source_message_id = VALUES(source_message_id),
                        expires_at = VALUES(expires_at),
                        invalidated_at = NULL
                    """,
                    (
                        conversation_row_id,
                        tool_name,
                        cache_key,
                        period,
                        self._json_or_none(params_json),
                        self._json_or_none(response_json),
                        source_message_db_id,
                        expires_at,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                logger.exception(
                    "Failed to upsert conversation tool cache conversation_id=%s tool_name=%s",
                    conversation_id,
                    tool_name,
                )
                raise
            finally:
                cursor.close()
        finally:
            connection.close()

    def _ensure_enabled(self) -> None:
        if not self._enabled:
            raise ConversationHistoryDisabledError("Conversation history persistence is not configured")
        if mysql is None:
            raise ConversationHistoryDisabledError(
                "mysql-connector-python is not installed; conversation history persistence is unavailable"
            )

    def _connect(self):
        connect_kwargs: dict[str, Any] = {
            "host": self._settings.mysql_host,
            "port": self._settings.mysql_port,
            "database": self._settings.mysql_database,
            "user": self._settings.mysql_user,
            "password": self._settings.mysql_password,
            "autocommit": False,
            "connection_timeout": int(self._settings.mysql_connect_timeout_seconds),
        }
        if self._settings.mysql_ssl_disabled:
            connect_kwargs["ssl_disabled"] = True
        else:
            connect_kwargs["ssl_disabled"] = False
            if self._settings.mysql_ssl_ca:
                connect_kwargs["ssl_ca"] = self._settings.mysql_ssl_ca

        return mysql.connector.connect(**connect_kwargs)

    @staticmethod
    def _json_or_none(value: Any) -> Optional[str]:
        return json.dumps(value, default=str) if value is not None else None

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _conversation_from_row(row: dict[str, Any]) -> ConversationRecord:
        return ConversationRecord(
            conversation_id=row["conversation_uuid"],
            title=row.get("title"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_message_at=row.get("last_message_at"),
        )

    @staticmethod
    def _message_from_row(row: dict[str, Any]) -> ConversationMessageRecord:
        return ConversationMessageRecord(
            db_id=row.get("id"),
            message_id=row["message_uuid"],
            role=row["role"],
            content=row["content"],
            period=row.get("period"),
            period_source=row.get("period_source"),
            created_at=row["created_at"],
        )

    @staticmethod
    def _json_from_db_value(value: Any) -> Any:
        if value is None or isinstance(value, (dict, list)):
            return value
        if isinstance(value, (bytes, bytearray)):
            value = value.decode()
        if isinstance(value, str):
            return json.loads(value)
        return value

    @staticmethod
    def _datetime_to_isoformat(value: Any) -> Optional[str]:
        if isinstance(value, datetime):
            return value.isoformat()
        return None

    @staticmethod
    def _get_message_row_id(cursor: Any, message_id: str) -> int:
        cursor.execute(
            """
            SELECT id
            FROM messages
            WHERE message_uuid = %s
            LIMIT 1
            """,
            (message_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ConversationNotFoundError(f"Message {message_id} was not found")
        return MySQLConversationHistoryRepository._coerce_int(row["id"], field_name="message id")

    @staticmethod
    def _get_conversation_row_id(cursor: Any, conversation_id: str) -> int:
        cursor.execute(
            """
            SELECT id
            FROM conversations
            WHERE conversation_uuid = %s AND status <> 'deleted'
            LIMIT 1
            """,
            (conversation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ConversationNotFoundError(f"Conversation {conversation_id} was not found")
        return MySQLConversationHistoryRepository._coerce_int(row["id"], field_name="conversation id")

    @staticmethod
    def _coerce_int(value: Any, *, field_name: str) -> int:
        if value is None:
            raise ValueError(f"{field_name} is None")
        if isinstance(value, bytes):
            return int(value.decode())
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cannot convert {field_name} to int: {type(value)}") from exc



