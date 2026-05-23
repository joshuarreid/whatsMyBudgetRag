from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
        result = []
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
                if raw_id is None:
                    raise ValueError("Conversation id is None")
                if isinstance(raw_id, int):
                    internal_conversation_id = raw_id
                elif isinstance(raw_id, float):
                    internal_conversation_id = int(raw_id)
                elif hasattr(raw_id, "__int__"):
                    internal_conversation_id = int(raw_id)
                elif isinstance(raw_id, str):
                    internal_conversation_id = int(raw_id)
                elif isinstance(raw_id, bytes):
                    internal_conversation_id = int(raw_id.decode())
                else:
                    try:
                        from decimal import Decimal
                        if isinstance(raw_id, Decimal):
                            internal_conversation_id = int(raw_id)
                        else:
                            internal_conversation_id = int(str(raw_id))
                    except Exception:
                        raise TypeError(f"Cannot convert id to int: {type(raw_id)}")
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
                if raw_seq is None:
                    raise ValueError("next_sequence_no is None")
                if isinstance(raw_seq, int):
                    next_sequence_no = raw_seq
                elif isinstance(raw_seq, float):
                    next_sequence_no = int(raw_seq)
                elif hasattr(raw_seq, "__int__"):
                    next_sequence_no = int(raw_seq)
                elif isinstance(raw_seq, str):
                    next_sequence_no = int(raw_seq)
                elif isinstance(raw_seq, bytes):
                    next_sequence_no = int(raw_seq.decode())
                else:
                    try:
                        from decimal import Decimal
                        if isinstance(raw_seq, Decimal):
                            next_sequence_no = int(raw_seq)
                        else:
                            next_sequence_no = int(str(raw_seq))
                    except Exception:
                        raise TypeError(f"Cannot convert next_sequence_no to int: {type(raw_seq)}")

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
            message_id=message_id,
            role=role,
            content=content,
            period=period,
            period_source=period_source,
            created_at=created_at,
        )

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
    def _json_or_none(value: Optional[dict[str, Any]]) -> Optional[str]:
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
            message_id=row["message_uuid"],
            role=row["role"],
            content=row["content"],
            period=row.get("period"),
            period_source=row.get("period_source"),
            created_at=row["created_at"],
        )



