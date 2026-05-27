from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api.routes.rag import get_rag_service
from app.main import app
from app.models.schemas import RagAnswerResponse, RagConversationMessageResponse, RagConversationResponse, RagTimeScope
from app.repositories import ConversationHistoryDisabledError, ConversationNotFoundError


class RagRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.service = Mock()
        app.dependency_overrides[get_rag_service] = lambda: self.service

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self.client.close()

    def test_ask_returns_conversation_id(self) -> None:
        self.service.answer.return_value = RagAnswerResponse(
            question="What was my spend?",
            conversation_id="conv-123",
            time_scope=RagTimeScope(scope_type="statement_period", statement_period="May2026"),
            period="May2026",
            plan=["overview"],
            context={},
            answer="You spent 42.",
        )

        response = self.client.post(
            "/rag/ask",
            json={"question": "What was my spend?", "conversation_id": "conv-123"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["conversation_id"], "conv-123")
        self.service.answer.assert_called_once()
        self.assertEqual(self.service.answer.call_args.kwargs["conversation_id"], "conv-123")

    def test_ask_logs_request_and_response_payloads(self) -> None:
        self.service.answer.return_value = RagAnswerResponse(
            question="What was my spend?",
            conversation_id="conv-123",
            time_scope=RagTimeScope(scope_type="statement_period", statement_period="May2026"),
            period="May2026",
            plan=["overview"],
            context={},
            answer="You spent 42.",
        )

        with self.assertLogs("app.api.routes.rag", level="INFO") as captured_logs:
            response = self.client.post(
                "/rag/ask",
                json={"question": "What was my spend?", "conversation_id": "conv-123"},
            )

        self.assertEqual(response.status_code, 200)
        joined_logs = "\n".join(captured_logs.output)
        self.assertIn("RAG ask request payload=", joined_logs)
        self.assertIn("RAG ask response payload=", joined_logs)

    def test_ask_forwards_time_scope_when_provided(self) -> None:
        self.service.answer.return_value = RagAnswerResponse(
            question="What did I spend in the first week of April?",
            conversation_id="conv-124",
            time_scope=RagTimeScope(scope_type="date_range", start_date=datetime(2026, 4, 1, tzinfo=timezone.utc).date(), end_date=datetime(2026, 4, 7, tzinfo=timezone.utc).date()),
            period=None,
            plan=["overview"],
            context={},
            answer="You spent 42.",
        )

        response = self.client.post(
            "/rag/ask",
            json={
                "question": "What did I spend in the first week of April?",
                "time_scope": {
                    "scope_type": "date_range",
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-07",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.answer.call_args.kwargs["time_scope"].scope_type, "date_range")

    def test_get_conversation_returns_history(self) -> None:
        self.service.get_conversation_history.return_value = RagConversationResponse(
            conversation_id="conv-123",
            title="Spending chat",
            created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            messages=[
                RagConversationMessageResponse(
                    message_id="msg-1",
                    role="user",
                    content="What was my spend?",
                    time_scope=RagTimeScope(scope_type="statement_period", statement_period="May2026"),
                    created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
                )
            ],
        )

        response = self.client.get("/rag/conversations/conv-123?limit=25")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["conversation_id"], "conv-123")
        self.service.get_conversation_history.assert_called_once_with("conv-123", limit=25)

    def test_get_conversation_logs_request_and_response_payloads(self) -> None:
        self.service.get_conversation_history.return_value = RagConversationResponse(
            conversation_id="conv-123",
            title="Spending chat",
            created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            messages=[
                RagConversationMessageResponse(
                    message_id="msg-1",
                    role="user",
                    content="What was my spend?",
                    time_scope=RagTimeScope(scope_type="statement_period", statement_period="May2026"),
                    created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
                )
            ],
        )

        with self.assertLogs("app.api.routes.rag", level="INFO") as captured_logs:
            response = self.client.get("/rag/conversations/conv-123?limit=25")

        self.assertEqual(response.status_code, 200)
        joined_logs = "\n".join(captured_logs.output)
        self.assertIn("RAG conversation request payload=", joined_logs)
        self.assertIn("RAG conversation response payload=", joined_logs)

    def test_ask_maps_missing_conversation_to_404(self) -> None:
        self.service.answer.side_effect = ConversationNotFoundError("Conversation missing")

        response = self.client.post(
            "/rag/ask",
            json={"question": "What was my spend?", "conversation_id": "missing"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Conversation missing")

    def test_get_conversation_maps_disabled_history_to_503(self) -> None:
        self.service.get_conversation_history.side_effect = ConversationHistoryDisabledError("History disabled")

        response = self.client.get("/rag/conversations/conv-123")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "History disabled")


if __name__ == "__main__":
    unittest.main()

