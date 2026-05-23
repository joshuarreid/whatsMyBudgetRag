from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from app.repositories import ConversationMessageRecord, ConversationRecord
from app.services.rag_service import RAGService


class RAGServicePeriodResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spring = Mock()
        self.service = RAGService(self.spring, Mock(), None)

    def test_explicit_request_period_wins(self) -> None:
        resolved_period, interpretation = self.service._resolve_period(
            question="What was my spend in October?",
            period="February2026",
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_period, "February2026")
        self.assertEqual(interpretation["source"], "request_parameter")

    def test_bare_month_resolves_to_most_recent_matching_period(self) -> None:
        resolved_period, interpretation = self.service._resolve_period(
            question="What was my spend in October?",
            period=None,
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_period, "October2025")
        self.assertEqual(interpretation["matched_text"], "October")
        self.assertEqual(interpretation["source"], "question_bare_month")

    def test_relative_month_resolves_from_current_statement_period(self) -> None:
        resolved_period, interpretation = self.service._resolve_period(
            question="What was my spend last month?",
            period=None,
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_period, "April2026")
        self.assertEqual(interpretation["matched_text"], "last month")
        self.assertEqual(interpretation["source"], "question_relative_month")

    def test_current_period_phrase_stays_on_current_statement_period(self) -> None:
        resolved_period, interpretation = self.service._resolve_period(
            question="Compare this period versus last month.",
            period=None,
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_period, "May2026")
        self.assertEqual(interpretation["matched_text"], "this period")
        self.assertEqual(interpretation["source"], "question_current_period")

    def test_build_timeline_context_includes_current_statement_period(self) -> None:
        timeline_context = self.service._build_timeline_context(
            period="October2025",
            period_interpretation={
                "source": "question_bare_month",
                "matched_text": "October",
                "resolved_period": "October2025",
            },
            today=date(2026, 5, 22),
        )

        self.assertEqual(timeline_context["current_date"], "2026-05-22")
        self.assertEqual(timeline_context["current_month"], "May2026")
        self.assertEqual(timeline_context["current_statement_period"], "May2026")
        self.assertEqual(timeline_context["selected_statement_period"], "October2025")
        self.assertEqual(timeline_context["statement_period_format"], "MonthYear")
        self.assertEqual(timeline_context["selection_source"], "question_bare_month")

    def test_current_statement_period_is_used_when_question_has_no_time_reference(self) -> None:
        resolved_period, interpretation = self.service._resolve_period(
            question="What was my total spend?",
            period=None,
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_period, "May2026")
        self.assertEqual(interpretation["source"], "current_statement_period_fallback")
        self.assertEqual(self.spring.get_periods.call_count, 0)


class RAGServiceConversationHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spring = Mock()
        self.registry = Mock()
        self.registry.select.return_value = []
        self.registry.resolve.return_value = []
        self.registry.execute.return_value = (
            {"overview": {"total_amount": "42.00"}},
            [],
            [
                {
                    "tool_name": "overview",
                    "context_key": "overview",
                    "category": "analytics",
                    "status": "ok",
                    "duration_ms": 12,
                    "cache_hit": False,
                    "cacheable": True,
                    "arguments": {
                        "period": "May2026",
                        "payment_method": None,
                        "account": None,
                        "transaction_id": None,
                    },
                    "result": {"total_amount": "42.00"},
                    "result_summary": {"total_amount": "42.00"},
                    "citation": {
                        "source_type": "api",
                        "source_ref": "api://overview?period=May2026",
                        "source_title": "Overview source",
                        "snippet": "overview -> total_amount=42.00",
                        "score": 1.0,
                    },
                    "description": "Overview source",
                }
            ],
        )
        self.registry.available_skills.return_value = []
        self.llm = Mock()
        self.llm.classify_intent.return_value = None
        self.llm.generate_answer.return_value = "Stored answer"
        self.history = Mock()
        self.history.is_enabled.return_value = True
        self.history.get_tool_cache.return_value = None
        self.service = RAGService(self.spring, self.registry, self.llm, self.history)

    def test_answer_creates_conversation_and_persists_both_messages(self) -> None:
        self.history.create_conversation.return_value = ConversationRecord(
            conversation_id="conv-123",
            title="What did I spend?",
            created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=None,
        )
        self.history.append_message.side_effect = [
            ConversationMessageRecord(
                message_id="msg-user",
                role="user",
                content="What did I spend?",
                period=None,
                period_source=None,
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
            ConversationMessageRecord(
                db_id=42,
                message_id="msg-assistant",
                role="assistant",
                content="Stored answer",
                period="May2026",
                period_source="request_parameter",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
        ]

        response = self.service.answer(question="What did I spend?", period="May2026")

        self.assertEqual(response.conversation_id, "conv-123")
        self.history.create_conversation.assert_called_once()
        self.assertEqual(self.history.append_message.call_count, 2)
        user_call = self.history.append_message.call_args_list[0]
        assistant_call = self.history.append_message.call_args_list[1]
        self.assertEqual(user_call.args[0], "conv-123")
        self.assertEqual(user_call.kwargs["role"], "user")
        self.assertEqual(assistant_call.kwargs["role"], "assistant")
        self.assertEqual(response.context["conversation"]["conversation_id"], "conv-123")
        self.history.append_message_tool_calls.assert_called_once()
        self.history.append_message_citations.assert_called_once()
        self.history.upsert_tool_cache.assert_called_once()
        self.assertGreaterEqual(len(response.tool_traces), 1)
        self.assertGreaterEqual(len(response.citations), 1)

    def test_answer_includes_prior_messages_for_existing_conversation(self) -> None:
        self.history.get_conversation.return_value = ConversationRecord(
            conversation_id="conv-456",
            title="Spending chat",
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
        )
        self.history.list_messages.return_value = [
            ConversationMessageRecord(
                message_id="msg-1",
                role="user",
                content="How much did I spend yesterday?",
                period="May2026",
                period_source="question_current_month",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            )
        ]
        self.history.append_message.side_effect = [
            ConversationMessageRecord(
                message_id="msg-user",
                role="user",
                content="What about today?",
                period=None,
                period_source=None,
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
            ConversationMessageRecord(
                db_id=99,
                message_id="msg-assistant",
                role="assistant",
                content="Stored answer",
                period="May2026",
                period_source="request_parameter",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
        ]

        response = self.service.answer(
            question="What about today?",
            conversation_id="conv-456",
            period="May2026",
        )

        self.history.create_conversation.assert_not_called()
        self.history.list_messages.assert_called_once_with("conv-456", limit=self.service.conversation_history_context_limit)
        self.assertEqual(response.context["conversation_history"][0]["content"], "How much did I spend yesterday?")

    def test_answer_uses_conversation_cache_for_repeated_tool_requests(self) -> None:
        self.history.create_conversation.return_value = ConversationRecord(
            conversation_id="conv-cache",
            title="Cached chat",
            created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=None,
        )
        self.history.append_message.side_effect = [
            ConversationMessageRecord(
                message_id="msg-user",
                role="user",
                content="What did I spend?",
                period=None,
                period_source=None,
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
            ConversationMessageRecord(
                db_id=100,
                message_id="msg-assistant",
                role="assistant",
                content="Stored answer",
                period="May2026",
                period_source="request_parameter",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
        ]
        self.history.get_tool_cache.return_value = {
            "response_json": {
                "context_key": "overview",
                "payload": {"total_amount": "42.00"},
                "metadata": {
                    "citation": {
                        "source_type": "api",
                        "source_ref": "api://overview?period=May2026",
                        "source_title": "Overview source",
                        "snippet": "overview -> total_amount=42.00",
                        "score": 1.0,
                    }
                },
            },
            "created_at": "2026-05-23T00:00:00+00:00",
            "expires_at": "2026-05-23T00:15:00+00:00",
        }

        def execute_with_cache(skills, request, cache_lookup=None):
            arguments = {
                "period": request.period,
                "payment_method": request.payment_method,
                "account": request.account,
                "transaction_id": request.transaction_id,
            }
            cached = cache_lookup(SimpleNamespace(skill_id="overview", context_key="overview"), arguments)
            self.assertIsNotNone(cached)
            return (
                {"overview": cached["payload"]},
                [],
                [
                    {
                        "tool_name": "overview",
                        "context_key": "overview",
                        "category": "analytics",
                        "status": "ok",
                        "duration_ms": 1,
                        "cache_hit": True,
                        "cacheable": True,
                        "arguments": arguments,
                        "result": cached["payload"],
                        "result_summary": {"total_amount": "42.00"},
                        "citation": cached["metadata"]["citation"],
                        "description": "Overview source",
                    }
                ],
            )

        self.registry.execute.side_effect = execute_with_cache

        response = self.service.answer(question="What did I spend?", period="May2026")

        self.history.get_tool_cache.assert_called()
        self.assertEqual(response.cache.hits, 1)
        self.assertTrue(response.tool_traces[0].cache_hit)
        self.history.upsert_tool_cache.assert_not_called()

    def test_get_conversation_history_returns_serialized_messages(self) -> None:
        self.history.get_conversation.return_value = ConversationRecord(
            conversation_id="conv-789",
            title="Summary",
            created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=None,
        )
        self.history.list_messages.return_value = [
            ConversationMessageRecord(
                message_id="msg-2",
                role="assistant",
                content="You spent 42.",
                period="May2026",
                period_source="request_parameter",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            )
        ]

        response = self.service.get_conversation_history("conv-789", limit=25)

        self.history.list_messages.assert_called_once_with("conv-789", limit=25)
        self.assertEqual(response.conversation_id, "conv-789")
        self.assertEqual(response.messages[0].message_id, "msg-2")


if __name__ == "__main__":
    unittest.main()


