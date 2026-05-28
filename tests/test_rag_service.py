from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
import time
from types import SimpleNamespace
from unittest.mock import Mock

from app.models.schemas import RagTimeScope
from app.repositories import ConversationMessageRecord, ConversationRecord
from app.services.langgraph_reasoning_service import LangGraphReasoningService
from app.services.rag_service import RAGService
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult
from app.skills.registry import SkillRegistry


class RecordingSkill(Skill):
    def __init__(self, *, skill_id: str, context_key: str, keywords: tuple[str, ...]) -> None:
        self.definition = SkillDefinition(
            skill_id=skill_id,
            category="analytics",
            context_key=context_key,
            description=f"Recording skill {skill_id}",
            keywords=keywords,
            required=True,
        )

    def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult(
            skill_id=self.skill_id,
            context_key=self.context_key,
            payload={
                "time_scope": request.time_scope.model_dump(mode="json", exclude_none=True) if request.time_scope else None,
                "period": request.period,
                "account": request.account,
                "payment_method": request.payment_method,
            },
        )


class SleepingSkill(Skill):
    def __init__(self, *, skill_id: str, context_key: str, keywords: tuple[str, ...], delay_seconds: float) -> None:
        self.definition = SkillDefinition(
            skill_id=skill_id,
            category="analytics",
            context_key=context_key,
            description=f"Sleeping skill {skill_id}",
            keywords=keywords,
        )
        self.delay_seconds = delay_seconds

    def execute(self, request: SkillRequest) -> SkillResult:
        time.sleep(self.delay_seconds)
        return SkillResult(
            skill_id=self.skill_id,
            context_key=self.context_key,
            payload={"period": request.period},
        )


class ScriptedDailySkill(Skill):
    def __init__(self, payloads_by_period: dict[str, list[dict[str, object]]]) -> None:
        self.definition = SkillDefinition(
            skill_id="daily",
            category="analytics",
            context_key="daily_totals",
            description="Scripted daily totals",
            keywords=("daily", "trend"),
        )
        self.payloads_by_period = payloads_by_period

    def execute(self, request: SkillRequest) -> SkillResult:
        period = request.period or (request.time_scope.statement_period if request.time_scope is not None else None)
        return SkillResult(
            skill_id=self.skill_id,
            context_key=self.context_key,
            payload=list(self.payloads_by_period.get(period or "", [])),
        )


class RAGServicePeriodResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spring = Mock()
        self.service = RAGService(self.spring, Mock(), None)

    def test_explicit_question_period_overrides_requested_period(self) -> None:
        resolved_time_scope, interpretation = self.service._resolve_time_scope(
            question="What was my spend in October?",
            time_scope=RagTimeScope(scope_type="statement_period", statement_period="February2026"),
            period="February2026",
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_time_scope.statement_period, "October2025")
        self.assertEqual(interpretation["source"], "question_bare_month")

    def test_bare_month_resolves_to_most_recent_matching_period(self) -> None:
        resolved_time_scope, interpretation = self.service._resolve_time_scope(
            question="What was my spend in October?",
            time_scope=None,
            period=None,
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_time_scope.statement_period, "October2025")
        self.assertEqual(interpretation["matched_text"], "October")
        self.assertEqual(interpretation["source"], "question_bare_month")

    def test_relative_month_resolves_from_current_statement_period(self) -> None:
        resolved_time_scope, interpretation = self.service._resolve_time_scope(
            question="What was my spend last month?",
            time_scope=None,
            period=None,
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_time_scope.statement_period, "April2026")
        self.assertEqual(interpretation["matched_text"], "last month")
        self.assertEqual(interpretation["source"], "question_relative_month")

    def test_current_period_phrase_stays_on_current_statement_period(self) -> None:
        resolved_time_scope, interpretation = self.service._resolve_time_scope(
            question="Compare this period versus last month.",
            time_scope=None,
            period=None,
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_time_scope.statement_period, "May2026")
        self.assertEqual(interpretation["matched_text"], "this period")
        self.assertEqual(interpretation["source"], "question_current_period")

    def test_build_timeline_context_includes_current_statement_period(self) -> None:
        timeline_context = self.service._build_timeline_context(
            time_scope=RagTimeScope(scope_type="statement_period", statement_period="October2025"),
            period_interpretation={
                "source": "question_bare_month",
                "matched_text": "October",
                "resolved_period": "October2025",
                "time_scope": {"scope_type": "statement_period", "statement_period": "October2025"},
            },
            today=date(2026, 5, 22),
        )

        self.assertEqual(timeline_context["current_date"], "2026-05-22")
        self.assertEqual(timeline_context["current_month"], "May2026")
        self.assertEqual(timeline_context["current_statement_period"], "May2026")
        self.assertEqual(
            timeline_context["selected_time_scope"],
            {"scope_type": "statement_period", "statement_period": "October2025"},
        )
        self.assertEqual(timeline_context["selected_statement_period"], "October2025")
        self.assertEqual(timeline_context["statement_period_format"], "MonthYear")
        self.assertEqual(timeline_context["selection_source"], "question_bare_month")

    def test_cache_policy_disables_current_statement_period(self) -> None:
        self.assertFalse(
            self.service._is_cache_allowed_for_time_scope(
                RagTimeScope(scope_type="statement_period", statement_period="May2026"),
                today=date(2026, 5, 27),
            )
        )

    def test_cache_policy_disables_date_ranges_overlapping_current_month(self) -> None:
        self.assertFalse(
            self.service._is_cache_allowed_for_time_scope(
                RagTimeScope(
                    scope_type="date_range",
                    start_date=date(2026, 4, 28),
                    end_date=date(2026, 5, 3),
                ),
                today=date(2026, 5, 27),
            )
        )

    def test_cache_policy_allows_completed_historical_date_ranges(self) -> None:
        self.assertTrue(
            self.service._is_cache_allowed_for_time_scope(
                RagTimeScope(
                    scope_type="date_range",
                    start_date=date(2026, 4, 1),
                    end_date=date(2026, 4, 15),
                ),
                today=date(2026, 5, 27),
            )
        )

    def test_current_statement_period_is_used_when_question_has_no_time_reference(self) -> None:
        resolved_time_scope, interpretation = self.service._resolve_time_scope(
            question="What was my total spend?",
            time_scope=None,
            period=None,
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_time_scope.statement_period, "May2026")
        self.assertEqual(interpretation["source"], "current_statement_period_fallback")
        self.assertEqual(self.spring.get_periods.call_count, 0)

    def test_prior_conversation_period_is_used_when_question_has_no_time_reference(self) -> None:
        resolved_time_scope, interpretation = self.service._resolve_time_scope(
            question="What about categories?",
            time_scope=None,
            period=None,
            transaction_id=None,
            conversation_history=[
                ConversationMessageRecord(
                    message_id="msg-1",
                    role="assistant",
                    content="December answer",
                    period="December2025",
                    period_source="question_bare_month",
                    created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
                )
            ],
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_time_scope.statement_period, "December2025")
        self.assertEqual(interpretation["source"], "conversation_history_time_scope")

    def test_requested_period_is_used_when_question_and_conversation_have_no_period(self) -> None:
        resolved_time_scope, interpretation = self.service._resolve_time_scope(
            question="What was my total spend?",
            time_scope=RagTimeScope(scope_type="statement_period", statement_period="May2026"),
            period="May2026",
            transaction_id=None,
            conversation_history=[],
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_time_scope.statement_period, "May2026")
        self.assertEqual(interpretation["source"], "request_parameter")

    def test_date_range_follow_up_reuses_prior_time_scope(self) -> None:
        resolved_time_scope, interpretation = self.service._resolve_time_scope(
            question="What about categories?",
            time_scope=None,
            period=None,
            transaction_id=None,
            conversation_history=[
                ConversationMessageRecord(
                    message_id="msg-1",
                    role="assistant",
                    content="April first week answer",
                    period=None,
                    period_source="question_week_of_month",
                    answer_json={
                        "time_scope": {
                            "scope_type": "date_range",
                            "start_date": "2026-04-01",
                            "end_date": "2026-04-07",
                        }
                    },
                    created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
                )
            ],
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_time_scope.scope_type, "date_range")
        self.assertEqual(resolved_time_scope.start_date.isoformat(), "2026-04-01")
        self.assertEqual(resolved_time_scope.end_date.isoformat(), "2026-04-07")
        self.assertEqual(interpretation["source"], "conversation_history_time_scope")


class RAGServiceAccountResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spring = Mock()
        self.service = RAGService(self.spring, Mock(), None)

    def _assistant_message(self, *, account: str, period: str = "May2026") -> ConversationMessageRecord:
        return ConversationMessageRecord(
            message_id="msg-assistant",
            role="assistant",
            content="Stored answer",
            period=period,
            period_source="request_parameter",
            context_json={
                "filters": {
                    "payment_method": None,
                    "account": account,
                }
            },
            answer_json={
                "resolved_filters": {
                    "payment_method": None,
                    "account": account,
                }
            },
            created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
        )

    def test_explicit_question_account_overrides_request_and_history(self) -> None:
        resolved_account, interpretation = self.service._resolve_account(
            question="What was my spend on Checking account?",
            account="Savings",
            conversation_history=[self._assistant_message(account="Travel Card")],
        )

        self.assertEqual(resolved_account, "Checking")
        self.assertEqual(interpretation["source"], "question_explicit_account")

    def test_this_account_reuses_last_conversation_account(self) -> None:
        resolved_account, interpretation = self.service._resolve_account(
            question="What about this account?",
            account="Savings",
            conversation_history=[self._assistant_message(account="Travel Card")],
        )

        self.assertEqual(resolved_account, "Travel Card")
        self.assertEqual(interpretation["source"], "question_contextual_account_reference")

    def test_prior_conversation_account_is_used_for_follow_up_without_new_account(self) -> None:
        resolved_account, interpretation = self.service._resolve_account(
            question="Which categories drove spending?",
            account=None,
            conversation_history=[self._assistant_message(account="Travel Card")],
        )

        self.assertEqual(resolved_account, "Travel Card")
        self.assertEqual(interpretation["source"], "conversation_history_account")

    def test_this_account_uses_request_account_when_conversation_has_no_account(self) -> None:
        resolved_account, interpretation = self.service._resolve_account(
            question="What about this account?",
            account="Savings",
            conversation_history=[],
        )

        self.assertEqual(resolved_account, "Savings")
        self.assertEqual(interpretation["source"], "question_contextual_request_account")

    def test_generic_account_questions_do_not_create_false_positive_account_filters(self) -> None:
        resolved_account, interpretation = self.service._resolve_account(
            question="Give me an account breakdown for this period.",
            account=None,
            conversation_history=[],
        )

        self.assertIsNone(resolved_account)
        self.assertEqual(interpretation["source"], "no_account_filter")


class RAGServiceIntentDependencyTests(unittest.TestCase):
    def test_classify_intent_uses_dedicated_intent_service_when_available(self) -> None:
        spring = Mock()
        registry = Mock()
        registry.available_skills.return_value = [{"skill_id": "overview"}]
        llm = Mock()
        llm.classify_intent.return_value = None
        intent_service = Mock()
        intent_service.classify_intent.return_value = SimpleNamespace(skill_ids=["overview"])
        service = RAGService(spring, registry, llm, None, intent_service=intent_service)

        intent = service._classify_intent("What did I spend?")

        self.assertEqual(intent.skill_ids, ["overview"])
        intent_service.classify_intent.assert_called_once_with(
            question="What did I spend?",
            available_skills=[{"skill_id": "overview"}],
        )
        llm.classify_intent.assert_not_called()


class RAGServiceLatencyOptimizationTests(unittest.TestCase):
    def test_answer_skips_llm_classification_for_confident_trend_question(self) -> None:
        registry = SkillRegistry(
            [
                RecordingSkill(skill_id="overview", context_key="overview", keywords=("spend", "summary", "overview")),
                RecordingSkill(skill_id="daily", context_key="daily_totals", keywords=("daily", "trend")),
            ]
        )
        llm = Mock()
        llm.classify_intent.return_value = SimpleNamespace(skill_ids=["overview"])
        llm.generate_answer.return_value = None
        service = RAGService(Mock(), registry, llm)

        response = service.answer(question="Show my daily spending trend from December through February")

        llm.classify_intent.assert_not_called()
        self.assertEqual(response.context["routing"]["source"], "keyword_match")
        self.assertTrue(response.context["routing"]["intent_classification"]["skipped"])
        self.assertEqual(response.tool_selection.deterministic_tools, ["daily"])
        self.assertEqual(response.plan, ["daily", "daily", "daily"])

    def test_answer_passes_compact_context_to_llm_generation(self) -> None:
        registry = SkillRegistry(
            [
                RecordingSkill(skill_id="overview", context_key="overview", keywords=("spend", "summary")),
                RecordingSkill(skill_id="averages", context_key="averages", keywords=("average",)),
            ]
        )
        llm = Mock()
        llm.classify_intent.return_value = None
        llm.generate_answer.return_value = "Compact answer"
        service = RAGService(Mock(), registry, llm)

        response = service.answer(question="What was my average spend this month?", period="May2026")

        self.assertEqual(response.answer, "Compact answer")
        llm.generate_answer.assert_called_once()
        compact_context = llm.generate_answer.call_args.args[1]
        self.assertIn("overview", compact_context)
        self.assertIn("averages", compact_context)
        self.assertNotIn("routing", compact_context)
        self.assertNotIn("execution_plan", compact_context)
        self.assertNotIn("supporting_sources", compact_context)
        self.assertNotIn("tool_trace_summaries", compact_context)
        self.assertNotIn("cache", compact_context)
        self.assertNotIn("conversation_history", compact_context)

    def test_answer_returns_timing_outside_context_and_persists_it_in_answer_json(self) -> None:
        registry = SkillRegistry(
            [RecordingSkill(skill_id="overview", context_key="overview", keywords=("spend", "summary", "total"))]
        )
        history = Mock()
        history.is_enabled.return_value = True
        history.get_tool_cache.return_value = None
        history.get_shared_tool_cache.return_value = None
        history.create_conversation.return_value = ConversationRecord(
            conversation_id="conv-timing",
            title="Timing chat",
            created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=None,
        )
        history.append_message.side_effect = [
            ConversationMessageRecord(
                message_id="msg-user",
                role="user",
                content="What did I spend?",
                period=None,
                period_source=None,
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
            ConversationMessageRecord(
                db_id=301,
                message_id="msg-assistant",
                role="assistant",
                content="Stored answer",
                period="May2026",
                period_source="request_parameter",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
        ]
        service = RAGService(Mock(), registry, None, history)

        response = service.answer(question="What did I spend?", period="April2026")

        self.assertNotIn("timing", response.context)
        self.assertGreaterEqual(response.timing.classifier_latency_ms, 0)
        self.assertGreaterEqual(response.timing.plan_execution_latency_ms, 0)
        self.assertGreaterEqual(response.timing.cache_lookup_latency_ms, 0)
        self.assertGreaterEqual(response.timing.tool_execution_latency_ms, 0)
        self.assertGreaterEqual(response.timing.answer_generation_latency_ms, 0)
        assistant_call = history.append_message.call_args_list[1]
        self.assertEqual(
            assistant_call.kwargs["answer_json"]["timing"],
            {
                "classifier_latency_ms": response.timing.classifier_latency_ms,
                "plan_execution_latency_ms": response.timing.plan_execution_latency_ms,
                "cache_lookup_latency_ms": response.timing.cache_lookup_latency_ms,
                "tool_execution_latency_ms": response.timing.tool_execution_latency_ms,
                "answer_generation_latency_ms": response.timing.answer_generation_latency_ms,
            },
        )

    def test_answer_timing_separates_cache_lookup_from_tool_execution(self) -> None:
        registry = Mock()
        registry.skills = []
        registry.select.return_value = []
        registry.resolve.return_value = []
        registry.available_skills.return_value = []
        history = Mock()
        history.is_enabled.return_value = True
        history.get_tool_cache.return_value = {
            "response_json": {
                "context_key": "overview",
                "payload": {"total_amount": "42.00"},
                "metadata": {"citation": None},
            },
            "created_at": "2026-05-23T00:00:00+00:00",
            "expires_at": "2026-05-23T00:15:00+00:00",
        }
        history.get_shared_tool_cache.return_value = None
        history.create_conversation.return_value = ConversationRecord(
            conversation_id="conv-cache-timing",
            title="Cache timing chat",
            created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=None,
        )
        history.append_message.side_effect = [
            ConversationMessageRecord(
                message_id="msg-user",
                role="user",
                content="What did I spend?",
                period=None,
                period_source=None,
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
            ConversationMessageRecord(
                db_id=302,
                message_id="msg-assistant",
                role="assistant",
                content="Stored answer",
                period="April2026",
                period_source="request_parameter",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
        ]
        service = RAGService(Mock(), registry, None, history)

        def execute_with_split_timing(skills, request, cache_lookup=None):
            arguments = {
                "time_scope": request.time_scope.model_dump(mode="json", exclude_none=True),
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
                        "duration_ms": 4,
                        "cache_lookup_duration_ms": 3,
                        "tool_execution_duration_ms": 0,
                        "cache_hit": True,
                        "cacheable": True,
                        "arguments": arguments,
                        "result": cached["payload"],
                        "result_summary": {"total_amount": "42.00"},
                        "citation": None,
                        "description": "Overview source",
                    }
                ],
            )

        registry.execute.side_effect = execute_with_split_timing

        response = service.answer(question="What did I spend?", period="April2026")

        self.assertEqual(response.timing.cache_lookup_latency_ms, 3)
        self.assertEqual(response.timing.tool_execution_latency_ms, 0)
        self.assertGreaterEqual(response.timing.plan_execution_latency_ms, 0)

    def test_execute_plan_parallelizes_independent_monthly_steps(self) -> None:
        registry = SkillRegistry(
            [
                SleepingSkill(
                    skill_id="daily",
                    context_key="daily_totals",
                    keywords=("daily", "trend"),
                    delay_seconds=0.25,
                )
            ]
        )
        service = RAGService(Mock(), registry, None)

        started_at = time.perf_counter()
        response = service.answer(question="Show the daily trend from December through February")
        elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.6)
        self.assertEqual(response.plan, ["daily", "daily", "daily"])
        self.assertIn("daily_totals_december2025", response.context)
        self.assertIn("daily_totals_january2026", response.context)
        self.assertIn("daily_totals_february2026", response.context)

    def test_answer_compacts_daily_context_before_calling_llm(self) -> None:
        registry = SkillRegistry(
            [
                ScriptedDailySkill(
                    {
                        "December2025": [
                            {"date": "2025-12-02", "total_amount": "48.00", "transaction_count": 1},
                            {"date": "2025-12-23", "total_amount": "503.22", "transaction_count": 10},
                        ],
                        "January2026": [{"date": "2026-01-20", "total_amount": "2234.38", "transaction_count": 8}],
                        "February2026": [{"date": "2026-02-28", "total_amount": "574.08", "transaction_count": 5}],
                    }
                )
            ]
        )
        llm = Mock()
        llm.classify_intent.return_value = SimpleNamespace(skill_ids=["daily"])
        llm.generate_answer.return_value = "Compact answer"
        service = RAGService(Mock(), registry, llm)

        answer_context = service._build_answer_context(
            {
                "question": "Show my daily trend from December through February",
                "time_scope": {
                    "scope_type": "statement_period_range",
                    "start_period": "December2025",
                    "end_period": "February2026",
                },
                "daily_totals_december2025": [
                    {"date": "2025-12-02", "total_amount": "48.00", "transaction_count": 1},
                    {"date": "2025-12-23", "total_amount": "503.22", "transaction_count": 10},
                ],
                "daily_totals_january2026": [
                    {"date": "2026-01-20", "total_amount": "2234.38", "transaction_count": 8}
                ],
                "execution_plan": {
                    "strategy": "multi_scope",
                    "steps": [
                        {"output_key": "daily_totals_december2025", "label": "December2025", "skill_id": "daily"},
                        {"output_key": "daily_totals_january2026", "label": "January2026", "skill_id": "daily"},
                    ],
                },
                "routing": {"source": "keyword_match"},
                "supporting_sources": [{"source_ref": "api://daily"}],
            }
        )

        self.assertNotIn("daily_totals_december2025", answer_context)
        self.assertNotIn("daily_totals_january2026", answer_context)
        self.assertIn("daily_trend_summary", answer_context)
        self.assertEqual(answer_context["daily_trend_summary"]["series"][0]["label"], "December2025")
        self.assertEqual(answer_context["daily_trend_summary"]["series"][0]["peak_day"]["total_amount"], "503.22")

    def test_answer_uses_deterministic_daily_range_response_and_skips_llm_generation(self) -> None:
        registry = SkillRegistry(
            [
                ScriptedDailySkill(
                    {
                        "December2025": [
                            {"date": "2025-12-02", "total_amount": "48.00", "transaction_count": 1},
                            {"date": "2025-12-23", "total_amount": "503.22", "transaction_count": 10},
                        ],
                        "January2026": [
                            {"date": "2026-01-20", "total_amount": "2234.38", "transaction_count": 8},
                            {"date": "2026-01-21", "total_amount": "1177.02", "transaction_count": 4},
                        ],
                        "February2026": [{"date": "2026-02-28", "total_amount": "574.08", "transaction_count": 5}],
                    }
                )
            ]
        )
        llm = Mock()
        llm.classify_intent.return_value = None
        llm.generate_answer.return_value = "LLM answer"
        service = RAGService(Mock(), registry, llm)

        response = service.answer(question="Show my daily spending trend from December through February")

        llm.generate_answer.assert_not_called()
        self.assertEqual(response.plan, ["daily", "daily", "daily"])
        self.assertIn("## Daily spending trend", response.answer)
        self.assertIn("December2025", response.answer)
        self.assertIn("January2026", response.answer)
        self.assertIn("February2026", response.answer)
        self.assertIn("## Daily totals", response.answer)
        self.assertIn("2026-01-20: $2234.38 across 8 transactions", response.answer)

    def test_deterministic_single_scope_daily_answer_uses_mobile_markdown_summary(self) -> None:
        service = RAGService(Mock(), Mock(), None)

        answer = service._deterministic_answer(
            {
                "time_scope": {
                    "scope_type": "date_range",
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-07",
                },
                "execution_plan": {
                    "strategy": "single_scope",
                    "steps": [
                        {"skill_id": "overview", "output_key": "overview"},
                        {"skill_id": "daily", "output_key": "daily_totals"},
                    ],
                },
                "overview": {"total_amount": "2235.01", "transaction_count": 44},
                "daily_totals": [
                    {"date": "2026-04-01", "total_amount": "100.00", "transaction_count": 4},
                    {"date": "2026-04-02", "total_amount": "250.00", "transaction_count": 7},
                    {"date": "2026-04-03", "total_amount": "300.00", "transaction_count": 8},
                    {"date": "2026-04-04", "total_amount": "200.00", "transaction_count": 5},
                    {"date": "2026-04-05", "total_amount": "199.26", "transaction_count": 6},
                    {"date": "2026-04-06", "total_amount": "400.00", "transaction_count": 8},
                    {"date": "2026-04-07", "total_amount": "785.75", "transaction_count": 6},
                ],
            }
        )

        assert answer is not None
        self.assertIn("Here's what I found for 2026-04-01 through 2026-04-07:", answer)
        self.assertIn("- Date range: 2026-04-01 -> 2026-04-07", answer)
        self.assertIn("- Total spend: $2235.01", answer)
        self.assertIn("- Transaction count: 44", answer)
        self.assertIn("- Average daily spend: $319.29", answer)
        self.assertIn("- Peak day: 2026-04-07 - $785.75 (6 transactions)", answer)
        self.assertIn("## Daily totals", answer)
        self.assertIn("- 2026-04-01: $100.00 across 4 transactions", answer)
        self.assertNotIn("\\n", answer)

    def test_deterministic_range_category_average_answer_uses_monthly_category_totals(self) -> None:
        service = RAGService(Mock(), Mock(), None)

        answer = service._deterministic_answer(
            {
                "question": "From December to May on average how much do I spend a month on Dining Out and Groceries?",
                "time_scope": {
                    "scope_type": "statement_period_range",
                    "start_period": "December2025",
                    "end_period": "May2026",
                },
                "filters": {"account": "josh", "payment_method": None},
                "execution_plan": {
                    "strategy": "multi_scope",
                    "steps": [
                        {"skill_id": "categories", "output_key": "categories_december2025", "label": "December2025"},
                        {"skill_id": "categories", "output_key": "categories_january2026", "label": "January2026"},
                        {"skill_id": "categories", "output_key": "categories_february2026", "label": "February2026"},
                        {"skill_id": "categories", "output_key": "categories_march2026", "label": "March2026"},
                        {"skill_id": "categories", "output_key": "categories_april2026", "label": "April2026"},
                        {"skill_id": "categories", "output_key": "categories_may2026", "label": "May2026"},
                    ],
                },
                "statement_period_summary_range": [
                    {
                        "statement_period": "DECEMBER2025",
                        "category_breakdown": {
                            "josh": [{"category": "Dining Out", "total_amount": "60.63", "transaction_count": 6}],
                            "joint": [
                                {"category": "Dining Out", "total_amount": "424.07", "transaction_count": 24},
                                {"category": "Groceries", "total_amount": "1261.67", "transaction_count": 29},
                            ],
                        },
                    },
                    {
                        "statement_period": "JANUARY2026",
                        "category_breakdown": {
                            "josh": [
                                {"category": "Dining Out", "total_amount": "7.70", "transaction_count": 1},
                                {"category": "Groceries", "total_amount": "15.25", "transaction_count": 1},
                            ],
                            "joint": [
                                {"category": "Dining Out", "total_amount": "436.01", "transaction_count": 27},
                                {"category": "Groceries", "total_amount": "960.81", "transaction_count": 35},
                            ],
                        },
                    },
                    {
                        "statement_period": "FEBRUARY2026",
                        "category_breakdown": {
                            "josh": [{"category": "Dining Out", "total_amount": "79.17", "transaction_count": 13}],
                            "joint": [
                                {"category": "Dining Out", "total_amount": "409.92", "transaction_count": 32},
                                {"category": "Groceries", "total_amount": "633.93", "transaction_count": 22},
                            ],
                        },
                    },
                    {
                        "statement_period": "MARCH2026",
                        "category_breakdown": {
                            "josh": [{"category": "Dining Out", "total_amount": "37.57", "transaction_count": 5}],
                            "joint": [
                                {"category": "Dining Out", "total_amount": "243.50", "transaction_count": 20},
                                {"category": "Groceries", "total_amount": "618.55", "transaction_count": 23},
                            ],
                        },
                    },
                    {
                        "statement_period": "APRIL2026",
                        "category_breakdown": {
                            "josh": [
                                {"category": "Dining Out", "total_amount": "8.29", "transaction_count": 2},
                                {"category": "Groceries", "total_amount": "3.00", "transaction_count": 1},
                            ],
                            "joint": [
                                {"category": "Dining Out", "total_amount": "426.81", "transaction_count": 30},
                                {"category": "Groceries", "total_amount": "502.41", "transaction_count": 20},
                            ],
                        },
                    },
                    {
                        "statement_period": "MAY2026",
                        "category_breakdown": {
                            "joint": [
                                {"category": "Dining Out", "total_amount": "297.60", "transaction_count": 24},
                                {"category": "Groceries", "total_amount": "499.06", "transaction_count": 14},
                            ],
                        },
                    },
                ],
                "categories_december2025": [
                    {"category": "Dining Out", "total_amount": "484.70", "transaction_count": 30},
                    {"category": "Groceries", "total_amount": "1261.67", "transaction_count": 29},
                ],
                "categories_january2026": [
                    {"category": "Dining Out", "total_amount": "443.71", "transaction_count": 28},
                    {"category": "Groceries", "total_amount": "976.06", "transaction_count": 36},
                ],
                "categories_february2026": [
                    {"category": "Dining Out", "total_amount": "489.09", "transaction_count": 45},
                    {"category": "Groceries", "total_amount": "633.93", "transaction_count": 22},
                ],
                "categories_march2026": [
                    {"category": "Dining Out", "total_amount": "281.07", "transaction_count": 25},
                    {"category": "Groceries", "total_amount": "618.55", "transaction_count": 23},
                ],
                "categories_april2026": [
                    {"category": "Dining Out", "total_amount": "451.99", "transaction_count": 33},
                    {"category": "Groceries", "total_amount": "505.41", "transaction_count": 21},
                ],
                "categories_may2026": [
                    {"category": "Dining Out", "total_amount": "297.60", "transaction_count": 24},
                    {"category": "Groceries", "total_amount": "499.06", "transaction_count": 14},
                ],
            }
        )

        assert answer is not None
        self.assertIn("Average monthly spend from December2025 through May2026 (6 months):", answer)
        self.assertIn("- Dining Out: $218.72/month average ($1312.32 total across 6 months).", answer)
        self.assertIn("- Groceries: $376.08/month average ($2256.46 total across 6 months).", answer)
        self.assertIn("## Monthly category totals", answer)
        self.assertIn("- December2025: Dining Out $272.66, Groceries $630.84", answer)


class RAGServiceLangGraphReasoningTests(unittest.TestCase):
    def test_select_skills_can_expand_reasoning_families_with_langgraph(self) -> None:
        registry = SkillRegistry(
            [
                RecordingSkill(skill_id="overview", context_key="overview", keywords=("overview", "summary", "spend")),
                RecordingSkill(skill_id="statement_period_summary_range", context_key="statement_period_summary_range", keywords=("compare",)),
                RecordingSkill(skill_id="top_categories", context_key="top_categories", keywords=("category",)),
                RecordingSkill(skill_id="account_breakdown", context_key="account_breakdown", keywords=("account",)),
                RecordingSkill(skill_id="payment_methods", context_key="payment_methods", keywords=("payment",)),
                RecordingSkill(skill_id="month_over_month", context_key="month_over_month", keywords=("month over month",)),
            ]
        )
        langgraph_service = LangGraphReasoningService(enabled=True)
        if not langgraph_service.enabled:
            self.skipTest("langgraph dependency is not installed")

        service = RAGService(Mock(), registry, None, langgraph_service=langgraph_service)

        selected_skills, routing_metadata = service._select_skills(
            "Why did my spending jump in April versus March?",
            time_scope=RagTimeScope(
                scope_type="statement_period_range",
                start_period="March2026",
                end_period="April2026",
            ),
            llm_intent=None,
        )

        self.assertEqual(routing_metadata["source"], "langgraph_reasoning")
        selected_skill_ids = [skill.skill_id for skill in selected_skills]
        self.assertIn("statement_period_summary_range", selected_skill_ids)
        self.assertIn("overview", selected_skill_ids)
        self.assertIn("top_categories", selected_skill_ids)
        self.assertIn("account_breakdown", selected_skill_ids)
        self.assertIn("payment_methods", selected_skill_ids)
        self.assertIn("month_over_month", selected_skill_ids)
        self.assertEqual(
            routing_metadata["reasoning_graph"]["selected_families"],
            ["baseline_summary", "driver_analysis", "derived_narrative"],
        )


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
                        "time_scope": {"scope_type": "statement_period", "statement_period": "April2026"},
                        "period": "April2026",
                        "payment_method": None,
                        "account": None,
                        "transaction_id": None,
                    },
                    "result": {"total_amount": "42.00"},
                    "result_summary": {"total_amount": "42.00"},
                    "citation": {
                        "source_type": "api",
                        "source_ref": "api://overview?period=April2026",
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
        self.history.get_shared_tool_cache.return_value = None
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

        response = self.service.answer(question="What did I spend?", period="April2026")

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
        self.history.upsert_shared_tool_cache.assert_called_once()
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
                period="April2026",
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

    def test_answer_reuses_last_period_and_account_for_follow_up_without_new_filters(self) -> None:
        self.history.get_conversation.return_value = ConversationRecord(
            conversation_id="conv-sticky",
            title="Sticky chat",
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
        )
        self.history.list_messages.return_value = [
            ConversationMessageRecord(
                message_id="msg-previous-assistant",
                role="assistant",
                content="Stored December answer",
                period="December2025",
                period_source="question_bare_month",
                context_json={
                    "time_scope": {
                        "scope_type": "statement_period",
                        "statement_period": "December2025",
                    },
                    "filters": {
                        "payment_method": None,
                        "account": "Travel Card",
                    }
                },
                answer_json={
                    "time_scope": {
                        "scope_type": "statement_period",
                        "statement_period": "December2025",
                    },
                    "resolved_filters": {
                        "payment_method": None,
                        "account": "Travel Card",
                    }
                },
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            )
        ]
        self.history.append_message.side_effect = [
            ConversationMessageRecord(
                message_id="msg-user",
                role="user",
                content="What about categories?",
                period=None,
                period_source=None,
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
            ConversationMessageRecord(
                db_id=101,
                message_id="msg-assistant",
                role="assistant",
                content="Stored answer",
                period="December2025",
                period_source="conversation_history_period",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
        ]

        response = self.service.answer(
            question="What about categories?",
            conversation_id="conv-sticky",
            period="May2026",
            account="Savings",
        )

        request = self.registry.execute.call_args.args[1]
        self.assertEqual(request.time_scope.statement_period, "December2025")
        self.assertEqual(request.period, "December2025")
        self.assertEqual(request.account, "Travel Card")
        self.assertEqual(response.period, "December2025")
        self.assertEqual(response.time_scope.statement_period, "December2025")
        assistant_call = self.history.append_message.call_args_list[1]
        self.assertEqual(assistant_call.kwargs["period"], "December2025")
        self.assertEqual(
            assistant_call.kwargs["answer_json"]["time_scope"],
            {"scope_type": "statement_period", "statement_period": "December2025"},
        )
        self.assertEqual(assistant_call.kwargs["answer_json"]["resolved_filters"]["account"], "Travel Card")

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
                        "source_ref": "api://overview?period=April2026",
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
                "time_scope": request.time_scope.model_dump(mode="json", exclude_none=True),
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

        response = self.service.answer(question="What did I spend?", period="April2026")

        self.history.get_tool_cache.assert_called()
        self.assertEqual(response.cache.hits, 1)
        self.assertTrue(response.cache.enabled)
        self.assertTrue(response.cache.eligible)
        self.assertIsNone(response.cache.reason)
        self.assertTrue(response.tool_traces[0].cache_hit)
        self.history.upsert_tool_cache.assert_not_called()
        self.history.upsert_shared_tool_cache.assert_not_called()

    def test_answer_uses_shared_cache_when_conversation_cache_misses(self) -> None:
        self.history.create_conversation.return_value = ConversationRecord(
            conversation_id="conv-shared-cache",
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
                db_id=101,
                message_id="msg-assistant",
                role="assistant",
                content="Stored answer",
                period="April2026",
                period_source="request_parameter",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
        ]
        self.history.get_tool_cache.return_value = None
        self.history.get_shared_tool_cache.return_value = {
            "response_json": {
                "context_key": "overview",
                "payload": {"total_amount": "55.00"},
                "metadata": {
                    "citation": {
                        "source_type": "api",
                        "source_ref": "api://overview?period=April2026",
                        "source_title": "Overview source",
                        "snippet": "overview -> total_amount=55.00",
                        "score": 1.0,
                    }
                },
            },
            "created_at": "2026-05-23T00:00:00+00:00",
            "expires_at": "2026-05-23T00:15:00+00:00",
        }

        def execute_with_cache(skills, request, cache_lookup=None):
            arguments = {
                "time_scope": request.time_scope.model_dump(mode="json", exclude_none=True),
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
                        "result_summary": {"total_amount": "55.00"},
                        "citation": cached["metadata"]["citation"],
                        "description": "Overview source",
                    }
                ],
            )

        self.registry.execute.side_effect = execute_with_cache

        response = self.service.answer(question="What did I spend?", period="April2026")

        self.history.get_tool_cache.assert_called()
        self.history.get_shared_tool_cache.assert_called()
        self.assertEqual(response.cache.hits, 1)
        self.assertTrue(response.cache.enabled)
        self.assertTrue(response.cache.eligible)
        self.assertIsNone(response.cache.reason)
        self.assertTrue(response.tool_traces[0].cache_hit)
        self.history.upsert_tool_cache.assert_not_called()
        self.history.upsert_shared_tool_cache.assert_not_called()

    def test_answer_does_not_cache_current_month_requests(self) -> None:
        current_period = self.service.intent_parser.format_statement_period(date.today())
        self.history.create_conversation.return_value = ConversationRecord(
            conversation_id="conv-current-month",
            title="Current month chat",
            created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=None,
        )
        self.history.append_message.side_effect = [
            ConversationMessageRecord(
                message_id="msg-user",
                role="user",
                content="What did I spend this month?",
                period=None,
                period_source=None,
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
            ConversationMessageRecord(
                db_id=102,
                message_id="msg-assistant",
                role="assistant",
                content="Stored answer",
                period=current_period,
                period_source="request_parameter",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
        ]

        response = self.service.answer(question="What did I spend this month?", period=current_period)

        self.history.get_tool_cache.assert_not_called()
        self.history.get_shared_tool_cache.assert_not_called()
        self.history.upsert_tool_cache.assert_not_called()
        self.history.upsert_shared_tool_cache.assert_not_called()
        self.assertTrue(response.cache.enabled)
        self.assertFalse(response.cache.eligible)
        self.assertEqual(response.cache.reason, "current_month_not_cacheable")
        self.assertEqual(response.cache.hits, 0)
        self.assertEqual(response.cache.misses, 0)
        self.assertEqual(response.cache.writes, 0)
        self.assertFalse(response.tool_traces[0].cache_hit)

    def test_answer_uses_distinct_cache_keys_for_repeated_planned_skill_steps(self) -> None:
        registry = SkillRegistry(
            [
                RecordingSkill(
                    skill_id="daily",
                    context_key="daily_totals",
                    keywords=("daily", "trend"),
                )
            ]
        )
        llm = Mock()
        llm.classify_intent.return_value = None
        llm.generate_answer.return_value = None
        history = Mock()
        history.is_enabled.return_value = True
        history.get_tool_cache.return_value = None
        history.get_shared_tool_cache.return_value = None
        history.create_conversation.return_value = ConversationRecord(
            conversation_id="conv-plan-cache",
            title="Compare daily spend for April versus May",
            created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            last_message_at=None,
        )
        history.append_message.side_effect = [
            ConversationMessageRecord(
                message_id="msg-user",
                role="user",
                content="Compare daily spend for April versus May",
                period=None,
                period_source=None,
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
            ConversationMessageRecord(
                db_id=201,
                message_id="msg-assistant",
                role="assistant",
                content="Planned answer",
                period="April2026",
                period_source="question_bare_month",
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            ),
        ]
        service = RAGService(Mock(), registry, llm, history)

        response = service.answer(question="Compare daily spend for April versus May")

        self.assertEqual(response.plan, ["daily", "daily"])
        self.assertEqual(response.context["execution_plan"]["strategy"], "multi_scope")
        self.assertIn("daily_totals_april2026", response.context)
        self.assertIn("daily_totals_may2026", response.context)
        self.assertEqual(response.context["daily_totals_april2026"]["period"], "April2026")
        self.assertEqual(response.context["daily_totals_may2026"]["period"], "May2026")
        self.assertEqual(len(response.tool_traces), 2)
        self.assertEqual(response.tool_traces[0].context_key, "daily_totals_april2026")
        self.assertEqual(response.tool_traces[1].context_key, "daily_totals_may2026")

        self.assertEqual(history.get_tool_cache.call_count, 2)
        cache_keys = [call.kwargs["cache_key"] for call in history.get_tool_cache.call_args_list]
        self.assertEqual(len(set(cache_keys)), 2)
        requested_tools = [call.kwargs["tool_name"] for call in history.get_tool_cache.call_args_list]
        self.assertEqual(requested_tools, ["daily", "daily"])

        self.assertEqual(history.upsert_tool_cache.call_count, 2)
        persisted_cache_keys = [call.kwargs["cache_key"] for call in history.upsert_tool_cache.call_args_list]
        self.assertEqual(len(set(persisted_cache_keys)), 2)
        persisted_context_keys = [call.kwargs["response_json"]["context_key"] for call in history.upsert_tool_cache.call_args_list]
        self.assertEqual(persisted_context_keys, ["daily_totals", "daily_totals"])

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
                answer_json={
                    "time_scope": {
                        "scope_type": "statement_period",
                        "statement_period": "May2026",
                    }
                },
                created_at=datetime(2026, 5, 23, tzinfo=timezone.utc),
            )
        ]

        response = self.service.get_conversation_history("conv-789", limit=25)

        self.history.list_messages.assert_called_once_with("conv-789", limit=25)
        self.assertEqual(response.conversation_id, "conv-789")
        self.assertEqual(response.messages[0].message_id, "msg-2")
        self.assertEqual(response.messages[0].time_scope.statement_period, "May2026")


if __name__ == "__main__":
    unittest.main()


