from __future__ import annotations

import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

from app.models.schemas import RagIntentFilters, RagIntentResponse
from app.services.intent_service import IntentService


class IntentServiceClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = IntentService()
        self.service.model = "gpt-4o-mini"
        self.service.client = Mock()

    def test_classify_intent_returns_parsed_schema_output(self) -> None:
        parsed_intent = RagIntentResponse(
            skill_ids=["overview", "period_summary"],
            time_reference="this month",
            confidence=0.94,
        )
        self.service.client.responses.parse.return_value = SimpleNamespace(
            output_parsed=parsed_intent,
            output_text='{"skill_ids": ["overview", "period_summary"]}',
        )

        intent = self.service.classify_intent(
            question="Summarize my spending this month",
            available_skills=[{"skill_id": "overview"}, {"skill_id": "period_summary"}],
        )

        self.assertEqual(intent, parsed_intent)
        self.service.client.responses.parse.assert_called_once()

    def test_classify_intent_validates_parsed_dict_output(self) -> None:
        self.service.client.responses.parse.return_value = SimpleNamespace(
            output_parsed={
                "skill_ids": ["overview"],
                "time_reference": "this month",
                "filters": {"payment_method": None, "account": None},
                "confidence": 0.81,
            },
            output_text='{"skill_ids": ["overview"]}',
        )

        intent = self.service.classify_intent(
            question="How are my finances lately?",
            available_skills=[{"skill_id": "overview"}],
        )

        self.assertIsNotNone(intent)
        self.assertEqual(intent.skill_ids, ["overview"])
        self.assertEqual(intent.time_reference, "this month")

    def test_classify_intent_returns_none_when_parse_returns_no_output(self) -> None:
        self.service.client.responses.parse.return_value = SimpleNamespace(
            output_parsed=None,
            output_text="",
        )

        intent = self.service.classify_intent(
            question="How are my finances lately?",
            available_skills=[{"skill_id": "overview"}],
        )

        self.assertIsNone(intent)

    def test_classify_intent_returns_none_when_parse_raises(self) -> None:
        self.service.client.responses.parse.side_effect = RuntimeError("parse failed")

        intent = self.service.classify_intent(
            question="How are my finances lately?",
            available_skills=[{"skill_id": "overview"}],
        )

        self.assertIsNone(intent)


class IntentServiceInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = IntentService()

    def test_infer_time_scope_prefers_explicit_question_reference_over_llm_hint(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="What was my spend in October?",
            today=date(2026, 5, 22),
            llm_intent=RagIntentResponse(time_reference="this month"),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(inferred_scope["resolved_period"], "October2025")
        self.assertEqual(inferred_scope["time_scope"]["scope_type"], "statement_period")
        self.assertEqual(inferred_scope["source"], "question_bare_month")

    def test_infer_time_scope_uses_llm_time_reference_when_question_has_no_date(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="How much did I spend?",
            today=date(2026, 5, 22),
            llm_intent=RagIntentResponse(time_reference="last month"),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(inferred_scope["resolved_period"], "April2026")
        self.assertEqual(inferred_scope["source"], "llm_relative_month")

    def test_infer_time_scope_resolves_bare_month_to_statement_period(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="Show me my spending for April",
            today=date(2026, 5, 27),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(inferred_scope["time_scope"], {"scope_type": "statement_period", "statement_period": "April2026"})

    def test_infer_time_scope_resolves_month_range_to_statement_period_range(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="Compare spending December through March",
            today=date(2026, 5, 27),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(
            inferred_scope["time_scope"],
            {
                "scope_type": "statement_period_range",
                "start_period": "December2025",
                "end_period": "March2026",
            },
        )

    def test_infer_time_scope_resolves_compact_monthyear_range_to_statement_period_range(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="For statement periods JANUARY2026 through MAY2026, how much did I spend?",
            today=date(2026, 5, 29),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(
            inferred_scope["time_scope"],
            {
                "scope_type": "statement_period_range",
                "start_period": "January2026",
                "end_period": "May2026",
            },
        )

    def test_infer_time_scope_resolves_explicit_current_year_to_year_to_date_statement_period_range(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="How much do I spend a month on average in 2026 on nonessential items?",
            today=date(2026, 5, 28),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(inferred_scope["source"], "question_calendar_year")
        self.assertEqual(
            inferred_scope["time_scope"],
            {
                "scope_type": "statement_period_range",
                "start_period": "January2026",
                "end_period": "May2026",
            },
        )

    def test_infer_time_scope_resolves_this_year_to_year_to_date_statement_period_range(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="How much have I spent this year?",
            today=date(2026, 5, 28),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(inferred_scope["source"], "question_calendar_year")
        self.assertEqual(
            inferred_scope["time_scope"],
            {
                "scope_type": "statement_period_range",
                "start_period": "January2026",
                "end_period": "May2026",
            },
        )

    def test_infer_time_scope_resolves_prior_year_to_full_calendar_year_statement_period_range(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="How much did I spend in 2025?",
            today=date(2026, 5, 28),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(inferred_scope["source"], "question_calendar_year")
        self.assertEqual(
            inferred_scope["time_scope"],
            {
                "scope_type": "statement_period_range",
                "start_period": "January2025",
                "end_period": "December2025",
            },
        )

    def test_infer_time_scope_resolves_end_of_month_to_date_range(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="How much did I spend at the end of April?",
            today=date(2026, 5, 27),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(
            inferred_scope["time_scope"],
            {
                "scope_type": "date_range",
                "start_date": "2026-04-24",
                "end_date": "2026-04-30",
            },
        )

    def test_infer_time_scope_resolves_first_week_of_month_to_date_range(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="What did I spend in the first week of April?",
            today=date(2026, 5, 27),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(
            inferred_scope["time_scope"],
            {
                "scope_type": "date_range",
                "start_date": "2026-04-01",
                "end_date": "2026-04-07",
            },
        )

    def test_infer_time_scope_resolves_first_half_of_month_to_date_range(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="What was my spend in the first half of April?",
            today=date(2026, 5, 27),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(
            inferred_scope["time_scope"],
            {
                "scope_type": "date_range",
                "start_date": "2026-04-01",
                "end_date": "2026-04-15",
            },
        )

    def test_infer_time_scope_resolves_today_to_single_day_date_range(self) -> None:
        inferred_scope = self.service.infer_time_scope(
            question="How much did I spend today?",
            today=date(2026, 5, 27),
        )

        self.assertIsNotNone(inferred_scope)
        self.assertEqual(
            inferred_scope["time_scope"],
            {
                "scope_type": "date_range",
                "start_date": "2026-05-27",
                "end_date": "2026-05-27",
            },
        )

    def test_infer_period_returns_none_for_non_statement_period_time_scopes(self) -> None:
        inferred_period = self.service.infer_period(
            question="What did I spend in the first week of April?",
            today=date(2026, 5, 27),
        )

        self.assertIsNone(inferred_period)

    def test_infer_account_prefers_explicit_question_reference_over_llm_hint(self) -> None:
        inferred_account = self.service.infer_account(
            question="What was my spend on Checking account?",
            llm_intent=RagIntentResponse(filters=RagIntentFilters(account="Savings")),
        )

        self.assertIsNotNone(inferred_account)
        self.assertEqual(inferred_account["resolved_account"], "Checking")
        self.assertEqual(inferred_account["source"], "question_explicit_account")

    def test_infer_account_uses_llm_hint_when_question_has_no_explicit_account(self) -> None:
        inferred_account = self.service.infer_account(
            question="How much did I spend?",
            llm_intent=RagIntentResponse(filters=RagIntentFilters(account="Travel Card")),
        )

        self.assertIsNotNone(inferred_account)
        self.assertEqual(inferred_account["resolved_account"], "Travel Card")
        self.assertEqual(inferred_account["source"], "llm_question_account")

    def test_infer_account_returns_contextual_reference_without_resolved_account(self) -> None:
        inferred_account = self.service.infer_account(
            question="What about this account?",
            llm_intent=None,
        )

        self.assertIsNotNone(inferred_account)
        assert inferred_account is not None
        self.assertEqual(inferred_account["source"], "question_contextual_account_reference")
        self.assertNotIn("resolved_account", inferred_account)

    def test_infer_account_ignores_generic_account_phrases(self) -> None:
        inferred_account = self.service.infer_account(
            question="Give me an account breakdown for this period.",
            llm_intent=None,
        )

        self.assertIsNone(inferred_account)

    def test_infer_account_recognizes_possessive_person_reference(self) -> None:
        inferred_account = self.service.infer_account(
            question="In March what was Anna's biggest transaction?",
            llm_intent=None,
        )

        self.assertIsNotNone(inferred_account)
        assert inferred_account is not None
        self.assertEqual(inferred_account["resolved_account"], "Anna")
        self.assertEqual(inferred_account["source"], "question_explicit_account")

    def test_infer_account_recognizes_trailing_for_name_reference(self) -> None:
        inferred_account = self.service.infer_account(
            question="How does december look for anna?",
            llm_intent=None,
        )

        self.assertIsNotNone(inferred_account)
        assert inferred_account is not None
        self.assertEqual(inferred_account["resolved_account"], "anna")
        self.assertEqual(inferred_account["source"], "question_explicit_account")

    def test_infer_account_recognizes_on_track_name_reference(self) -> None:
        inferred_account = self.service.infer_account(
            question="Is Josh on track to spend less than $1500 on nonessential stuff this month?",
            llm_intent=None,
        )

        self.assertIsNotNone(inferred_account)
        assert inferred_account is not None
        self.assertEqual(inferred_account["resolved_account"], "Josh")
        self.assertEqual(inferred_account["source"], "question_explicit_account")


if __name__ == "__main__":
    unittest.main()



