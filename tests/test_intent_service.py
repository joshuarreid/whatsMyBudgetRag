from __future__ import annotations

import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

from app.models.schemas import RagIntentResponse
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

    def test_infer_period_prefers_explicit_question_reference_over_llm_hint(self) -> None:
        inferred_period = self.service.infer_period(
            question="What was my spend in October?",
            today=date(2026, 5, 22),
            llm_intent=RagIntentResponse(time_reference="this month"),
        )

        self.assertIsNotNone(inferred_period)
        self.assertEqual(inferred_period["resolved_period"], "October2025")
        self.assertEqual(inferred_period["source"], "question_bare_month")

    def test_infer_period_uses_llm_time_reference_when_question_has_no_date(self) -> None:
        inferred_period = self.service.infer_period(
            question="How much did I spend?",
            today=date(2026, 5, 22),
            llm_intent=RagIntentResponse(time_reference="last month"),
        )

        self.assertIsNotNone(inferred_period)
        self.assertEqual(inferred_period["resolved_period"], "April2026")
        self.assertEqual(inferred_period["source"], "llm_relative_month")

    def test_infer_account_prefers_explicit_question_reference_over_llm_hint(self) -> None:
        inferred_account = self.service.infer_account(
            question="What was my spend on Checking account?",
            llm_intent=RagIntentResponse(filters={"account": "Savings"}),
        )

        self.assertIsNotNone(inferred_account)
        self.assertEqual(inferred_account["resolved_account"], "Checking")
        self.assertEqual(inferred_account["source"], "question_explicit_account")

    def test_infer_account_uses_llm_hint_when_question_has_no_explicit_account(self) -> None:
        inferred_account = self.service.infer_account(
            question="How much did I spend?",
            llm_intent=RagIntentResponse(filters={"account": "Travel Card"}),
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
        self.assertEqual(inferred_account["source"], "question_contextual_account_reference")
        self.assertNotIn("resolved_account", inferred_account)

    def test_infer_account_ignores_generic_account_phrases(self) -> None:
        inferred_account = self.service.infer_account(
            question="Give me an account breakdown for this period.",
            llm_intent=None,
        )

        self.assertIsNone(inferred_account)


if __name__ == "__main__":
    unittest.main()



