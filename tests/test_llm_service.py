from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.models.schemas import RagIntentResponse
from app.services.llm_service import LLMService


class LLMServiceIntentClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LLMService()
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


if __name__ == "__main__":
    unittest.main()

