from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.services.llm_service import LLMService


class LLMServiceAnswerGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LLMService()
        self.service.model = "gpt-4o-mini"
        self.service.client = Mock()

    def test_generate_answer_returns_trimmed_output_text(self) -> None:
        self.service.client.responses.create.return_value = SimpleNamespace(
            output_text="  Answer from model.  ",
        )

        answer = self.service.generate_answer(
            question="Summarize my spending this month",
            context={"overview": {"total_amount": "42.00"}},
        )

        self.assertEqual(answer, "Answer from model.")
        self.service.client.responses.create.assert_called_once()

    def test_generate_answer_returns_none_when_client_is_disabled(self) -> None:
        self.service.client = None

        answer = self.service.generate_answer(
            question="How are my finances lately?",
            context={"overview": {"total_amount": "42.00"}},
        )

        self.assertIsNone(answer)


if __name__ == "__main__":
    unittest.main()

