from __future__ import annotations

import logging
import unittest
from typing import Any
from unittest.mock import Mock

from app.services.rag_service import RAGService
from app.skills.analytics import (
    AccountBreakdownSkill,
    CategoriesSkill,
    CriticalitySkill,
    DailyTotalsSkill,
    DuplicatesSkill,
    OutliersSkill,
    OverviewSkill,
    PaymentMethodsSkill,
    TopCategoriesSkill,
    UncategorizedSkill,
)
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult
from app.skills.registry import SkillRegistry


logger = logging.getLogger(__name__)

QUESTION_VARIATIONS = {
    "overview": [
        "How much did I spend this month?",
        "Give me a summary of this period.",
        "What's my total for May?",
    ],
    "categories": [
        "Show my categories for this period.",
        "Which category cost me the most?",
        "Can you break this down by categories?",
    ],
    "top_categories": [
        "What are my top categories this month?",
        "Show the top category for this period.",
        "Give me the top categories, please.",
    ],
    "account_breakdown": [
        "Give me an account breakdown for this period.",
        "Which account drove the most spending?",
        "Breakdown by account, please.",
    ],
    "payment_methods": [
        "Which payment method drove most of my spending?",
        "How much did I spend on card purchases?",
        "Did I use cash much this month?",
    ],
    "daily": [
        "Show my daily spending trend this month.",
        "Give me the daily totals.",
        "Can I see the time series for this period?",
    ],
    "criticality": [
        "How much was essential versus non-essential?",
        "Show the criticality split for this month.",
        "Was most of my spend essential?",
    ],
    "duplicates": [
        "Do I have duplicate transactions this period?",
        "Any duplicates I should review?",
        "Check for duplicate charges this month.",
    ],
    "uncategorized": [
        "Show me uncategorized transactions.",
        "Any uncategorized spending this month?",
        "List uncategorized items for this period.",
    ],
    "outliers": [
        "What are my outlier transactions this month?",
        "Show the largest transactions this period.",
        "Any spending outliers I should know about?",
    ],
}

EXPECTED_ANSWER_FRAGMENTS = {
    "overview": [
        "Total spend for period May2026 is 123.45.",
        "Transaction count is 7.",
    ],
    "categories": ["Category breakdown data was included from Spring Boot analytics."],
    "top_categories": ["Top category spend data was included from Spring Boot analytics."],
    "account_breakdown": ["Account breakdown data was included for the selected period."],
    "payment_methods": ["Payment method breakdown data was included for the selected period."],
    "daily": ["Daily totals were included for the selected period."],
    "criticality": ["Criticality breakdown data was included for the selected period."],
    "duplicates": ["Duplicate transaction candidates were included for the selected period."],
    "uncategorized": ["Uncategorized transactions were included for the selected period."],
    "outliers": ["Outlier transactions were included for the selected period."],
}


class StubPayloadSkill(Skill):
    def __init__(self, definition: SkillDefinition, payload: Any) -> None:
        self.definition = definition
        self.payload = payload

    def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=self.payload)


class AnalyticsSkillQuestionVariationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig(level=logging.INFO, format="TEST %(message)s", force=True)

    def setUp(self) -> None:
        spring_client = Mock()
        analytics_service = Mock()

        self.skills = {
            "overview": OverviewSkill(analytics_service),
            "categories": CategoriesSkill(spring_client),
            "top_categories": TopCategoriesSkill(spring_client),
            "account_breakdown": AccountBreakdownSkill(spring_client),
            "payment_methods": PaymentMethodsSkill(spring_client),
            "daily": DailyTotalsSkill(spring_client),
            "criticality": CriticalitySkill(spring_client),
            "duplicates": DuplicatesSkill(spring_client),
            "uncategorized": UncategorizedSkill(spring_client),
            "outliers": OutliersSkill(spring_client),
        }
        self.registry = SkillRegistry(list(self.skills.values()))
        self.question_variations = QUESTION_VARIATIONS

    def _log_case(self, *, phase: str, skill_id: str, question: str, answer: str) -> None:
        logger.info("[%s] skill=%s question=%r answer=%s", phase, skill_id, question, answer)

    def test_each_analytics_skill_matches_human_question_variations(self) -> None:
        for skill_id, questions in self.question_variations.items():
            skill = self.skills[skill_id]
            for question in questions:
                with self.subTest(skill=skill_id, question=question):
                    matched = skill.matches(question)
                    self._log_case(
                        phase="match",
                        skill_id=skill_id,
                        question=question,
                        answer=f"matched={matched}",
                    )
                    self.assertTrue(
                        matched,
                        msg=f"{skill_id} should match question: {question}",
                    )

    def test_registry_select_includes_expected_analytics_skill_for_human_variations(self) -> None:
        for expected_skill_id, questions in self.question_variations.items():
            for question in questions:
                with self.subTest(expected_skill_id=expected_skill_id, question=question):
                    selected_ids = [skill.skill_id for skill in self.registry.select(question)]
                    self._log_case(
                        phase="select",
                        skill_id=expected_skill_id,
                        question=question,
                        answer=f"selected={selected_ids}",
                    )
                    self.assertIn(
                        expected_skill_id,
                        selected_ids,
                        msg=(
                            f"Expected '{expected_skill_id}' to be selected for question "
                            f"{question!r}, but got {selected_ids}"
                        ),
                    )


class AnalyticsSkillRAGIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.basicConfig(level=logging.INFO, format="TEST %(message)s", force=True)

    def setUp(self) -> None:
        spring_client = Mock()
        analytics_service = Mock()
        concrete_skills = {
            "overview": OverviewSkill(analytics_service),
            "categories": CategoriesSkill(spring_client),
            "top_categories": TopCategoriesSkill(spring_client),
            "account_breakdown": AccountBreakdownSkill(spring_client),
            "payment_methods": PaymentMethodsSkill(spring_client),
            "daily": DailyTotalsSkill(spring_client),
            "criticality": CriticalitySkill(spring_client),
            "duplicates": DuplicatesSkill(spring_client),
            "uncategorized": UncategorizedSkill(spring_client),
            "outliers": OutliersSkill(spring_client),
        }
        payloads = {
            "overview": {"total_amount": "123.45", "transaction_count": 7},
            "categories": [{"category": "Groceries", "total_amount": "42.00", "transaction_count": 3}],
            "top_categories": [{"category": "Dining", "total_amount": "58.00", "transaction_count": 2}],
            "account_breakdown": [{"account": "Checking", "total_amount": "88.00", "transaction_count": 4}],
            "payment_methods": [
                {"payment_method": "Card", "total_amount": "73.00", "transaction_count": 5}
            ],
            "daily": [{"date": "2026-05-10", "total_amount": "15.00", "transaction_count": 2}],
            "criticality": [
                {"criticality": "ESSENTIAL", "total_amount": "61.00", "transaction_count": 4}
            ],
            "duplicates": [{"row_hash": "abc123", "occurrences": 2, "total_amount": "19.98"}],
            "uncategorized": [
                {
                    "id": 1,
                    "name": "Unknown Merchant",
                    "amount": "17.32",
                    "category": "Uncategorized",
                    "criticality": "UNKNOWN",
                    "transactionDate": "2026-05-14",
                    "account": "Checking",
                    "status": "POSTED",
                    "createdTime": "2026-05-14T12:00:00Z",
                    "paymentMethod": "Card",
                    "statementPeriod": "May2026",
                    "rowHash": "uncat-1",
                }
            ],
            "outliers": [
                {
                    "id": 2,
                    "name": "Flight Booking",
                    "amount": "412.99",
                    "category": "Travel",
                    "criticality": "NON_ESSENTIAL",
                    "transactionDate": "2026-05-20",
                    "account": "Travel Card",
                    "status": "POSTED",
                    "createdTime": "2026-05-20T18:45:00Z",
                    "paymentMethod": "Card",
                    "statementPeriod": "May2026",
                    "rowHash": "outlier-1",
                }
            ],
        }
        stub_skills = [
            StubPayloadSkill(concrete_skills[skill_id].definition, payloads[skill_id])
            for skill_id in concrete_skills
        ]
        self.registry = SkillRegistry(stub_skills)
        self.service = RAGService(Mock(), self.registry, None)

    def _log_rag_case(self, *, skill_id: str, question: str, plan: list[str], answer: str) -> None:
        logger.info("[rag] skill=%s question=%r plan=%s answer=%s", skill_id, question, plan, answer)

    def test_answer_logs_final_response_text_for_each_analytics_question_variation(self) -> None:
        for expected_skill_id, questions in QUESTION_VARIATIONS.items():
            for question in questions:
                with self.subTest(expected_skill_id=expected_skill_id, question=question):
                    response = self.service.answer(question=question, period="May2026")
                    self._log_rag_case(
                        skill_id=expected_skill_id,
                        question=question,
                        plan=response.plan,
                        answer=response.answer,
                    )
                    self.assertEqual(response.question, question)
                    self.assertEqual(response.period, "May2026")
                    self.assertIn(expected_skill_id, response.plan)
                    for fragment in EXPECTED_ANSWER_FRAGMENTS[expected_skill_id]:
                        self.assertIn(fragment, response.answer)


if __name__ == "__main__":
    unittest.main()

