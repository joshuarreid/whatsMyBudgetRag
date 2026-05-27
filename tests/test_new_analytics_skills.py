from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from unittest.mock import Mock

from app.models.schemas import RagTimeScope
from app.services.rag_service import RAGService
from app.skills.analytics import (
    AvailablePeriodsSkill,
    OverviewSkill,
    StatementPeriodSummaryRangeSkill,
    StatementPeriodSummarySkill,
)
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult
from app.skills.factories import build_skill_registry
from app.skills.insights import PeriodSummarySkill
from app.skills.registry import SkillRegistry


class StubSkill(Skill):
    def __init__(self, definition: SkillDefinition) -> None:
        self.definition = definition

    def execute(self, request: SkillRequest) -> SkillResult:
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload={})


class AvailablePeriodsSkillTests(unittest.TestCase):
    def test_normalizes_raw_period_list_payload(self) -> None:
        spring = Mock()
        spring.get_periods.return_value = ["MAY2026", "APRIL2026"]

        result = AvailablePeriodsSkill(spring).execute(SkillRequest(question="What periods are available?", period="May2026"))

        self.assertEqual(result.payload, {"periods": ["MAY2026", "APRIL2026"], "count": 2})

    def test_normalizes_wrapped_period_payload(self) -> None:
        spring = Mock()
        spring.get_periods.return_value = {"periods": ["MAY2026"], "count": 1}

        result = AvailablePeriodsSkill(spring).execute(SkillRequest(question="List periods", period="May2026"))

        self.assertEqual(result.payload, {"periods": ["MAY2026"], "count": 1})


class StatementPeriodSummarySkillTests(unittest.TestCase):
    def test_single_period_summary_skill_uses_statement_period_scope(self) -> None:
        spring = Mock()
        spring.get_statement_period_summary_for_time_scope.return_value = {
            "statementPeriod": "May2026",
            "periodStartDate": "2026-05-01",
            "periodEndDate": "2026-05-31",
            "totalAmount": "123.45",
            "transactionCount": 7,
            "essentialAmount": "80.00",
            "essentialCount": 4,
            "nonessentialAmount": "43.45",
            "nonessentialCount": 3,
            "categoryBreakdown": {
                "checking": [
                    {"category": "groceries", "totalAmount": "70.00", "transactionCount": 3}
                ]
            },
            "criticalityBreakdown": {
                "checking": [
                    {"criticality": "Essential", "totalAmount": "80.00", "transactionCount": 4}
                ]
            },
            "accountBreakdown": {
                "checking": {"account": "checking", "totalAmount": "123.45", "transactionCount": 7}
            },
            "paymentMethodBreakdown": {
                "checking": [
                    {"paymentMethod": "visa", "totalAmount": "123.45", "transactionCount": 7}
                ]
            },
            "outliers": {
                "checking": [
                    {
                        "id": 1,
                        "name": "Flight",
                        "amount": "123.45",
                        "category": "travel",
                        "criticality": "Nonessential",
                        "transactionDate": "2026-05-10",
                        "account": "checking",
                        "paymentMethod": "visa",
                        "statementPeriod": "May2026",
                        "rowHash": "abc",
                    }
                ]
            },
            "generatedAt": "2026-05-24T18:42:55.569027+00:00",
        }
        skill = StatementPeriodSummarySkill(spring)

        result = skill.execute(
            SkillRequest(
                question="Give me the full summary for May2026",
                time_scope=RagTimeScope(scope_type="statement_period", statement_period="May2026"),
            )
        )

        self.assertEqual(result.payload["statement_period"], "May2026")
        self.assertEqual(result.payload["total_amount"], "123.45")
        self.assertEqual(result.payload["transaction_count"], 7)
        spring.get_statement_period_summary_for_time_scope.assert_called_once()

    def test_single_period_summary_skill_rejects_non_statement_period_scope(self) -> None:
        spring = Mock()
        skill = StatementPeriodSummarySkill(spring)

        with self.assertRaises(ValueError):
            skill.execute(
                SkillRequest(
                    question="Give me the full summary for the first week of April",
                    time_scope=RagTimeScope(
                        scope_type="date_range",
                        start_date=date(2026, 4, 1),
                        end_date=date(2026, 4, 7),
                    ),
                )
            )


class StatementPeriodSummaryRangeSkillTests(unittest.TestCase):
    def test_range_summary_skill_uses_statement_period_range_scope(self) -> None:
        spring = Mock()
        spring.get_statement_period_summary_for_time_scope.return_value = [
            {
                "statementPeriod": "January2026",
                "periodStartDate": "2026-01-01",
                "periodEndDate": "2026-01-31",
                "totalAmount": "100.00",
                "transactionCount": 5,
                "essentialAmount": "60.00",
                "essentialCount": 3,
                "nonessentialAmount": "40.00",
                "nonessentialCount": 2,
                "categoryBreakdown": {},
                "criticalityBreakdown": {},
                "accountBreakdown": {},
                "paymentMethodBreakdown": {},
                "outliers": {},
                "generatedAt": "2026-05-24T18:42:55.569027+00:00",
            },
            {
                "statementPeriod": "February2026",
                "periodStartDate": "2026-02-01",
                "periodEndDate": "2026-02-28",
                "totalAmount": "200.00",
                "transactionCount": 8,
                "essentialAmount": "140.00",
                "essentialCount": 5,
                "nonessentialAmount": "60.00",
                "nonessentialCount": 3,
                "categoryBreakdown": {},
                "criticalityBreakdown": {},
                "accountBreakdown": {},
                "paymentMethodBreakdown": {},
                "outliers": {},
                "generatedAt": "2026-05-24T18:42:55.569027+00:00",
            },
        ]
        skill = StatementPeriodSummaryRangeSkill(spring)

        result = skill.execute(
            SkillRequest(
                question="Show summaries from January to February",
                time_scope=RagTimeScope(
                    scope_type="statement_period_range",
                    start_period="January2026",
                    end_period="February2026",
                ),
            )
        )

        self.assertEqual(len(result.payload), 2)
        self.assertEqual(result.payload[0]["statement_period"], "January2026")
        self.assertEqual(result.payload[1]["statement_period"], "February2026")
        spring.get_statement_period_summary_for_time_scope.assert_called_once()


class AnalyticsRoutingCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        spring = Mock()
        analytics = Mock()
        self.registry = SkillRegistry(
            [
                StubSkill(AvailablePeriodsSkill(spring).definition),
                StubSkill(OverviewSkill(analytics).definition),
                StubSkill(PeriodSummarySkill(Mock()).definition),
                StubSkill(StatementPeriodSummarySkill(spring).definition),
                StubSkill(StatementPeriodSummaryRangeSkill(spring).definition),
            ]
        )

    def test_available_periods_prompt_selects_available_periods_skill(self) -> None:
        selected_ids = [skill.skill_id for skill in self.registry.select("What periods are available?")]

        self.assertIn("available_periods", selected_ids)

    def test_full_summary_prompt_prefers_statement_summary_over_derived_period_summary(self) -> None:
        selected_ids = [skill.skill_id for skill in self.registry.select("Give me the full summary for May2026")]

        self.assertIn("statement_period_summary", selected_ids)
        self.assertNotIn("period_summary", selected_ids)

    def test_summary_range_prompt_selects_statement_period_summary_range(self) -> None:
        selected_ids = [skill.skill_id for skill in self.registry.select("Show summaries from January to March")]

        self.assertIn("statement_period_summary_range", selected_ids)


class SkillFactoryIntegrationTests(unittest.TestCase):
    def test_build_skill_registry_registers_new_skills(self) -> None:
        registry = build_skill_registry(Mock(), Mock(), Mock())
        available_ids = [item["skill_id"] for item in registry.available_skills()]

        self.assertIn("available_periods", available_ids)
        self.assertIn("statement_period_summary", available_ids)
        self.assertIn("statement_period_summary_range", available_ids)


class RagFallbackAnswerTests(unittest.TestCase):
    def test_fallback_answer_mentions_new_summary_context(self) -> None:
        service = RAGService(Mock(), Mock(), None)

        answer = service._fallback_answer(
            {
                "time_scope": {"scope_type": "statement_period", "statement_period": "May2026"},
                "period": "May2026",
                "available_periods": {"periods": ["May2026", "April2026"], "count": 2},
                "statement_period_summary": {
                    "statement_period": "May2026",
                    "total_amount": "123.45",
                    "transaction_count": 7,
                },
                "statement_period_summary_range": [
                    {
                        "statement_period": "April2026",
                        "total_amount": "100.00",
                        "transaction_count": 5,
                        "generated_at": datetime(2026, 5, 24, tzinfo=timezone.utc).isoformat(),
                    }
                ],
            }
        )

        self.assertIn("There are 2 available statement periods.", answer)
        self.assertIn("Statement period summary for May2026 shows total spend of 123.45.", answer)
        self.assertIn("Statement period summaries were included for 1 periods in the selected range.", answer)


if __name__ == "__main__":
    unittest.main()


