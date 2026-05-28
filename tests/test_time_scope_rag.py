from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock

from pydantic import ValidationError

from app.models.schemas import RagIntentResponse, RagTimeScope
from app.services.analytics_service import AnalyticsService
from app.services.insight_service import InsightService
from app.services.rag_service import RAGService
from app.skills.analytics import TopCategoriesSkill
from app.skills.factories import build_skill_registry
from app.skills.registry import SkillRegistry


class RAGServiceDateRangeExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spring = Mock()
        self.analytics = AnalyticsService(self.spring)
        self.insights = InsightService(self.spring, self.analytics)
        self.registry = build_skill_registry(self.spring, self.analytics, self.insights)
        self.llm = Mock()
        self.llm.classify_intent.return_value = RagIntentResponse(
            skill_ids=["overview", "period_summary", "categories", "top_categories", "averages"],
            time_reference="2026-04-01 to 2026-04-15",
            confidence=0.9,
        )
        self.llm.generate_answer.return_value = None
        self.service = RAGService(self.spring, self.registry, self.llm)

        self.spring.get_range_overview.return_value = {
            "statementPeriod": None,
            "paymentMethod": None,
            "account": None,
            "totalAmount": "150.00",
            "transactionCount": 4,
        }
        self.spring.get_range_category_breakdown.return_value = [
            {"category": "groceries", "totalAmount": "75.00", "transactionCount": 2},
            {"category": "dining", "totalAmount": "75.00", "transactionCount": 2},
        ]
        self.spring.get_range_top_categories.return_value = [
            {"category": "groceries", "totalAmount": "75.00", "transactionCount": 2},
            {"category": "dining", "totalAmount": "75.00", "transactionCount": 2},
        ]
        self.spring.get_range_account_breakdown.return_value = []
        self.spring.get_range_payment_method_breakdown.return_value = []
        self.spring.get_range_outliers.return_value = []
        self.spring.get_range_daily_totals.return_value = [
            {"date": "2026-04-01", "totalAmount": "50.00", "transactionCount": 1},
            {"date": "2026-04-10", "totalAmount": "100.00", "transactionCount": 3},
        ]
        self.spring.get_overview_for_time_scope.return_value = self.spring.get_range_overview.return_value
        self.spring.get_category_breakdown_for_time_scope.return_value = self.spring.get_range_category_breakdown.return_value
        self.spring.get_top_categories_for_time_scope.return_value = self.spring.get_range_top_categories.return_value
        self.spring.get_account_breakdown_for_time_scope.return_value = self.spring.get_range_account_breakdown.return_value
        self.spring.get_payment_method_breakdown_for_time_scope.return_value = (
            self.spring.get_range_payment_method_breakdown.return_value
        )
        self.spring.get_daily_totals_for_time_scope.return_value = self.spring.get_range_daily_totals.return_value
        self.spring.get_outliers_for_time_scope.return_value = self.spring.get_range_outliers.return_value

    def test_date_range_summary_uses_range_endpoints_without_period_validation_errors(self) -> None:
        response = self.service.answer(
            question="Summarize my spending for the first half of April.",
            time_scope=RagTimeScope(
                scope_type="date_range",
                start_date=date(2026, 4, 1),
                end_date=date(2026, 4, 15),
            ),
        )

        self.assertEqual(response.period, None)
        self.assertEqual(response.time_scope.scope_type, "date_range")
        self.assertNotIn("unavailable_tools", response.context)
        self.assertEqual(response.context["overview"]["statement_period"], None)
        self.assertEqual(response.context["period_summary"]["period"], "2026-04-01 through 2026-04-15")
        self.assertEqual(response.context["averages"]["period"], "2026-04-01 through 2026-04-15")
        self.assertEqual(response.context["averages"]["average_transaction_amount"], "37.50")
        self.assertEqual(response.context["averages"]["active_days"], 2)
        self.spring.get_overview_for_time_scope.assert_called()
        self.spring.get_category_breakdown_for_time_scope.assert_called()
        self.spring.get_top_categories_for_time_scope.assert_called()
        self.spring.get_daily_totals_for_time_scope.assert_called()
        self.spring.get_account_breakdown_for_time_scope.assert_called()
        self.spring.get_payment_method_breakdown_for_time_scope.assert_called()
        self.spring.get_outliers_for_time_scope.assert_called()


class RagTimeScopeValidationTests(unittest.TestCase):
    def test_statement_period_scope_strips_whitespace(self) -> None:
        scope = RagTimeScope(scope_type="statement_period", statement_period="  May2026  ")

        self.assertEqual(scope.statement_period, "May2026")
        self.assertEqual(scope.derived_period, "May2026")
        self.assertEqual(scope.label, "May2026")

    def test_statement_period_scope_rejects_mixed_range_fields(self) -> None:
        with self.assertRaises(ValidationError):
            RagTimeScope(
                scope_type="statement_period",
                statement_period="May2026",
                start_period="April2026",
            )

    def test_statement_period_range_rejects_reverse_order(self) -> None:
        with self.assertRaises(ValidationError):
            RagTimeScope(
                scope_type="statement_period_range",
                start_period="June2026",
                end_period="May2026",
            )

    def test_date_range_rejects_reverse_dates(self) -> None:
        with self.assertRaises(ValidationError):
            RagTimeScope(
                scope_type="date_range",
                start_date=date(2026, 5, 2),
                end_date=date(2026, 5, 1),
            )

    def test_time_scope_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            RagTimeScope.model_validate(
                {
                    "scope_type": "statement_period",
                    "statement_period": "May2026",
                    "unexpected": True,
                }
            )


class RAGServiceWeeklyTopCategoriesTests(unittest.TestCase):
    def test_weekly_top_categories_question_expands_month_and_returns_week_by_week_answer(self) -> None:
        spring = Mock()
        registry = SkillRegistry([TopCategoriesSkill(spring)])
        service = RAGService(spring, registry, None)

        weekly_payloads = {
            (date(2026, 3, 1), date(2026, 3, 7)): [
                {"category": "Reimbursement", "totalAmount": "900.00", "transactionCount": 1},
                {"category": "Gifts", "totalAmount": "200.00", "transactionCount": 2},
            ],
            (date(2026, 3, 8), date(2026, 3, 14)): [
                {"category": "Healthcare", "totalAmount": "300.00", "transactionCount": 3},
                {"category": "Gas", "totalAmount": "150.00", "transactionCount": 4},
            ],
            (date(2026, 3, 15), date(2026, 3, 21)): [
                {"category": "Miata", "totalAmount": "180.00", "transactionCount": 1},
            ],
            (date(2026, 3, 22), date(2026, 3, 28)): [
                {"category": "Gifts", "totalAmount": "250.00", "transactionCount": 2},
            ],
            (date(2026, 3, 29), date(2026, 3, 31)): [
                {"category": "Gas", "totalAmount": "60.00", "transactionCount": 2},
            ],
        }

        def top_categories_for_scope(*, time_scope, limit=10, payment_method=None, account=None, transaction_id=None):
            self.assertEqual(time_scope.scope_type, "date_range")
            return weekly_payloads[(time_scope.start_date, time_scope.end_date)]

        spring.get_top_categories_for_time_scope.side_effect = top_categories_for_scope

        response = service.answer(
            question="What were my highest spending categories per week?",
            time_scope=RagTimeScope(scope_type="statement_period", statement_period="March2026"),
        )

        self.assertEqual(response.period, "March2026")
        self.assertEqual(response.time_scope.scope_type, "statement_period")
        self.assertEqual(response.context["execution_plan"]["strategy"], "multi_scope")
        self.assertEqual(len(response.context["execution_plan"]["steps"]), 5)
        self.assertEqual(response.plan, ["top_categories", "top_categories", "top_categories", "top_categories", "top_categories"])
        self.assertIn("Here are the highest spending categories per week for March2026:", response.answer)
        self.assertIn("Week 1 (2026-03-01 to 2026-03-07): Reimbursement at $900.00 across 1 transactions.", response.answer)
        self.assertIn("Week 5 (2026-03-29 to 2026-03-31): Gas at $60.00 across 2 transactions.", response.answer)
        self.assertEqual(spring.get_top_categories_for_time_scope.call_count, 5)


if __name__ == "__main__":
    unittest.main()



