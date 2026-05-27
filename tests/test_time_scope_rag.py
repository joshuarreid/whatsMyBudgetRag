from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock

from app.models.schemas import RagIntentResponse, RagTimeScope
from app.services.analytics_service import AnalyticsService
from app.services.insight_service import InsightService
from app.services.rag_service import RAGService
from app.skills.factories import build_skill_registry


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


if __name__ == "__main__":
    unittest.main()



