from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock

from app.services.rag_service import RAGService


class RAGServicePeriodResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = RAGService(Mock(), Mock(), None)

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

    def test_default_setting_is_used_when_question_has_no_time_reference(self) -> None:
        self.service.default_period = "January2026"

        resolved_period, interpretation = self.service._resolve_period(
            question="What was my total spend?",
            period=None,
            transaction_id=None,
            today=date(2026, 5, 22),
        )

        self.assertEqual(resolved_period, "January2026")
        self.assertEqual(interpretation["source"], "default_setting")


if __name__ == "__main__":
    unittest.main()

