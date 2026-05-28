from __future__ import annotations

import unittest
from unittest.mock import Mock

from app.skills.analytics import (
    AccountBreakdownSkill,
    CategoriesSkill,
    CriticalitySkill,
    DailyTotalsSkill,
    OutliersSkill,
    OverviewSkill,
    PaymentMethodsSkill,
    TopCategoriesSkill,
)
from app.skills.insights import BehaviorSummarySkill, MonthOverMonthSkill, PeriodSummarySkill
from app.skills.registry import SkillRegistry


SAMPLE_QUESTION_EXPECTATIONS = {
    "What’s my overview": {"overview"},
    "What’s my overview for May": {"overview"},
    "What’s the gist of this month": {"overview"},
    "What’s my spending this month": {"overview"},
    "what was my largest spending category in february": {"categories", "top_categories"},
    "what were my highest spending categories per week in march": {"categories", "top_categories"},
    "How much did I spend in January and what were the categories": {"overview", "categories"},
    "In March what was my biggest transaction": {"outliers"},
    "What were the biggest transactions of the month?": {"outliers"},
    "Show my daily spending for last week.": {"daily"},
    "Show my daily spending trend from December through February.": {"daily"},
    "What is my spending summary in April?": {"overview", "period_summary"},
    "Give me an account breakdown for this period. include categories as well": {"account_breakdown", "categories"},
    "What payment methods drove most of my spending this period and which categories drove it?": {"payment_methods", "categories"},
    "Is Josh on track to spend less than $1500 on nonessential stuff this month": {"overview", "criticality"},
    "Compare this period versus last month.": {"month_over_month"},
    "What spending patterns do you see in this period?": {"behavior_summary"},
}


class SampleQuestionKeywordRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        spring = Mock()
        analytics = Mock()
        insights = Mock()
        self.registry = SkillRegistry(
            [
                OverviewSkill(analytics),
                CategoriesSkill(spring),
                TopCategoriesSkill(spring),
                AccountBreakdownSkill(spring),
                PaymentMethodsSkill(spring),
                DailyTotalsSkill(spring),
                CriticalitySkill(spring),
                OutliersSkill(spring),
                PeriodSummarySkill(insights),
                BehaviorSummarySkill(insights),
                MonthOverMonthSkill(insights),
            ]
        )

    def test_sample_questions_select_expected_skills(self) -> None:
        for question, expected_skill_ids in SAMPLE_QUESTION_EXPECTATIONS.items():
            with self.subTest(question=question):
                selected_skill_ids = {skill.skill_id for skill in self.registry.select(question)}
                self.assertTrue(
                    expected_skill_ids.issubset(selected_skill_ids),
                    msg=f"Expected {expected_skill_ids} to be selected for {question!r}, but got {selected_skill_ids}",
                )


if __name__ == "__main__":
    unittest.main()

