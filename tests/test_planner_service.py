from __future__ import annotations

import unittest
from datetime import date

from app.models.schemas import RagTimeScope
from app.services.intent_service import IntentService
from app.services.planner_service import PlannerService
from app.skills.base import Skill, SkillDefinition


class StubSkill(Skill):
    def __init__(self, *, skill_id: str, context_key: str, expand_with_multi_scope: bool = True) -> None:
        self.definition = SkillDefinition(
            skill_id=skill_id,
            category="analytics",
            context_key=context_key,
            description=f"Stub skill {skill_id}",
            keywords=(skill_id,),
            expand_with_multi_scope=expand_with_multi_scope,
        )

    def execute(self, request):  # pragma: no cover - planning tests never execute skills
        raise NotImplementedError


class PlannerServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = PlannerService(intent_service=IntentService(enable_llm=False))
        self.daily_skill = StubSkill(skill_id="daily", context_key="daily_totals")
        self.range_summary_skill = StubSkill(
            skill_id="statement_period_summary_range",
            context_key="statement_period_summary_range",
            expand_with_multi_scope=False,
        )

    def test_build_plan_repeats_same_skill_for_explicit_period_comparison(self) -> None:
        plan = self.planner.build_plan(
            question="Compare daily spend for April versus May",
            skills=[self.daily_skill],
            time_scope=RagTimeScope(scope_type="statement_period", statement_period="April2026"),
            period="April2026",
            payment_method=None,
            account=None,
            today=date(2026, 5, 27),
        )

        self.assertEqual(plan.strategy, "multi_scope")
        self.assertEqual([step.skill_id for step in plan.steps], ["daily", "daily"])
        self.assertEqual([step.period for step in plan.steps], ["April2026", "May2026"])
        self.assertEqual([step.output_key for step in plan.steps], ["daily_totals_april2026", "daily_totals_may2026"])

    def test_build_plan_expands_statement_period_range_for_trend_questions(self) -> None:
        plan = self.planner.build_plan(
            question="Show the daily trend from December through February",
            skills=[self.daily_skill],
            time_scope=RagTimeScope(
                scope_type="statement_period_range",
                start_period="December2025",
                end_period="February2026",
            ),
            period=None,
            payment_method="Visa",
            account="Checking",
            today=date(2026, 5, 27),
        )

        self.assertEqual(plan.strategy, "multi_scope")
        self.assertEqual([step.period for step in plan.steps], ["December2025", "January2026", "February2026"])
        self.assertEqual([step.payment_method for step in plan.steps], ["Visa", "Visa", "Visa"])
        self.assertEqual([step.account for step in plan.steps], ["Checking", "Checking", "Checking"])

    def test_build_plan_expands_statement_period_range_for_monthly_average_questions(self) -> None:
        categories_skill = StubSkill(skill_id="categories", context_key="categories")

        plan = self.planner.build_plan(
            question="From December to May on average how much do I spend a month on Dining Out and Groceries?",
            skills=[categories_skill, self.range_summary_skill],
            time_scope=RagTimeScope(
                scope_type="statement_period_range",
                start_period="December2025",
                end_period="May2026",
            ),
            period=None,
            payment_method=None,
            account="josh",
            today=date(2026, 5, 27),
        )

        self.assertEqual(plan.strategy, "multi_scope")
        self.assertEqual(plan.steps[0].skill_id, "statement_period_summary_range")
        self.assertEqual(
            [step.period for step in plan.steps[1:]],
            ["December2025", "January2026", "February2026", "March2026", "April2026", "May2026"],
        )
        self.assertEqual([step.skill_id for step in plan.steps[1:]], ["categories"] * 6)

    def test_build_plan_keeps_range_native_skills_on_original_range_scope(self) -> None:
        plan = self.planner.build_plan(
            question="Show the daily trend from December through February",
            skills=[self.daily_skill, self.range_summary_skill],
            time_scope=RagTimeScope(
                scope_type="statement_period_range",
                start_period="December2025",
                end_period="February2026",
            ),
            period=None,
            payment_method=None,
            account=None,
            today=date(2026, 5, 27),
        )

        self.assertEqual(plan.strategy, "multi_scope")
        self.assertEqual(plan.steps[0].skill_id, "statement_period_summary_range")
        self.assertEqual(plan.steps[0].output_key, "statement_period_summary_range")
        self.assertEqual(plan.steps[0].time_scope.scope_type, "statement_period_range")
        self.assertEqual(plan.steps[0].time_scope.start_period, "December2025")
        self.assertEqual(plan.steps[0].time_scope.end_period, "February2026")
        self.assertEqual(
            [step.period for step in plan.steps[1:]],
            ["December2025", "January2026", "February2026"],
        )

    def test_build_plan_keeps_single_scope_when_no_multi_period_signal_exists(self) -> None:
        plan = self.planner.build_plan(
            question="Show daily spend for April",
            skills=[self.daily_skill],
            time_scope=RagTimeScope(scope_type="statement_period", statement_period="April2026"),
            period="April2026",
            payment_method=None,
            account=None,
            today=date(2026, 5, 27),
        )

        self.assertEqual(plan.strategy, "single_scope")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].output_key, "daily_totals")
        self.assertEqual(plan.steps[0].period, "April2026")

    def test_build_plan_expands_statement_period_into_weeks_for_weekly_breakdown_questions(self) -> None:
        weekly_categories_skill = StubSkill(skill_id="top_categories", context_key="top_categories")

        plan = self.planner.build_plan(
            question="What were my highest spending categories per week?",
            skills=[weekly_categories_skill],
            time_scope=RagTimeScope(scope_type="statement_period", statement_period="March2026"),
            period="March2026",
            payment_method=None,
            account=None,
            today=date(2026, 5, 27),
        )

        self.assertEqual(plan.strategy, "multi_scope")
        self.assertEqual(len(plan.steps), 5)
        self.assertEqual([step.skill_id for step in plan.steps], ["top_categories"] * 5)
        self.assertTrue(all(step.time_scope is not None and step.time_scope.scope_type == "date_range" for step in plan.steps))
        self.assertEqual(plan.steps[0].label, "Week 1 (2026-03-01 to 2026-03-07)")
        self.assertEqual(plan.steps[0].time_scope.start_date, date(2026, 3, 1))
        self.assertEqual(plan.steps[0].time_scope.end_date, date(2026, 3, 7))
        self.assertEqual(plan.steps[-1].label, "Week 5 (2026-03-29 to 2026-03-31)")
        self.assertEqual(plan.steps[-1].time_scope.start_date, date(2026, 3, 29))
        self.assertEqual(plan.steps[-1].time_scope.end_date, date(2026, 3, 31))


if __name__ == "__main__":
    unittest.main()

