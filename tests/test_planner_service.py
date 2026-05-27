from __future__ import annotations

import unittest
from datetime import date

from app.models.schemas import RagTimeScope
from app.services.intent_service import IntentService
from app.services.planner_service import PlannerService
from app.skills.base import Skill, SkillDefinition


class StubSkill(Skill):
    def __init__(self, *, skill_id: str, context_key: str) -> None:
        self.definition = SkillDefinition(
            skill_id=skill_id,
            category="analytics",
            context_key=context_key,
            description=f"Stub skill {skill_id}",
            keywords=(skill_id,),
        )

    def execute(self, request):  # pragma: no cover - planning tests never execute skills
        raise NotImplementedError


class PlannerServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = PlannerService(intent_service=IntentService(enable_llm=False))
        self.daily_skill = StubSkill(skill_id="daily", context_key="daily_totals")

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


if __name__ == "__main__":
    unittest.main()

