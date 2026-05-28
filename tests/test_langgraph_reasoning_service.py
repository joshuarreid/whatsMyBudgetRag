from __future__ import annotations

import unittest

from app.models.schemas import RagTimeScope
from app.services.langgraph_reasoning_service import LangGraphReasoningService
from app.skills.base import Skill, SkillDefinition


class StubSkill(Skill):
    def __init__(self, skill_id: str) -> None:
        self.definition = SkillDefinition(
            skill_id=skill_id,
            category="analytics",
            context_key=skill_id,
            description=f"{skill_id} skill",
            keywords=(skill_id,),
        )

    def execute(self, request):  # pragma: no cover - not needed for graph selection tests
        raise NotImplementedError


class LangGraphReasoningServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skills = [
            StubSkill("overview"),
            StubSkill("available_periods"),
            StubSkill("statement_period_summary"),
            StubSkill("statement_period_summary_range"),
            StubSkill("categories"),
            StubSkill("top_categories"),
            StubSkill("account_breakdown"),
            StubSkill("payment_methods"),
            StubSkill("criticality"),
            StubSkill("daily"),
            StubSkill("duplicates"),
            StubSkill("outliers"),
            StubSkill("uncategorized"),
            StubSkill("averages"),
            StubSkill("month_over_month"),
            StubSkill("period_summary"),
            StubSkill("behavior_summary"),
        ]

    def test_comparison_diagnostics_expand_into_baseline_driver_and_narrative_families(self) -> None:
        service = LangGraphReasoningService(enabled=True)

        selected_skills, metadata = service.plan(
            question="Why did my spending jump in April versus March?",
            time_scope=RagTimeScope(
                scope_type="statement_period_range",
                start_period="March2026",
                end_period="April2026",
            ),
            available_skills=self.skills,
            seed_skills=[],
        )

        selected_skill_ids = [skill.skill_id for skill in selected_skills]
        self.assertIn("statement_period_summary_range", selected_skill_ids)
        self.assertIn("top_categories", selected_skill_ids)
        self.assertIn("account_breakdown", selected_skill_ids)
        self.assertIn("payment_methods", selected_skill_ids)
        self.assertIn("outliers", selected_skill_ids)
        self.assertIn("month_over_month", selected_skill_ids)
        self.assertEqual(
            metadata["selected_families"],
            ["baseline_summary", "driver_analysis", "pattern_anomaly", "derived_narrative"],
        )
        self.assertIn("finalize_selection", metadata["graph_trace"][-1])

    def test_anomaly_and_data_quality_questions_select_pattern_skills(self) -> None:
        service = LangGraphReasoningService(enabled=True)

        selected_skills, metadata = service.plan(
            question="Show my daily trend, highlight outliers, duplicates, and uncategorized transactions.",
            time_scope=RagTimeScope(scope_type="statement_period", statement_period="May2026"),
            available_skills=self.skills,
            seed_skills=[StubSkill("overview")],
        )

        selected_skill_ids = [skill.skill_id for skill in selected_skills]
        self.assertEqual(selected_skill_ids[0], "overview")
        self.assertIn("daily", selected_skill_ids)
        self.assertIn("outliers", selected_skill_ids)
        self.assertIn("duplicates", selected_skill_ids)
        self.assertIn("uncategorized", selected_skill_ids)
        self.assertIn("pattern_anomaly", metadata["selected_families"])

    def test_assess_question_detects_overlapping_financial_intents_from_natural_language(self) -> None:
        service = LangGraphReasoningService(enabled=True)

        assessment = service._assess_question(
            {
                "question": "Why did my spend jump last month vs the month before, and what drove my expenses?",
                "time_scope": RagTimeScope(scope_type="statement_period", statement_period="May2026"),
            }
        )

        intents = assessment["intents"]
        self.assertTrue(intents["comparison"])
        self.assertTrue(intents["diagnostic"])
        self.assertTrue(intents["anomaly"])

    def test_assess_question_detects_where_did_my_money_go_as_diagnostic(self) -> None:
        service = LangGraphReasoningService(enabled=True)

        assessment = service._assess_question(
            {
                "question": "Where did my money go last month?",
                "time_scope": RagTimeScope(scope_type="statement_period", statement_period="May2026"),
            }
        )

        intents = assessment["intents"]
        self.assertTrue(intents["diagnostic"])
        self.assertFalse(intents["comparison"])
        self.assertFalse(intents["available_periods"])

    def test_assess_question_does_not_trigger_summary_or_diagnostic_from_focus_alone(self) -> None:
        service = LangGraphReasoningService(enabled=True)

        assessment = service._assess_question(
            {
                "question": "Focus on groceries this month.",
                "time_scope": RagTimeScope(scope_type="statement_period", statement_period="May2026"),
            }
        )

        intents = assessment["intents"]
        self.assertFalse(intents["summary"])
        self.assertFalse(intents["diagnostic"])

    def test_assess_question_does_not_treat_past_as_available_periods_request(self) -> None:
        service = LangGraphReasoningService(enabled=True)

        assessment = service._assess_question(
            {
                "question": "Show my past spending trend.",
                "time_scope": RagTimeScope(scope_type="statement_period", statement_period="May2026"),
            }
        )

        intents = assessment["intents"]
        self.assertTrue(intents["trend"])
        self.assertFalse(intents["available_periods"])


if __name__ == "__main__":
    unittest.main()

