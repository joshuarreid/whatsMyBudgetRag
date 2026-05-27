from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsStatementPeriodSummaryResponse
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class StatementPeriodSummarySkill(Skill):
    definition = SkillDefinition(
        skill_id="statement_period_summary",
        category="analytics",
        context_key="statement_period_summary",
        description="Load the persisted or live analytics summary for a single statement period.",
        keywords=(
            "full summary",
            "statement period summary",
            "summary for period",
            "period archive",
            "archived summary",
        ),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        if request.time_scope is None:
            raise ValueError("Statement period summary skill requires a resolved time_scope")
        if request.time_scope.scope_type != "statement_period" or not request.time_scope.statement_period:
            raise ValueError("Statement period summary skill requires a statement_period time_scope")
        payload = AnalyticsStatementPeriodSummaryResponse.model_validate(
            self.spring.get_statement_period_summary_for_time_scope(
                time_scope=request.time_scope,
                transaction_id=request.transaction_id,
            )
        ).model_dump(mode="json")
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

