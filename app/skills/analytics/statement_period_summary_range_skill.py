from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsStatementPeriodSummaryResponse
from app.services.normalizers import normalize_list_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class StatementPeriodSummaryRangeSkill(Skill):
    definition = SkillDefinition(
        skill_id="statement_period_summary_range",
        category="analytics",
        context_key="statement_period_summary_range",
        description="Load analytics summaries across an inclusive statement period range.",
        keywords=(
            "monthly summaries",
            "summary range",
            "summaries from",
            "summaries between",
            "period range summary",
        ),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        if request.time_scope is None:
            raise ValueError("Statement period summary range skill requires a resolved time_scope")
        if request.time_scope.scope_type != "statement_period_range":
            raise ValueError("Statement period summary range skill requires a statement_period_range time_scope")
        payload = [
            item.model_dump(mode="json")
            for item in normalize_list_response(
                self.spring.get_statement_period_summary_for_time_scope(
                    time_scope=request.time_scope,
                    transaction_id=request.transaction_id,
                ),
                AnalyticsStatementPeriodSummaryResponse,
            )
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)


