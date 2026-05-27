from __future__ import annotations

from app.services.analytics_service import AnalyticsService
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class OverviewSkill(Skill):
    definition = SkillDefinition(
        skill_id="overview",
        category="analytics",
        context_key="overview",
        description="Load the high-level spend overview for the selected statement period.",
        keywords=("overview", "summary", "spend", "total"),
        required=True,
    )

    def __init__(self, analytics_service: AnalyticsService) -> None:
        self.analytics = analytics_service

    def execute(self, request: SkillRequest) -> SkillResult:
        payload = self.analytics.period_overview(
            period=request.period,
            time_scope=request.time_scope,
            payment_method=request.payment_method,
            account=request.account,
            transaction_id=request.transaction_id,
        ).model_dump(mode="json")
        return SkillResult(
            skill_id=self.skill_id,
            context_key=self.context_key,
            payload=payload,
        )

