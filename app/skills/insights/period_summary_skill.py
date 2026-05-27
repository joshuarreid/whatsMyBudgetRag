from __future__ import annotations

from app.services.insight_service import InsightService
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class PeriodSummarySkill(Skill):
    definition = SkillDefinition(
        skill_id="period_summary",
        category="insights",
        context_key="period_summary",
        description="Build a derived summary for the selected statement period.",
        keywords=("summary", "summarize", "snapshot", "anomaly", "anomalies", "flag", "flags"),
    )

    def __init__(self, insight_service: InsightService) -> None:
        self.insights = insight_service

    def execute(self, request: SkillRequest) -> SkillResult:
        payload = self.insights.period_summary(
            period=request.period,
            time_scope=request.time_scope,
            payment_method=request.payment_method,
            account=request.account,
            transaction_id=request.transaction_id,
        ).model_dump(mode="json")
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

