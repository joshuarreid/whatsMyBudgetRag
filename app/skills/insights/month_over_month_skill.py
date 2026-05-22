from __future__ import annotations

from app.services.insight_service import InsightService
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class MonthOverMonthSkill(Skill):
    definition = SkillDefinition(
        skill_id="month_over_month",
        category="insights",
        context_key="month_over_month",
        description="Compare the selected statement period with the prior statement period.",
        keywords=("month over month", "mom", "versus last month", "compare last month"),
    )

    def __init__(self, insight_service: InsightService) -> None:
        self.insights = insight_service

    def execute(self, request: SkillRequest) -> SkillResult:
        payload = self.insights.month_over_month(
            period=request.period,
            payment_method=request.payment_method,
            account=request.account,
            transaction_id=request.transaction_id,
        ).model_dump(mode="json")
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

