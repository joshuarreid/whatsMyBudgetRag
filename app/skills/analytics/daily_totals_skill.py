from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsDailyTotalResponse
from app.services.normalizers import normalize_list_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class DailyTotalsSkill(Skill):
    definition = SkillDefinition(
        skill_id="daily",
        category="analytics",
        context_key="daily_totals",
        description="Load daily spending totals for the selected statement period.",
        keywords=("daily", "trend", "time series"),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        if request.time_scope is None:
            raise ValueError("Daily totals skill requires a resolved time_scope")
        payload = [
            item.model_dump(mode="json")
            for item in normalize_list_response(
                self.spring.get_daily_totals_for_time_scope(
                    time_scope=request.time_scope,
                    payment_method=request.payment_method,
                    account=request.account,
                    transaction_id=request.transaction_id,
                ),
                AnalyticsDailyTotalResponse,
            )
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

