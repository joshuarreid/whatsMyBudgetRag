from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsCriticalityBreakdownResponse
from app.services.normalizers import normalize_list_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class CriticalitySkill(Skill):
    definition = SkillDefinition(
        skill_id="criticality",
        category="analytics",
        context_key="criticality",
        description="Load the criticality breakdown for the selected statement period.",
        keywords=("criticality", "essential", "non-essential"),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        payload = [
            item.model_dump(mode="json")
            for item in normalize_list_response(
                self.spring.get_criticality_breakdown(
                    period=request.period,
                    payment_method=request.payment_method,
                    account=request.account,
                    transaction_id=request.transaction_id,
                ),
                AnalyticsCriticalityBreakdownResponse,
            )
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

