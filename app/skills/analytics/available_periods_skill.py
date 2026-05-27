from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.services.normalizers import normalize_periods_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class AvailablePeriodsSkill(Skill):
    definition = SkillDefinition(
        skill_id="available_periods",
        category="analytics",
        context_key="available_periods",
        description="Load the available statement periods from analytics.",
        keywords=(
            "available periods",
            "statement periods",
            "periods available",
            "list periods",
            "what periods",
            "which periods",
        ),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        payload = normalize_periods_response(
            self.spring.get_periods(transaction_id=request.transaction_id)
        ).model_dump(mode="json")
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

