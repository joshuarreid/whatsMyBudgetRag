from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsDuplicateResponse
from app.services.normalizers import normalize_list_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class DuplicatesSkill(Skill):
    definition = SkillDefinition(
        skill_id="duplicates",
        category="analytics",
        context_key="duplicates",
        description="Load duplicate transaction candidates for the selected statement period.",
        keywords=("duplicate",),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        payload = [
            item.model_dump(mode="json")
            for item in normalize_list_response(
                self.spring.get_duplicates(
                    period=request.period,
                    transaction_id=request.transaction_id,
                ),
                AnalyticsDuplicateResponse,
            )
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

