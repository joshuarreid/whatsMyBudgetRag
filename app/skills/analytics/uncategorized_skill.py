from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import BudgetTransactionResponse
from app.services.normalizers import normalize_list_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class UncategorizedSkill(Skill):
    definition = SkillDefinition(
        skill_id="uncategorized",
        category="analytics",
        context_key="uncategorized",
        description="Load uncategorized transactions for the selected statement period.",
        keywords=("uncategorized",),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        payload = [
            item.model_dump(mode="json")
            for item in normalize_list_response(
                self.spring.get_uncategorized(
                    period=request.period,
                    transaction_id=request.transaction_id,
                ),
                BudgetTransactionResponse,
            )
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

