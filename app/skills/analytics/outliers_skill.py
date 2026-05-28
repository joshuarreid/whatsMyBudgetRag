from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import BudgetTransactionResponse
from app.services.normalizers import normalize_list_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class OutliersSkill(Skill):
    definition = SkillDefinition(
        skill_id="outliers",
        category="analytics",
        context_key="outliers",
        description="Load outlier transactions for the selected statement period.",
        keywords=(
            "outlier",
            "largest",
            "largest transaction",
            "largest transactions",
            "biggest transaction",
            "biggest transactions",
            "largest purchase",
            "biggest purchase",
        ),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        if request.time_scope is None:
            raise ValueError("Outliers skill requires a resolved time_scope")
        payload = [
            item.model_dump(mode="json")
            for item in normalize_list_response(
                self.spring.get_outliers_for_time_scope(
                    time_scope=request.time_scope,
                    transaction_id=request.transaction_id,
                ),
                BudgetTransactionResponse,
            )
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

