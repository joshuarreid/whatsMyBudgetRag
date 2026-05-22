from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsCategoryBreakdownResponse
from app.services.normalizers import normalize_list_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class TopCategoriesSkill(Skill):
    definition = SkillDefinition(
        skill_id="top_categories",
        category="analytics",
        context_key="top_categories",
        description="Load the top spending categories for the selected statement period.",
        keywords=("top category", "top categories"),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        payload = [
            item.model_dump(mode="json")
            for item in normalize_list_response(
                self.spring.get_top_categories(
                    period=request.period,
                    payment_method=request.payment_method,
                    account=request.account,
                    transaction_id=request.transaction_id,
                ),
                AnalyticsCategoryBreakdownResponse,
            )
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

