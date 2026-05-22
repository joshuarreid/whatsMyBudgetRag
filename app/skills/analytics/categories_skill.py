from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsCategoryBreakdownResponse
from app.services.normalizers import normalize_list_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class CategoriesSkill(Skill):
    definition = SkillDefinition(
        skill_id="categories",
        category="analytics",
        context_key="categories",
        description="Load the category spend breakdown for the selected statement period.",
        keywords=("category", "categories"),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        payload = [
            item.model_dump(mode="json")
            for item in normalize_list_response(
                self.spring.get_category_breakdown(
                    period=request.period,
                    payment_method=request.payment_method,
                    account=request.account,
                    transaction_id=request.transaction_id,
                ),
                AnalyticsCategoryBreakdownResponse,
            )
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

