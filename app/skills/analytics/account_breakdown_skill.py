from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsAccountBreakdownResponse
from app.services.normalizers import normalize_list_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class AccountBreakdownSkill(Skill):
    definition = SkillDefinition(
        skill_id="account_breakdown",
        category="analytics",
        context_key="account_breakdown",
        description="Load the account breakdown for the selected statement period.",
        keywords=("account", "accounts", "account breakdown", "breakdown", "which account", "which accounts"),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        if request.time_scope is None:
            raise ValueError("Account breakdown skill requires a resolved time_scope")
        payload = [
            item.model_dump(mode="json")
            for item in normalize_list_response(
                self.spring.get_account_breakdown_for_time_scope(
                    time_scope=request.time_scope,
                    payment_method=request.payment_method,
                    transaction_id=request.transaction_id,
                ),
                AnalyticsAccountBreakdownResponse,
            )
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

