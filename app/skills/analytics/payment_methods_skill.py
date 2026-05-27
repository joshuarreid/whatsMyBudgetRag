from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import AnalyticsPaymentMethodBreakdownResponse
from app.services.normalizers import normalize_list_response
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult


class PaymentMethodsSkill(Skill):
    definition = SkillDefinition(
        skill_id="payment_methods",
        category="analytics",
        context_key="payment_methods",
        description="Load the payment method breakdown for the selected statement period.",
        keywords=("payment method", "card", "cash"),
    )

    def __init__(self, spring_client: SpringBootClient) -> None:
        self.spring = spring_client

    def execute(self, request: SkillRequest) -> SkillResult:
        if request.time_scope is None:
            raise ValueError("Payment methods skill requires a resolved time_scope")
        payload = [
            item.model_dump(mode="json")
            for item in normalize_list_response(
                self.spring.get_payment_method_breakdown_for_time_scope(
                    time_scope=request.time_scope,
                    account=request.account,
                    transaction_id=request.transaction_id,
                ),
                AnalyticsPaymentMethodBreakdownResponse,
            )
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

