from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

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
        normalized_items = normalize_list_response(
            self.spring.get_daily_totals_for_time_scope(
                time_scope=request.time_scope,
                payment_method=request.payment_method,
                account=request.account,
                transaction_id=request.transaction_id,
            ),
            AnalyticsDailyTotalResponse,
        )
        payload = [
            item.model_dump(mode="json")
            for item in normalized_items
            if self._is_in_scope(item.date, request.time_scope)
        ]
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=payload)

    @classmethod
    def _is_in_scope(cls, candidate_date: date, time_scope) -> bool:
        if time_scope.scope_type == "statement_period":
            if not time_scope.statement_period:
                return True
            return candidate_date.strftime("%B%Y") == time_scope.statement_period

        if time_scope.scope_type == "statement_period_range":
            start_date = cls._statement_period_start(time_scope.start_period)
            end_date = cls._statement_period_end(time_scope.end_period)
            if start_date is None or end_date is None:
                return True
            return start_date <= candidate_date <= end_date

        if time_scope.scope_type == "date_range":
            if time_scope.start_date is None or time_scope.end_date is None:
                return True
            return time_scope.start_date <= candidate_date <= time_scope.end_date

        return True

    @staticmethod
    def _statement_period_start(statement_period: Optional[str]) -> Optional[date]:
        if not statement_period:
            return None
        try:
            return datetime.strptime(statement_period, "%B%Y").date()
        except ValueError:
            return None

    @classmethod
    def _statement_period_end(cls, statement_period: Optional[str]) -> Optional[date]:
        start_date = cls._statement_period_start(statement_period)
        if start_date is None:
            return None
        next_month = (start_date.replace(day=28) + timedelta(days=4)).replace(day=1)
        return next_month - timedelta(days=1)

