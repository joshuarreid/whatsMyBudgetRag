from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Optional

from app.models.schemas import RagExecutionPlan, RagPlanStep, RagTimeScope
from app.services.intent_service import IntentService, STATEMENT_PERIOD_MONTH_PATTERN
from app.skills.base import Skill

COMPARISON_KEYWORD_PATTERN = re.compile(r"\b(?:compare|comparison|versus|vs\.?|against)\b", re.IGNORECASE)
EXPLICIT_PERIOD_PAIR_PATTERN = re.compile(
    rf"\b(?P<left>{STATEMENT_PERIOD_MONTH_PATTERN})(?:\s+(?P<left_year>\d{{4}}))?\s+"
    rf"(?:vs\.?|versus|against)\s+"
    rf"(?P<right>{STATEMENT_PERIOD_MONTH_PATTERN})(?:\s+(?P<right_year>\d{{4}}))?\b",
    re.IGNORECASE,
)
COMPARISON_AND_PAIR_PATTERN = re.compile(
    rf"\bcompare\b.*?\b(?P<left>{STATEMENT_PERIOD_MONTH_PATTERN})(?:\s+(?P<left_year>\d{{4}}))?\s+"
    rf"(?:and|to|with)\s+"
    rf"(?P<right>{STATEMENT_PERIOD_MONTH_PATTERN})(?:\s+(?P<right_year>\d{{4}}))?\b",
    re.IGNORECASE,
)
MAX_MULTI_PERIOD_STEPS = 24


@dataclass(frozen=True)
class _PlannedScope:
    label: str
    time_scope: RagTimeScope


class PlannerService:
    """Builds deterministic execution plans that can repeat the same skill across multiple scopes."""

    def __init__(self, *, intent_service: Optional[IntentService] = None) -> None:
        self.intent_service = intent_service or IntentService(enable_llm=False)

    def build_plan(
        self,
        *,
        question: str,
        skills: list[Skill],
        time_scope: RagTimeScope,
        period: Optional[str],
        payment_method: Optional[str],
        account: Optional[str],
        today: Optional[date] = None,
    ) -> RagExecutionPlan:
        reference_date = today or date.today()
        planned_scopes = self._planned_scopes(question=question, time_scope=time_scope, today=reference_date)
        if len(planned_scopes) <= 1:
            return RagExecutionPlan(
                strategy="single_scope",
                steps=[
                    RagPlanStep(
                        step_id=skill.skill_id,
                        skill_id=skill.skill_id,
                        context_key=skill.context_key,
                        output_key=skill.context_key,
                        time_scope=time_scope,
                        period=period,
                        payment_method=payment_method,
                        account=account,
                    )
                    for skill in skills
                ],
            )

        steps: list[RagPlanStep] = []
        seen_output_keys: set[str] = set()
        original_scope_label = self._scope_label(time_scope)
        if time_scope.scope_type == "statement_period_range":
            for skill in skills:
                if skill.expand_with_multi_scope:
                    continue
                output_key = self._unique_output_key(
                    base_key=skill.context_key,
                    seen_output_keys=seen_output_keys,
                )
                steps.append(
                    RagPlanStep(
                        step_id=f"{skill.skill_id}_{self._slugify(original_scope_label)}",
                        skill_id=skill.skill_id,
                        context_key=skill.context_key,
                        output_key=output_key,
                        label=original_scope_label,
                        time_scope=time_scope,
                        period=period,
                        payment_method=payment_method,
                        account=account,
                    )
                )
        for planned_scope in planned_scopes:
            step_period = self._derive_period(planned_scope.time_scope)
            step_label = planned_scope.label
            for skill in skills:
                if time_scope.scope_type == "statement_period_range" and not skill.expand_with_multi_scope:
                    continue
                output_key = self._unique_output_key(
                    base_key=f"{skill.context_key}_{self._slugify(step_label)}",
                    seen_output_keys=seen_output_keys,
                )
                steps.append(
                    RagPlanStep(
                        step_id=f"{skill.skill_id}_{self._slugify(step_label)}",
                        skill_id=skill.skill_id,
                        context_key=skill.context_key,
                        output_key=output_key,
                        label=step_label,
                        time_scope=planned_scope.time_scope,
                        period=step_period,
                        payment_method=payment_method,
                        account=account,
                    )
                )
        return RagExecutionPlan(strategy="multi_scope", steps=steps)

    def _planned_scopes(
        self,
        *,
        question: str,
        time_scope: RagTimeScope,
        today: date,
    ) -> list[_PlannedScope]:
        comparison_scopes = self._explicit_comparison_scopes(question=question, today=today)
        if comparison_scopes:
            return comparison_scopes[:MAX_MULTI_PERIOD_STEPS]

        if time_scope.scope_type == "statement_period_range" and self._requires_period_expansion(question):
            return self._expand_statement_period_range(time_scope)[:MAX_MULTI_PERIOD_STEPS]

        return [
            _PlannedScope(
                label=self._scope_label(time_scope),
                time_scope=time_scope,
            )
        ]

    def _explicit_comparison_scopes(self, *, question: str, today: date) -> list[_PlannedScope]:
        for pattern in (EXPLICIT_PERIOD_PAIR_PATTERN, COMPARISON_AND_PAIR_PATTERN):
            match = pattern.search(question)
            if match is None:
                continue
            left_reference = self.intent_service.resolve_month_reference(
                month_text=match.group("left"),
                today=today,
                year_text=match.group("left_year"),
            )
            right_reference = self.intent_service.resolve_month_reference(
                month_text=match.group("right"),
                today=today,
                year_text=match.group("right_year"),
            )
            left_period = self.intent_service.format_statement_period(left_reference)
            right_period = self.intent_service.format_statement_period(right_reference)
            return [
                _PlannedScope(
                    label=left_period,
                    time_scope=RagTimeScope.from_period(left_period),
                ),
                _PlannedScope(
                    label=right_period,
                    time_scope=RagTimeScope.from_period(right_period),
                ),
            ]
        return []

    @staticmethod
    def _requires_period_expansion(question: str) -> bool:
        lowered = question.lower()
        return bool(COMPARISON_KEYWORD_PATTERN.search(question)) or any(
            phrase in lowered
            for phrase in (
                "daily",
                "trend",
                "over time",
                "each month",
                "per month",
            )
        )

    def _expand_statement_period_range(self, time_scope: RagTimeScope) -> list[_PlannedScope]:
        if time_scope.scope_type != "statement_period_range":
            return []

        assert time_scope.start_period is not None
        assert time_scope.end_period is not None
        start_reference = RagTimeScope.parse_statement_period(time_scope.start_period)
        end_reference = RagTimeScope.parse_statement_period(time_scope.end_period)

        scopes: list[_PlannedScope] = []
        current = start_reference
        while current <= end_reference and len(scopes) < MAX_MULTI_PERIOD_STEPS:
            statement_period = self.intent_service.format_statement_period(current)
            scopes.append(
                _PlannedScope(
                    label=statement_period,
                    time_scope=RagTimeScope.from_period(statement_period),
                )
            )
            current = self.intent_service.shift_month(current, offset=1)
        return scopes

    @staticmethod
    def _derive_period(time_scope: RagTimeScope) -> Optional[str]:
        return time_scope.derived_period

    @staticmethod
    def _scope_label(time_scope: RagTimeScope) -> str:
        return time_scope.label.replace(" through ", "_through_")

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "step"

    @staticmethod
    def _unique_output_key(*, base_key: str, seen_output_keys: set[str]) -> str:
        if base_key not in seen_output_keys:
            seen_output_keys.add(base_key)
            return base_key
        suffix = 2
        while f"{base_key}_{suffix}" in seen_output_keys:
            suffix += 1
        output_key = f"{base_key}_{suffix}"
        seen_output_keys.add(output_key)
        return output_key

