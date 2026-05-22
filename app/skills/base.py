from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    category: str
    context_key: str
    description: str
    keywords: tuple[str, ...] = ()
    required: bool = False


@dataclass(frozen=True)
class SkillRequest:
    question: str
    period: str
    payment_method: Optional[str] = None
    account: Optional[str] = None
    transaction_id: Optional[str] = None


@dataclass
class SkillResult:
    skill_id: str
    context_key: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    definition: SkillDefinition

    @property
    def skill_id(self) -> str:
        return self.definition.skill_id

    @property
    def category(self) -> str:
        return self.definition.category

    @property
    def context_key(self) -> str:
        return self.definition.context_key

    @property
    def required(self) -> bool:
        return self.definition.required

    def matches(self, question: str) -> bool:
        lowered = question.lower()
        return any(keyword in lowered for keyword in self.definition.keywords)

    @abstractmethod
    def execute(self, request: SkillRequest) -> SkillResult:
        raise NotImplementedError

