from __future__ import annotations

import logging
from typing import Any

import requests

from app.skills.base import Skill, SkillRequest

logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self, skills: list[Skill]) -> None:
        self.skills = skills
        self._skills_by_id = {skill.skill_id: skill for skill in skills}

    def select(self, question: str) -> list[Skill]:
        selected_skills = [skill for skill in self.skills if skill.matches(question)]
        if selected_skills:
            return selected_skills
        default_skill = self._skills_by_id.get("overview")
        return [default_skill] if default_skill is not None else []

    def execute(self, skills: list[Skill], request: SkillRequest) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        context_sections: dict[str, Any] = {}
        unavailable_skills: list[dict[str, Any]] = []

        for skill in skills:
            try:
                result = skill.execute(request)
                context_sections[result.context_key] = result.payload
            except requests.RequestException as exc:
                if skill.required:
                    raise
                unavailable_skills.append(self._serialize_request_error(skill.skill_id, exc))
                logger.warning(
                    "Skipping unavailable RAG skill skill=%s context_key=%s error=%s",
                    skill.skill_id,
                    skill.context_key,
                    unavailable_skills[-1],
                )
            except ValueError as exc:
                if skill.required:
                    raise
                unavailable_skills.append(
                    {
                        "tool": skill.skill_id,
                        "error_type": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
                logger.warning(
                    "Skipping invalid RAG skill payload skill=%s context_key=%s detail=%s",
                    skill.skill_id,
                    skill.context_key,
                    exc,
                )

        return context_sections, unavailable_skills

    @staticmethod
    def _serialize_request_error(skill_id: str, exc: requests.RequestException) -> dict[str, Any]:
        error: dict[str, Any] = {
            "tool": skill_id,
            "error_type": type(exc).__name__,
            "detail": str(exc),
        }
        response = getattr(exc, "response", None)
        if response is not None:
            error["status_code"] = response.status_code
            response_text = response.text.strip()
            if response_text:
                error["response_body"] = response_text[:300]
        return error

