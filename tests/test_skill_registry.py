from __future__ import annotations

import unittest
from typing import Any, Optional
from unittest.mock import Mock

import requests

from app.models.schemas import RagIntentResponse
from app.services.rag_service import RAGService
from app.skills.base import Skill, SkillDefinition, SkillRequest, SkillResult
from app.skills.registry import SkillRegistry


class StubSkill(Skill):
    def __init__(
        self,
        *,
        skill_id: str,
        context_key: str,
        keywords: tuple[str, ...],
        payload: Any = None,
        required: bool = False,
        error: Optional[Exception] = None,
    ) -> None:
        self.definition = SkillDefinition(
            skill_id=skill_id,
            category="test",
            context_key=context_key,
            description=f"Stub skill {skill_id}",
            keywords=keywords,
            required=required,
        )
        self.payload = payload
        self.error = error

    def execute(self, request: SkillRequest) -> SkillResult:
        if self.error is not None:
            raise self.error
        return SkillResult(skill_id=self.skill_id, context_key=self.context_key, payload=self.payload)


class SkillRegistryTests(unittest.TestCase):
    def test_select_returns_matching_skills_in_registration_order(self) -> None:
        registry = SkillRegistry(
            [
                StubSkill(skill_id="overview", context_key="overview", keywords=("spend",), payload={}),
                StubSkill(skill_id="averages", context_key="averages", keywords=("average",), payload={}),
                StubSkill(skill_id="categories", context_key="categories", keywords=("category",), payload=[]),
            ]
        )

        selected = registry.select("What was my average category spend?")

        self.assertEqual([skill.skill_id for skill in selected], ["overview", "averages", "categories"])

    def test_select_falls_back_to_overview(self) -> None:
        registry = SkillRegistry(
            [
                StubSkill(skill_id="overview", context_key="overview", keywords=("spend",), payload={"total": 1}),
                StubSkill(skill_id="daily", context_key="daily_totals", keywords=("daily",), payload=[]),
            ]
        )

        selected = registry.select("Tell me something unexpected")

        self.assertEqual([skill.skill_id for skill in selected], ["overview"])

    def test_available_skills_and_resolve_use_registered_skills_only(self) -> None:
        registry = SkillRegistry(
            [
                StubSkill(skill_id="overview", context_key="overview", keywords=("spend",), payload={}),
                StubSkill(skill_id="averages", context_key="averages", keywords=("average",), payload={}),
            ]
        )

        available_skills = registry.available_skills()
        resolved = registry.resolve(["averages", "unknown", "averages", "overview"])

        self.assertEqual([item["skill_id"] for item in available_skills], ["overview", "averages"])
        self.assertEqual([skill.skill_id for skill in resolved], ["averages", "overview"])

    def test_execute_collects_optional_errors_without_failing(self) -> None:
        response = Mock(status_code=503, text="service unavailable")
        request_error = requests.HTTPError("boom")
        request_error.response = response
        registry = SkillRegistry(
            [
                StubSkill(skill_id="overview", context_key="overview", keywords=("spend",), payload={"total": 1}),
                StubSkill(
                    skill_id="categories",
                    context_key="categories",
                    keywords=("category",),
                    error=request_error,
                ),
            ]
        )

        context_sections, unavailable, tool_traces = registry.execute(
            registry.skills,
            SkillRequest(question="test", period="May2026"),
        )

        self.assertEqual(context_sections["overview"], {"total": 1})
        self.assertEqual(unavailable[0]["tool"], "categories")
        self.assertEqual(unavailable[0]["status_code"], 503)
        self.assertEqual(tool_traces[0]["tool_name"], "overview")
        self.assertEqual(tool_traces[1]["status"], "error")

    def test_execute_raises_for_required_skills(self) -> None:
        registry = SkillRegistry(
            [
                StubSkill(
                    skill_id="overview",
                    context_key="overview",
                    keywords=("spend",),
                    required=True,
                    error=ValueError("missing overview"),
                )
            ]
        )

        with self.assertRaises(ValueError):
            registry.execute(registry.skills, SkillRequest(question="test", period="May2026"))

    def test_execute_uses_cache_lookup_for_cacheable_skills(self) -> None:
        registry = SkillRegistry(
            [
                StubSkill(skill_id="overview", context_key="overview", keywords=("spend",), payload={"total": 1}),
            ]
        )

        context_sections, unavailable, tool_traces = registry.execute(
            registry.skills,
            SkillRequest(question="test", period="May2026"),
            cache_lookup=lambda skill, arguments: {
                "context_key": skill.context_key,
                "payload": {"total": 99},
                "metadata": {},
                "created_at": "2026-05-23T00:00:00+00:00",
                "expires_at": "2026-05-23T00:15:00+00:00",
            },
        )

        self.assertEqual(context_sections["overview"], {"total": 99})
        self.assertEqual(unavailable, [])
        self.assertTrue(tool_traces[0]["cache_hit"])


class RAGServiceSkillIntegrationTests(unittest.TestCase):
    def test_answer_includes_skill_outputs_and_plan(self) -> None:
        registry = SkillRegistry(
            [
                StubSkill(
                    skill_id="overview",
                    context_key="overview",
                    keywords=("spend",),
                    payload={"total_amount": "100.00", "transaction_count": 4},
                    required=True,
                ),
                StubSkill(
                    skill_id="averages",
                    context_key="averages",
                    keywords=("average",),
                    payload={"average_transaction_amount": "25.00"},
                ),
            ]
        )
        service = RAGService(Mock(), registry, None)

        response = service.answer(
            question="What was my average spend this month?",
            period="May2026",
        )

        self.assertEqual(response.plan, ["overview", "averages"])
        self.assertEqual(response.tool_selection.llm_suggested_tools, [])
        self.assertEqual(response.tool_selection.deterministic_tools, ["overview", "averages"])
        self.assertEqual(response.tool_selection.union_tools, ["overview", "averages"])
        self.assertEqual(response.context["skills"]["selected"], ["overview", "averages"])
        self.assertEqual(response.context["overview"]["total_amount"], "100.00")
        self.assertEqual(response.context["averages"]["average_transaction_amount"], "25.00")
        self.assertEqual(response.citations[0].source_type, "api")
        self.assertEqual(response.tool_traces[0].tool_name, "overview")
        self.assertFalse(response.tool_traces[0].cache_hit)

    def test_answer_prefers_llm_intent_when_it_resolves_to_registered_skills(self) -> None:
        registry = SkillRegistry(
            [
                StubSkill(
                    skill_id="overview",
                    context_key="overview",
                    keywords=("spend",),
                    payload={"total_amount": "100.00", "transaction_count": 4},
                    required=True,
                ),
                StubSkill(
                    skill_id="averages",
                    context_key="averages",
                    keywords=("average",),
                    payload={"average_transaction_amount": "25.00"},
                ),
            ]
        )
        llm = Mock()
        llm.classify_intent.return_value = RagIntentResponse(
            skill_ids=["averages"],
            time_reference="this month",
            confidence=0.92,
        )
        llm.generate_answer.return_value = None
        service = RAGService(Mock(), registry, llm)

        response = service.answer(
            question="Give me the gist of this month",
            period="May2026",
        )

        self.assertEqual(response.plan, ["averages"])
        self.assertEqual(response.tool_selection.llm_suggested_tools, ["averages"])
        self.assertEqual(response.tool_selection.deterministic_tools, ["overview"])
        self.assertEqual(response.tool_selection.union_tools, ["averages", "overview"])
        self.assertEqual(response.context["routing"]["source"], "llm_intent")
        self.assertEqual(response.context["routing"]["llm_intent"]["skill_ids"], ["averages"])

    def test_answer_falls_back_to_keyword_selection_when_llm_intent_is_invalid(self) -> None:
        registry = SkillRegistry(
            [
                StubSkill(
                    skill_id="overview",
                    context_key="overview",
                    keywords=("spend",),
                    payload={"total_amount": "100.00", "transaction_count": 4},
                    required=True,
                ),
                StubSkill(
                    skill_id="averages",
                    context_key="averages",
                    keywords=("average",),
                    payload={"average_transaction_amount": "25.00"},
                ),
            ]
        )
        llm = Mock()
        llm.classify_intent.return_value = RagIntentResponse(skill_ids=["does_not_exist"], confidence=0.6)
        llm.generate_answer.return_value = None
        service = RAGService(Mock(), registry, llm)

        response = service.answer(
            question="What was my average spend this month?",
            period="May2026",
        )

        self.assertEqual(response.plan, ["overview", "averages"])
        self.assertEqual(response.tool_selection.llm_suggested_tools, [])
        self.assertEqual(response.tool_selection.deterministic_tools, ["overview", "averages"])
        self.assertEqual(response.tool_selection.union_tools, ["overview", "averages"])
        self.assertEqual(response.context["routing"]["source"], "keyword_fallback")
        self.assertEqual(response.context["routing"]["resolved_skill_ids"], ["overview", "averages"])


if __name__ == "__main__":
    unittest.main()


