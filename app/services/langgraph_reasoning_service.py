from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, TypedDict

from app.models.schemas import RagTimeScope
from app.skills.base import Skill

try:  # pragma: no cover - exercised indirectly when dependency is installed
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - graceful fallback when optional dependency is unavailable
    END = START = StateGraph = None

logger = logging.getLogger(__name__)

BASELINE_FAMILY = "baseline_summary"
DRIVER_FAMILY = "driver_analysis"
PATTERN_FAMILY = "pattern_anomaly"
NARRATIVE_FAMILY = "derived_narrative"
GRAPH_VERSION = "skill_reasoning_v1"

_NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class IntentPattern:
    threshold: int = 2
    strong_phrases: tuple[str, ...] = ()
    supporting_phrases: tuple[str, ...] = ()
    token_groups: tuple[tuple[str, ...], ...] = ()
    regexes: tuple[str, ...] = ()


INTENT_PATTERNS: dict[str, IntentPattern] = {
    "comparison": IntentPattern(
        strong_phrases=(
            "compare",
            "comparison",
            "compared to",
            "compare to",
            "compare with",
            "versus",
            "vs",
            "against",
            "difference between",
            "month over month",
            "year over year",
            "relative to",
            "side by side",
            "what changed",
            "what has changed",
            "what s changed",
        ),
        supporting_phrases=(
            "changed from",
            "change from",
            "changed versus",
            "changed vs",
            "different from",
        ),
        regexes=(
            r"\b(?:how|what)\s+did\b.*\bcompare\b",
            r"\b(?:this|that|current)\s+(?:month|period|week|year)\s+(?:vs|versus|against)\s+(?:last|previous|prior)\s+(?:month|period|week|year)\b",
            r"\b(?:last|previous|prior)\s+(?:month|period|week|year)\s+(?:vs|versus|against)\s+(?:this|current)\s+(?:month|period|week|year)\b",
        ),
    ),
    "diagnostic": IntentPattern(
        threshold=2,
        strong_phrases=(
            "why did",
            "why was",
            "why were",
            "what caused",
            "what s causing",
            "what is causing",
            "what drove",
            "what s driving",
            "what is driving",
            "what contributed",
            "where did my money go",
            "help me understand",
            "explain why",
            "break down why",
            "what made",
            "what happened to my spending",
        ),
        supporting_phrases=(
            "drivers of",
            "driver of",
            "spend drivers",
            "expense drivers",
            "spending drivers",
            "drove my expenses",
            "drove my spending",
        ),
        token_groups=(
            ("drivers", "spend"),
            ("drivers", "spending"),
            ("drivers", "expenses"),
            ("drove", "expenses"),
            ("drove", "spending"),
            ("money", "go"),
        ),
    ),
    "trend": IntentPattern(
        strong_phrases=(
            "daily",
            "daily spending",
            "daily spend",
            "trend",
            "over time",
            "time series",
            "day by day",
            "week by week",
            "month by month",
        ),
    ),
    "anomaly": IntentPattern(
        strong_phrases=(
            "spike",
            "spiked",
            "jump",
            "jumped",
            "surge",
            "surged",
            "shot up",
            "shoot up",
            "drop",
            "dropped",
            "dip",
            "outlier",
            "outliers",
            "anomaly",
            "anomalies",
            "unusual",
            "odd",
            "unexpected",
            "weird",
            "higher than usual",
            "lower than usual",
        ),
        supporting_phrases=(
            "way up",
            "way down",
            "far above normal",
            "far below normal",
            "off pattern",
        ),
    ),
    "summary": IntentPattern(
        strong_phrases=(
            "summary",
            "overview",
            "summarize",
            "insight",
            "insights",
            "recap",
            "big picture",
            "high level",
            "high level summary",
            "give me the gist",
            "how am i doing",
            "where do i stand",
            "spending patterns",
            "behavior summary",
            "behaviour summary",
        ),
        supporting_phrases=(
            "behavior",
            "behaviour",
            "patterns do you see",
            "standout takeaways",
        ),
    ),
    "average": IntentPattern(
        strong_phrases=(
            "average",
            "averages",
            "avg",
            "on average",
            "typical",
            "ticket size",
            "frequency",
            "average transaction",
            "average spend",
            "average spending",
            "how often",
        ),
    ),
    "available_periods": IntentPattern(
        strong_phrases=(
            "available periods",
            "what periods",
            "which periods",
            "which months",
            "what months",
            "available months",
            "historical periods",
            "historical months",
            "available history",
            "how far back",
            "what date range",
            "what date ranges",
        ),
        regexes=(
            r"\b(?:what|which)\s+(?:months|periods|date ranges?)\b",
            r"\bhow\s+far\s+back\b",
        ),
    ),
    "category_focus": IntentPattern(
        strong_phrases=("category", "categories", "categorized", "categorization"),
    ),
    "account_focus": IntentPattern(
        strong_phrases=("account", "accounts", "account breakdown", "by account", "which account"),
    ),
    "payment_focus": IntentPattern(
        strong_phrases=(
            "payment method",
            "payment methods",
            "credit card",
            "credit cards",
            "debit card",
            "debit cards",
            "cash spending",
            "cash purchases",
            "card spending",
            "by payment method",
        ),
        supporting_phrases=("cash vs card", "card vs cash", "how did i pay"),
    ),
    "criticality_focus": IntentPattern(
        strong_phrases=(
            "criticality",
            "essential",
            "essentials",
            "nonessential",
            "nonessentials",
            "non essential",
            "non essentials",
            "non essential spending",
            "non essential spend",
            "necessity",
            "necessities",
            "needs vs wants",
            "wants vs needs",
            "discretionary",
        ),
    ),
    "duplicates_focus": IntentPattern(
        strong_phrases=(
            "duplicate",
            "duplicates",
            "duplicate transaction",
            "duplicate transactions",
            "duplicated charge",
            "duplicated charges",
            "double charged",
            "charged twice",
        ),
    ),
    "uncategorized_focus": IntentPattern(
        strong_phrases=(
            "uncategorized",
            "without a category",
            "without categories",
            "no category",
            "no categories",
            "missing category",
            "missing categories",
        ),
    ),
    "broad_follow_up": IntentPattern(
        strong_phrases=(
            "what changed",
            "what stands out",
            "what stood out",
            "what should i focus on",
            "where should i focus",
            "what should i look at",
            "what should i pay attention to",
            "anything stand out",
        ),
        supporting_phrases=("stand out",),
    ),
}


class _ReasoningState(TypedDict, total=False):
    question: str
    time_scope: RagTimeScope
    available_skill_ids: list[str]
    seed_skill_ids: list[str]
    selected_skill_ids: list[str]
    selected_families: list[str]
    intents: dict[str, bool]
    graph_trace: list[str]


class LangGraphReasoningService:
    """Selects skill families with a lightweight LangGraph state machine."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled) and StateGraph is not None
        if enabled and StateGraph is None:
            logger.warning("LangGraph reasoning requested but langgraph is not installed; falling back to legacy routing")
        self._graph = self._build_graph() if self.enabled else None

    def plan(
        self,
        *,
        question: str,
        time_scope: RagTimeScope,
        available_skills: list[Skill],
        seed_skills: list[Skill],
    ) -> tuple[list[Skill], dict[str, Any]]:
        if not self.enabled or self._graph is None:
            return seed_skills, {
                "enabled": False,
                "applied": False,
                "reason": "langgraph_unavailable",
                "selected_skill_ids": [skill.skill_id for skill in seed_skills],
                "selected_families": [],
                "graph_trace": [],
                "graph_version": GRAPH_VERSION,
            }

        state = self._graph.invoke(
            {
                "question": question,
                "time_scope": time_scope,
                "available_skill_ids": [skill.skill_id for skill in available_skills],
                "seed_skill_ids": [skill.skill_id for skill in seed_skills],
                "selected_skill_ids": [skill.skill_id for skill in seed_skills],
                "selected_families": [],
                "graph_trace": [],
            }
        )
        selected_skill_ids = self._unique(state.get("selected_skill_ids", []))
        resolved_skills = self._resolve_skills(selected_skill_ids, available_skills)
        if not resolved_skills:
            resolved_skills = seed_skills
            selected_skill_ids = [skill.skill_id for skill in resolved_skills]
        return resolved_skills, {
            "enabled": True,
            "applied": True,
            "reason": None,
            "selected_skill_ids": selected_skill_ids,
            "selected_families": state.get("selected_families", []),
            "graph_trace": state.get("graph_trace", []),
            "seed_skill_ids": [skill.skill_id for skill in seed_skills],
            "graph_version": GRAPH_VERSION,
        }

    def _build_graph(self):
        if StateGraph is None:  # pragma: no cover - constructor already disables this path without langgraph installed
            raise RuntimeError("LangGraph reasoning is unavailable because langgraph is not installed")
        graph = StateGraph(_ReasoningState)
        graph.add_node("assess_question", self._assess_question)
        graph.add_node("select_baseline", self._select_baseline)
        graph.add_node("expand_reasoning_families", self._expand_reasoning_families)
        graph.add_node("finalize_selection", self._finalize_selection)
        graph.add_edge(START, "assess_question")
        graph.add_edge("assess_question", "select_baseline")
        graph.add_edge("select_baseline", "expand_reasoning_families")
        graph.add_edge("expand_reasoning_families", "finalize_selection")
        graph.add_edge("finalize_selection", END)
        return graph.compile()

    def _assess_question(self, state: _ReasoningState) -> dict[str, Any]:
        question = self._normalize_text(state["question"])
        intents = {intent_name: self._matches_intent(question, pattern) for intent_name, pattern in INTENT_PATTERNS.items()}
        active_intents = sorted(intent for intent, enabled in intents.items() if enabled)
        return {
            "intents": intents,
            "graph_trace": [
                *state.get("graph_trace", []),
                f"assess_question:{','.join(active_intents) if active_intents else 'none'}",
            ],
        }

    def _select_baseline(self, state: _ReasoningState) -> dict[str, Any]:
        intents = state.get("intents", {})
        time_scope = state["time_scope"]
        available_skill_ids = set(state.get("available_skill_ids", []))
        baseline_skill_ids: list[str] = []
        selected_families = list(state.get("selected_families", []))

        if intents.get("available_periods") and "available_periods" in available_skill_ids:
            baseline_skill_ids.append("available_periods")

        if time_scope.scope_type == "statement_period_range" or intents.get("comparison"):
            if "statement_period_summary_range" in available_skill_ids:
                baseline_skill_ids.append("statement_period_summary_range")
            elif "overview" in available_skill_ids:
                baseline_skill_ids.append("overview")
        elif intents.get("diagnostic") or intents.get("summary") or intents.get("broad_follow_up"):
            if "statement_period_summary" in available_skill_ids:
                baseline_skill_ids.append("statement_period_summary")
            if "overview" in available_skill_ids:
                baseline_skill_ids.append("overview")
        elif not state.get("seed_skill_ids") and "overview" in available_skill_ids:
            baseline_skill_ids.append("overview")

        if baseline_skill_ids:
            selected_families = self._append_unique(selected_families, BASELINE_FAMILY)

        selected_skill_ids = self._append_unique(state.get("selected_skill_ids", []), *baseline_skill_ids)
        return {
            "selected_skill_ids": selected_skill_ids,
            "selected_families": selected_families,
            "graph_trace": [
                *state.get("graph_trace", []),
                f"select_baseline:{','.join(baseline_skill_ids) if baseline_skill_ids else 'none'}",
            ],
        }

    def _expand_reasoning_families(self, state: _ReasoningState) -> dict[str, Any]:
        intents = state.get("intents", {})
        available_skill_ids = set(state.get("available_skill_ids", []))
        selected_families = list(state.get("selected_families", []))
        selected_skill_ids = list(state.get("selected_skill_ids", []))
        graph_trace = list(state.get("graph_trace", []))

        if intents.get("comparison") or intents.get("diagnostic") or intents.get("category_focus") or intents.get("account_focus") or intents.get("payment_focus") or intents.get("criticality_focus"):
            driver_skill_ids = self._driver_skill_ids(intents=intents, available_skill_ids=available_skill_ids)
            if driver_skill_ids:
                selected_skill_ids = self._append_unique(selected_skill_ids, *driver_skill_ids)
                selected_families = self._append_unique(selected_families, DRIVER_FAMILY)
                graph_trace.append(f"expand_reasoning_families:{DRIVER_FAMILY}={','.join(driver_skill_ids)}")

        if intents.get("trend") or intents.get("anomaly") or intents.get("duplicates_focus") or intents.get("uncategorized_focus"):
            pattern_skill_ids = self._pattern_skill_ids(intents=intents, available_skill_ids=available_skill_ids)
            if pattern_skill_ids:
                selected_skill_ids = self._append_unique(selected_skill_ids, *pattern_skill_ids)
                selected_families = self._append_unique(selected_families, PATTERN_FAMILY)
                graph_trace.append(f"expand_reasoning_families:{PATTERN_FAMILY}={','.join(pattern_skill_ids)}")

        if intents.get("comparison") or intents.get("summary") or intents.get("diagnostic") or intents.get("average") or intents.get("broad_follow_up"):
            narrative_skill_ids = self._narrative_skill_ids(intents=intents, available_skill_ids=available_skill_ids)
            if narrative_skill_ids:
                selected_skill_ids = self._append_unique(selected_skill_ids, *narrative_skill_ids)
                selected_families = self._append_unique(selected_families, NARRATIVE_FAMILY)
                graph_trace.append(f"expand_reasoning_families:{NARRATIVE_FAMILY}={','.join(narrative_skill_ids)}")

        return {
            "selected_skill_ids": selected_skill_ids,
            "selected_families": selected_families,
            "graph_trace": graph_trace,
        }

    def _finalize_selection(self, state: _ReasoningState) -> dict[str, Any]:
        available_skill_ids = set(state.get("available_skill_ids", []))
        selected_skill_ids = [skill_id for skill_id in self._unique(state.get("selected_skill_ids", [])) if skill_id in available_skill_ids]
        return {
            "selected_skill_ids": selected_skill_ids,
            "graph_trace": [
                *state.get("graph_trace", []),
                f"finalize_selection:{','.join(selected_skill_ids) if selected_skill_ids else 'none'}",
            ],
        }

    def _driver_skill_ids(self, *, intents: dict[str, bool], available_skill_ids: set[str]) -> list[str]:
        driver_skill_ids: list[str] = []
        if intents.get("category_focus"):
            driver_skill_ids.extend(skill_id for skill_id in ("categories", "top_categories") if skill_id in available_skill_ids)
        elif intents.get("comparison") or intents.get("diagnostic"):
            driver_skill_ids.extend(skill_id for skill_id in ("top_categories",) if skill_id in available_skill_ids)

        if intents.get("account_focus") or intents.get("comparison") or intents.get("diagnostic"):
            if "account_breakdown" in available_skill_ids:
                driver_skill_ids.append("account_breakdown")

        if intents.get("payment_focus") or intents.get("comparison") or intents.get("diagnostic"):
            if "payment_methods" in available_skill_ids:
                driver_skill_ids.append("payment_methods")

        if intents.get("criticality_focus") or intents.get("summary"):
            if "criticality" in available_skill_ids:
                driver_skill_ids.append("criticality")

        return self._unique(driver_skill_ids)

    def _pattern_skill_ids(self, *, intents: dict[str, bool], available_skill_ids: set[str]) -> list[str]:
        pattern_skill_ids: list[str] = []
        if intents.get("trend") and "daily" in available_skill_ids:
            pattern_skill_ids.append("daily")
        if intents.get("anomaly") and "outliers" in available_skill_ids:
            pattern_skill_ids.append("outliers")
        if intents.get("duplicates_focus") and "duplicates" in available_skill_ids:
            pattern_skill_ids.append("duplicates")
        if intents.get("uncategorized_focus") and "uncategorized" in available_skill_ids:
            pattern_skill_ids.append("uncategorized")
        return self._unique(pattern_skill_ids)

    def _narrative_skill_ids(self, *, intents: dict[str, bool], available_skill_ids: set[str]) -> list[str]:
        narrative_skill_ids: list[str] = []
        if intents.get("comparison") and "month_over_month" in available_skill_ids:
            narrative_skill_ids.append("month_over_month")
        if (intents.get("summary") or intents.get("diagnostic") or intents.get("broad_follow_up")) and "period_summary" in available_skill_ids:
            narrative_skill_ids.append("period_summary")
        if (intents.get("summary") or intents.get("broad_follow_up")) and "behavior_summary" in available_skill_ids:
            narrative_skill_ids.append("behavior_summary")
        if intents.get("average") and "averages" in available_skill_ids:
            narrative_skill_ids.append("averages")
        return self._unique(narrative_skill_ids)

    @classmethod
    def _matches_intent(cls, question: str, pattern: IntentPattern) -> bool:
        score = 0

        for phrase in pattern.strong_phrases:
            if cls._contains_phrase(question, phrase):
                score += 2

        for phrase in pattern.supporting_phrases:
            if cls._contains_phrase(question, phrase):
                score += 1

        for token_group in pattern.token_groups:
            if all(cls._contains_phrase(question, token) for token in token_group):
                score += 2

        for expression in pattern.regexes:
            if re.search(expression, question):
                score += 2

        return score >= pattern.threshold

    @classmethod
    def _contains_phrase(cls, question: str, phrase: str) -> bool:
        normalized_phrase = cls._normalize_text(phrase)
        if not normalized_phrase:
            return False
        padded_question = f" {question} "
        return f" {normalized_phrase} " in padded_question

    @staticmethod
    def _normalize_text(value: str) -> str:
        normalized = value.lower().replace("’", " ").replace("'", " ")
        normalized = _NON_ALPHANUMERIC_PATTERN.sub(" ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _append_unique(existing: list[str], *new_items: str) -> list[str]:
        combined = list(existing)
        seen = set(existing)
        for item in new_items:
            if item in seen:
                continue
            combined.append(item)
            seen.add(item)
        return combined

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            ordered.append(value)
            seen.add(value)
        return ordered

    @staticmethod
    def _resolve_skills(skill_ids: list[str], available_skills: list[Skill]) -> list[Skill]:
        skill_lookup = {skill.skill_id: skill for skill in available_skills}
        return [skill_lookup[skill_id] for skill_id in skill_ids if skill_id in skill_lookup]


