from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from app.models.schemas import RagIntentResponse, RagTimeScope
from app.repositories import ConversationMessageRecord
from app.skills.base import Skill
from app.services.rag.constants import SUMMARY_ROUTING_KEYWORDS, TREND_ROUTING_KEYWORDS

logger = logging.getLogger(__name__)


class RAGRoutingMixin:
    def _resolve_time_scope(
        self,
        question: str,
        time_scope: Optional[RagTimeScope],
        period: Optional[str],
        transaction_id: Optional[str],
        *,
        conversation_history: Optional[list[ConversationMessageRecord]] = None,
        llm_intent: Optional[RagIntentResponse] = None,
        today: Optional[date] = None,
    ) -> tuple[RagTimeScope, dict[str, Any]]:
        reference_date = today or date.today()
        inferred_time_scope = self.intent_parser.infer_time_scope(
            question=question,
            today=reference_date,
            llm_intent=llm_intent,
        )
        if inferred_time_scope is not None:
            resolved_time_scope = RagTimeScope.model_validate(inferred_time_scope["time_scope"])
            logger.debug(
                "Resolved analytics time scope from question matched_text=%s time_scope=%s",
                inferred_time_scope.get("matched_text"),
                resolved_time_scope.model_dump(mode="json", exclude_none=True),
            )
            return resolved_time_scope, inferred_time_scope

        prior_time_scope = self._last_resolved_time_scope(conversation_history)
        if prior_time_scope is not None:
            logger.debug(
                "Reusing prior conversation analytics time_scope=%s",
                prior_time_scope.model_dump(mode="json", exclude_none=True),
            )
            return prior_time_scope, self._build_time_scope_interpretation(
                source="conversation_history_time_scope",
                matched_text=self._time_scope_label(prior_time_scope),
                time_scope=prior_time_scope,
                resolution_rule="follow-up questions without a new time reference reuse the last resolved time scope from the conversation",
            )

        if time_scope is not None:
            logger.debug(
                "Using requested analytics time_scope=%s",
                time_scope.model_dump(mode="json", exclude_none=True),
            )
            return time_scope, self._build_time_scope_interpretation(
                source="request_time_scope" if period is None else "request_parameter",
                matched_text=self._time_scope_label(time_scope),
                time_scope=time_scope,
                resolution_rule="request time_scope overrides the default fallback when the question and conversation do not introduce a new time reference",
            )

        resolved_time_scope = RagTimeScope(
            scope_type="statement_period",
            statement_period=self.intent_parser.format_statement_period(reference_date),
        )
        logger.debug(
            "Falling back to current statement period=%s",
            resolved_time_scope.statement_period,
        )
        return resolved_time_scope, self._build_time_scope_interpretation(
            source="current_statement_period_fallback",
            matched_text=resolved_time_scope.statement_period or "",
            time_scope=resolved_time_scope,
            resolution_rule="questions without a time reference fall back to the current statement period",
        )

    def _resolve_period(
        self,
        question: str,
        period: Optional[str],
        transaction_id: Optional[str],
        *,
        conversation_history: Optional[list[ConversationMessageRecord]] = None,
        llm_intent: Optional[RagIntentResponse] = None,
        today: Optional[date] = None,
    ) -> tuple[Optional[str], dict[str, Any]]:
        resolved_time_scope, interpretation = self._resolve_time_scope(
            question=question,
            time_scope=None,
            period=period,
            transaction_id=transaction_id,
            conversation_history=conversation_history,
            llm_intent=llm_intent,
            today=today,
        )
        return self._derive_period_from_time_scope(resolved_time_scope), interpretation

    def _resolve_account(
        self,
        question: str,
        account: Optional[str],
        *,
        conversation_history: Optional[list[ConversationMessageRecord]] = None,
        llm_intent: Optional[RagIntentResponse] = None,
    ) -> tuple[Optional[str], dict[str, Optional[str]]]:
        inferred_account = self.intent_parser.infer_account(question=question, llm_intent=llm_intent)
        prior_account = self._last_resolved_filter(conversation_history, "account")

        if inferred_account is not None and inferred_account.get("resolved_account"):
            resolved_account = inferred_account["resolved_account"]
            logger.debug(
                "Resolved analytics account from question matched_text=%s resolved_account=%s",
                inferred_account.get("matched_text"),
                resolved_account,
            )
            return resolved_account, inferred_account

        if inferred_account is not None and inferred_account.get("source") == "question_contextual_account_reference":
            if prior_account:
                return prior_account, {
                    "source": "question_contextual_account_reference",
                    "matched_text": inferred_account.get("matched_text"),
                    "resolved_account": prior_account,
                    "resolution_rule": "contextual references like this account reuse the last resolved account from the conversation",
                }
            if account:
                return account, {
                    "source": "question_contextual_request_account",
                    "matched_text": inferred_account.get("matched_text"),
                    "resolved_account": account,
                    "resolution_rule": "contextual references like this account reuse the request account when the conversation has not resolved one yet",
                }

        if prior_account:
            logger.debug("Reusing prior conversation analytics account=%s", prior_account)
            return prior_account, {
                "source": "conversation_history_account",
                "matched_text": prior_account,
                "resolved_account": prior_account,
                "resolution_rule": "follow-up questions without a new account reference reuse the last resolved account from the conversation",
            }

        if account:
            logger.debug("Using requested analytics account=%s", account)
            return account, {
                "source": "request_parameter",
                "matched_text": account,
                "resolved_account": account,
            }

        return None, {
            "source": "no_account_filter",
            "matched_text": None,
            "resolved_account": None,
            "resolution_rule": "questions without an account reference or prior account context do not apply an account filter",
        }

    def _select_skills(
        self,
        question: str,
        *,
        time_scope: RagTimeScope,
        llm_intent: Optional[RagIntentResponse] = None,
    ) -> tuple[list[Skill], dict[str, Any]]:
        llm_selected_skills = self._optimize_skill_selection(
            question,
            self.skill_registry.resolve(llm_intent.skill_ids) if llm_intent is not None else [],
        )
        deterministic_skills = self._optimize_skill_selection(question, self.skill_registry.select(question))
        tool_selection = {
            "llm_suggested_tools": [skill.skill_id for skill in llm_selected_skills],
            "deterministic_tools": [skill.skill_id for skill in deterministic_skills],
            "union_tools": self._union_skill_ids(llm_selected_skills, deterministic_skills),
        }
        registry_skills = getattr(self.skill_registry, "skills", None)
        if self.langgraph_service is not None and isinstance(registry_skills, list):
            seed_skills = self._optimize_skill_selection(
                question,
                [*llm_selected_skills, *deterministic_skills],
            )
            graph_selected_skills, graph_metadata = self.langgraph_service.plan(
                question=question,
                time_scope=time_scope,
                available_skills=registry_skills,
                seed_skills=seed_skills,
            )
            if graph_metadata.get("applied"):
                return self._optimize_skill_selection(question, graph_selected_skills), {
                    "source": "langgraph_reasoning",
                    "llm_intent": (
                        llm_intent.model_dump(mode="json", exclude_none=True) if llm_intent is not None else None
                    ),
                    "resolved_skill_ids": [skill.skill_id for skill in graph_selected_skills],
                    "llm_raw_suggested_tools": llm_intent.skill_ids if llm_intent is not None else [],
                    "tool_selection": tool_selection,
                    "reasoning_graph": graph_metadata,
                }

        if llm_intent is not None and llm_selected_skills:
            return llm_selected_skills, {
                "source": "llm_intent",
                "llm_intent": llm_intent.model_dump(mode="json", exclude_none=True),
                "resolved_skill_ids": [skill.skill_id for skill in llm_selected_skills],
                "llm_raw_suggested_tools": llm_intent.skill_ids,
                "tool_selection": tool_selection,
            }

        return deterministic_skills, {
            "source": "keyword_match" if llm_intent is None else "keyword_fallback",
            "llm_intent": (
                llm_intent.model_dump(mode="json", exclude_none=True) if llm_intent is not None else None
            ),
            "resolved_skill_ids": [skill.skill_id for skill in deterministic_skills],
            "llm_raw_suggested_tools": llm_intent.skill_ids if llm_intent is not None else [],
            "tool_selection": tool_selection,
        }

    def _optimize_skill_selection(self, question: str, skills: list[Skill]) -> list[Skill]:
        optimized: list[Skill] = []
        seen_skill_ids: set[str] = set()
        for skill in skills:
            if skill.skill_id in seen_skill_ids:
                continue
            optimized.append(skill)
            seen_skill_ids.add(skill.skill_id)

        if self._is_trend_only_question(question) and any(skill.skill_id != "overview" for skill in optimized):
            optimized = [skill for skill in optimized if skill.skill_id != "overview"]
        return optimized

    @staticmethod
    def _is_trend_only_question(question: str) -> bool:
        lowered = question.lower()
        has_trend_signal = any(keyword in lowered for keyword in TREND_ROUTING_KEYWORDS)
        has_summary_signal = any(keyword in lowered for keyword in SUMMARY_ROUTING_KEYWORDS)
        return has_trend_signal and not has_summary_signal

    def _intent_classification_policy(
        self,
        question: str,
        *,
        today: Optional[date] = None,
    ) -> dict[str, Any]:
        registry_skills = getattr(self.skill_registry, "skills", None)
        direct_matches = self._optimize_skill_selection(
            question,
            [skill for skill in registry_skills if skill.matches(question)] if isinstance(registry_skills, list) else [],
        )
        direct_match_ids = [skill.skill_id for skill in direct_matches]
        classifier = self._intent_classifier()
        if classifier is None:
            return {
                "skip": True,
                "reason": "intent_service_unavailable",
                "direct_matches": direct_match_ids,
            }
        if not direct_matches:
            return {"skip": False, "reason": None, "direct_matches": direct_match_ids}
        if len(direct_matches) > 2:
            return {
                "skip": False,
                "reason": "too_many_deterministic_matches",
                "direct_matches": direct_match_ids,
            }
        lowered = question.lower()
        if any(marker in lowered for marker in (" and then ", ";", " plus ", " also ")):
            return {
                "skip": False,
                "reason": "compound_question",
                "direct_matches": direct_match_ids,
            }
        if len(direct_matches) == 1:
            inferred_time_scope = self.intent_parser.infer_time_scope(
                question=question,
                today=today or date.today(),
                llm_intent=None,
            )
            return {
                "skip": True,
                "reason": "deterministic_routing_confident",
                "direct_matches": direct_match_ids,
                "inferred_time_scope": inferred_time_scope.get("time_scope") if inferred_time_scope is not None else None,
            }
        if not self._is_trend_only_question(question):
            return {
                "skip": False,
                "reason": "requires_llm_disambiguation",
                "direct_matches": direct_match_ids,
            }
        inferred_time_scope = self.intent_parser.infer_time_scope(
            question=question,
            today=today or date.today(),
            llm_intent=None,
        )
        return {
            "skip": True,
            "reason": "deterministic_routing_confident",
            "direct_matches": direct_match_ids,
            "inferred_time_scope": inferred_time_scope.get("time_scope") if inferred_time_scope is not None else None,
        }

    def _intent_classifier(self) -> Optional[Any]:
        service = self.intent_service
        if service is None and self.llm is not None and hasattr(self.llm, "classify_intent"):
            service = self.llm
        if service is None or not hasattr(service, "classify_intent"):
            return None
        return service

    def _classify_intent(self, question: str) -> Optional[RagIntentResponse]:
        service = self._intent_classifier()
        if service is None:
            return None

        try:
            return service.classify_intent(
                question=question,
                available_skills=self.skill_registry.available_skills(),
            )
        except Exception:
            logger.exception("Intent classification raised unexpectedly; continuing with deterministic routing")
            return None

    @staticmethod
    def _union_skill_ids(*skill_groups: list[Skill]) -> list[str]:
        union_skill_ids: list[str] = []
        seen_skill_ids: set[str] = set()
        for skill_group in skill_groups:
            for skill in skill_group:
                if skill.skill_id in seen_skill_ids:
                    continue
                union_skill_ids.append(skill.skill_id)
                seen_skill_ids.add(skill.skill_id)
        return union_skill_ids

