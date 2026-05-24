from __future__ import annotations

from datetime import date
import json
import logging
import re
from typing import Any, Optional

from openai import OpenAI

from app.core.config import get_settings
from app.models.schemas import RagIntentResponse

logger = logging.getLogger(__name__)

INTENT_CLASSIFICATION_PROMPT_TEMPLATE = (
    "You are a finance routing assistant. Return JSON only with keys: "
    "skill_ids (array of strings), time_reference (string or null), filters (object with payment_method/account), "
    "confidence (0 to 1), rationale (string or null). "
    "Only choose skill_ids from the provided available_skills list. "
    "Do not invent new skills. If uncertain, return an empty skill_ids array.\n\n"
    "Question:\n{question}\n\n"
    "Available skills:\n{available_skills_json}"
)

STATEMENT_PERIOD_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
STATEMENT_PERIOD_MONTH_PATTERN = "|".join(STATEMENT_PERIOD_MONTHS)
STATEMENT_PERIOD_MONTH_LOOKUP: dict[str, int] = {
    month.lower(): index for index, month in enumerate(STATEMENT_PERIOD_MONTHS, start=1)
}
EXPLICIT_STATEMENT_PERIOD_PATTERN = re.compile(
    rf"\b(?P<month>{STATEMENT_PERIOD_MONTH_PATTERN})\s*(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
CONTEXTUAL_MONTH_PATTERN = re.compile(
    rf"\b(?:in|for|during|on|from)\s+(?:the\s+month\s+of\s+)?(?P<month>{STATEMENT_PERIOD_MONTH_PATTERN})\b",
    re.IGNORECASE,
)
MONTH_OF_PATTERN = re.compile(
    rf"\bmonth\s+of\s+(?P<month>{STATEMENT_PERIOD_MONTH_PATTERN})\b",
    re.IGNORECASE,
)
SIMPLE_MONTH_PATTERN = re.compile(rf"^(?P<month>{STATEMENT_PERIOD_MONTH_PATTERN})$", re.IGNORECASE)
CONTEXTUAL_ACCOUNT_REFERENCE_PATTERN = re.compile(
    r"\b(?:this|that|selected|current|same)\s+account\b",
    re.IGNORECASE,
)
EXPLICIT_ACCOUNT_REFERENCE_PATTERNS = (
    re.compile(
        r"\b(?:account|acct)\s+(?:named|called)\s+[\"“'](?P<account>[^\"”']{1,80})[\"”']",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:account|acct)\s+(?:named|called)\s+(?P<account>[A-Za-z0-9][A-Za-z0-9 &'/-]{0,60})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:on|for|from|using|in)\s+(?:my\s+|the\s+)?(?P<account>[A-Za-z0-9][A-Za-z0-9 &'/-]{0,60}?)\s+account\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:my|the)\s+(?P<account>[A-Za-z0-9][A-Za-z0-9 &'/-]{0,60}?)\s+account\b",
        re.IGNORECASE,
    ),
)
GENERIC_ACCOUNT_REFERENCE_VALUES = {
    "this",
    "that",
    "the",
    "my",
    "current",
    "selected",
    "same",
    "another",
    "other",
    "new",
    "an",
    "a",
}


class IntentService:
    """Optional LLM-backed intent classification and filter extraction."""

    def __init__(self, *, enable_llm: bool = True) -> None:
        settings = get_settings()
        self.model = settings.openai_chat_model
        self.client = OpenAI(api_key=settings.openai_api_key) if enable_llm and settings.openai_api_key else None
        logger.info(
            "Initialized IntentService model=%s enabled=%s",
            self.model,
            self.client is not None,
        )

    def classify_intent(
        self,
        question: str,
        available_skills: list[dict[str, Any]],
    ) -> Optional[RagIntentResponse]:
        if self.client is None:
            logger.debug("Skipping LLM intent classification because no OpenAI client is configured")
            return None

        prompt = self._build_classification_prompt(question=question, available_skills=available_skills)
        logger.info(
            "Submitting LLM intent classification model=%s skill_count=%s question_length=%s",
            self.model,
            len(available_skills),
            len(question),
        )
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=prompt,
                text_format=RagIntentResponse,
            )
            parsed_intent = response.output_parsed
            if parsed_intent is None:
                logger.warning(
                    "LLM intent classification returned no parsed output model=%s output_text_length=%s",
                    self.model,
                    len(response.output_text or ""),
                )
                return None
            intent = (
                parsed_intent
                if isinstance(parsed_intent, RagIntentResponse)
                else RagIntentResponse.model_validate(parsed_intent)
            )
        except Exception:
            logger.exception("LLM intent classification failed; falling back to deterministic routing")
            return None

        logger.info(
            "LLM intent classification completed skill_ids=%s confidence=%s",
            intent.skill_ids,
            intent.confidence,
        )
        return intent

    def infer_period(
        self,
        question: str,
        today: date,
        *,
        llm_intent: Optional[RagIntentResponse] = None,
    ) -> Optional[dict[str, str]]:
        question_period = self._infer_period_from_text(question, today=today, source_prefix="question")
        if question_period is not None:
            return question_period

        llm_time_reference = llm_intent.time_reference if llm_intent is not None else None
        if isinstance(llm_time_reference, str) and llm_time_reference.strip():
            return self._infer_period_from_text(
                llm_time_reference.strip(),
                today=today,
                source_prefix="llm",
                matched_text=llm_time_reference.strip(),
            )

        return None

    def infer_account(
        self,
        question: str,
        *,
        llm_intent: Optional[RagIntentResponse] = None,
    ) -> Optional[dict[str, str]]:
        explicit_account = self._infer_explicit_account_from_question(question)
        if explicit_account is not None:
            return explicit_account

        llm_account = self._infer_account_from_intent(llm_intent)
        if llm_account is not None:
            return llm_account

        contextual_account = self._infer_contextual_account_reference(question)
        if contextual_account is not None:
            return contextual_account

        return None

    def _build_classification_prompt(self, question: str, available_skills: list[dict[str, Any]]) -> str:
        return INTENT_CLASSIFICATION_PROMPT_TEMPLATE.format(
            question=question,
            available_skills_json=json.dumps(available_skills, default=str, indent=2),
        )

    def _infer_period_from_text(
        self,
        text: str,
        *,
        today: date,
        source_prefix: str,
        matched_text: Optional[str] = None,
    ) -> Optional[dict[str, str]]:
        lowered = text.lower()
        current_statement_period = self.format_statement_period(today)

        explicit_period_match = EXPLICIT_STATEMENT_PERIOD_PATTERN.search(text)
        if explicit_period_match:
            resolved_period = self.format_statement_period(
                date(
                    int(explicit_period_match.group("year")),
                    self.month_number(explicit_period_match.group("month")),
                    1,
                )
            )
            return {
                "source": f"{source_prefix}_explicit_period",
                "matched_text": matched_text or explicit_period_match.group(0),
                "resolved_period": resolved_period,
                "resolution_rule": self._resolution_rule(
                    source_prefix,
                    "explicit MonthYear or Month YYYY reference resolved to a statement period",
                ),
            }

        if "this period" in lowered or "current period" in lowered:
            return {
                "source": f"{source_prefix}_current_period",
                "matched_text": matched_text or ("this period" if "this period" in lowered else "current period"),
                "resolved_period": current_statement_period,
                "resolution_rule": self._resolution_rule(
                    source_prefix,
                    "current period resolves to the current statement period",
                ),
            }

        if "this month" in lowered or "current month" in lowered:
            return {
                "source": f"{source_prefix}_current_month",
                "matched_text": matched_text or ("this month" if "this month" in lowered else "current month"),
                "resolved_period": current_statement_period,
                "resolution_rule": self._resolution_rule(
                    source_prefix,
                    "current month resolves to the current statement period",
                ),
            }

        if "last month" in lowered or "previous month" in lowered:
            matched_relative_text = matched_text or ("last month" if "last month" in lowered else "previous month")
            return {
                "source": f"{source_prefix}_relative_month",
                "matched_text": matched_relative_text,
                "resolved_period": self.format_statement_period(self.shift_month(today, offset=-1)),
                "resolution_rule": self._resolution_rule(
                    source_prefix,
                    "relative month resolves from the current statement period",
                ),
            }

        contextual_month_match = CONTEXTUAL_MONTH_PATTERN.search(text) or MONTH_OF_PATTERN.search(text)
        if contextual_month_match:
            matched_month = contextual_month_match.group("month")
            return {
                "source": f"{source_prefix}_bare_month",
                "matched_text": matched_text or matched_month,
                "resolved_period": self.resolve_recent_month_reference(matched_month, today),
                "resolution_rule": self._resolution_rule(
                    source_prefix,
                    "bare month names resolve to the most recent matching statement period not in the future",
                ),
            }

        simple_month_match = SIMPLE_MONTH_PATTERN.search(text.strip())
        if simple_month_match:
            matched_month = simple_month_match.group("month")
            return {
                "source": f"{source_prefix}_bare_month",
                "matched_text": matched_text or matched_month,
                "resolved_period": self.resolve_recent_month_reference(matched_month, today),
                "resolution_rule": self._resolution_rule(
                    source_prefix,
                    "bare month names resolve to the most recent matching statement period not in the future",
                ),
            }

        return None

    def _infer_explicit_account_from_question(self, question: str) -> Optional[dict[str, str]]:
        for pattern in EXPLICIT_ACCOUNT_REFERENCE_PATTERNS:
            match = pattern.search(question)
            if match is None:
                continue
            resolved_account = self.normalize_explicit_account_reference(match.group("account"))
            if resolved_account is None:
                continue
            return {
                "source": "question_explicit_account",
                "matched_text": match.group(0),
                "resolved_account": resolved_account,
                "resolution_rule": "explicit account references from the question override the prior conversation account",
            }
        return None

    def _infer_account_from_intent(self, llm_intent: Optional[RagIntentResponse]) -> Optional[dict[str, str]]:
        llm_filters = llm_intent.filters if llm_intent is not None else None
        llm_account = llm_filters.account if llm_filters is not None else None
        normalized_account = self.normalize_explicit_account_reference(llm_account)
        if normalized_account is None:
            return None
        return {
            "source": "llm_question_account",
            "matched_text": normalized_account,
            "resolved_account": normalized_account,
            "resolution_rule": "LLM intent classification extracted an account reference from the question",
        }

    def _infer_contextual_account_reference(self, question: str) -> Optional[dict[str, str]]:
        contextual_match = CONTEXTUAL_ACCOUNT_REFERENCE_PATTERN.search(question)
        if contextual_match is None:
            return None
        return {
            "source": "question_contextual_account_reference",
            "matched_text": contextual_match.group(0),
            "resolution_rule": "contextual references like this account defer to the current conversation account when available",
        }

    @staticmethod
    def _resolution_rule(source_prefix: str, base_rule: str) -> str:
        if source_prefix == "llm":
            return f"LLM intent classification time_reference provided a hint; {base_rule.lower()}"
        return base_rule

    @staticmethod
    def month_number(month_text: str) -> int:
        month_number = STATEMENT_PERIOD_MONTH_LOOKUP.get(month_text.lower())
        if month_number is None:
            raise ValueError(f"Unsupported statement period month: {month_text}")
        return month_number

    @staticmethod
    def format_statement_period(reference_date: date) -> str:
        return reference_date.strftime("%B%Y")

    @staticmethod
    def shift_month(reference_date: date, offset: int) -> date:
        target_index = reference_date.month - 1 + offset
        target_year = reference_date.year + (target_index // 12)
        target_month = (target_index % 12) + 1
        return date(target_year, target_month, 1)

    def resolve_recent_month_reference(self, month_text: str, today: date) -> str:
        target_month = self.month_number(month_text)
        target_year = today.year if target_month <= today.month else today.year - 1
        return self.format_statement_period(date(target_year, target_month, 1))

    @staticmethod
    def normalize_explicit_account_reference(candidate: Optional[str]) -> Optional[str]:
        if not isinstance(candidate, str):
            return None
        normalized = re.sub(r"\s+", " ", candidate).strip(" \t\n\r\"'.,:;!?()[]{}")
        if not normalized:
            return None
        lowered_words = [word for word in normalized.lower().split() if word]
        if lowered_words and all(word in GENERIC_ACCOUNT_REFERENCE_VALUES for word in lowered_words):
            return None
        return normalized



