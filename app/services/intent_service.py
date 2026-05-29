from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
import json
import logging
import re
from typing import Any, Optional

from openai import OpenAI

from app.core.config import get_settings
from app.models.schemas import RagIntentResponse, RagTimeScope

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
YEAR_REFERENCE_PATTERN = re.compile(r"\b(?P<year>20\d{2})\b")
CURRENT_YEAR_PATTERN = re.compile(r"\b(?:this|current)\s+year\b", re.IGNORECASE)
LAST_YEAR_PATTERN = re.compile(r"\b(?:last|previous)\s+year\b", re.IGNORECASE)
MONTH_RANGE_PATTERN = re.compile(
    rf"\b(?P<start_month>{STATEMENT_PERIOD_MONTH_PATTERN})(?:\s*(?P<start_year>\d{{4}}))?\s+"
    rf"(?:through|thru|to|until)\s+"
    rf"(?P<end_month>{STATEMENT_PERIOD_MONTH_PATTERN})(?:\s*(?P<end_year>\d{{4}}))?\b",
    re.IGNORECASE,
)
END_OF_MONTH_PATTERN = re.compile(
    rf"\bend\s+of\s+(?P<month>{STATEMENT_PERIOD_MONTH_PATTERN})(?:\s+(?P<year>\d{{4}}))?\b",
    re.IGNORECASE,
)
WEEK_OF_MONTH_PATTERN = re.compile(
    rf"\b(?P<ordinal>first|1st|second|2nd|third|3rd|fourth|4th|last)\s+week\s+of\s+"
    rf"(?P<month>{STATEMENT_PERIOD_MONTH_PATTERN})(?:\s+(?P<year>\d{{4}}))?\b",
    re.IGNORECASE,
)
HALF_OF_MONTH_PATTERN = re.compile(
    rf"\b(?P<half>first|1st|second|2nd|last)\s+half\s+of\s+"
    rf"(?P<month>{STATEMENT_PERIOD_MONTH_PATTERN})(?:\s+(?P<year>\d{{4}}))?\b",
    re.IGNORECASE,
)
CONTEXTUAL_ACCOUNT_REFERENCE_PATTERN = re.compile(
    r"\b(?:this|that|selected|current|same)\s+account\b",
    re.IGNORECASE,
)
EXPLICIT_ACCOUNT_REFERENCE_PATTERNS = (
    re.compile(
        r"\b(?P<account>[A-Za-z][A-Za-z0-9&/-]{0,40})['’]s\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfor\s+(?P<account>[A-Za-z][A-Za-z0-9&/-]{0,30}(?:\s+[A-Za-z][A-Za-z0-9&/-]{0,30})?)\b[?!.,\s]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:is\s+)?(?P<account>[A-Za-z][A-Za-z0-9 &'/-]{0,40})\s+on\s+track\b",
        re.IGNORECASE,
    ),
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
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "period",
    "month",
    "week",
    "today",
    "yesterday",
    *(month.lower() for month in STATEMENT_PERIOD_MONTHS),
}
DAY_REFERENCE_PATTERNS = (
    (re.compile(r"\btoday\b", re.IGNORECASE), 0, "today resolves to a single-day inclusive date range"),
    (re.compile(r"\byesterday\b", re.IGNORECASE), -1, "yesterday resolves to a single-day inclusive date range"),
)
WEEK_REFERENCE_PATTERNS = (
    (re.compile(r"\b(?:this|current)\s+week\b", re.IGNORECASE), 0, "current week resolves to the Monday-through-today inclusive date range"),
    (re.compile(r"\b(?:last|previous)\s+week\b", re.IGNORECASE), -1, "last week resolves to the previous Monday-through-Sunday inclusive date range"),
)
WEEK_ORDINAL_LOOKUP = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
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

    def infer_time_scope(
        self,
        question: str,
        today: date,
        *,
        llm_intent: Optional[RagIntentResponse] = None,
    ) -> Optional[dict[str, Any]]:
        question_time_scope = self._infer_time_scope_from_text(question, today=today, source_prefix="question")
        if question_time_scope is not None:
            return question_time_scope

        llm_time_reference = llm_intent.time_reference if llm_intent is not None else None
        if isinstance(llm_time_reference, str) and llm_time_reference.strip():
            return self._infer_time_scope_from_text(
                llm_time_reference.strip(),
                today=today,
                source_prefix="llm",
                matched_text=llm_time_reference.strip(),
            )

        return None

    def infer_period(
        self,
        question: str,
        today: date,
        *,
        llm_intent: Optional[RagIntentResponse] = None,
    ) -> Optional[dict[str, str]]:
        inferred_time_scope = self.infer_time_scope(
            question=question,
            today=today,
            llm_intent=llm_intent,
        )
        if inferred_time_scope is None:
            return None
        if inferred_time_scope.get("scope_type") != "statement_period":
            return None
        return inferred_time_scope  # type: ignore[return-value]

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

    def _infer_time_scope_from_text(
        self,
        text: str,
        *,
        today: date,
        source_prefix: str,
        matched_text: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        lowered = text.lower()
        current_statement_period = self.format_statement_period(today)

        month_range_match = MONTH_RANGE_PATTERN.search(text)
        if month_range_match:
            start_reference = self.resolve_month_reference(
                month_text=month_range_match.group("start_month"),
                today=today,
                year_text=month_range_match.group("start_year"),
            )
            end_reference = self.resolve_month_reference(
                month_text=month_range_match.group("end_month"),
                today=today,
                year_text=month_range_match.group("end_year"),
            )
            start_period, end_period = self.resolve_statement_period_range(
                start_reference=start_reference,
                end_reference=end_reference,
            )
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_statement_period_range",
                matched_text=matched_text or month_range_match.group(0),
                time_scope=RagTimeScope(
                    scope_type="statement_period_range",
                    start_period=start_period,
                    end_period=end_period,
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "month-through-month references resolve to an inclusive statement period range",
                ),
            )

        week_of_month_match = WEEK_OF_MONTH_PATTERN.search(text)
        if week_of_month_match:
            month_reference = self.resolve_month_reference(
                month_text=week_of_month_match.group("month"),
                today=today,
                year_text=week_of_month_match.group("year"),
            )
            start_date, end_date = self.resolve_week_of_month_range(
                month_reference=month_reference,
                ordinal_text=week_of_month_match.group("ordinal"),
            )
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_week_of_month",
                matched_text=matched_text or week_of_month_match.group(0),
                time_scope=RagTimeScope(
                    scope_type="date_range",
                    start_date=start_date,
                    end_date=end_date,
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "week-of-month references resolve to an inclusive date range within the target month",
                ),
            )

        half_of_month_match = HALF_OF_MONTH_PATTERN.search(text)
        if half_of_month_match:
            month_reference = self.resolve_month_reference(
                month_text=half_of_month_match.group("month"),
                today=today,
                year_text=half_of_month_match.group("year"),
            )
            start_date, end_date = self.resolve_half_of_month_range(
                month_reference=month_reference,
                half_text=half_of_month_match.group("half"),
            )
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_half_of_month",
                matched_text=matched_text or half_of_month_match.group(0),
                time_scope=RagTimeScope(
                    scope_type="date_range",
                    start_date=start_date,
                    end_date=end_date,
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "half-month references resolve to an inclusive date range within the target month",
                ),
            )

        end_of_month_match = END_OF_MONTH_PATTERN.search(text)
        if end_of_month_match:
            month_reference = self.resolve_month_reference(
                month_text=end_of_month_match.group("month"),
                today=today,
                year_text=end_of_month_match.group("year"),
            )
            start_date, end_date = self.resolve_end_of_month_range(month_reference)
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_end_of_month",
                matched_text=matched_text or end_of_month_match.group(0),
                time_scope=RagTimeScope(
                    scope_type="date_range",
                    start_date=start_date,
                    end_date=end_date,
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "end-of-month references resolve to the last seven days of the target month",
                ),
            )

        explicit_period_match = EXPLICIT_STATEMENT_PERIOD_PATTERN.search(text)
        if explicit_period_match:
            resolved_period = self.format_statement_period(
                date(
                    int(explicit_period_match.group("year")),
                    self.month_number(explicit_period_match.group("month")),
                    1,
                )
            )
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_explicit_period",
                matched_text=matched_text or explicit_period_match.group(0),
                time_scope=RagTimeScope(
                    scope_type="statement_period",
                    statement_period=resolved_period,
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "explicit MonthYear or Month YYYY reference resolved to a statement period",
                ),
            )

        current_year_match = CURRENT_YEAR_PATTERN.search(text)
        if current_year_match:
            start_period, end_period = self.resolve_calendar_year_range(year=today.year, today=today)
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_calendar_year",
                matched_text=matched_text or current_year_match.group(0),
                time_scope=RagTimeScope(
                    scope_type="statement_period_range",
                    start_period=start_period,
                    end_period=end_period,
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "current year references resolve to a year-to-date inclusive statement period range through the current statement period",
                ),
            )

        last_year_match = LAST_YEAR_PATTERN.search(text)
        if last_year_match:
            target_year = today.year - 1
            start_period, end_period = self.resolve_calendar_year_range(year=target_year, today=today)
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_calendar_year",
                matched_text=matched_text or last_year_match.group(0),
                time_scope=RagTimeScope(
                    scope_type="statement_period_range",
                    start_period=start_period,
                    end_period=end_period,
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "last year references resolve to the full prior calendar year's inclusive statement period range",
                ),
            )

        year_match = YEAR_REFERENCE_PATTERN.search(text)
        if year_match:
            target_year = int(year_match.group("year"))
            start_period, end_period = self.resolve_calendar_year_range(year=target_year, today=today)
            year_resolution_rule = (
                "explicit calendar year references resolve to a year-to-date inclusive statement period range through the current statement period"
                if target_year == today.year
                else "explicit calendar year references resolve to the full calendar year's inclusive statement period range"
            )
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_calendar_year",
                matched_text=matched_text or year_match.group(0),
                time_scope=RagTimeScope(
                    scope_type="statement_period_range",
                    start_period=start_period,
                    end_period=end_period,
                ),
                resolution_rule=self._resolution_rule(source_prefix, year_resolution_rule),
            )

        for pattern, offset_days, resolution_rule in DAY_REFERENCE_PATTERNS:
            day_match = pattern.search(text)
            if day_match is None:
                continue
            resolved_day = today + timedelta(days=offset_days)
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_day_reference",
                matched_text=matched_text or day_match.group(0),
                time_scope=RagTimeScope(
                    scope_type="date_range",
                    start_date=resolved_day,
                    end_date=resolved_day,
                ),
                resolution_rule=self._resolution_rule(source_prefix, resolution_rule),
            )

        for pattern, week_offset, resolution_rule in WEEK_REFERENCE_PATTERNS:
            week_match = pattern.search(text)
            if week_match is None:
                continue
            start_date, end_date = self.resolve_relative_week_range(today=today, week_offset=week_offset)
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_week_reference",
                matched_text=matched_text or week_match.group(0),
                time_scope=RagTimeScope(
                    scope_type="date_range",
                    start_date=start_date,
                    end_date=end_date,
                ),
                resolution_rule=self._resolution_rule(source_prefix, resolution_rule),
            )

        if "this period" in lowered or "current period" in lowered:
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_current_period",
                matched_text=matched_text or ("this period" if "this period" in lowered else "current period"),
                time_scope=RagTimeScope(
                    scope_type="statement_period",
                    statement_period=current_statement_period,
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "current period resolves to the current statement period",
                ),
            )

        if "this month" in lowered or "current month" in lowered:
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_current_month",
                matched_text=matched_text or ("this month" if "this month" in lowered else "current month"),
                time_scope=RagTimeScope(
                    scope_type="statement_period",
                    statement_period=current_statement_period,
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "current month resolves to the current statement period",
                ),
            )

        if "last month" in lowered or "previous month" in lowered:
            matched_relative_text = matched_text or ("last month" if "last month" in lowered else "previous month")
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_relative_month",
                matched_text=matched_relative_text,
                time_scope=RagTimeScope(
                    scope_type="statement_period",
                    statement_period=self.format_statement_period(self.shift_month(today, offset=-1)),
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "relative month resolves from the current statement period",
                ),
            )

        contextual_month_match = CONTEXTUAL_MONTH_PATTERN.search(text) or MONTH_OF_PATTERN.search(text)
        if contextual_month_match:
            matched_month = contextual_month_match.group("month")
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_bare_month",
                matched_text=matched_text or matched_month,
                time_scope=RagTimeScope(
                    scope_type="statement_period",
                    statement_period=self.resolve_recent_month_reference(matched_month, today),
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "bare month names resolve to the most recent matching statement period not in the future",
                ),
            )

        simple_month_match = SIMPLE_MONTH_PATTERN.search(text.strip())
        if simple_month_match:
            matched_month = simple_month_match.group("month")
            return self._build_time_scope_resolution(
                source=f"{source_prefix}_bare_month",
                matched_text=matched_text or matched_month,
                time_scope=RagTimeScope(
                    scope_type="statement_period",
                    statement_period=self.resolve_recent_month_reference(matched_month, today),
                ),
                resolution_rule=self._resolution_rule(
                    source_prefix,
                    "bare month names resolve to the most recent matching statement period not in the future",
                ),
            )

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
    def _build_time_scope_resolution(
        *,
        source: str,
        matched_text: str,
        time_scope: RagTimeScope,
        resolution_rule: str,
    ) -> dict[str, Any]:
        resolved = {
            "source": source,
            "matched_text": matched_text,
            "scope_type": time_scope.scope_type,
            "time_scope": time_scope.model_dump(mode="json", exclude_none=True),
            "resolution_rule": resolution_rule,
        }
        if time_scope.scope_type == "statement_period" and time_scope.statement_period is not None:
            resolved["resolved_period"] = time_scope.statement_period
        if time_scope.scope_type == "statement_period_range":
            if time_scope.start_period is not None:
                resolved["resolved_start_period"] = time_scope.start_period
            if time_scope.end_period is not None:
                resolved["resolved_end_period"] = time_scope.end_period
        if time_scope.scope_type == "date_range":
            if time_scope.start_date is not None:
                resolved["resolved_start_date"] = time_scope.start_date.isoformat()
            if time_scope.end_date is not None:
                resolved["resolved_end_date"] = time_scope.end_date.isoformat()
        return resolved

    @staticmethod
    def shift_month(reference_date: date, offset: int) -> date:
        target_index = reference_date.month - 1 + offset
        target_year = reference_date.year + (target_index // 12)
        target_month = (target_index % 12) + 1
        return date(target_year, target_month, 1)

    def resolve_month_reference(
        self,
        *,
        month_text: str,
        today: date,
        year_text: Optional[str] = None,
    ) -> date:
        if year_text is not None:
            return date(int(year_text), self.month_number(month_text), 1)
        target_month = self.month_number(month_text)
        target_year = today.year if target_month <= today.month else today.year - 1
        return date(target_year, target_month, 1)

    def resolve_recent_month_reference(self, month_text: str, today: date) -> str:
        return self.format_statement_period(self.resolve_month_reference(month_text=month_text, today=today))

    def resolve_statement_period_range(self, *, start_reference: date, end_reference: date) -> tuple[str, str]:
        start_year = start_reference.year
        if start_reference.month > end_reference.month and start_reference.year >= end_reference.year:
            start_year = end_reference.year - 1
        elif start_reference.month <= end_reference.month:
            start_year = end_reference.year
        start_period_reference = date(start_year, start_reference.month, 1)
        end_period_reference = date(end_reference.year, end_reference.month, 1)
        return (
            self.format_statement_period(start_period_reference),
            self.format_statement_period(end_period_reference),
        )

    def resolve_calendar_year_range(self, *, year: int, today: date) -> tuple[str, str]:
        end_month = today.month if year == today.year else 12
        start_period_reference = date(year, 1, 1)
        end_period_reference = date(year, end_month, 1)
        return (
            self.format_statement_period(start_period_reference),
            self.format_statement_period(end_period_reference),
        )

    @staticmethod
    def resolve_relative_week_range(*, today: date, week_offset: int) -> tuple[date, date]:
        current_week_start = today - timedelta(days=today.weekday())
        week_start = current_week_start + timedelta(weeks=week_offset)
        week_end = today if week_offset == 0 else week_start + timedelta(days=6)
        return week_start, week_end

    @staticmethod
    def resolve_week_of_month_range(*, month_reference: date, ordinal_text: str) -> tuple[date, date]:
        last_day = monthrange(month_reference.year, month_reference.month)[1]
        if ordinal_text.lower() == "last":
            end_date = date(month_reference.year, month_reference.month, last_day)
            start_date = max(month_reference, end_date - timedelta(days=6))
            return start_date, end_date
        ordinal = WEEK_ORDINAL_LOOKUP[ordinal_text.lower()]
        start_day = ((ordinal - 1) * 7) + 1
        start_date = date(month_reference.year, month_reference.month, start_day)
        end_day = min(start_day + 6, last_day)
        return start_date, date(month_reference.year, month_reference.month, end_day)

    @staticmethod
    def resolve_half_of_month_range(*, month_reference: date, half_text: str) -> tuple[date, date]:
        last_day = monthrange(month_reference.year, month_reference.month)[1]
        midpoint = 15
        start_date = date(month_reference.year, month_reference.month, 1)
        if half_text.lower() in {"first", "1st"}:
            return start_date, date(month_reference.year, month_reference.month, min(midpoint, last_day))
        return date(month_reference.year, month_reference.month, min(midpoint + 1, last_day)), date(
            month_reference.year,
            month_reference.month,
            last_day,
        )

    @staticmethod
    def resolve_end_of_month_range(month_reference: date) -> tuple[date, date]:
        last_day = monthrange(month_reference.year, month_reference.month)[1]
        end_date = date(month_reference.year, month_reference.month, last_day)
        start_date = max(month_reference, end_date - timedelta(days=6))
        return start_date, end_date

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



