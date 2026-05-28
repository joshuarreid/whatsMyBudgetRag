from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from openai import OpenAI

from app.core.config import get_settings


logger = logging.getLogger(__name__)


class LLMService:
    """Optional answer synthesis over already-fetched finance context."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_chat_model
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        logger.info(
            "Initialized LLMService model=%s enabled=%s",
            self.model,
            self.client is not None,
        )

    def generate_answer(self, question: str, context: dict[str, Any]) -> Optional[str]:
        if self.client is None:
            logger.debug("Skipping LLM generation because no OpenAI client is configured")
            return None

        prompt = (
            "You are a finance assistant. Answer using only the provided API context. "
            "Statement periods always use MonthYear format such as October2025 or May2026. "
            "Use timeline_context and period_interpretation to resolve references like this month, current month, this period, last month, previous month, or a bare month name such as October. "
            "Use the provided skill outputs and any unavailable_tools notes when deciding what evidence was available. "
            "If the data is insufficient, say so clearly. "
            "Format the response as compact GitHub-flavored markdown for a narrow mobile chat window. "
            "Use a short opening sentence followed by short bullet lists when helpful. "
            "Do not use tables. Use real line breaks and never emit escaped newline sequences like \\n. "
            "If daily totals are present in the context, summarize or list them directly instead of claiming they need to be fetched. "
            "Do not ask follow-up fetch questions unless the context explicitly indicates missing or unavailable data.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{json.dumps(context, default=str, indent=2)}"
        )
        logger.info(
            "Submitting LLM response request model=%s context_keys=%s question_length=%s",
            self.model,
            sorted(context.keys()),
            len(question),
        )
        response = self.client.responses.create(model=self.model, input=prompt)
        logger.info("LLM response received model=%s output_length=%s", self.model, len(response.output_text))
        return self._normalize_answer_text(response.output_text)

    @classmethod
    def _normalize_answer_text(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"} and "\\n" in normalized:
            normalized = normalized[1:-1]
        normalized = normalized.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "  ")
        normalized = normalized.replace('\\"', '"')
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

