from __future__ import annotations

import json
import logging
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
            "Prefer the structured supporting_sources and tool_trace_summaries when deciding what evidence was used. "
            "If the data is insufficient, say so clearly.\n\n"
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
        return response.output_text.strip()

