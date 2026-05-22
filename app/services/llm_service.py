from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.core.config import get_settings


class LLMService:
    """Optional answer synthesis over already-fetched finance context."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_chat_model
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def generate_answer(self, question: str, context: dict[str, Any]) -> str | None:
        if self.client is None:
            return None

        prompt = (
            "You are a finance assistant. Answer using only the provided API context. "
            "If the data is insufficient, say so clearly.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{json.dumps(context, default=str, indent=2)}"
        )
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text.strip()