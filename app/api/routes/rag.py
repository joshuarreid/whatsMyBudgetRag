from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends

from app.clients.spring_boot_client import SpringBootClient
from app.models.schemas import RagAskRequest, RagAnswerResponse
from app.services.analytics_service import AnalyticsService
from app.services.insight_service import InsightService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService

router = APIRouter()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    client = SpringBootClient()
    analytics = AnalyticsService(client)
    insights = InsightService(client, analytics)
    llm_service = LLMService()
    return RAGService(client, analytics, insights, llm_service)


@router.post("/ask", response_model=RagAnswerResponse)
def ask(
    request: RagAskRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RagAnswerResponse:
    return rag_service.answer(
        question=request.question,
        period=request.period,
        payment_method=request.payment_method,
        account=request.account,
        transaction_id=request.transaction_id,
    )