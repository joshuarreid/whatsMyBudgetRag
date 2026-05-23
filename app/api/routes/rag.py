from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query

from app.clients.spring_boot_client import SpringBootClient
from app.core.config import get_settings
from app.models.schemas import RagAnswerResponse, RagAskRequest, RagConversationResponse
from app.repositories import (
    ConversationHistoryDisabledError,
    ConversationNotFoundError,
    MySQLConversationHistoryRepository,
)
from app.services.analytics_service import AnalyticsService
from app.services.insight_service import InsightService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.skills.factories import build_skill_registry

router = APIRouter()


@lru_cache(maxsize=1)
def get_conversation_history_repository() -> MySQLConversationHistoryRepository:
    return MySQLConversationHistoryRepository(get_settings())


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    client = SpringBootClient()
    analytics = AnalyticsService(client)
    insights = InsightService(client, analytics)
    skill_registry = build_skill_registry(client, analytics, insights)
    llm_service = LLMService()
    history_repository = get_conversation_history_repository()
    return RAGService(client, skill_registry, llm_service, history_repository)


@router.post("/ask", response_model=RagAnswerResponse)
def ask(
    request: RagAskRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RagAnswerResponse:
    try:
        return rag_service.answer(
            question=request.question,
            conversation_id=request.conversation_id,
            period=request.period,
            payment_method=request.payment_method,
            account=request.account,
            transaction_id=request.transaction_id,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationHistoryDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/conversations/{conversation_id}", response_model=RagConversationResponse)
def get_conversation(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    rag_service: RAGService = Depends(get_rag_service),
) -> RagConversationResponse:
    try:
        return rag_service.get_conversation_history(conversation_id, limit=limit)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationHistoryDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
