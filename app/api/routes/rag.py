from __future__ import annotations

from functools import lru_cache
import logging

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
from app.services.intent_service import IntentService
from app.services.langgraph_reasoning_service import LangGraphReasoningService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.skills.factories import build_skill_registry

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_conversation_history_repository() -> MySQLConversationHistoryRepository:
    return MySQLConversationHistoryRepository(get_settings())


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    settings = get_settings()
    client = SpringBootClient()
    analytics = AnalyticsService(client)
    insights = InsightService(client, analytics)
    skill_registry = build_skill_registry(client, analytics, insights)
    llm_service = LLMService()
    intent_service = IntentService()
    history_repository = get_conversation_history_repository()
    langgraph_service = LangGraphReasoningService(enabled=settings.langgraph_enabled)
    return RAGService(
        client,
        skill_registry,
        llm_service,
        history_repository,
        intent_service=intent_service,
        langgraph_service=langgraph_service,
    )


@router.post("/ask", response_model=RagAnswerResponse)
def ask(
    request: RagAskRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RagAnswerResponse:
    logger.info("RAG ask request payload=%s", request.model_dump(mode="json", exclude_none=True))
    try:
        response = rag_service.answer(
            question=request.question,
            conversation_id=request.conversation_id,
            time_scope=request.time_scope,
            period=request.period,
            payment_method=request.payment_method,
            account=request.account,
            transaction_id=request.transaction_id,
        )
        logger.info("RAG ask response payload=%s", response.model_dump(mode="json", exclude_none=True))
        return response
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
    logger.info(
        "RAG conversation request payload=%s",
        {"conversation_id": conversation_id, "limit": limit},
    )
    try:
        response = rag_service.get_conversation_history(conversation_id, limit=limit)
        logger.info("RAG conversation response payload=%s", response.model_dump(mode="json", exclude_none=True))
        return response
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationHistoryDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
