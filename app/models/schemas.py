from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AnalyticsBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class RagTimeScope(AnalyticsBaseModel):
    scope_type: str
    statement_period: Optional[str] = None
    start_period: Optional[str] = None
    end_period: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class RagAskRequest(AnalyticsBaseModel):
    question: str = Field(min_length=1)
    conversation_id: Optional[str] = None
    time_scope: Optional[RagTimeScope] = None
    period: Optional[str] = None
    payment_method: Optional[str] = None
    account: Optional[str] = None
    transaction_id: Optional[str] = None


class RagIntentFilters(AnalyticsBaseModel):
    payment_method: Optional[str] = None
    account: Optional[str] = None


class RagIntentResponse(AnalyticsBaseModel):
    skill_ids: list[str] = Field(default_factory=list)
    time_reference: Optional[str] = None
    filters: RagIntentFilters = Field(default_factory=RagIntentFilters)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    rationale: Optional[str] = None


class RagToolSelectionResponse(AnalyticsBaseModel):
    llm_suggested_tools: list[str] = Field(default_factory=list)
    deterministic_tools: list[str] = Field(default_factory=list)
    union_tools: list[str] = Field(default_factory=list)


class RagPlanStep(AnalyticsBaseModel):
    step_id: str
    skill_id: str
    context_key: str
    output_key: str
    label: Optional[str] = None
    time_scope: Optional[RagTimeScope] = None
    period: Optional[str] = None
    payment_method: Optional[str] = None
    account: Optional[str] = None


class RagExecutionPlan(AnalyticsBaseModel):
    strategy: str
    steps: list[RagPlanStep] = Field(default_factory=list)


class RagCitationResponse(AnalyticsBaseModel):
    source_type: str
    source_ref: str
    source_title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None


class RagToolTraceResponse(AnalyticsBaseModel):
    tool_name: str
    context_key: str
    category: str
    status: str
    duration_ms: Optional[int] = None
    cache_hit: bool = False
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    error_text: Optional[str] = None


class RagCacheMetadataResponse(AnalyticsBaseModel):
    enabled: bool = False
    eligible: bool = False
    reason: Optional[str] = None
    hits: int = 0
    misses: int = 0
    writes: int = 0


class RagTimingMetadataResponse(AnalyticsBaseModel):
    classifier_latency_ms: int = 0
    plan_execution_latency_ms: int = 0
    cache_lookup_latency_ms: int = 0
    tool_execution_latency_ms: int = 0
    answer_generation_latency_ms: int = 0


class RagConversationMessageResponse(AnalyticsBaseModel):
    message_id: str
    role: str
    content: str
    time_scope: Optional[RagTimeScope] = None
    period: Optional[str] = None
    period_source: Optional[str] = None
    created_at: datetime


class RagConversationResponse(AnalyticsBaseModel):
    conversation_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None
    messages: list[RagConversationMessageResponse] = Field(default_factory=list)


class AnalyticsPeriodsResponse(AnalyticsBaseModel):
    periods: list[str] = Field(default_factory=list)
    count: int


class AnalyticsPeriodOverviewResponse(AnalyticsBaseModel):
    statement_period: Optional[str] = Field(default=None, alias="statementPeriod")
    payment_method: Optional[str] = Field(default=None, alias="paymentMethod")
    account: Optional[str] = None
    total_amount: Decimal = Field(alias="totalAmount")
    transaction_count: int = Field(alias="transactionCount")


class AnalyticsCategoryBreakdownResponse(AnalyticsBaseModel):
    category: str
    total_amount: Decimal = Field(alias="totalAmount")
    transaction_count: int = Field(alias="transactionCount")


class AnalyticsAccountBreakdownResponse(AnalyticsBaseModel):
    account: str
    total_amount: Decimal = Field(alias="totalAmount")
    transaction_count: int = Field(alias="transactionCount")


class AnalyticsPaymentMethodBreakdownResponse(AnalyticsBaseModel):
    payment_method: str = Field(alias="paymentMethod")
    total_amount: Decimal = Field(alias="totalAmount")
    transaction_count: int = Field(alias="transactionCount")


class AnalyticsDailyTotalResponse(AnalyticsBaseModel):
    date: date
    total_amount: Decimal = Field(alias="totalAmount")
    transaction_count: int = Field(alias="transactionCount")


class AnalyticsCriticalityBreakdownResponse(AnalyticsBaseModel):
    criticality: str
    total_amount: Decimal = Field(alias="totalAmount")
    transaction_count: int = Field(alias="transactionCount")


class AnalyticsDuplicateResponse(AnalyticsBaseModel):
    row_hash: str = Field(alias="rowHash")
    occurrences: int
    total_amount: Decimal = Field(alias="totalAmount")


class BudgetTransactionResponse(AnalyticsBaseModel):
    id: Optional[int] = None
    name: str
    amount: Decimal
    category: str
    criticality: str
    transaction_date: date = Field(alias="transactionDate")
    account: str
    status: Optional[str] = None
    created_time: Optional[str] = Field(default=None, alias="createdTime")
    payment_method: str = Field(alias="paymentMethod")
    statement_period: str = Field(alias="statementPeriod")
    row_hash: Optional[str] = Field(default=None, alias="rowHash")


class AnalyticsStatementPeriodSummaryResponse(AnalyticsBaseModel):
    statement_period: str = Field(alias="statementPeriod")
    period_start_date: Optional[date] = Field(default=None, alias="periodStartDate")
    period_end_date: Optional[date] = Field(default=None, alias="periodEndDate")
    total_amount: Decimal = Field(alias="totalAmount")
    transaction_count: int = Field(alias="transactionCount")
    essential_amount: Decimal = Field(alias="essentialAmount")
    essential_count: int = Field(alias="essentialCount")
    nonessential_amount: Decimal = Field(alias="nonessentialAmount")
    nonessential_count: int = Field(alias="nonessentialCount")
    category_breakdown: dict[str, list[AnalyticsCategoryBreakdownResponse]] = Field(
        default_factory=dict,
        alias="categoryBreakdown",
    )
    criticality_breakdown: dict[str, list[AnalyticsCriticalityBreakdownResponse]] = Field(
        default_factory=dict,
        alias="criticalityBreakdown",
    )
    account_breakdown: dict[str, AnalyticsAccountBreakdownResponse] = Field(
        default_factory=dict,
        alias="accountBreakdown",
    )
    payment_method_breakdown: dict[str, list[AnalyticsPaymentMethodBreakdownResponse]] = Field(
        default_factory=dict,
        alias="paymentMethodBreakdown",
    )
    outliers: dict[str, list[BudgetTransactionResponse]] = Field(default_factory=dict)
    generated_at: datetime = Field(alias="generatedAt")


class InsightMetric(AnalyticsBaseModel):
    label: str
    value: Union[Decimal, int, str]
    direction: Optional[str] = None
    summary: Optional[str] = None


class InsightComparisonMetric(AnalyticsBaseModel):
    label: str
    current_value: Union[Decimal, int, str] = Field(alias="currentValue")
    previous_value: Union[Decimal, int, str] = Field(alias="previousValue")
    absolute_change: Union[Decimal, int, str] = Field(alias="absoluteChange")
    percent_change: Optional[Decimal] = Field(default=None, alias="percentChange")
    direction: str
    summary: str


class InsightFlag(AnalyticsBaseModel):
    kind: str
    severity: str
    title: str
    summary: str
    related_value: Optional[Union[Decimal, int, str]] = None


class InsightCategorySummary(AnalyticsBaseModel):
    category: str
    total_amount: Decimal
    transaction_count: int
    share_of_spend: Decimal


class InsightAccountSummary(AnalyticsBaseModel):
    account: str
    total_amount: Decimal
    transaction_count: int
    share_of_spend: Decimal


class InsightPaymentMethodSummary(AnalyticsBaseModel):
    payment_method: str
    total_amount: Decimal
    transaction_count: int
    share_of_spend: Decimal


class InsightPeriodSummaryResponse(AnalyticsBaseModel):
    period: str
    overview: AnalyticsPeriodOverviewResponse
    top_categories: list[InsightCategorySummary] = Field(default_factory=list)
    top_accounts: list[InsightAccountSummary] = Field(default_factory=list)
    top_payment_methods: list[InsightPaymentMethodSummary] = Field(default_factory=list)
    metrics: list[InsightMetric] = Field(default_factory=list)
    flags: list[InsightFlag] = Field(default_factory=list)


class InsightBehaviorSummaryResponse(AnalyticsBaseModel):
    period: str
    behavior_summary: list[str] = Field(default_factory=list)
    metrics: list[InsightMetric] = Field(default_factory=list)
    flags: list[InsightFlag] = Field(default_factory=list)


class InsightAveragesResponse(AnalyticsBaseModel):
    period: str
    active_days: int = Field(alias="activeDays")
    total_spend: Decimal = Field(alias="totalSpend")
    transaction_count: int = Field(alias="transactionCount")
    average_transaction_amount: Decimal = Field(alias="averageTransactionAmount")
    average_daily_spend: Decimal = Field(alias="averageDailySpend")
    average_daily_transaction_count: Decimal = Field(alias="averageDailyTransactionCount")
    metrics: list[InsightMetric] = Field(default_factory=list)


class InsightMonthOverMonthResponse(AnalyticsBaseModel):
    period: str
    previous_period: Optional[str] = Field(default=None, alias="previousPeriod")
    metrics: list[InsightComparisonMetric] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)


class RagAnswerResponse(AnalyticsBaseModel):
    question: str
    conversation_id: Optional[str] = None
    time_scope: RagTimeScope
    period: Optional[str] = None
    plan: list[str] = Field(default_factory=list)
    tool_selection: RagToolSelectionResponse = Field(default_factory=RagToolSelectionResponse)
    context: dict[str, Any] = Field(default_factory=dict)
    citations: list[RagCitationResponse] = Field(default_factory=list)
    tool_traces: list[RagToolTraceResponse] = Field(default_factory=list)
    cache: RagCacheMetadataResponse = Field(default_factory=RagCacheMetadataResponse)
    timing: RagTimingMetadataResponse = Field(default_factory=RagTimingMetadataResponse)
    answer: str