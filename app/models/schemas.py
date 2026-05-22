from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AnalyticsBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class RagAskRequest(AnalyticsBaseModel):
    question: str = Field(min_length=1)
    period: Optional[str] = None
    payment_method: Optional[str] = None
    account: Optional[str] = None
    transaction_id: Optional[str] = None


class AnalyticsPeriodsResponse(AnalyticsBaseModel):
    periods: list[str] = Field(default_factory=list)
    count: int


class AnalyticsPeriodOverviewResponse(AnalyticsBaseModel):
    statement_period: str = Field(alias="statementPeriod")
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
    period: str
    plan: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    answer: str