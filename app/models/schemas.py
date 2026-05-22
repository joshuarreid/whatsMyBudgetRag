from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalyticsBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class RagAskRequest(AnalyticsBaseModel):
    question: str = Field(min_length=1)
    period: str | None = None
    payment_method: str | None = None
    account: str | None = None
    transaction_id: str | None = None


class AnalyticsPeriodsResponse(AnalyticsBaseModel):
    periods: list[str] = Field(default_factory=list)
    count: int


class AnalyticsPeriodOverviewResponse(AnalyticsBaseModel):
    statement_period: str = Field(alias="statementPeriod")
    payment_method: str | None = Field(default=None, alias="paymentMethod")
    account: str | None = None
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
    id: int | None = None
    name: str
    amount: Decimal
    category: str
    criticality: str
    transaction_date: date = Field(alias="transactionDate")
    account: str
    status: str | None = None
    created_time: str | None = Field(default=None, alias="createdTime")
    payment_method: str = Field(alias="paymentMethod")
    statement_period: str = Field(alias="statementPeriod")
    row_hash: str | None = Field(default=None, alias="rowHash")


class RagAnswerResponse(AnalyticsBaseModel):
    question: str
    period: str
    plan: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    answer: str