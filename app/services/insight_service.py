from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Union, cast

from app.clients.spring_boot_client import SpringBootClient
from app.core.config import get_settings
from app.models.schemas import (
    AnalyticsAccountBreakdownResponse,
    AnalyticsCategoryBreakdownResponse,
    AnalyticsDailyTotalResponse,
    AnalyticsPaymentMethodBreakdownResponse,
    AnalyticsPeriodOverviewResponse,
    BudgetTransactionResponse,
    InsightAccountSummary,
    InsightAveragesResponse,
    InsightBehaviorSummaryResponse,
    InsightCategorySummary,
    InsightComparisonMetric,
    InsightFlag,
    InsightMetric,
    InsightMonthOverMonthResponse,
    InsightPaymentMethodSummary,
    InsightPeriodSummaryResponse,
    RagTimeScope,
)
from app.services.analytics_service import AnalyticsService
from app.services.normalizers import normalize_list_response, normalize_periods_response


class InsightService:
    """Builds derived summaries and anomaly flags from structured analytics endpoints."""

    def __init__(
        self,
        client: SpringBootClient,
        analytics_service: AnalyticsService,
    ) -> None:
        self.client = client
        self.analytics = analytics_service
        settings = get_settings()
        self.high_share_threshold = Decimal(str(settings.insight_high_share_threshold))
        self.outlier_amount_threshold = Decimal(str(settings.insight_outlier_amount_threshold))

    def period_summary(
        self,
        period: Optional[str] = None,
        *,
        time_scope: Optional[RagTimeScope] = None,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> InsightPeriodSummaryResponse:
        resolved_time_scope = self._resolve_time_scope(period=period, time_scope=time_scope)
        scope_label = self._time_scope_label(resolved_time_scope)
        overview = self.analytics.period_overview(
            period=period,
            time_scope=resolved_time_scope,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )
        categories = cast(
            list[AnalyticsCategoryBreakdownResponse],
            normalize_list_response(
                self.client.get_top_categories_for_time_scope(
                    time_scope=resolved_time_scope,
                    limit=5,
                    payment_method=payment_method,
                    account=account,
                    transaction_id=transaction_id,
                ),
                AnalyticsCategoryBreakdownResponse,
            ),
        )
        accounts = cast(
            list[AnalyticsAccountBreakdownResponse],
            normalize_list_response(
                self.client.get_account_breakdown_for_time_scope(
                    time_scope=resolved_time_scope,
                    payment_method=payment_method,
                    transaction_id=transaction_id,
                ),
                AnalyticsAccountBreakdownResponse,
            ),
        )
        payment_methods = cast(
            list[AnalyticsPaymentMethodBreakdownResponse],
            normalize_list_response(
                self.client.get_payment_method_breakdown_for_time_scope(
                    time_scope=resolved_time_scope,
                    account=account,
                    transaction_id=transaction_id,
                ),
                AnalyticsPaymentMethodBreakdownResponse,
            ),
        )
        outliers = cast(
            list[BudgetTransactionResponse],
            normalize_list_response(
                self.client.get_outliers_for_time_scope(
                    time_scope=resolved_time_scope,
                    limit=10,
                    transaction_id=transaction_id,
                ),
                BudgetTransactionResponse,
            ),
        )

        total_amount = overview.total_amount
        top_categories = [self._to_category_summary(item, total_amount) for item in categories]
        top_accounts = [self._to_account_summary(item, total_amount) for item in accounts[:5]]
        top_payment_methods = [
            self._to_payment_method_summary(item, total_amount) for item in payment_methods[:5]
        ]
        metrics = self._build_summary_metrics(overview, top_categories, top_accounts, top_payment_methods)
        flags = self._build_summary_flags(top_categories, top_accounts, outliers)

        return InsightPeriodSummaryResponse(
            period=scope_label,
            overview=overview,
            top_categories=top_categories,
            top_accounts=top_accounts,
            top_payment_methods=top_payment_methods,
            metrics=metrics,
            flags=flags,
        )

    def behavior_summary(
        self,
        period: Optional[str] = None,
        *,
        time_scope: Optional[RagTimeScope] = None,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> InsightBehaviorSummaryResponse:
        resolved_time_scope = self._resolve_time_scope(period=period, time_scope=time_scope)
        summary = self.period_summary(
            period=period,
            time_scope=resolved_time_scope,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )
        daily_totals = cast(
            list[AnalyticsDailyTotalResponse],
            normalize_list_response(
                self.client.get_daily_totals_for_time_scope(
                    time_scope=resolved_time_scope,
                    payment_method=payment_method,
                    account=account,
                    transaction_id=transaction_id,
                ),
                AnalyticsDailyTotalResponse,
            ),
        )
        behavior_summary = self._build_behavior_lines(summary, daily_totals)
        behavior_metrics = list(summary.metrics)
        if daily_totals:
            average_daily_spend = (
                sum(item.total_amount for item in daily_totals) / Decimal(len(daily_totals))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            behavior_metrics.append(
                InsightMetric(
                    label="average_daily_spend",
                    value=average_daily_spend,
                    summary="Average spend per active day in the selected period.",
                )
            )

        return InsightBehaviorSummaryResponse(
            period=self._time_scope_label(resolved_time_scope),
            behavior_summary=behavior_summary,
            metrics=behavior_metrics,
            flags=summary.flags,
        )

    def averages(
        self,
        period: Optional[str] = None,
        *,
        time_scope: Optional[RagTimeScope] = None,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> InsightAveragesResponse:
        resolved_time_scope = self._resolve_time_scope(period=period, time_scope=time_scope)
        overview = self.analytics.period_overview(
            period=period,
            time_scope=resolved_time_scope,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )
        daily_totals = cast(
            list[AnalyticsDailyTotalResponse],
            normalize_list_response(
                self.client.get_daily_totals_for_time_scope(
                    time_scope=resolved_time_scope,
                    payment_method=payment_method,
                    account=account,
                    transaction_id=transaction_id,
                ),
                AnalyticsDailyTotalResponse,
            ),
        )
        active_days = len(daily_totals)
        average_transaction_amount = _safe_average(
            overview.total_amount,
            Decimal(overview.transaction_count),
        )
        average_daily_spend = _safe_average(
            sum((item.total_amount for item in daily_totals), Decimal("0.00")),
            Decimal(active_days),
        )
        average_daily_transaction_count = _safe_average(
            Decimal(sum(item.transaction_count for item in daily_totals)),
            Decimal(active_days),
        )
        metrics = [
            InsightMetric(
                label="average_transaction_amount",
                value=average_transaction_amount,
                summary="Average spend per transaction in the selected period.",
            ),
            InsightMetric(
                label="average_daily_spend",
                value=average_daily_spend,
                summary="Average spend per active day in the selected period.",
            ),
            InsightMetric(
                label="average_daily_transaction_count",
                value=average_daily_transaction_count,
                summary="Average number of transactions per active day in the selected period.",
            ),
        ]
        return InsightAveragesResponse(
            period=self._time_scope_label(resolved_time_scope),
            activeDays=active_days,
            totalSpend=overview.total_amount,
            transactionCount=overview.transaction_count,
            averageTransactionAmount=average_transaction_amount,
            averageDailySpend=average_daily_spend,
            averageDailyTransactionCount=average_daily_transaction_count,
            metrics=metrics,
        )

    def month_over_month(
        self,
        period: str,
        payment_method: Optional[str] = None,
        account: Optional[str] = None,
        compare_to: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> InsightMonthOverMonthResponse:
        previous_period = compare_to or self._resolve_previous_period(period, transaction_id)
        if previous_period is None:
            return InsightMonthOverMonthResponse(
                period=period,
                previousPeriod=None,
                metrics=[],
                highlights=["No previous statement period was available for comparison."],
            )

        current_overview = self.analytics.period_overview(
            period=period,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )
        previous_overview = self.analytics.period_overview(
            period=previous_period,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )
        current_averages = self.averages(
            period=period,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )
        previous_averages = self.averages(
            period=previous_period,
            payment_method=payment_method,
            account=account,
            transaction_id=transaction_id,
        )
        metrics = [
            _comparison_metric(
                label="total_spend",
                current_value=current_overview.total_amount,
                previous_value=previous_overview.total_amount,
                summary_label="total spend",
            ),
            _comparison_metric(
                label="transaction_count",
                current_value=current_overview.transaction_count,
                previous_value=previous_overview.transaction_count,
                summary_label="transaction count",
            ),
            _comparison_metric(
                label="average_transaction_amount",
                current_value=current_averages.average_transaction_amount,
                previous_value=previous_averages.average_transaction_amount,
                summary_label="average transaction amount",
            ),
            _comparison_metric(
                label="average_daily_spend",
                current_value=current_averages.average_daily_spend,
                previous_value=previous_averages.average_daily_spend,
                summary_label="average daily spend",
            ),
        ]
        highlights = [metric.summary for metric in metrics]
        return InsightMonthOverMonthResponse(
            period=period,
            previousPeriod=previous_period,
            metrics=metrics,
            highlights=highlights,
        )

    def _build_summary_metrics(
        self,
        overview: AnalyticsPeriodOverviewResponse,
        categories: list[InsightCategorySummary],
        accounts: list[InsightAccountSummary],
        payment_methods: list[InsightPaymentMethodSummary],
    ) -> list[InsightMetric]:
        metrics: list[InsightMetric] = [
            InsightMetric(
                label="total_spend",
                value=overview.total_amount,
                summary="Total actual spend recorded for the selected statement period.",
            ),
            InsightMetric(
                label="transaction_count",
                value=overview.transaction_count,
                summary="Number of actual transactions recorded for the selected statement period.",
            ),
        ]
        if categories:
            metrics.append(
                InsightMetric(
                    label="leading_category",
                    value=categories[0].category,
                    summary=f"Largest spend category at {categories[0].share_of_spend}% of total spend.",
                )
            )
        if accounts:
            metrics.append(
                InsightMetric(
                    label="leading_account",
                    value=accounts[0].account,
                    summary=f"Largest account contribution at {accounts[0].share_of_spend}% of total spend.",
                )
            )
        if payment_methods:
            metrics.append(
                InsightMetric(
                    label="leading_payment_method",
                    value=payment_methods[0].payment_method,
                    summary=(
                        f"Largest payment method contribution at {payment_methods[0].share_of_spend}% of total spend."
                    ),
                )
            )
        return metrics

    def _build_summary_flags(
        self,
        categories: list[InsightCategorySummary],
        accounts: list[InsightAccountSummary],
        outliers: list[BudgetTransactionResponse],
    ) -> list[InsightFlag]:
        flags: list[InsightFlag] = []
        if categories and categories[0].share_of_spend >= self.high_share_threshold:
            flags.append(
                InsightFlag(
                    kind="category_concentration",
                    severity="medium",
                    title="Category concentration detected",
                    summary=(
                        f"{categories[0].category} accounts for {categories[0].share_of_spend}% of total spend."
                    ),
                    related_value=categories[0].share_of_spend,
                )
            )
        if accounts and accounts[0].share_of_spend >= self.high_share_threshold:
            flags.append(
                InsightFlag(
                    kind="account_concentration",
                    severity="medium",
                    title="Account concentration detected",
                    summary=(
                        f"{accounts[0].account} accounts for {accounts[0].share_of_spend}% of total spend."
                    ),
                    related_value=accounts[0].share_of_spend,
                )
            )
        large_outliers = [item for item in outliers if item.amount >= self.outlier_amount_threshold]
        if large_outliers:
            flags.append(
                InsightFlag(
                    kind="large_outlier_transactions",
                    severity="high",
                    title="Large outlier transactions detected",
                    summary=(
                        f"{len(large_outliers)} transactions exceed the configured outlier threshold of "
                        f"{self.outlier_amount_threshold}."
                    ),
                    related_value=len(large_outliers),
                )
            )
        return flags

    def _build_behavior_lines(
        self,
        summary: InsightPeriodSummaryResponse,
        daily_totals: list[AnalyticsDailyTotalResponse],
    ) -> list[str]:
        lines: list[str] = []
        if summary.top_categories:
            lead = summary.top_categories[0]
            lines.append(
                f"Spending is concentrated in {lead.category}, which represents {lead.share_of_spend}% of total spend."
            )
        if summary.top_accounts:
            lead_account = summary.top_accounts[0]
            lines.append(
                f"{lead_account.account} is the primary funding source at {lead_account.share_of_spend}% of total spend."
            )
        if daily_totals:
            peak_day = max(daily_totals, key=lambda item: item.total_amount)
            lines.append(
                f"Peak daily spend occurred on {peak_day.date} with total spend of {peak_day.total_amount}."
            )
        if not lines:
            lines.append("No derived behavior insights were available for the selected period.")
        return lines

    def _resolve_previous_period(
        self,
        period: str,
        transaction_id: Optional[str],
    ) -> Optional[str]:
        periods = normalize_periods_response(
            self.client.get_periods(transaction_id=transaction_id)
        ).periods
        ordered_periods = sorted(periods)
        if period not in ordered_periods:
            return ordered_periods[-1] if ordered_periods else None
        current_index = ordered_periods.index(period)
        if current_index == 0:
            return None
        return ordered_periods[current_index - 1]

    @staticmethod
    def _resolve_time_scope(period: Optional[str], time_scope: Optional[RagTimeScope]) -> RagTimeScope:
        if time_scope is not None:
            return time_scope
        if period:
            return RagTimeScope.from_period(period)
        raise ValueError("Insight calculations require either a statement period or a date range time_scope")

    @staticmethod
    def _time_scope_label(time_scope: RagTimeScope) -> str:
        return time_scope.label

    @staticmethod
    def _to_category_summary(
        item: AnalyticsCategoryBreakdownResponse,
        total_amount: Decimal,
    ) -> InsightCategorySummary:
        return InsightCategorySummary(
            category=item.category,
            total_amount=item.total_amount,
            transaction_count=item.transaction_count,
            share_of_spend=_share_of_spend(item.total_amount, total_amount),
        )

    @staticmethod
    def _to_account_summary(
        item: AnalyticsAccountBreakdownResponse,
        total_amount: Decimal,
    ) -> InsightAccountSummary:
        return InsightAccountSummary(
            account=item.account,
            total_amount=item.total_amount,
            transaction_count=item.transaction_count,
            share_of_spend=_share_of_spend(item.total_amount, total_amount),
        )

    @staticmethod
    def _to_payment_method_summary(
        item: AnalyticsPaymentMethodBreakdownResponse,
        total_amount: Decimal,
    ) -> InsightPaymentMethodSummary:
        return InsightPaymentMethodSummary(
            payment_method=item.payment_method,
            total_amount=item.total_amount,
            transaction_count=item.transaction_count,
            share_of_spend=_share_of_spend(item.total_amount, total_amount),
        )


def _share_of_spend(amount: Decimal, total_amount: Decimal) -> Decimal:
    if total_amount == 0:
        return Decimal("0.00")
    return ((amount / total_amount) * Decimal("100")).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


def _safe_average(total: Decimal, divisor: Decimal) -> Decimal:
    if divisor == 0:
        return Decimal("0.00")
    return (total / divisor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _comparison_metric(
    label: str,
    current_value: Union[Decimal, int],
    previous_value: Union[Decimal, int],
    summary_label: str,
) -> InsightComparisonMetric:
    current_decimal = Decimal(str(current_value))
    previous_decimal = Decimal(str(previous_value))
    absolute_change = (current_decimal - previous_decimal).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    if previous_decimal == 0:
        percent_change: Optional[Decimal] = None
    else:
        percent_change = ((absolute_change / previous_decimal) * Decimal("100")).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    if absolute_change > 0:
        direction = "up"
    elif absolute_change < 0:
        direction = "down"
    else:
        direction = "flat"
    percent_text = f" ({percent_change}%)" if percent_change is not None else ""
    summary = (
        f"{summary_label.capitalize()} moved {direction} by {absolute_change}{percent_text} compared with the previous period."
    )
    return InsightComparisonMetric(
        label=label,
        currentValue=current_value,
        previousValue=previous_value,
        absoluteChange=absolute_change,
        percentChange=percent_change,
        direction=direction,
        summary=summary,
    )