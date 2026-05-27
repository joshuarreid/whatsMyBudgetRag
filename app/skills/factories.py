from __future__ import annotations

from app.clients.spring_boot_client import SpringBootClient
from app.services.analytics_service import AnalyticsService
from app.services.insight_service import InsightService
from app.skills.analytics import (
    AccountBreakdownSkill,
    AvailablePeriodsSkill,
    CategoriesSkill,
    CriticalitySkill,
    DailyTotalsSkill,
    DuplicatesSkill,
    OutliersSkill,
    OverviewSkill,
    PaymentMethodsSkill,
    StatementPeriodSummaryRangeSkill,
    StatementPeriodSummarySkill,
    TopCategoriesSkill,
    UncategorizedSkill,
)
from app.skills.insights import (
    AveragesSkill,
    BehaviorSummarySkill,
    MonthOverMonthSkill,
    PeriodSummarySkill,
)
from app.skills.registry import SkillRegistry


def build_skill_registry(
    spring_client: SpringBootClient,
    analytics_service: AnalyticsService,
    insight_service: InsightService,
) -> SkillRegistry:
    return SkillRegistry(
        skills=[
            AvailablePeriodsSkill(spring_client),
            OverviewSkill(analytics_service),
            CategoriesSkill(spring_client),
            TopCategoriesSkill(spring_client),
            AccountBreakdownSkill(spring_client),
            PaymentMethodsSkill(spring_client),
            DailyTotalsSkill(spring_client),
            AveragesSkill(insight_service),
            MonthOverMonthSkill(insight_service),
            PeriodSummarySkill(insight_service),
            BehaviorSummarySkill(insight_service),
            CriticalitySkill(spring_client),
            DuplicatesSkill(spring_client),
            UncategorizedSkill(spring_client),
            OutliersSkill(spring_client),
            StatementPeriodSummarySkill(spring_client),
            StatementPeriodSummaryRangeSkill(spring_client),
        ]
    )


