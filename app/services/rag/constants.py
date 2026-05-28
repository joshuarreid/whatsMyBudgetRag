from decimal import Decimal

TOOL_CACHE_TTL_SECONDS = 900
MAX_RESPONSE_CITATIONS = 5
TREND_ROUTING_KEYWORDS = ("daily", "trend", "over time", "time series")
SUMMARY_ROUTING_KEYWORDS = ("overview", "summary", "summarize", "total", "how much")
ANSWER_CONTEXT_EXCLUDED_KEYS = {
    "cache",
    "conversation",
    "conversation_history",
    "execution_plan",
    "routing",
    "supporting_sources",
    "tool_trace_summaries",
}
DAILY_CONTEXT_KEY_PREFIX = "daily_totals"
DECIMAL_CENTS = Decimal("0.01")

