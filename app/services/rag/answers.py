from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Optional

from app.services.rag.constants import (
    ANSWER_CONTEXT_EXCLUDED_KEYS,
    DAILY_CONTEXT_KEY_PREFIX,
    DECIMAL_CENTS,
)


class RAGAnswerMixin:
    def _build_answer_context(self, context: dict[str, Any]) -> dict[str, Any]:
        answer_context: dict[str, Any] = {}
        daily_compact_sections: dict[str, Any] = {}
        for key, value in context.items():
            if key in ANSWER_CONTEXT_EXCLUDED_KEYS:
                continue
            if key == "unavailable_tools" and isinstance(value, list):
                answer_context[key] = [
                    {
                        "tool": item.get("tool"),
                        "detail": item.get("detail"),
                        "label": item.get("label"),
                    }
                    for item in value
                    if isinstance(item, dict)
                ]
                continue
            if self._is_daily_context_key(key) and isinstance(value, list):
                daily_compact_sections[key] = self._compact_daily_series(
                    context_key=key,
                    rows=value,
                    label=self._daily_context_label(context, key),
                )
                continue
            answer_context[key] = value
        if daily_compact_sections:
            answer_context["daily_trend_summary"] = self._compact_daily_trend_summary(context, daily_compact_sections)
        return answer_context

    @classmethod
    def _is_daily_context_key(cls, key: str) -> bool:
        return key == DAILY_CONTEXT_KEY_PREFIX or key.startswith(f"{DAILY_CONTEXT_KEY_PREFIX}_")

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        return format(value.quantize(DECIMAL_CENTS), "f")

    @staticmethod
    def _to_decimal(value: Any) -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return Decimal("0")

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _parse_iso_date(value: Any) -> Optional[date]:
        if isinstance(value, date):
            return value
        if not isinstance(value, str) or not value:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_period_label(period: Optional[str]) -> Optional[date]:
        if not isinstance(period, str) or not period.strip():
            return None
        try:
            return datetime.strptime(period.strip(), "%B%Y").date()
        except ValueError:
            return None

    @classmethod
    def _sort_period_labels(cls, periods: list[str]) -> list[str]:
        return sorted(
            periods,
            key=lambda period: cls._parse_period_label(period) or date.max,
        )

    def _daily_context_label(self, context: dict[str, Any], context_key: str) -> str:
        execution_plan = context.get("execution_plan")
        if isinstance(execution_plan, dict):
            steps = execution_plan.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    if step.get("output_key") == context_key and isinstance(step.get("label"), str):
                        return str(step["label"])
        if context_key == DAILY_CONTEXT_KEY_PREFIX:
            return context.get("period") or "selected period"
        suffix = context_key[len(f"{DAILY_CONTEXT_KEY_PREFIX}_") :] if context_key.startswith(f"{DAILY_CONTEXT_KEY_PREFIX}_") else context_key
        if suffix:
            parts = [part.capitalize() for part in suffix.split("_") if part]
            if parts:
                return "".join(parts)
        return context_key

    def _normalize_daily_rows(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_date = self._parse_iso_date(row.get("date"))
            if row_date is None:
                continue
            normalized_row = {
                "date": row_date.isoformat(),
                "date_value": row_date,
                "total_amount": self._to_decimal(row.get("total_amount")),
                "transaction_count": self._to_int(row.get("transaction_count")),
            }
            grouped_rows.setdefault(row_date.isoformat(), []).append(normalized_row)

        normalized_rows: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        for row_date in sorted(grouped_rows.keys()):
            entries = grouped_rows[row_date]
            unique_entries: list[tuple[Decimal, int]] = []
            for entry in entries:
                signature = (entry["total_amount"], entry["transaction_count"])
                if signature not in unique_entries:
                    unique_entries.append(signature)
            chosen_entry = max(entries, key=lambda entry: (entry["total_amount"], entry["transaction_count"]))
            normalized_rows.append(chosen_entry)
            if len(unique_entries) > 1:
                conflicts.append(
                    {
                        "date": row_date,
                        "reported_values": [
                            {
                                "total_amount": self._format_decimal(total_amount),
                                "transaction_count": transaction_count,
                            }
                            for total_amount, transaction_count in unique_entries
                        ],
                        "selected_value": {
                            "total_amount": self._format_decimal(chosen_entry["total_amount"]),
                            "transaction_count": chosen_entry["transaction_count"],
                        },
                    }
                )
        return normalized_rows, conflicts

    def _compact_daily_series(self, *, context_key: str, rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        normalized_rows, conflicts = self._normalize_daily_rows(rows)
        total_spend = sum((row["total_amount"] for row in normalized_rows), start=Decimal("0"))
        total_transactions = sum(row["transaction_count"] for row in normalized_rows)
        peak_day = max(normalized_rows, key=lambda row: row["total_amount"], default=None)
        return {
            "context_key": context_key,
            "label": label,
            "active_days": len(normalized_rows),
            "total_spend": self._format_decimal(total_spend),
            "transaction_count": total_transactions,
            "average_daily_spend": self._format_decimal(total_spend / len(normalized_rows)) if normalized_rows else "0.00",
            "date_range": {
                "start_date": normalized_rows[0]["date"] if normalized_rows else None,
                "end_date": normalized_rows[-1]["date"] if normalized_rows else None,
            },
            "peak_day": (
                {
                    "date": peak_day["date"],
                    "total_amount": self._format_decimal(peak_day["total_amount"]),
                    "transaction_count": peak_day["transaction_count"],
                }
                if peak_day is not None
                else None
            ),
            "conflicts": conflicts,
        }

    def _compact_daily_trend_summary(
        self,
        context: dict[str, Any],
        daily_compact_sections: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        time_scope = self._context_time_scope(context)
        ordered_sections = [
            daily_compact_sections[key]
            for key in sorted(
                daily_compact_sections.keys(),
                key=lambda item: self._parse_period_label(self._daily_context_label(context, item)) or date.max,
            )
        ]
        highest_peak = max(
            (section["peak_day"] for section in ordered_sections if isinstance(section.get("peak_day"), dict)),
            key=lambda peak: self._to_decimal(peak.get("total_amount")),
            default=None,
        )
        return {
            "range_label": self._time_scope_label(time_scope) if time_scope is not None else "selected range",
            "series": ordered_sections,
            "highest_peak_day": highest_peak,
            "conflicts": [
                {"label": section["label"], "items": section["conflicts"]}
                for section in ordered_sections
                if section.get("conflicts")
            ],
        }

    def _deterministic_answer(self, context: dict[str, Any]) -> Optional[str]:
        period_category_average_answer = self._deterministic_period_category_average_answer(context)
        if period_category_average_answer is not None:
            return period_category_average_answer
        if self._is_deterministic_weekly_category_context(context):
            return self._deterministic_weekly_category_answer(context)
        if self._is_deterministic_daily_range_context(context):
            return self._deterministic_daily_range_answer(context)
        if self._is_deterministic_single_scope_daily_context(context):
            return self._deterministic_single_scope_daily_answer(context)
        return None

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        normalized = value.lower().replace("’", "'")
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _is_deterministic_weekly_category_context(self, context: dict[str, Any]) -> bool:
        time_scope = self._context_time_scope(context)
        if time_scope is None or time_scope.scope_type != "statement_period":
            return False
        execution_plan = context.get("execution_plan")
        if not isinstance(execution_plan, dict):
            return False
        steps = execution_plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return False
        weekly_steps = [step for step in steps if isinstance(step, dict) and str(step.get("label", "")).startswith("Week ")]
        if not weekly_steps:
            return False
        plan_skill_ids = {
            str(step.get("skill_id"))
            for step in weekly_steps
            if isinstance(step, dict) and step.get("skill_id")
        }
        return bool(plan_skill_ids) and plan_skill_ids.issubset({"categories", "top_categories"})

    def _deterministic_weekly_category_answer(self, context: dict[str, Any]) -> str:
        execution_plan = context.get("execution_plan")
        steps = execution_plan.get("steps") if isinstance(execution_plan, dict) else None
        if not isinstance(steps, list):
            return self._fallback_answer(context)

        weekly_steps = [step for step in steps if isinstance(step, dict) and str(step.get("label", "")).startswith("Week ")]
        if not weekly_steps:
            return self._fallback_answer(context)

        preferred_skill_id = (
            "top_categories"
            if any(step.get("skill_id") == "top_categories" for step in weekly_steps)
            else "categories"
        )
        ordered_steps = [step for step in weekly_steps if step.get("skill_id") == preferred_skill_id]
        if not ordered_steps:
            return self._fallback_answer(context)

        selected_period = context.get("period") or "the selected month"
        lines = [f"Here are the highest spending categories per week for {selected_period}:", ""]

        for step in ordered_steps:
            label = str(step.get("label") or "Week")
            output_key = step.get("output_key")
            payload = context.get(output_key) if isinstance(output_key, str) else None
            if not isinstance(payload, list) or not payload:
                lines.append(f"- {label}: no category spend was returned for that week.")
                continue

            ranked_rows = [row for row in payload if isinstance(row, dict) and row.get("category")]
            if not ranked_rows:
                lines.append(f"- {label}: no category spend was returned for that week.")
                continue

            max_total = max((self._to_decimal(row.get("total_amount")) for row in ranked_rows), default=Decimal("0"))
            top_rows = [row for row in ranked_rows if self._to_decimal(row.get("total_amount")) == max_total]
            if len(top_rows) == 1:
                top_row = top_rows[0]
                lines.append(
                    f"- {label}: {top_row.get('category')} at ${self._format_decimal(max_total)} across {self._to_int(top_row.get('transaction_count'))} transactions."
                )
                continue

            tied_categories = ", ".join(
                f"{row.get('category')} (${self._format_decimal(self._to_decimal(row.get('total_amount')))}, {self._to_int(row.get('transaction_count'))} txns)"
                for row in top_rows
            )
            lines.append(f"- {label}: tie between {tied_categories}.")

        return "\n".join(lines)

    def _daily_sections(self, context: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
        daily_sections: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
        for key, value in context.items():
            if not self._is_daily_context_key(key) or not isinstance(value, list):
                continue
            normalized_rows, conflicts = self._normalize_daily_rows(value)
            daily_sections.append((self._daily_context_label(context, key), normalized_rows, conflicts))
        daily_sections.sort(key=lambda item: self._parse_period_label(item[0]) or date.max)
        return daily_sections

    @staticmethod
    def _plan_skill_ids(context: dict[str, Any]) -> list[str]:
        execution_plan = context.get("execution_plan")
        if not isinstance(execution_plan, dict):
            return []
        steps = execution_plan.get("steps")
        if not isinstance(steps, list):
            return []
        return [str(step.get("skill_id")) for step in steps if isinstance(step, dict) and step.get("skill_id")]

    def _category_sections(self, context: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
        execution_plan = context.get("execution_plan")
        if not isinstance(execution_plan, dict):
            return []
        steps = execution_plan.get("steps")
        if not isinstance(steps, list):
            return []

        sections: list[tuple[str, list[dict[str, Any]]]] = []
        for step in steps:
            if not isinstance(step, dict) or step.get("skill_id") != "categories":
                continue
            output_key = step.get("output_key")
            if not isinstance(output_key, str):
                continue
            payload = context.get(output_key)
            if not isinstance(payload, list):
                continue
            label = str(step.get("label") or step.get("period") or output_key)
            rows = [row for row in payload if isinstance(row, dict) and row.get("category")]
            sections.append((label, rows))
        return sections

    def _statement_period_summary_lookup(self, context: dict[str, Any]) -> dict[str, dict[str, Any]]:
        payload = context.get("statement_period_summary_range")
        if not isinstance(payload, list):
            return {}

        lookup: dict[str, dict[str, Any]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            statement_period = item.get("statement_period")
            if not isinstance(statement_period, str) or not statement_period.strip():
                continue
            parsed_period = self._parse_period_label(statement_period.title())
            if parsed_period is None:
                continue
            lookup[parsed_period.isoformat()] = item
        return lookup

    def _question_category_names_from_sources(
        self,
        *,
        question: str,
        category_sections: list[tuple[str, list[dict[str, Any]]]],
        summary_lookup: dict[str, dict[str, Any]],
    ) -> list[str]:
        normalized_question = f" {self._normalize_text(question)} "
        matched_categories: list[str] = []
        seen_normalized_categories: set[str] = set()

        def consider(category: Any) -> None:
            if not isinstance(category, str) or not category.strip():
                return
            normalized_category = self._normalize_text(category)
            if not normalized_category or normalized_category in seen_normalized_categories:
                return
            if f" {normalized_category} " not in normalized_question:
                return
            seen_normalized_categories.add(normalized_category)
            matched_categories.append(category)

        for _, rows in category_sections:
            for row in rows:
                consider(row.get("category"))

        for item in summary_lookup.values():
            category_breakdown = item.get("category_breakdown")
            if not isinstance(category_breakdown, dict):
                continue
            for entries in category_breakdown.values():
                if not isinstance(entries, list):
                    continue
                for row in entries:
                    if isinstance(row, dict):
                        consider(row.get("category"))

        return matched_categories

    def _summary_category_total(
        self,
        period_summary: dict[str, Any],
        *,
        account: Optional[str],
        category: str,
    ) -> Optional[Decimal]:
        category_breakdown = period_summary.get("category_breakdown")
        if not isinstance(category_breakdown, dict):
            return None

        normalized_category = self._normalize_text(category)

        def total_for_account(account_name: str) -> Decimal:
            entries = category_breakdown.get(account_name)
            if not isinstance(entries, list):
                return Decimal("0")
            return sum(
                (
                    self._to_decimal(row.get("total_amount"))
                    for row in entries
                    if isinstance(row, dict)
                    and self._normalize_text(row.get("category")) == normalized_category
                ),
                start=Decimal("0"),
            )

        if not account:
            return sum(
                (total_for_account(account_name) for account_name in category_breakdown.keys() if isinstance(account_name, str)),
                start=Decimal("0"),
            )

        normalized_account = str(account).strip().lower()
        if normalized_account == "joint":
            return total_for_account("joint")

        return total_for_account(normalized_account) + (total_for_account("joint") / Decimal("2"))

    def _question_category_names(
        self,
        *,
        question: str,
        category_sections: list[tuple[str, list[dict[str, Any]]]],
    ) -> list[str]:
        normalized_question = f" {self._normalize_text(question)} "
        matched_categories: list[str] = []
        seen_normalized_categories: set[str] = set()

        for _, rows in category_sections:
            for row in rows:
                category = row.get("category")
                if not isinstance(category, str) or not category.strip():
                    continue
                normalized_category = self._normalize_text(category)
                if not normalized_category or normalized_category in seen_normalized_categories:
                    continue
                if f" {normalized_category} " not in normalized_question:
                    continue
                seen_normalized_categories.add(normalized_category)
                matched_categories.append(category)
        return matched_categories

    def _deterministic_period_category_average_answer(self, context: dict[str, Any]) -> Optional[str]:
        time_scope = self._context_time_scope(context)
        if time_scope is None or time_scope.scope_type != "statement_period_range":
            return None

        normalized_question = self._normalize_text(context.get("question"))
        if not any(keyword in normalized_question for keyword in ("average", "avg", "mean")):
            return None
        if "month" not in normalized_question:
            return None

        category_sections = self._category_sections(context)
        if len(category_sections) < 2:
            return None

        summary_lookup = self._statement_period_summary_lookup(context)

        matched_categories = self._question_category_names_from_sources(
            question=str(context.get("question") or ""),
            category_sections=category_sections,
            summary_lookup=summary_lookup,
        )
        if not matched_categories:
            return None

        month_count = len(category_sections)
        period_labels = [label for label, _ in category_sections]
        account = None
        filters = context.get("filters")
        if isinstance(filters, dict) and isinstance(filters.get("account"), str):
            account = str(filters.get("account"))
        lines = [
            f"Average monthly spend from {period_labels[0]} through {period_labels[-1]} ({month_count} months):",
            "",
        ]

        for category in matched_categories:
            normalized_category = self._normalize_text(category)
            monthly_amounts: list[Decimal] = []
            for label, rows in category_sections:
                summary_item = None
                parsed_label = self._parse_period_label(label)
                if parsed_label is not None:
                    summary_item = summary_lookup.get(parsed_label.isoformat())
                if isinstance(summary_item, dict):
                    summary_total = self._summary_category_total(summary_item, account=account, category=category)
                    if summary_total is not None:
                        monthly_amounts.append(summary_total)
                        continue
                matched_row = next(
                    (
                        row
                        for row in rows
                        if self._normalize_text(row.get("category")) == normalized_category
                    ),
                    None,
                )
                monthly_amounts.append(self._to_decimal(matched_row.get("total_amount") if matched_row is not None else 0))

            total_amount = sum(monthly_amounts, start=Decimal("0"))
            average_amount = total_amount / month_count if month_count else Decimal("0")
            lines.append(
                f"- {category}: ${self._format_decimal(average_amount)}/month average (${self._format_decimal(total_amount)} total across {month_count} months)."
            )

        lines.extend(["", "## Monthly category totals"])
        for label, rows in category_sections:
            row_totals: dict[str, Decimal] = {}
            parsed_label = self._parse_period_label(label)
            summary_item = summary_lookup.get(parsed_label.isoformat()) if parsed_label is not None else None
            if isinstance(summary_item, dict):
                for category in matched_categories:
                    summary_total = self._summary_category_total(summary_item, account=account, category=category)
                    if summary_total is not None:
                        row_totals[self._normalize_text(category)] = summary_total
            if not row_totals:
                row_totals = {
                    self._normalize_text(row.get("category")): self._to_decimal(row.get("total_amount"))
                    for row in rows
                }
            category_summaries = ", ".join(
                f"{category} ${self._format_decimal(row_totals.get(self._normalize_text(category), Decimal('0')))}"
                for category in matched_categories
            )
            lines.append(f"- {label}: {category_summaries}")

        return "\n".join(lines)

    def _is_deterministic_daily_range_context(self, context: dict[str, Any]) -> bool:
        time_scope = self._context_time_scope(context)
        if time_scope is None or time_scope.scope_type != "statement_period_range":
            return False
        plan_skill_ids = self._plan_skill_ids(context)
        return bool(plan_skill_ids) and all(skill_id == "daily" for skill_id in plan_skill_ids)

    def _is_deterministic_single_scope_daily_context(self, context: dict[str, Any]) -> bool:
        time_scope = self._context_time_scope(context)
        if time_scope is None or time_scope.scope_type != "date_range":
            return False
        daily_sections = self._daily_sections(context)
        if len(daily_sections) != 1:
            return False
        plan_skill_ids = set(self._plan_skill_ids(context))
        return bool(plan_skill_ids) and plan_skill_ids.issubset({"overview", "daily"})

    @staticmethod
    def _trend_direction(current_value: Decimal, next_value: Decimal) -> str:
        if next_value > current_value:
            return "increased"
        if next_value < current_value:
            return "decreased"
        return "stayed flat"

    def _deterministic_daily_range_answer(self, context: dict[str, Any]) -> str:
        daily_sections = self._daily_sections(context)
        if not daily_sections:
            return self._fallback_answer(context)
        scope = self._context_time_scope(context)
        resolved_label = self._time_scope_label(scope) if scope is not None else "selected range"
        lines = [f"Period resolved: {resolved_label}.", "", "## Daily spending trend"]

        monthly_totals: list[tuple[str, Decimal]] = []
        overall_peak: Optional[tuple[str, dict[str, Any]]] = None
        conflict_lines: list[str] = []

        for label, rows, conflicts in daily_sections:
            total_spend = sum((row["total_amount"] for row in rows), start=Decimal("0"))
            average_daily_spend = total_spend / len(rows) if rows else Decimal("0")
            peak_day = max(rows, key=lambda row: row["total_amount"], default=None)
            monthly_totals.append((label, total_spend))
            if peak_day is not None and (overall_peak is None or peak_day["total_amount"] > overall_peak[1]["total_amount"]):
                overall_peak = (label, peak_day)
            summary_line = (
                f"- {label}: {len(rows)} active days, ${self._format_decimal(total_spend)} total spend, "
                f"${self._format_decimal(average_daily_spend)} average spend on active days"
            )
            if peak_day is not None:
                summary_line += (
                    f", peak day {peak_day['date']} at ${self._format_decimal(peak_day['total_amount'])} "
                    f"across {peak_day['transaction_count']} transactions"
                )
            summary_line += "."
            lines.append(summary_line)
            for conflict in conflicts:
                reported = ", ".join(
                    f"${item['total_amount']} ({item['transaction_count']} txns)"
                    for item in conflict["reported_values"]
                )
                selected = conflict["selected_value"]
                conflict_lines.append(
                    f"- {label} {conflict['date']}: reported values {reported}; used ${selected['total_amount']} ({selected['transaction_count']} txns)."
                )

        trend_lines: list[str] = []
        for index in range(len(monthly_totals) - 1):
            current_label, current_total = monthly_totals[index]
            next_label, next_total = monthly_totals[index + 1]
            trend_lines.append(
                f"- Total spend {self._trend_direction(current_total, next_total)} from {current_label} (${self._format_decimal(current_total)}) to {next_label} (${self._format_decimal(next_total)})."
            )
        if overall_peak is not None:
            peak_label, peak_day = overall_peak
            trend_lines.append(
                f"- Highest daily spend in the range was {peak_day['date']} in {peak_label} at ${self._format_decimal(peak_day['total_amount'])}."
            )
        if trend_lines:
            lines.extend(["", "## Overall observations", *trend_lines])
        if conflict_lines:
            lines.extend(["", "## Data notes", *conflict_lines])

        lines.append("")
        lines.append("## Daily totals")
        for label, rows, _ in daily_sections:
            lines.extend(["", f"### {label}"])
            if not rows:
                lines.append("- No in-scope daily totals were returned for this period.")
                continue
            for row in rows:
                lines.append(
                    f"- {row['date']}: ${self._format_decimal(row['total_amount'])} across {row['transaction_count']} transactions"
                )

        return "\n".join(lines)

    def _deterministic_single_scope_daily_answer(self, context: dict[str, Any]) -> str:
        daily_sections = self._daily_sections(context)
        if len(daily_sections) != 1:
            return self._fallback_answer(context)

        _, rows, conflicts = daily_sections[0]
        if not rows:
            return self._fallback_answer(context)

        time_scope = self._context_time_scope(context)
        resolved_label = self._time_scope_label(time_scope) if time_scope is not None else "the selected range"
        total_spend = sum((row["total_amount"] for row in rows), start=Decimal("0"))
        transaction_count = sum(row["transaction_count"] for row in rows)
        average_daily_spend = total_spend / len(rows)
        peak_day = max(rows, key=lambda row: row["total_amount"], default=None)

        lines = [f"Here's what I found for {resolved_label}:", ""]
        lines.append(f"- Date range: {rows[0]['date']} -> {rows[-1]['date']}")
        lines.append(f"- Total spend: ${self._format_decimal(total_spend)}")
        lines.append(f"- Transaction count: {transaction_count}")
        lines.append(f"- Average daily spend: ${self._format_decimal(average_daily_spend)}")
        if peak_day is not None:
            lines.append(
                f"- Peak day: {peak_day['date']} - ${self._format_decimal(peak_day['total_amount'])} ({peak_day['transaction_count']} transactions)"
            )

        if conflicts:
            lines.extend(["", "## Data notes"])
            for conflict in conflicts:
                reported = ", ".join(
                    f"${item['total_amount']} ({item['transaction_count']} txns)"
                    for item in conflict["reported_values"]
                )
                selected = conflict["selected_value"]
                lines.append(
                    f"- {conflict['date']}: reported values {reported}; used ${selected['total_amount']} ({selected['transaction_count']} txns)."
                )

        lines.extend(["", "## Daily totals"])
        for row in rows:
            lines.append(
                f"- {row['date']}: ${self._format_decimal(row['total_amount'])} across {row['transaction_count']} transactions"
            )

        return "\n".join(lines)

    def _fallback_answer(self, context: dict[str, Any]) -> str:
        fragments: list[str] = []
        selected_scope = self._context_time_scope(context)
        selected_scope_label = context.get("period") or (
            self._time_scope_label(selected_scope) if selected_scope is not None else "the selected scope"
        )
        overview = context.get("overview")
        if isinstance(overview, dict):
            total_spend = overview.get("total_amount")
            transaction_count = overview.get("transaction_count")
            if total_spend is not None:
                if context.get("period"):
                    fragments.append(f"Total spend for period {context.get('period')} is {total_spend}.")
                else:
                    fragments.append(f"Total spend for {selected_scope_label} is {total_spend}.")
            if transaction_count is not None:
                fragments.append(f"Transaction count is {transaction_count}.")

        period_summary = context.get("period_summary")
        if isinstance(period_summary, dict):
            flags = period_summary.get("flags")
            if isinstance(flags, list) and flags:
                fragments.append(f"There are {len(flags)} derived anomaly or concentration flags for this period.")

        behavior_summary = context.get("behavior_summary")
        if isinstance(behavior_summary, dict):
            behavior_lines = behavior_summary.get("behavior_summary")
            if isinstance(behavior_lines, list) and behavior_lines:
                fragments.append(str(behavior_lines[0]))

        averages = context.get("averages")
        if isinstance(averages, dict):
            average_transaction_amount = averages.get("average_transaction_amount")
            if average_transaction_amount is not None:
                fragments.append(f"Average transaction amount is {average_transaction_amount}.")

        month_over_month = context.get("month_over_month")
        if isinstance(month_over_month, dict):
            highlights = month_over_month.get("highlights")
            if isinstance(highlights, list) and highlights:
                fragments.append(str(highlights[0]))

        available_periods = context.get("available_periods")
        if isinstance(available_periods, dict):
            periods = available_periods.get("periods")
            count = available_periods.get("count")
            if isinstance(periods, list) and periods:
                preview = ", ".join(str(period) for period in periods[:5])
                if count is not None:
                    fragments.append(f"There are {count} available statement periods. Examples: {preview}.")
                else:
                    fragments.append(f"Available statement periods include: {preview}.")

        statement_period_summary = context.get("statement_period_summary")
        if isinstance(statement_period_summary, dict):
            summary_period = statement_period_summary.get("statement_period")
            total_amount = statement_period_summary.get("total_amount")
            transaction_count = statement_period_summary.get("transaction_count")
            if summary_period and total_amount is not None:
                fragments.append(f"Statement period summary for {summary_period} shows total spend of {total_amount}.")
            if transaction_count is not None:
                fragments.append(f"That summary includes {transaction_count} transactions.")

        statement_period_summary_range = context.get("statement_period_summary_range")
        if isinstance(statement_period_summary_range, list):
            fragments.append(
                f"Statement period summaries were included for {len(statement_period_summary_range)} periods in the selected range."
            )

        execution_plan = context.get("execution_plan")
        if isinstance(execution_plan, dict):
            steps = execution_plan.get("steps")
            if execution_plan.get("strategy") == "multi_scope" and isinstance(steps, list) and len(steps) > 1:
                fragments.append(f"Executed {len(steps)} planned analytics steps for this question.")

        if "categories" in context:
            fragments.append("Category breakdown data was included from Spring Boot analytics.")

        if "top_categories" in context:
            fragments.append("Top category spend data was included from Spring Boot analytics.")

        if "account_breakdown" in context:
            fragments.append("Account breakdown data was included for the selected period.")

        unavailable_tools = context.get("unavailable_tools")
        if isinstance(unavailable_tools, list) and unavailable_tools:
            missing_tools = ", ".join(
                item.get("tool", "unknown")
                for item in unavailable_tools
                if isinstance(item, dict)
            )
            if missing_tools:
                fragments.append(f"Some analytics sources were unavailable: {missing_tools}.")
            if any(
                isinstance(item, dict) and item.get("tool") == "account_breakdown"
                for item in unavailable_tools
            ):
                fragments.append(
                    "I could not determine which accounts drove the most spending because the account breakdown endpoint was unavailable."
                )

        if "payment_methods" in context:
            fragments.append("Payment method breakdown data was included for the selected period.")

        if "daily_totals" in context:
            fragments.append("Daily totals were included for the selected period.")

        if "criticality" in context:
            fragments.append("Criticality breakdown data was included for the selected period.")

        if "duplicates" in context:
            fragments.append("Duplicate transaction candidates were included for the selected period.")

        if "uncategorized" in context:
            fragments.append("Uncategorized transactions were included for the selected period.")

        if "outliers" in context:
            fragments.append("Outlier transactions were included for the selected period.")

        if not fragments:
            return "No matching finance context was available for this question."
        return " ".join(fragments)

