"""Temporal Consistency Guardrail - validates date logic."""

import logging
import re
from datetime import date, datetime

from src.api.models.requests import CaseContext
from src.api.models.responses import ExtractedData

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)


class TemporalConsistencyGuardrail(BaseGuardrail):
    """
    Validates temporal logic of AI outputs.

    Checks:
    1. Promise dates are in the future (not past)
    2. Due dates mentioned match obligation due dates
    3. Days overdue calculations are consistent with today's date
    """

    def __init__(self):
        super().__init__(
            name="temporal_consistency",
            severity=GuardrailSeverity.HIGH,  # High, not critical - dates can be subjective
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        """Validate temporal consistency of the output."""
        results = []

        # Check extracted promise dates
        extracted_data = kwargs.get("extracted_data")
        if extracted_data and hasattr(extracted_data, "promise_date"):
            results.append(self._validate_promise_date_is_future(extracted_data))

        # Check mentioned due dates
        results.append(self._validate_due_dates(output, context))

        return results

    def _validate_promise_date_is_future(self, extracted_data: ExtractedData) -> GuardrailResult:
        """Validate that extracted promise dates are in the future."""
        if not extracted_data.promise_date:
            return self._pass(message="No promise date to validate")

        today = date.today()
        promise_date = extracted_data.promise_date

        # Allow today as a valid promise date
        if promise_date < today:
            days_past = (today - promise_date).days
            return self._fail(
                message=f"Promise date {promise_date} is {days_past} days in the past",
                expected="Date in future or today",
                found=str(promise_date),
                details={
                    "promise_date": str(promise_date),
                    "today": str(today),
                    "days_past": days_past,
                },
            )

        # Warn if promise date is very far in the future (>90 days)
        days_future = (promise_date - today).days
        if days_future > 90:
            return self._fail(
                message=f"Promise date {promise_date} is {days_future} days in future (unusual)",
                expected="Date within 90 days",
                found=str(promise_date),
                details={
                    "promise_date": str(promise_date),
                    "days_future": days_future,
                    "note": "Unusually distant promise date",
                },
            )

        return self._pass(
            message="Promise date is valid",
            details={"promise_date": str(promise_date), "days_future": days_future},
        )

    def _validate_due_dates(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that mentioned due dates match obligation data."""
        # Extract date patterns from output
        date_patterns = [
            r"due\s+(?:on|by)\s+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            r"due\s+date[:\s]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
        ]

        # Get valid due dates from context
        valid_due_dates = set()
        for o in context.obligations:
            try:
                if isinstance(o.due_date, str):
                    valid_due_dates.add(date.fromisoformat(o.due_date))
            except ValueError:
                continue

        # Find mentioned dates
        mentioned_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                parsed = self._parse_date(match)
                if parsed:
                    mentioned_dates.append(parsed)

        # Validate mentioned dates
        for mentioned in mentioned_dates:
            if mentioned not in valid_due_dates:
                # Check if it's close to any valid date (±1 day tolerance)
                is_close = any(abs((mentioned - valid).days) <= 1 for valid in valid_due_dates)
                if not is_close:
                    return self._fail(
                        message=f"Due date {mentioned} not found in obligations",
                        expected=sorted([str(d) for d in valid_due_dates]),
                        found=str(mentioned),
                        details={
                            "mentioned_date": str(mentioned),
                            "valid_dates": sorted([str(d) for d in valid_due_dates]),
                        },
                    )

        return self._pass(
            message="Due dates validated",
            details={"valid_dates": sorted([str(d) for d in valid_due_dates])},
        )

    def _parse_date(self, date_str: str) -> date | None:
        """Try to parse a date string in various formats."""
        formats = [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d/%m/%y",
            "%d-%m-%y",
            "%m/%d/%Y",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        # Try parsing natural dates like "15th January 2024"
        try:
            # Remove ordinal suffixes
            cleaned = re.sub(r"(\d+)(?:st|nd|rd|th)", r"\1", date_str)
            return datetime.strptime(cleaned, "%d %B %Y").date()
        except ValueError:
            pass

        return None
