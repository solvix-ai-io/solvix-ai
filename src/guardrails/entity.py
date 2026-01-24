"""Entity Verification Guardrail - validates customer/party identifiers."""

import logging
import re
from typing import Any

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)


class EntityVerificationGuardrail(BaseGuardrail):
    """
    Validates that entity identifiers match input exactly.

    Checks:
    1. Customer code matches context.party.customer_code
    2. Party name matches context.party.name (or reasonable substring)
    3. Email addresses are not fabricated
    """

    def __init__(self):
        super().__init__(
            name="entity_verification",
            severity=GuardrailSeverity.CRITICAL,
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        """Validate entity identifiers in the output."""
        results = []

        # Run all validation checks
        results.append(self._validate_customer_code(output, context))
        results.append(self._validate_party_name(output, context))

        # Only validate emails if extracted_data is provided
        extracted_data = kwargs.get("extracted_data")
        if extracted_data:
            results.append(self._validate_emails(output, context, extracted_data))

        return results

    def _validate_customer_code(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that any customer code in output matches context."""
        # Get valid customer code
        valid_code = context.party.customer_code.upper()

        # Common patterns for customer codes
        code_patterns = [
            r"customer\s*(?:code|number|id|#)?\s*:?\s*([A-Z0-9\-_]+)",
            r"account\s*(?:number|#)?\s*:?\s*([A-Z0-9\-_]+)",
            r"reference\s*(?:number|#)?\s*:?\s*([A-Z0-9\-_]+)",
        ]

        output_upper = output.upper()

        # Check if valid code is mentioned
        if valid_code in output_upper:
            return self._pass(
                message="Customer code validated",
                details={"customer_code": valid_code},
            )

        # Look for any customer code patterns that don't match
        for pattern in code_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                if match.upper() != valid_code and len(match) > 3:
                    # Found a code that doesn't match
                    return self._fail(
                        message=f"Customer code mismatch: found '{match}', expected '{valid_code}'",
                        expected=valid_code,
                        found=match,
                        details={
                            "expected_code": valid_code,
                            "found_code": match,
                        },
                    )

        # No code found, that's okay
        return self._pass(
            message="No invalid customer codes found",
            details={"expected_code": valid_code},
        )

    def _validate_party_name(self, output: str, context: CaseContext) -> GuardrailResult:
        """Validate that party name references are accurate."""
        valid_name = context.party.name.lower()
        output_lower = output.lower()

        # Check if valid name is mentioned
        if valid_name in output_lower:
            return self._pass(
                message="Party name validated",
                details={"party_name": context.party.name},
            )

        # Check for partial matches (e.g., "Acme" in "Acme Corp Ltd")
        name_parts = valid_name.split()
        for part in name_parts:
            if len(part) > 3 and part in output_lower:
                return self._pass(
                    message="Party name (partial) validated",
                    details={
                        "party_name": context.party.name,
                        "matched_part": part,
                    },
                )

        # Look for "Dear Company" or company name patterns that might be wrong
        company_patterns = [
            r"dear\s+([A-Za-z][A-Za-z\s&]+?)(?:,|\s+team|\s+ltd|\s+limited|\s+inc|\.)",
            r"[Aa]t\s+([A-Z][A-Za-z\s&]+?)(?:,|\.|we)",
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, output)
            for match in matches:
                match_clean = match.strip().lower()
                # Check if this looks like a company name that doesn't match
                if (
                    len(match_clean) > 3
                    and match_clean != valid_name
                    and not any(part in match_clean for part in name_parts)
                ):
                    return self._fail(
                        message=f"Party name mismatch: found '{match}', expected '{context.party.name}'",
                        expected=context.party.name,
                        found=match,
                        details={
                            "expected_name": context.party.name,
                            "found_name": match,
                        },
                    )

        return self._pass(
            message="No invalid party names found",
            details={"expected_name": context.party.name},
        )

    def _validate_emails(
        self, output: str, context: CaseContext, extracted_data: Any
    ) -> GuardrailResult:
        """Validate that email addresses are not fabricated."""
        # Extract email addresses from output
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        found_emails = set(re.findall(email_pattern, output))

        if not found_emails:
            return self._pass(message="No email addresses to validate")

        # Valid emails could come from:
        # 1. The email being classified (from_address)
        # 2. Extracted redirect email
        valid_emails = set()

        # This would need to be passed in context - for now, check extracted_data
        if hasattr(extracted_data, "redirect_email") and extracted_data.redirect_email:
            valid_emails.add(extracted_data.redirect_email.lower())

        # If no valid emails defined, any email could be suspicious
        if not valid_emails and found_emails:
            return self._fail(
                message=f"Fabricated email addresses found: {found_emails}",
                expected="No emails or extracted redirect email",
                found=list(found_emails),
                details={
                    "found_emails": list(found_emails),
                    "note": "Emails should come from extracted_data only",
                },
            )

        # Check for invalid emails
        invalid_emails = found_emails - valid_emails
        if invalid_emails:
            return self._fail(
                message=f"Unverified email addresses found: {invalid_emails}",
                expected=list(valid_emails),
                found=list(invalid_emails),
            )

        return self._pass(
            message="Email addresses validated",
            details={"validated_emails": list(found_emails)},
        )
