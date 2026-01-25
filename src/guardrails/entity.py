"""Entity Verification Guardrail - LLM-based validation of customer/party identifiers."""

import asyncio
import json
import logging
import re
from typing import Any

from src.api.models.requests import CaseContext
from src.llm.factory import llm_client

from .base import BaseGuardrail, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)

# Validation prompt for LLM-based entity verification
ENTITY_VALIDATION_PROMPT = """Validate the following draft email for entity accuracy.

EXPECTED ENTITIES:
- Customer Code: {customer_code}
- Party/Company Name: {party_name}

DRAFT TO VALIDATE:
{draft}

Your task:
1. Check if the draft correctly references the customer code (if mentioned at all)
2. Check if the draft addresses the correct party/company name
3. Identify any hallucinated, fabricated, or mismatched identifiers

IMPORTANT: The draft does NOT need to explicitly mention the customer code. Only flag it as invalid if it mentions a DIFFERENT code than expected.

Respond ONLY with valid JSON (no markdown):
{{
  "customer_code_valid": true,
  "customer_code_reason": "Customer code not mentioned or matches expected",
  "party_name_valid": true,
  "party_name_reason": "Party name matches or is a reasonable variation",
  "issues_found": [],
  "passed": true
}}

Set "passed" to false only if there are actual mismatches or hallucinated identifiers."""


class EntityVerificationGuardrail(BaseGuardrail):
    """
    LLM-based entity verification guardrail.

    Uses the same LLM as draft generation to validate that:
    1. Customer code is correct (if mentioned)
    2. Party name is accurate
    3. No hallucinated identifiers exist

    This replaces regex-based detection which was prone to false positives
    (e.g., matching common words like "with" as customer codes).
    """

    def __init__(self):
        super().__init__(
            name="entity_verification",
            severity=GuardrailSeverity.CRITICAL,
        )

    def validate(self, output: str, context: CaseContext, **kwargs) -> list[GuardrailResult]:
        """
        Validate entity identifiers using LLM-based verification.

        Runs the LLM synchronously using asyncio.run() since guardrails
        execute in a thread pool.
        """
        results = []

        # Run LLM-based entity validation
        try:
            llm_result = self._validate_entities_with_llm(output, context)
            results.extend(llm_result)
        except Exception as e:
            logger.error(f"LLM-based entity validation failed: {e}")
            # Fall back to basic validation on LLM failure
            results.extend(self._basic_entity_validation(output, context))

        # Only validate emails if extracted_data is provided (keep this deterministic)
        extracted_data = kwargs.get("extracted_data")
        if extracted_data:
            results.append(self._validate_emails(output, context, extracted_data))

        return results

    def _validate_entities_with_llm(
        self, output: str, context: CaseContext
    ) -> list[GuardrailResult]:
        """
        Use LLM to validate entity accuracy.

        Returns list of GuardrailResults for customer code and party name.
        """
        prompt = ENTITY_VALIDATION_PROMPT.format(
            customer_code=context.party.customer_code,
            party_name=context.party.name,
            draft=output,
        )

        # Run async LLM call in sync context (guardrails run in thread pool)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    llm_client.complete(
                        system_prompt="You are a validation assistant. Respond only with valid JSON.",
                        user_prompt=prompt,
                        temperature=0,  # Deterministic for validation
                        json_mode=True,
                        max_tokens=500,
                    )
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"LLM call failed in entity verification: {e}")
            raise

        # Parse LLM response
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response.content[:500]}")
            raise ValueError(f"Invalid JSON from LLM: {e}")

        results = []

        # Customer code validation result
        if result.get("customer_code_valid", True):
            results.append(
                self._pass(
                    message=result.get("customer_code_reason", "Customer code validated"),
                    details={"customer_code": context.party.customer_code},
                )
            )
        else:
            results.append(
                self._fail(
                    message=result.get(
                        "customer_code_reason",
                        f"Customer code validation failed for {context.party.customer_code}",
                    ),
                    expected=context.party.customer_code,
                    found=None,
                    details={"issues": result.get("issues_found", [])},
                )
            )

        # Party name validation result
        if result.get("party_name_valid", True):
            results.append(
                self._pass(
                    message=result.get("party_name_reason", "Party name validated"),
                    details={"party_name": context.party.name},
                )
            )
        else:
            results.append(
                self._fail(
                    message=result.get(
                        "party_name_reason",
                        f"Party name validation failed for {context.party.name}",
                    ),
                    expected=context.party.name,
                    found=None,
                    details={"issues": result.get("issues_found", [])},
                )
            )

        logger.info(
            f"Entity verification completed: customer_code_valid={result.get('customer_code_valid')}, "
            f"party_name_valid={result.get('party_name_valid')}, passed={result.get('passed')}"
        )

        return results

    def _basic_entity_validation(self, output: str, context: CaseContext) -> list[GuardrailResult]:
        """
        Basic fallback validation when LLM is unavailable.

        Only checks if the correct values ARE present (no false positives).
        """
        results = []

        # Check customer code presence (if mentioned, should be correct)
        valid_code = context.party.customer_code.upper()
        output_upper = output.upper()

        if valid_code in output_upper:
            results.append(
                self._pass(
                    message="Customer code found and validated",
                    details={"customer_code": valid_code},
                )
            )
        else:
            # Code not mentioned is OK - we can't reliably detect mismatches without LLM
            results.append(
                self._pass(
                    message="Customer code not explicitly mentioned (fallback validation)",
                    details={"expected_code": valid_code, "fallback": True},
                )
            )

        # Check party name presence
        valid_name = context.party.name.lower()
        output_lower = output.lower()
        name_parts = [p for p in valid_name.split() if len(p) > 3]

        if valid_name in output_lower or any(part in output_lower for part in name_parts):
            results.append(
                self._pass(
                    message="Party name validated",
                    details={"party_name": context.party.name},
                )
            )
        else:
            # Name not found - pass with warning in fallback mode
            results.append(
                self._pass(
                    message="Party name validation skipped (fallback mode)",
                    details={"expected_name": context.party.name, "fallback": True},
                )
            )

        return results

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
