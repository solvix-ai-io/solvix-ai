"""
Draft generation engine.

Generates collection email drafts with 5 tones based on ai_logic.md:
friendly_reminder, professional, firm, final_notice, concerned_inquiry

Includes guardrail retry mechanism: if guardrails fail, the generator
will retry with feedback about what went wrong, giving the LLM a chance
to correct its output.
"""

import logging

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import GenerateDraftRequest
from src.api.models.responses import GenerateDraftResponse, GuardrailValidation
from src.guardrails.base import GuardrailPipelineResult, GuardrailSeverity
from src.guardrails.pipeline import guardrail_pipeline
from src.llm.factory import llm_client
from src.llm.schemas import DraftGenerationLLMResponse
from src.prompts import GENERATE_DRAFT_SYSTEM, GENERATE_DRAFT_USER
from src.utils import JSONExtractionError, extract_json

logger = logging.getLogger(__name__)

# Maximum retries when guardrails fail
MAX_GUARDRAIL_RETRIES = 2


class DraftGenerator:
    """Generates collection email drafts with guardrail retry mechanism."""

    async def generate(self, request: GenerateDraftRequest) -> GenerateDraftResponse:
        """
        Generate a collection email draft with automatic retry on guardrail failures.

        Args:
            request: Generation request with context and parameters

        Returns:
            Generated draft with subject, body, and guardrail validation

        The generator will retry up to MAX_GUARDRAIL_RETRIES times if guardrails
        fail, passing the failure reasons back to the LLM to help it correct
        the output.
        """
        # Calculate derived values
        total_outstanding = sum(o.amount_due for o in request.context.obligations)

        # Build invoices list (top 10 by days overdue)
        sorted_obligations = sorted(
            request.context.obligations, key=lambda o: o.days_past_due, reverse=True
        )[:10]

        invoices_list = (
            "\n".join(
                [
                    f"- {o.invoice_number}: {request.context.party.currency} {o.amount_due:,.2f} "
                    f"({o.days_past_due} days overdue)"
                    for o in sorted_obligations
                ]
            )
            if sorted_obligations
            else "No specific invoices provided"
        )

        # Get communication info
        comm = request.context.communication

        # Calculate days since last touch
        days_since_last_touch = request.context.days_in_state or 0
        if comm and comm.last_touch_at:
            from datetime import datetime, timezone

            delta = datetime.now(timezone.utc) - comm.last_touch_at
            days_since_last_touch = delta.days

        # Get behavior info
        behavior = request.context.behavior

        # Build base user prompt
        base_user_prompt = GENERATE_DRAFT_USER.format(
            party_name=request.context.party.name,
            customer_code=request.context.party.customer_code,
            currency=request.context.party.currency,
            total_outstanding=total_outstanding,
            relationship_tier=request.context.relationship_tier,
            is_verified=request.context.party.is_verified,
            invoices_list=invoices_list,
            monthly_touch_count=request.context.monthly_touch_count,
            touch_count=comm.touch_count if comm else 0,
            last_touch_at=comm.last_touch_at.strftime("%Y-%m-%d")
            if comm and comm.last_touch_at
            else "Never",
            last_tone_used=comm.last_tone_used if comm else "None",
            last_response_type=comm.last_response_type if comm else "No response",
            case_state=request.context.case_state or "ACTIVE",
            days_since_last_touch=days_since_last_touch,
            broken_promises_count=request.context.broken_promises_count,
            active_dispute=request.context.active_dispute,
            hardship_indicated=request.context.hardship_indicated,
            segment=behavior.segment if behavior else "standard",
            on_time_rate=f"{behavior.on_time_rate:.0%}"
            if behavior and behavior.on_time_rate
            else "Unknown",
            avg_days_to_pay=behavior.avg_days_to_pay if behavior else "Unknown",
            tone=request.tone,
            objective=request.objective or "collect payment",
            brand_tone=request.context.brand_tone,
            custom_instructions=f"\nAdditional: {request.custom_instructions}"
            if request.custom_instructions
            else "",
        )

        # Retry loop for guardrail failures
        guardrail_feedback = None
        total_tokens_used = 0
        result = None
        guardrail_result = None

        for attempt in range(MAX_GUARDRAIL_RETRIES + 1):
            # Build prompt with any guardrail feedback from previous attempt
            user_prompt = base_user_prompt
            if guardrail_feedback:
                user_prompt += guardrail_feedback
                logger.info(
                    f"Retrying draft generation (attempt {attempt + 1}) with guardrail feedback"
                )

            # Call LLM with higher temperature for creative generation
            response = await llm_client.complete(
                system_prompt=GENERATE_DRAFT_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.7,
                json_mode=True,
            )

            # Track total tokens across retries
            total_tokens_used += response.usage.get("total_tokens", 0)

            # Parse JSON response using robust extraction
            try:
                raw_result = extract_json(response.content)
            except JSONExtractionError as e:
                # Log with repr() to see actual characters including non-printable ones
                content_preview = repr(response.content[:500]) if response.content else "None"
                logger.error(
                    f"Failed to parse JSON response (len={len(response.content) if response.content else 0}): {content_preview}"
                )
                raise LLMResponseInvalidError(
                    message="LLM returned invalid JSON",
                    details={
                        "error": str(e),
                        "raw_content": e.raw_content[:1000] if e.raw_content else "",
                        "extraction_attempts": e.attempts,
                    },
                )

            # Validate LLM response using Pydantic schema
            try:
                result = DraftGenerationLLMResponse(**raw_result)
            except ValidationError as e:
                logger.error(f"LLM response validation failed: {e}")
                raise LLMResponseInvalidError(
                    message="LLM returned invalid draft generation response",
                    details={"validation_errors": e.errors(), "raw_response": raw_result},
                )

            # Run guardrails on generated draft body (critical for factual accuracy)
            guardrail_result = guardrail_pipeline.validate(
                output=result.body,
                context=request.context,
            )

            # If all guardrails passed, we're done
            if guardrail_result.all_passed:
                if attempt > 0:
                    logger.info(
                        f"Guardrails passed on retry attempt {attempt + 1} for "
                        f"{request.context.party.customer_code}"
                    )
                break

            # If this was the last attempt, exit loop with failed guardrails
            if attempt >= MAX_GUARDRAIL_RETRIES:
                logger.warning(
                    f"Guardrails still failing after {MAX_GUARDRAIL_RETRIES + 1} attempts for "
                    f"{request.context.party.customer_code}: {guardrail_result.blocking_guardrails}"
                )
                break

            # Build feedback for next attempt
            guardrail_feedback = self._build_guardrail_feedback(guardrail_result)

        # Extract referenced invoices from generated body
        invoices_referenced = [
            o.invoice_number for o in request.context.obligations if o.invoice_number in result.body
        ]

        # Calculate factual accuracy
        total_checks = len(guardrail_result.results)
        passed_checks = sum(1 for r in guardrail_result.results if r.passed)
        factual_accuracy = passed_checks / total_checks if total_checks > 0 else 1.0

        # Separate warnings from blocking failures
        warnings = [
            r.guardrail_name
            for r in guardrail_result.results
            if not r.passed and r.severity == GuardrailSeverity.MEDIUM
        ]

        guardrail_validation = GuardrailValidation(
            all_passed=guardrail_result.all_passed,
            guardrails_run=total_checks,
            guardrails_passed=passed_checks,
            blocking_failures=guardrail_result.blocking_guardrails,
            warnings=warnings,
            factual_accuracy=factual_accuracy,
        )

        if not guardrail_result.all_passed:
            logger.warning(
                f"Guardrails failed for draft {request.context.party.customer_code}: "
                f"blocking={guardrail_result.blocking_guardrails}, warnings={warnings}"
            )

        logger.info(
            f"Generated draft for {request.context.party.customer_code}: "
            f"tone={request.tone}, invoices_referenced={len(invoices_referenced)}, "
            f"guardrails_passed={guardrail_result.all_passed}, tokens={total_tokens_used}"
        )

        return GenerateDraftResponse(
            subject=result.subject,
            body=result.body,
            tone_used=request.tone,
            invoices_referenced=invoices_referenced,
            tokens_used=total_tokens_used,
            guardrail_validation=guardrail_validation,
        )

    def _build_guardrail_feedback(self, guardrail_result: GuardrailPipelineResult) -> str:
        """
        Build feedback prompt addition from guardrail failures.

        This feedback is appended to the user prompt on retry attempts
        to help the LLM correct its output.
        """
        failures = [r for r in guardrail_result.results if not r.passed]
        if not failures:
            return ""

        feedback_lines = [
            "\n\n**CRITICAL: Your previous draft had validation errors. Fix these issues:**\n"
        ]

        for failure in failures:
            feedback_lines.append(f"- {failure.guardrail_name}: {failure.message}")
            if failure.expected:
                feedback_lines.append(f"  Expected: {failure.expected}")
            if failure.found:
                feedback_lines.append(f"  Found: {failure.found}")

        feedback_lines.append(
            "\nEnsure the new draft addresses ALL validation issues listed above."
        )

        return "\n".join(feedback_lines)


# Singleton instance
generator = DraftGenerator()
