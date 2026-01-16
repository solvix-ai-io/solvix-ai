"""
Email classification engine.

Classifies inbound debtor emails into 13 categories based on ai_logic.md:
INSOLVENCY, DISPUTE, ALREADY_PAID, UNSUBSCRIBE, HOSTILE, PROMISE_TO_PAY,
HARDSHIP, PLAN_REQUEST, REDIRECT, REQUEST_INFO, OUT_OF_OFFICE, COOPERATIVE, UNCLEAR
"""

import json
import logging
from datetime import date

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import ClassifyRequest
from src.api.models.responses import ClassifyResponse, ExtractedData
from src.llm.factory import llm_client
from src.llm.schemas import ClassificationLLMResponse
from src.prompts import CLASSIFY_EMAIL_SYSTEM, CLASSIFY_EMAIL_USER

logger = logging.getLogger(__name__)


class EmailClassifier:
    """Classifies inbound emails from debtors."""

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        """
        Classify an inbound email.

        Args:
            request: Classification request with email and context

        Returns:
            Classification result with confidence and extracted data
        """
        # Calculate derived values
        total_outstanding = sum(o.amount_due for o in request.context.obligations)
        days_overdue_max = max((o.days_past_due for o in request.context.obligations), default=0)

        # Build user prompt with context
        user_prompt = CLASSIFY_EMAIL_USER.format(
            party_name=request.context.party.name,
            customer_code=request.context.party.customer_code,
            currency=request.context.party.currency,
            total_outstanding=total_outstanding,
            days_overdue_max=days_overdue_max,
            broken_promises_count=request.context.broken_promises_count,
            segment=request.context.behavior.segment if request.context.behavior else "unknown",
            active_dispute=request.context.active_dispute,
            hardship_indicated=request.context.hardship_indicated,
            from_name=request.email.from_name or "Unknown",
            from_address=request.email.from_address,
            subject=request.email.subject,
            body=request.email.body,
        )

        # Call LLM with lower temperature for classification
        response = await llm_client.complete(
            system_prompt=CLASSIFY_EMAIL_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.2,
            json_mode=True,
        )

        # Parse JSON response
        tokens_used = response.usage.get("total_tokens", 0)
        try:
            content = response.content.replace("```json", "").replace("```", "").strip()
            raw_result = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response.content}")
            raise LLMResponseInvalidError(
                message="LLM returned invalid JSON",
                details={"error": str(e), "raw_content": response.content},
            )

        # Validate LLM response using Pydantic schema
        try:
            result = ClassificationLLMResponse(**raw_result)
        except ValidationError as e:
            logger.error(f"LLM response validation failed: {e}")
            raise LLMResponseInvalidError(
                message="LLM returned invalid classification response",
                details={"validation_errors": e.errors(), "raw_response": raw_result},
            )

        # Parse extracted data
        extracted = None
        if result.extracted_data:
            extracted_raw = result.extracted_data
            # Only create ExtractedData if there's actual data
            if any(v is not None for v in extracted_raw.model_dump().values()):
                # Parse promise_date string to date if present
                promise_date_parsed = None
                if extracted_raw.promise_date:
                    try:
                        promise_date_parsed = date.fromisoformat(extracted_raw.promise_date)
                    except ValueError:
                        logger.warning(
                            f"Could not parse promise_date: {extracted_raw.promise_date}"
                        )

                extracted = ExtractedData(
                    promise_date=promise_date_parsed,
                    promise_amount=extracted_raw.promise_amount,
                    dispute_type=extracted_raw.dispute_type,
                    dispute_reason=extracted_raw.dispute_reason,
                    redirect_contact=extracted_raw.redirect_contact,
                    redirect_email=extracted_raw.redirect_email,
                )

        logger.info(
            f"Classified email for {request.context.party.customer_code}: "
            f"{result.classification} (confidence: {result.confidence:.2f})"
        )

        return ClassifyResponse(
            classification=result.classification,
            confidence=result.confidence,
            reasoning=result.reasoning,
            extracted_data=extracted,
            tokens_used=tokens_used,
        )


# Singleton instance
classifier = EmailClassifier()
