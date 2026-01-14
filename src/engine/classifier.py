"""
Email classification engine.

Classifies inbound debtor emails into 13 categories based on ai_logic.md:
INSOLVENCY, DISPUTE, ALREADY_PAID, UNSUBSCRIBE, HOSTILE, PROMISE_TO_PAY,
HARDSHIP, PLAN_REQUEST, REDIRECT, REQUEST_INFO, OUT_OF_OFFICE, COOPERATIVE, UNCLEAR
"""
import logging
from typing import Dict, Any

from src.api.models.requests import ClassifyRequest
from src.api.models.responses import ClassifyResponse, ExtractedData
from src.llm.client import llm_client
from src.prompts import CLASSIFY_EMAIL_SYSTEM, CLASSIFY_EMAIL_USER

logger = logging.getLogger(__name__)


class EmailClassifier:
    """Classifies inbound emails from debtors."""
    
    def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        """
        Classify an inbound email.
        
        Args:
            request: Classification request with email and context
        
        Returns:
            Classification result with confidence and extracted data
        """
        # Calculate derived values
        total_outstanding = sum(o.amount_due for o in request.context.obligations)
        days_overdue_max = max(
            (o.days_past_due for o in request.context.obligations), 
            default=0
        )
        
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
            body=request.email.body
        )
        
        # Call LLM with lower temperature for classification
        result = llm_client.complete(
            system_prompt=CLASSIFY_EMAIL_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.2
        )
        
        # Parse extracted data
        extracted = None
        if result.get("extracted_data"):
            extracted_raw = result["extracted_data"]
            # Only create ExtractedData if there's actual data
            if any(v is not None for v in extracted_raw.values()):
                extracted = ExtractedData(
                    promise_date=extracted_raw.get("promise_date"),
                    promise_amount=extracted_raw.get("promise_amount"),
                    dispute_type=extracted_raw.get("dispute_type"),
                    dispute_reason=extracted_raw.get("dispute_reason"),
                    redirect_contact=extracted_raw.get("redirect_contact"),
                    redirect_email=extracted_raw.get("redirect_email"),
                )
        
        logger.info(
            f"Classified email for {request.context.party.customer_code}: "
            f"{result['classification']} (confidence: {result['confidence']:.2f})"
        )
        
        return ClassifyResponse(
            classification=result["classification"],
            confidence=result["confidence"],
            reasoning=result.get("reasoning"),
            extracted_data=extracted,
            tokens_used=result.get("_tokens_used")
        )


# Singleton instance
classifier = EmailClassifier()
