"""
Draft generation engine.

Generates collection email drafts with 5 tones based on ai_logic.md:
friendly_reminder, professional, firm, final_notice, concerned_inquiry
"""
import logging
from typing import Dict, Any

from src.api.models.requests import GenerateDraftRequest
from src.api.models.responses import GenerateDraftResponse
from src.llm.client import llm_client
from src.prompts import GENERATE_DRAFT_SYSTEM, GENERATE_DRAFT_USER

logger = logging.getLogger(__name__)


class DraftGenerator:
    """Generates collection email drafts."""
    
    def generate(self, request: GenerateDraftRequest) -> GenerateDraftResponse:
        """
        Generate a collection email draft.
        
        Args:
            request: Generation request with context and parameters
        
        Returns:
            Generated draft with subject and body
        """
        # Calculate derived values
        total_outstanding = sum(o.amount_due for o in request.context.obligations)
        
        # Build invoices list (top 10 by days overdue)
        sorted_obligations = sorted(
            request.context.obligations, 
            key=lambda o: o.days_past_due, 
            reverse=True
        )[:10]
        
        invoices_list = "\n".join([
            f"- {o.invoice_number}: {request.context.party.currency} {o.amount_due:,.2f} "
            f"({o.days_past_due} days overdue)"
            for o in sorted_obligations
        ]) if sorted_obligations else "No specific invoices provided"
        
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
        
        # Build user prompt
        user_prompt = GENERATE_DRAFT_USER.format(
            party_name=request.context.party.name,
            customer_code=request.context.party.customer_code,
            currency=request.context.party.currency,
            total_outstanding=total_outstanding,
            invoices_list=invoices_list,
            touch_count=comm.touch_count if comm else 0,
            last_touch_at=comm.last_touch_at.strftime("%Y-%m-%d") if comm and comm.last_touch_at else "Never",
            last_tone_used=comm.last_tone_used if comm else "None",
            last_response_type=comm.last_response_type if comm else "No response",
            case_state=request.context.case_state or "ACTIVE",
            days_since_last_touch=days_since_last_touch,
            broken_promises_count=request.context.broken_promises_count,
            active_dispute=request.context.active_dispute,
            hardship_indicated=request.context.hardship_indicated,
            segment=behavior.segment if behavior else "standard",
            on_time_rate=f"{behavior.on_time_rate:.0%}" if behavior and behavior.on_time_rate else "Unknown",
            avg_days_to_pay=behavior.avg_days_to_pay if behavior else "Unknown",
            tone=request.tone,
            objective=request.objective or "collect payment",
            brand_tone=request.context.brand_tone,
            custom_instructions=f"\nAdditional: {request.custom_instructions}" if request.custom_instructions else ""
        )
        
        # Call LLM with higher temperature for creative generation
        result = llm_client.complete(
            system_prompt=GENERATE_DRAFT_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.7
        )
        
        # Extract referenced invoices from generated body
        invoices_referenced = [
            o.invoice_number for o in request.context.obligations
            if o.invoice_number in result.get("body", "")
        ]
        
        logger.info(
            f"Generated draft for {request.context.party.customer_code}: "
            f"tone={request.tone}, invoices_referenced={len(invoices_referenced)}"
        )
        
        return GenerateDraftResponse(
            subject=result["subject"],
            body=result["body"],
            tone_used=request.tone,
            invoices_referenced=invoices_referenced,
            tokens_used=result.get("_tokens_used")
        )


# Singleton instance
generator = DraftGenerator()
