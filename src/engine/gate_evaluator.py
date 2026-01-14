"""
Gate evaluation engine.

Evaluates 6 gates before allowing collection actions based on ai_logic.md:
touch_cap, cooling_off, dispute_active, hardship, unsubscribe, escalation_appropriate
"""
import logging
from typing import Dict, Any
from datetime import datetime, timezone

from src.api.models.requests import EvaluateGatesRequest
from src.api.models.responses import EvaluateGatesResponse, GateResult
from src.llm.client import llm_client
from src.prompts import EVALUATE_GATES_SYSTEM, EVALUATE_GATES_USER

logger = logging.getLogger(__name__)


class GateEvaluator:
    """Evaluates gates before allowing collection actions."""
    
    def evaluate(self, request: EvaluateGatesRequest) -> EvaluateGatesResponse:
        """
        Evaluate gates for a proposed action.
        
        Args:
            request: Gate evaluation request with context and proposed action
        
        Returns:
            Gate evaluation results
        """
        comm = request.context.communication
        
        # Calculate days since last touch
        days_since_last_touch = 999  # Default to large number if never contacted
        if comm and comm.last_touch_at:
            delta = datetime.now(timezone.utc) - comm.last_touch_at
            days_since_last_touch = delta.days
        
        # Calculate total outstanding
        total_outstanding = sum(o.amount_due for o in request.context.obligations)
        
        # Get behavior info
        behavior = request.context.behavior
        
        # Build user prompt
        user_prompt = EVALUATE_GATES_USER.format(
            proposed_action=request.proposed_action,
            proposed_tone=request.proposed_tone or "not specified",
            touch_count=comm.touch_count if comm else 0,
            touch_cap=request.context.touch_cap,
            days_since_last_touch=days_since_last_touch,
            touch_interval_days=request.context.touch_interval_days,
            active_dispute=request.context.active_dispute,
            hardship_indicated=request.context.hardship_indicated,
            unsubscribe_requested=False,  # TODO: Add to context model
            broken_promises_count=request.context.broken_promises_count,
            last_tone_used=comm.last_tone_used if comm else "None",
            case_state=request.context.case_state or "ACTIVE",
            currency=request.context.party.currency,
            total_outstanding=total_outstanding,
            segment=behavior.segment if behavior else "standard"
        )
        
        # Call LLM with very low temperature for consistent evaluation
        result = llm_client.complete(
            system_prompt=EVALUATE_GATES_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.1
        )
        
        # Parse gate results
        gate_results = {}
        for gate_name, gate_data in result.get("gate_results", {}).items():
            gate_results[gate_name] = GateResult(
                passed=gate_data["passed"],
                reason=gate_data["reason"],
                current_value=gate_data.get("current_value"),
                threshold=gate_data.get("threshold")
            )
        
        logger.info(
            f"Evaluated gates for {request.context.party.customer_code}: "
            f"action={request.proposed_action}, allowed={result['allowed']}"
        )
        
        return EvaluateGatesResponse(
            allowed=result["allowed"],
            gate_results=gate_results,
            recommended_action=result.get("recommended_action"),
            tokens_used=result.get("_tokens_used")
        )


# Singleton instance
gate_evaluator = GateEvaluator()
