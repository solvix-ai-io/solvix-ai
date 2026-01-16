"""
Gate evaluation engine.

Evaluates 6 gates before allowing collection actions based on ai_logic.md:
touch_cap, cooling_off, dispute_active, hardship, unsubscribe, escalation_appropriate
"""

import json
import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import EvaluateGatesRequest
from src.api.models.responses import EvaluateGatesResponse, GateResult
from src.llm.factory import llm_client
from src.llm.schemas import GateEvaluationLLMResponse
from src.prompts import EVALUATE_GATES_SYSTEM, EVALUATE_GATES_USER

logger = logging.getLogger(__name__)


class GateEvaluator:
    """Evaluates gates before allowing collection actions."""

    async def evaluate(self, request: EvaluateGatesRequest) -> EvaluateGatesResponse:
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
            segment=behavior.segment if behavior else "standard",
        )

        # Call LLM with very low temperature for consistent evaluation
        response = await llm_client.complete(
            system_prompt=EVALUATE_GATES_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.1,
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
            result = GateEvaluationLLMResponse(**raw_result)
        except ValidationError as e:
            logger.error(f"LLM response validation failed: {e}")
            raise LLMResponseInvalidError(
                message="LLM returned invalid gate evaluation response",
                details={"validation_errors": e.errors(), "raw_response": raw_result},
            )

        # Convert validated gate results to response model
        gate_results = {}
        for gate_name, gate_data in result.gate_results.items():
            gate_results[gate_name] = GateResult(
                passed=gate_data.passed,
                reason=gate_data.reason,
                current_value=gate_data.current_value,
                threshold=gate_data.threshold,
            )

        logger.info(
            f"Evaluated gates for {request.context.party.customer_code}: "
            f"action={request.proposed_action}, allowed={result.allowed}"
        )

        return EvaluateGatesResponse(
            allowed=result.allowed,
            gate_results=gate_results,
            recommended_action=result.recommended_action,
            tokens_used=tokens_used,
        )


# Singleton instance
gate_evaluator = GateEvaluator()
