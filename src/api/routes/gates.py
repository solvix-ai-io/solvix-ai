"""
Gate evaluation API endpoint.

POST /evaluate-gates - Evaluate gates before allowing a collection action.
POST /evaluate-gates/batch - Evaluate gates for multiple parties at once.

Security:
- Rate limited: configurable via settings (default 100/minute for internal service calls)
"""

import asyncio
import logging

from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.errors import ErrorResponse
from src.api.models.requests import EvaluateGatesBatchRequest, EvaluateGatesRequest
from src.api.models.responses import (
    EvaluateGatesBatchResponse,
    EvaluateGatesResponse,
    PartyGateResult,
)
from src.config.settings import settings
from src.engine.gate_evaluator import gate_evaluator

logger = logging.getLogger(__name__)
router = APIRouter()

# Rate limiter (uses app.state.limiter from main.py)
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/evaluate-gates",
    response_model=EvaluateGatesResponse,
    responses={
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
@limiter.limit(settings.rate_limit_gates)
async def evaluate_gates(
    request: Request, gates_request: EvaluateGatesRequest
) -> EvaluateGatesResponse:
    """
    Evaluate gates before allowing a collection action.

    Returns whether action is allowed and individual gate results.
    """
    logger.info(f"Evaluating gates for action: {gates_request.proposed_action}")
    result = await gate_evaluator.evaluate(gates_request)
    logger.info(f"Gates evaluation: allowed={result.allowed}")
    return result


@router.post(
    "/evaluate-gates/batch",
    response_model=EvaluateGatesBatchResponse,
    responses={
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
@limiter.limit(settings.rate_limit_gates)
async def evaluate_gates_batch(
    request: Request, batch_request: EvaluateGatesBatchRequest
) -> EvaluateGatesBatchResponse:
    """
    Evaluate gates for multiple parties at once.

    Since gate evaluation is deterministic (no LLM calls), this endpoint
    efficiently evaluates all parties in parallel and returns which ones
    are allowed to proceed with draft generation.

    This reduces HTTP overhead compared to calling /evaluate-gates N times.
    """
    logger.info(
        f"Batch evaluating gates for {len(batch_request.contexts)} parties, "
        f"action: {batch_request.proposed_action}"
    )

    # Create individual requests for each context
    async def evaluate_single(context):
        single_request = EvaluateGatesRequest(
            context=context,
            proposed_action=batch_request.proposed_action,
            proposed_tone=batch_request.proposed_tone,
        )
        result = await gate_evaluator.evaluate(single_request)

        # Find blocking gate if not allowed
        blocking_gate = None
        if not result.allowed:
            for gate_name, gate_result in result.gate_results.items():
                if not gate_result.passed:
                    blocking_gate = gate_name
                    break

        return PartyGateResult(
            party_id=context.party.party_id,
            customer_code=context.party.customer_code,
            allowed=result.allowed,
            gate_results=result.gate_results,
            recommended_action=result.recommended_action,
            blocking_gate=blocking_gate,
        )

    # Evaluate all parties concurrently
    results = await asyncio.gather(*[evaluate_single(ctx) for ctx in batch_request.contexts])

    allowed_count = sum(1 for r in results if r.allowed)
    blocked_count = len(results) - allowed_count

    logger.info(f"Batch gate evaluation complete: {allowed_count} allowed, {blocked_count} blocked")

    return EvaluateGatesBatchResponse(
        total=len(results),
        allowed_count=allowed_count,
        blocked_count=blocked_count,
        results=list(results),
    )
