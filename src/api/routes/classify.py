"""
Email classification API endpoint.

POST /classify - Classify an inbound email from a debtor.
"""

import logging

from fastapi import APIRouter

from src.api.errors import ErrorResponse
from src.api.models.requests import ClassifyRequest
from src.api.models.responses import ClassifyResponse
from src.engine.classifier import classifier

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
async def classify_email(request: ClassifyRequest) -> ClassifyResponse:
    """
    Classify an inbound email from a debtor.

    Returns classification (COOPERATIVE, PROMISE, DISPUTE, etc.),
    confidence score, and any extracted data.
    """
    logger.info(f"Classifying email for party: {request.context.party.party_id}")
    result = await classifier.classify(request)
    logger.info(f"Classification: {result.classification} ({result.confidence:.2f})")
    return result
