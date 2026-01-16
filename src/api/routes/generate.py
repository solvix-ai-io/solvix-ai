"""
Draft generation API endpoint.

POST /generate-draft - Generate a collection email draft.
"""

import logging

from fastapi import APIRouter

from src.api.errors import ErrorResponse
from src.api.models.requests import GenerateDraftRequest
from src.api.models.responses import GenerateDraftResponse
from src.engine.generator import generator

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/generate-draft",
    response_model=GenerateDraftResponse,
    responses={
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
        503: {"model": ErrorResponse, "description": "LLM provider unavailable"},
    },
)
async def generate_draft(request: GenerateDraftRequest) -> GenerateDraftResponse:
    """
    Generate a collection email draft.

    Returns subject, body, and metadata about the generated draft.
    """
    logger.info(f"Generating draft for party: {request.context.party.party_id}")
    result = await generator.generate(request)
    logger.info(f"Generated draft with tone: {result.tone_used}")
    return result
