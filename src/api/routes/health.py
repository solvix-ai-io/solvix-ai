"""
Health check API endpoint.

GET /health - Check service health and configuration.
"""
import time
from fastapi import APIRouter

from src.api.models.responses import HealthResponse
from src.llm.factory import llm_client

router = APIRouter()

# Track service start time for uptime calculation
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint with LLM provider info.

    Returns:
        - status: healthy/degraded/unhealthy
        - version: API version
        - provider: primary LLM provider (gemini/openai)
        - model: primary model name
        - fallback_provider: fallback LLM provider
        - fallback_model: fallback model name
        - fallback_count: number of times fallback was used
        - model_available: whether primary LLM is responding
        - fallback_available: whether fallback is healthy
        - uptime_seconds: API uptime
    """
    uptime = time.time() - _start_time

    # Check LLM health
    llm_health = await llm_client.health_check()
    primary_healthy = llm_health["primary"]["status"] == "healthy"
    fallback_status = llm_health["fallback"].get("status", "disabled")

    return HealthResponse(
        status="healthy" if primary_healthy else "degraded",
        version="0.1.0",
        provider=llm_client.provider_name,
        model=llm_client.model_name,
        fallback_provider=llm_client.fallback.provider_name if llm_client.fallback else None,
        fallback_model=llm_client.fallback.model_name if llm_client.fallback else None,
        fallback_count=llm_client.fallback_count,
        model_available=primary_healthy,
        fallback_available=fallback_status == "healthy",
        uptime_seconds=round(uptime, 2)
    )
