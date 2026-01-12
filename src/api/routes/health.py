"""
Health check API endpoint.

GET /health - Check service health and configuration.
"""
import time
from fastapi import APIRouter

from src.api.models.responses import HealthResponse
from src.config.settings import settings

router = APIRouter()

# Track service start time for uptime calculation
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service health status for:
    - Load balancer health checks
    - Circuit breaker integration (Django backend)
    - Monitoring dashboards
    """
    uptime = time.time() - _start_time

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model=settings.openai_model,
        model_available=True,
        uptime_seconds=round(uptime, 2),
    )
