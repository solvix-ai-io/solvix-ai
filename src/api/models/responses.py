from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import date


class ExtractedData(BaseModel):
    """Data extracted from email by AI."""
    promise_date: Optional[date] = None
    promise_amount: Optional[float] = None
    dispute_type: Optional[str] = None
    dispute_reason: Optional[str] = None
    redirect_contact: Optional[str] = None
    redirect_email: Optional[str] = None


class ClassifyResponse(BaseModel):
    """Response from email classification."""
    classification: str  # COOPERATIVE, PROMISE, DISPUTE, HOSTILE, QUERY, OUT_OF_OFFICE, UNSUBSCRIBE, OTHER
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    extracted_data: Optional[ExtractedData] = None
    tokens_used: Optional[int] = None


class GenerateDraftResponse(BaseModel):
    """Response from draft generation."""
    subject: str
    body: str
    tone_used: str
    invoices_referenced: List[str] = []
    tokens_used: Optional[int] = None


class GateResult(BaseModel):
    """Result of a single gate evaluation."""
    passed: bool
    reason: str
    current_value: Optional[Any] = None
    threshold: Optional[Any] = None


class EvaluateGatesResponse(BaseModel):
    """Response from gate evaluation."""
    allowed: bool
    gate_results: Dict[str, GateResult]
    recommended_action: Optional[str] = None
    tokens_used: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str  # "healthy", "degraded", "unhealthy"
    version: str
    provider: str  # "gemini", "openai", etc.
    model: str
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    fallback_count: int = 0
    model_available: bool = True
    fallback_available: bool = False
    uptime_seconds: Optional[float] = None
