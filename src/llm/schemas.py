"""
Pydantic models for validating LLM responses.

These models ensure type safety when parsing LLM outputs and provide
clear error messages when the LLM returns malformed data.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class LLMExtractedData(BaseModel):
    """Data extracted from email content by the LLM."""

    promise_date: Optional[str] = None  # String from LLM, parsed to date in engine
    promise_amount: Optional[float] = None
    dispute_type: Optional[str] = None
    dispute_reason: Optional[str] = None
    redirect_contact: Optional[str] = None
    redirect_email: Optional[str] = None


class ClassificationLLMResponse(BaseModel):
    """
    Expected response structure from classification LLM calls.

    The LLM must return JSON matching this schema.
    """

    classification: str = Field(
        ...,
        description="Email classification category",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Explanation for the classification",
    )
    extracted_data: Optional[LLMExtractedData] = Field(
        None,
        description="Data extracted from the email",
    )

    @field_validator("classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        valid_classifications = {
            "INSOLVENCY",
            "DISPUTE",
            "ALREADY_PAID",
            "UNSUBSCRIBE",
            "HOSTILE",
            "PROMISE_TO_PAY",
            "HARDSHIP",
            "PLAN_REQUEST",
            "REDIRECT",
            "REQUEST_INFO",
            "OUT_OF_OFFICE",
            "COOPERATIVE",
            "UNCLEAR",
        }
        upper_v = v.upper()
        if upper_v not in valid_classifications:
            raise ValueError(
                f"Invalid classification '{v}'. Must be one of: {', '.join(sorted(valid_classifications))}"
            )
        return upper_v


class DraftGenerationLLMResponse(BaseModel):
    """
    Expected response structure from draft generation LLM calls.

    The LLM must return JSON matching this schema.
    """

    subject: str = Field(
        ...,
        min_length=1,
        description="Email subject line",
    )
    body: str = Field(
        ...,
        min_length=1,
        description="Email body content",
    )


class GateResultLLM(BaseModel):
    """Single gate evaluation result from LLM."""

    passed: bool = Field(..., description="Whether the gate passed")
    reason: str = Field(..., description="Explanation for the gate result")
    current_value: Optional[Any] = Field(None, description="Current value being evaluated")
    threshold: Optional[Any] = Field(None, description="Threshold for the gate")


class GateEvaluationLLMResponse(BaseModel):
    """
    Expected response structure from gate evaluation LLM calls.

    The LLM must return JSON matching this schema.
    """

    allowed: bool = Field(
        ...,
        description="Whether the proposed action is allowed",
    )
    gate_results: Dict[str, GateResultLLM] = Field(
        ...,
        description="Individual gate evaluation results",
    )
    recommended_action: Optional[str] = Field(
        None,
        description="Recommended alternative action if not allowed",
    )

    # Note: We don't validate that all 6 gates are present because
    # the LLM may return only relevant gates depending on context.
    # The required gates are: touch_cap, cooling_off, dispute_active,
    # hardship, unsubscribe, escalation_appropriate
