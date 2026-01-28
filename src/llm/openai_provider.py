"""OpenAI LLM provider using LangChain."""

import logging
from typing import Any, Dict, Optional, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config.settings import settings

from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Import LengthFinishReasonError for handling reasoning model token exhaustion
try:
    from openai import LengthFinishReasonError
except ImportError:
    LengthFinishReasonError = None  # Older SDK versions may not have this


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using LangChain."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_model
        self._temperature = temperature if temperature is not None else settings.openai_temperature
        self._max_tokens = max_tokens if max_tokens is not None else settings.openai_max_tokens

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided (set via environment or .env file)")

        # LangChain handles all OpenAI API complexity
        self.client = ChatOpenAI(
            model=self._model,
            openai_api_key=self.api_key,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        logger.info(f"Initialized OpenAI provider with model: {self._model}")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        json_mode: bool = False,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> LLMResponse:
        """
        Generate completion using OpenAI via LangChain.

        LangChain automatically handles:
        - gpt-5 model quirks (no temperature support)
        - JSON mode via response_format
        - Structured output via with_structured_output()
        - Token counting

        Args:
            response_schema: Optional Pydantic model for structured output.
                When provided, uses LangChain's with_structured_output().
        """
        try:
            # Build messages
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            # Create client with appropriate settings
            client_kwargs = {
                "model": self._model,
                "openai_api_key": self.api_key,
                "temperature": temperature if temperature is not None else self._temperature,
                "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            }

            # For JSON mode without schema, configure response_format
            if json_mode and not response_schema:
                client_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

            # Create client for this specific request
            client = ChatOpenAI(**client_kwargs)

            logger.debug(
                "Calling OpenAI: model=%s, json_mode=%s, has_schema=%s",
                self._model,
                json_mode,
                response_schema is not None,
            )

            # Use structured output if schema provided (more reliable)
            if response_schema:
                structured_client = client.with_structured_output(
                    response_schema,
                    method="json_schema",
                )
                result = await structured_client.ainvoke(messages)
                content = result.model_dump_json()
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                return LLMResponse(
                    content=content,
                    model=self._model,
                    provider="openai",
                    usage=usage,
                    raw_response={"structured": True},
                )

            # Standard invoke for non-structured output
            response = await client.ainvoke(messages)

            # Extract usage metadata (LangChain standardizes this)
            usage = {
                "prompt_tokens": response.usage_metadata.get("input_tokens", 0)
                if response.usage_metadata
                else 0,
                "completion_tokens": response.usage_metadata.get("output_tokens", 0)
                if response.usage_metadata
                else 0,
                "total_tokens": response.usage_metadata.get("total_tokens", 0)
                if response.usage_metadata
                else 0,
            }

            logger.debug(f"OpenAI response: tokens={usage['total_tokens']}")

            return LLMResponse(
                content=response.content,
                model=self._model,
                provider="openai",
                usage=usage,
                raw_response={"response_metadata": response.response_metadata}
                if hasattr(response, "response_metadata")
                else None,
            )

        except Exception as e:
            # Check for LengthFinishReasonError (reasoning models exhaust token budget)
            if LengthFinishReasonError and isinstance(e, LengthFinishReasonError):
                effective_max = max_tokens if max_tokens is not None else self._max_tokens
                logger.error(
                    "OpenAI LengthFinishReasonError: model=%s exhausted max_tokens=%d "
                    "(reasoning tokens consumed entire budget). Increase openai_max_tokens.",
                    self._model,
                    effective_max,
                )
                raise ValueError(
                    f"OpenAI model '{self._model}' exhausted max_tokens={effective_max} on reasoning. "
                    f"No output generated. Increase openai_max_tokens in settings (current: {effective_max})."
                ) from e

            # Fallback detection for LengthFinishReasonError via error message
            # (handles cases where LangChain wraps the error)
            error_str = str(e).lower()
            if "length" in error_str and ("finish_reason" in error_str or "limit" in error_str):
                effective_max = max_tokens if max_tokens is not None else self._max_tokens
                logger.error(
                    "OpenAI output truncated (likely reasoning model): max_tokens=%d, model=%s, error=%s",
                    effective_max,
                    self._model,
                    e,
                )
                raise ValueError(
                    f"OpenAI output truncated: max_tokens={effective_max} insufficient for model '{self._model}'. "
                    f"Increase openai_max_tokens in settings."
                ) from e

            logger.error(f"OpenAI provider error: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI service health."""
        try:
            response = await self.complete(
                system_prompt="You are a test assistant.",
                user_prompt="Reply with 'OK'",
                max_tokens=10,
            )
            return {
                "status": "healthy",
                "provider": "openai",
                "model": self._model,
                "test_response": response.content[:20],
            }
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return {
                "status": "unhealthy",
                "provider": "openai",
                "model": self._model,
                "error": str(e),
            }
