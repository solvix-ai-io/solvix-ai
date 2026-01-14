"""Gemini LLM provider using LangChain."""
import os
import logging
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Gemini 3 Flash LLM provider using LangChain."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.3,
        max_tokens: int = 2048
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided")

        # LangChain handles all Gemini API complexity
        self.client = ChatGoogleGenerativeAI(
            model=self._model,
            google_api_key=self.api_key,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )

        logger.info(f"Initialized Gemini provider with model: {self._model}")

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Generate completion using Gemini via LangChain.

        LangChain handles:
        - Gemini's response_mime_type for JSON mode
        - Thinking configuration
        - Token counting
        - Error handling
        """
        try:
            # Build messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            # Create client with appropriate settings
            client_kwargs = {
                "model": self._model,
                "google_api_key": self.api_key,
                "temperature": temperature if temperature is not None else self._temperature,
                "max_output_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            }

            # For JSON mode, configure response_mime_type
            if json_mode:
                client_kwargs["response_mime_type"] = "application/json"

            # Create client for this specific request
            client = ChatGoogleGenerativeAI(**client_kwargs)

            logger.debug(f"Calling Gemini: model={self._model}, json_mode={json_mode}")

            # LangChain's invoke handles async automatically
            response = await client.ainvoke(messages)

            # Extract usage metadata (LangChain standardizes this)
            usage = {
                "prompt_tokens": response.usage_metadata.get("input_tokens", 0) if response.usage_metadata else 0,
                "completion_tokens": response.usage_metadata.get("output_tokens", 0) if response.usage_metadata else 0,
                "total_tokens": response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0,
            }

            logger.debug(f"Gemini response: tokens={usage['total_tokens']}")

            return LLMResponse(
                content=response.content,
                model=self._model,
                provider="gemini",
                usage=usage,
                raw_response={"response_metadata": response.response_metadata} if hasattr(response, 'response_metadata') else None
            )

        except Exception as e:
            logger.error(f"Gemini provider error: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check Gemini service health."""
        try:
            response = await self.complete(
                system_prompt="You are a test assistant.",
                user_prompt="Reply with 'OK'",
                max_tokens=10
            )
            return {
                "status": "healthy",
                "provider": "gemini",
                "model": self._model,
                "test_response": response.content[:20]
            }
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return {
                "status": "unhealthy",
                "provider": "gemini",
                "model": self._model,
                "error": str(e)
            }
