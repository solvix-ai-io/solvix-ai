"""OpenAI LLM provider using LangChain."""
import os
import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using LangChain."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-5-nano",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided")

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
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Generate completion using OpenAI via LangChain.

        LangChain automatically handles:
        - gpt-5 model quirks (no temperature support)
        - JSON mode via response_format
        - Retries and error handling
        - Token counting
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
                "openai_api_key": self.api_key,
                "temperature": temperature if temperature is not None else self._temperature,
                "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            }

            # For JSON mode, configure response_format
            if json_mode:
                client_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

            # Create client for this specific request
            client = ChatOpenAI(**client_kwargs)

            logger.debug(f"Calling OpenAI: model={self._model}, json_mode={json_mode}")

            # LangChain's invoke handles async automatically
            response = await client.ainvoke(messages)

            # Extract usage metadata (LangChain standardizes this)
            usage = {
                "prompt_tokens": response.usage_metadata.get("input_tokens", 0) if response.usage_metadata else 0,
                "completion_tokens": response.usage_metadata.get("output_tokens", 0) if response.usage_metadata else 0,
                "total_tokens": response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0,
            }

            logger.debug(f"OpenAI response: tokens={usage['total_tokens']}")

            return LLMResponse(
                content=response.content,
                model=self._model,
                provider="openai",
                usage=usage,
                raw_response={"response_metadata": response.response_metadata} if hasattr(response, 'response_metadata') else None
            )

        except Exception as e:
            logger.error(f"OpenAI provider error: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI service health."""
        try:
            response = await self.complete(
                system_prompt="You are a test assistant.",
                user_prompt="Reply with 'OK'",
                max_tokens=10
            )
            return {
                "status": "healthy",
                "provider": "openai",
                "model": self._model,
                "test_response": response.content[:20]
            }
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return {
                "status": "unhealthy",
                "provider": "openai",
                "model": self._model,
                "error": str(e)
            }
