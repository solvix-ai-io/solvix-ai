"""LLM Provider factory with automatic fallback."""
import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load .env file to make environment variables available
load_dotenv()

from .base import BaseLLMProvider, LLMResponse
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)


class LLMProviderWithFallback:
    """
    LLM Provider with automatic fallback from Gemini → OpenAI.

    Primary: Gemini 3 Flash (free tier, fast)
    Fallback: OpenAI gpt-5-nano (if Gemini fails)
    """

    def __init__(
        self,
        primary_provider: str = None,
        fallback_provider: str = "openai"
    ):
        self.primary_provider_name = primary_provider or os.getenv("LLM_PROVIDER", "gemini")
        self.fallback_provider_name = fallback_provider

        # Lazy initialization - providers created on first use
        self._primary = None
        self._fallback = None
        self.fallback_count = 0

        logger.info(f"LLM factory created with primary={self.primary_provider_name}, fallback={self.fallback_provider_name}")

    @property
    def primary(self):
        """Lazy-initialize primary provider."""
        if self._primary is None:
            if self.primary_provider_name == "gemini":
                self._primary = GeminiProvider(
                    model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
                    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
                    max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
                )
            elif self.primary_provider_name == "openai":
                self._primary = OpenAIProvider(
                    model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
                    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
                )
            else:
                raise ValueError(f"Unknown primary provider: {self.primary_provider_name}")
        return self._primary

    @property
    def fallback(self):
        """Lazy-initialize fallback provider."""
        if self._fallback is None and self.fallback_provider_name:
            if self.fallback_provider_name == "openai":
                self._fallback = OpenAIProvider(
                    model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
                    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
                )
            elif self.fallback_provider_name == "gemini":
                self._fallback = GeminiProvider(
                    model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
                    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
                    max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
                )
        return self._fallback

    @property
    def fallback_enabled(self):
        return self.fallback_provider_name is not None

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion with automatic fallback.

        Tries primary provider first, falls back to secondary on failure.
        """
        try:
            response = await self.primary.complete(system_prompt, user_prompt, **kwargs)
            logger.info(f"LLM request succeeded: provider={response.provider}, model={response.model}, tokens={response.usage['total_tokens']}")
            return response
        except Exception as e:
            logger.error(f"Primary provider ({self.primary.provider_name}) failed: {e}")

            if not self.fallback_enabled:
                logger.error("No fallback provider configured, raising error")
                raise

            logger.warning(f"Falling back to {self.fallback.provider_name}")
            self.fallback_count += 1

            try:
                response = await self.fallback.complete(system_prompt, user_prompt, **kwargs)
                logger.info(f"Fallback succeeded: provider={response.provider}, model={response.model}, tokens={response.usage['total_tokens']}")
                return response
            except Exception as fallback_error:
                logger.error(f"Fallback provider also failed: {fallback_error}")
                raise fallback_error

    async def health_check(self) -> dict:
        """Check health of both providers."""
        primary_health = await self.primary.health_check()
        fallback_health = await self.fallback.health_check() if self.fallback else {"status": "disabled"}

        return {
            "primary": primary_health,
            "fallback": fallback_health,
            "fallback_count": self.fallback_count
        }

    @property
    def provider_name(self) -> str:
        return self.primary.provider_name

    @property
    def model_name(self) -> str:
        return self.primary.model_name


# Singleton instance (backwards compatibility with existing code)
llm_client = LLMProviderWithFallback()
