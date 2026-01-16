"""LLM Provider factory with automatic fallback."""

import logging

from src.config.settings import settings

from .base import LLMResponse
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class LLMProviderWithFallback:
    """
    LLM Provider with automatic fallback from Gemini → OpenAI.

    Primary: Gemini 3 Flash (free tier, fast)
    Fallback: OpenAI gpt-5-nano (if Gemini fails)
    """

    def __init__(self, primary_provider: str = None, fallback_provider: str = "openai"):
        self.primary_provider_name = primary_provider or settings.llm_provider
        self.fallback_provider_name = fallback_provider

        # Lazy initialization - providers created on first use
        self._primary = None
        self._fallback = None
        self.fallback_count = 0

        logger.info(
            "LLM factory created with primary=%s, fallback=%s",
            self.primary_provider_name,
            self.fallback_provider_name,
        )

    @property
    def primary(self):
        """Lazy-initialize primary provider."""
        if self._primary is None:
            if self.primary_provider_name == "gemini":
                self._primary = GeminiProvider(
                    model=settings.gemini_model,
                    temperature=settings.gemini_temperature,
                    max_tokens=settings.gemini_max_tokens,
                )
            elif self.primary_provider_name == "openai":
                self._primary = OpenAIProvider(
                    model=settings.openai_model,
                    temperature=settings.openai_temperature,
                    max_tokens=settings.openai_max_tokens,
                )
            else:
                raise ValueError(f"Unknown primary provider: {self.primary_provider_name}")
        return self._primary

    @property
    def fallback(self):
        """Lazy-initialize fallback provider."""
        if self._fallback is None and self.fallback_provider_name:
            try:
                if self.fallback_provider_name == "openai":
                    self._fallback = OpenAIProvider(
                        model=settings.openai_model,
                        temperature=settings.openai_temperature,
                        max_tokens=settings.openai_max_tokens,
                    )
                elif self.fallback_provider_name == "gemini":
                    self._fallback = GeminiProvider(
                        model=settings.gemini_model,
                        temperature=settings.gemini_temperature,
                        max_tokens=settings.gemini_max_tokens,
                    )
            except ValueError as e:
                # API key not configured - disable fallback gracefully
                logger.warning("Fallback provider unavailable: %s", e)
                self.fallback_provider_name = None  # Disable fallback
                return None
        return self._fallback

    @property
    def fallback_enabled(self):
        return self.fallback_provider_name is not None

    async def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        """
        Generate completion with automatic fallback.

        Tries primary provider first, falls back to secondary on failure.
        """
        try:
            response = await self.primary.complete(system_prompt, user_prompt, **kwargs)
            logger.info(
                "LLM request succeeded: provider=%s, model=%s, tokens=%s",
                response.provider,
                response.model,
                response.usage["total_tokens"],
            )
            return response
        except Exception as e:
            logger.error("Primary provider (%s) failed: %s", self.primary.provider_name, e)

            if not self.fallback_enabled:
                logger.error("No fallback provider configured, raising error")
                raise

            logger.warning("Falling back to %s", self.fallback.provider_name)
            self.fallback_count += 1

            try:
                response = await self.fallback.complete(system_prompt, user_prompt, **kwargs)
                logger.info(
                    "Fallback succeeded: provider=%s, model=%s, tokens=%s",
                    response.provider,
                    response.model,
                    response.usage["total_tokens"],
                )
                return response
            except Exception as fallback_error:
                logger.error("Fallback provider also failed: %s", fallback_error)
                raise fallback_error

    async def health_check(self) -> dict:
        """Check health of both providers."""
        primary_health = await self.primary.health_check()
        fallback_health = (
            await self.fallback.health_check() if self.fallback else {"status": "disabled"}
        )

        return {
            "primary": primary_health,
            "fallback": fallback_health,
            "fallback_count": self.fallback_count,
        }

    @property
    def provider_name(self) -> str:
        return self.primary.provider_name

    @property
    def model_name(self) -> str:
        return self.primary.model_name


# Singleton instance (backwards compatibility with existing code)
llm_client = LLMProviderWithFallback()
