"""Base LLM provider abstraction."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standardized LLM response across all providers."""
    content: str
    model: str
    provider: str  # "openai", "gemini", etc.
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False
    ) -> LLMResponse:
        """Generate completion from prompts."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider availability and return model info."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (openai, gemini, etc.)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return current model name."""
        pass
