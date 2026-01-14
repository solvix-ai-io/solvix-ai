"""
Backwards compatibility wrapper for existing code.

This module provides an adapter that wraps the new LangChain-based
provider system with the old synchronous dict-based interface.
"""
import json
import logging
import asyncio
from typing import Optional, Dict, Any

from .factory import llm_client as llm_provider

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Backwards-compatible wrapper for the new provider system.

    Maintains the same interface as the old OpenAI-only client
    while using the new multi-provider system underneath.
    """

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_response: bool = True
    ) -> Dict[str, Any]:
        """
        Make a completion request using the configured LLM provider.

        Args:
            system_prompt: System message setting context
            user_prompt: User message with the actual request
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_response: If True, request JSON output

        Returns:
            Parsed response dict with _tokens_used field
        """
        # Call the async provider (LangChain-based)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, use the current loop
            import nest_asyncio
            nest_asyncio.apply()

        response = loop.run_until_complete(
            llm_provider.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_response
            )
        )

        # Convert LLMResponse to the old dict format
        content = response.content
        tokens_used = response.usage.get("total_tokens", 0)

        if json_response:
            try:
                # Remove markdown code block if present
                clean_content = content.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_content)
                parsed["_tokens_used"] = tokens_used
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                raise e

        return {"content": content, "_tokens_used": tokens_used}


# Singleton instance (backwards compatibility)
llm_client = LLMClient()
