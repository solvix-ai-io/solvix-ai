import json
import logging
from typing import Optional, Dict, Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for OpenAI API calls."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_response: bool = True
    ) -> Dict[str, Any]:
        """
        Make a completion request to OpenAI.
        
        Args:
            system_prompt: System message setting context
            user_prompt: User message with the actual request
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_response: If True, request JSON output
        
        Returns:
            Parsed response dict and token usage
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_completion_tokens": max_tokens or self.max_tokens,
        }
        
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}
        
        # Handle models that don't support temperature or max_tokens or json_object
        if self.model.startswith(("o1", "gpt-5")):
            # These models enforce temperature=1 and require max_completion_tokens
            if "temperature" in kwargs:
                del kwargs["temperature"]
            # Some reasoning models don't support response_format="json_object" yet
            if "response_format" in kwargs:
                del kwargs["response_format"]
        
        logger.debug(f"Calling OpenAI: model={self.model}")
        
        response = self.client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        tokens_used = response.usage.total_tokens if response.usage else None
        
        logger.debug(f"OpenAI response: tokens={tokens_used}, finish_reason={finish_reason}")
        
        if not content and finish_reason == "content_filter":
             raise ValueError("OpenAI request was blocked by content filter")

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


# Singleton instance
llm_client = LLMClient()
