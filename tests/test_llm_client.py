"""Unit tests for LLMClient."""
import pytest
from unittest.mock import MagicMock, patch
from openai import APIConnectionError

from src.llm.client import LLMClient

class TestLLMClient:
    """Tests for LLMClient."""

    @pytest.fixture
    def llm_client(self):
        """Create LLMClient instance."""
        with patch("src.llm.client.OpenAI"): # Mock OpenAI constructor
             client = LLMClient()
             return client

    def test_complete_retry_mechanism(self, llm_client):
        """Test that complete retries on failure."""
        # Mock the create method on the OpenAI client instance
        mock_create = llm_client.client.chat.completions.create
        
        # Configure mock to raise exception twice then return success
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"test": "success"}'
        mock_response.usage.total_tokens = 10
        
        # APIConnectionError requires 'request' arg in recent versions or message? 
        # OpenAI exceptions usually need args. 
        # Let's use a generic Exception or specific one if tenacity is configured to catch it.
        # tenacity defaults to retrying on all Exceptions unless configured?
        # The code: @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        # It doesn't specify 'retry' arg, so it retries on all exceptions.
        
        mock_create.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            mock_response
        ]
        
        result = llm_client.complete(
            system_prompt="sys",
            user_prompt="user"
        )
        
        assert result["test"] == "success"
        assert mock_create.call_count == 3
