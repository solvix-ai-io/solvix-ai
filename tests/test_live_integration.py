"""Live integration tests against OpenAI API."""
import os
import pytest
from datetime import date

from src.engine.classifier import classifier
from src.engine.generator import generator
from src.engine.gate_evaluator import gate_evaluator
from src.llm.client import llm_client

# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)

# Increase token limit for reasoning models that might need more "thinking" space
# This is necessary because 'finish_reason=length' was observed with 2000 tokens
llm_client.max_tokens = 10000

class TestLiveIntegration:
    """Live integration tests using real OpenAI API."""

    def test_live_classify_promise(self, sample_classify_request):
        """Test classifying a promise to pay with real LLM."""
        # Setup specific email content
        sample_classify_request.email.body = "I will pay the full amount of £1500 by January 20th 2025."
        sample_classify_request.email.subject = "Payment plan"
        
        result = classifier.classify(sample_classify_request)
        
        # Verify real classification logic works
        assert result.classification == "PROMISE_TO_PAY"
        assert result.confidence > 0.5
        assert result.extracted_data.promise_amount == 1500
        # Date parsing might vary slightly depending on current year in prompt vs extraction, 
        # but 2025-01-20 should be consistent if prompt is clear.
        # Note: The prompt doesn't explicitly inject 'today's date' in all places, 
        # but the extraction usually handles absolute dates well.
        assert result.extracted_data.promise_date == date(2025, 1, 20)

    def test_live_generate_draft(self, sample_generate_draft_request):
        """Test generating a draft with real LLM."""
        sample_generate_draft_request.tone = "professional"
        
        result = generator.generate(sample_generate_draft_request)
        
        assert result.subject
        assert result.body
        assert result.tone_used == "professional"
        assert len(result.body) > 50  # Should have substantial content
        
        # Check if it references the invoice from the sample data (INV-12345)
        # We can't guarantee exact string match of invoice number if LLM hallucinates or formats differently,
        # but usually it should be there.
        # Actually, let's relax this check or just log it, but the unit test checks exact string.
        # Real LLM should respect the instruction to include invoices.
        assert "INV-12345" in result.body

    def test_live_gate_evaluation(self, sample_evaluate_gates_request):
        """Test evaluating gates with real LLM."""
        # Make sure it's a passable scenario
        sample_evaluate_gates_request.context.active_dispute = False
        sample_evaluate_gates_request.context.touch_cap = 100
        sample_evaluate_gates_request.context.communication.touch_count = 1
        
        result = gate_evaluator.evaluate(sample_evaluate_gates_request)
        
        assert result.allowed is True
        assert result.gate_results["dispute_active"].passed is True
