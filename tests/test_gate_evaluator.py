"""Unit tests for GateEvaluator."""
import pytest
from unittest.mock import MagicMock, patch

from src.engine.gate_evaluator import GateEvaluator
from src.api.models.responses import EvaluateGatesResponse

class TestGateEvaluator:
    """Tests for GateEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return GateEvaluator()

    def test_evaluate_touch_cap_exceeded(self, evaluator, sample_evaluate_gates_request):
        """Test blocking when touch cap is exceeded."""
        # Setup context where touch count equals cap
        sample_evaluate_gates_request.context.communication.touch_count = 10
        sample_evaluate_gates_request.context.touch_cap = 10
        
        mock_result = {
            "allowed": False,
            "gate_results": {
                "touch_cap": {
                    "passed": False,
                    "reason": "Touch cap of 10 reached",
                    "current_value": 10,
                    "threshold": 10
                }
            },
            "recommended_action": "escalate",
            "_tokens_used": 100
        }

        with patch("src.engine.gate_evaluator.llm_client.complete") as mock_complete:
            mock_complete.return_value = mock_result
            
            result = evaluator.evaluate(sample_evaluate_gates_request)
            
            assert isinstance(result, EvaluateGatesResponse)
            assert result.allowed is False
            assert result.gate_results["touch_cap"].passed is False
            assert result.gate_results["touch_cap"].reason == "Touch cap of 10 reached"
            
            # Verify prompt contained correct info
            call_args = mock_complete.call_args
            user_prompt = call_args.kwargs["user_prompt"]
            # Check for values loosely as string format might vary
            assert "10" in user_prompt 

    def test_evaluate_active_dispute(self, evaluator, sample_evaluate_gates_request):
        """Test blocking when there is an active dispute."""
        sample_evaluate_gates_request.context.active_dispute = True
        
        mock_result = {
            "allowed": False,
            "gate_results": {
                "dispute_active": {
                    "passed": False,
                    "reason": "Active dispute prevents collection",
                    "current_value": True,
                    "threshold": False
                }
            },
            "recommended_action": "resolve_dispute",
            "_tokens_used": 100
        }

        with patch("src.engine.gate_evaluator.llm_client.complete") as mock_complete:
            mock_complete.return_value = mock_result
            
            result = evaluator.evaluate(sample_evaluate_gates_request)
            
            assert result.allowed is False
            assert result.gate_results["dispute_active"].passed is False
            
            # Verify prompt contained correct info
            call_args = mock_complete.call_args
            user_prompt = call_args.kwargs["user_prompt"]
            assert "True" in user_prompt or "active_dispute" in user_prompt

    def test_evaluate_allowed(self, evaluator, sample_evaluate_gates_request):
        """Test allowing when all gates pass."""
        mock_result = {
            "allowed": True,
            "gate_results": {
                "touch_cap": {
                    "passed": True,
                    "reason": "Touch count within limits",
                    "current_value": 3,
                    "threshold": 10
                },
                "dispute_active": {
                    "passed": True,
                    "reason": "No active dispute",
                    "current_value": False,
                    "threshold": False
                }
            },
            "recommended_action": "proceed",
            "_tokens_used": 100
        }

        with patch("src.engine.gate_evaluator.llm_client.complete") as mock_complete:
            mock_complete.return_value = mock_result
            
            result = evaluator.evaluate(sample_evaluate_gates_request)
            
            assert isinstance(result, EvaluateGatesResponse)
            assert result.allowed is True
            assert result.gate_results["touch_cap"].passed is True
            assert result.gate_results["dispute_active"].passed is True
