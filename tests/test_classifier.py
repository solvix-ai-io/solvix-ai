"""Unit tests for EmailClassifier."""
import pytest
from unittest.mock import MagicMock, patch

from src.engine.classifier import EmailClassifier
from src.api.models.responses import ClassifyResponse

class TestEmailClassifier:
    """Tests for EmailClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return EmailClassifier()

    def test_classify_hardship_email(self, classifier, sample_classify_request):
        """Test classification of hardship email."""
        mock_result = {
            "classification": "HARDSHIP",
            "confidence": 0.92,
            "reasoning": "Customer mentions job loss and requests payment plan",
            "extracted_data": {
                "promise_date": None,
                "promise_amount": None,
                "dispute_type": None,
                "dispute_reason": None,
                "redirect_contact": None,
                "redirect_email": None
            },
            "_tokens_used": 100
        }

        with patch("src.engine.classifier.llm_client.complete") as mock_complete:
            mock_complete.return_value = mock_result
            
            result = classifier.classify(sample_classify_request)
            
            assert isinstance(result, ClassifyResponse)
            assert result.classification == "HARDSHIP"
            assert result.confidence >= 0.9
            assert "job" in result.reasoning.lower() or "hardship" in result.reasoning.lower()

    def test_classify_promise_to_pay(self, classifier, sample_classify_request):
        """Test classification of promise to pay email."""
        from datetime import date
        sample_classify_request.email.body = "I will pay the full amount of £1500 by Friday January 20th."
        
        mock_result = {
            "classification": "PROMISE_TO_PAY",
            "confidence": 0.95,
            "reasoning": "Customer commits to specific payment amount and date",
            "extracted_data": {
                "promise_amount": 1500,
                "promise_date": "2024-01-20"
            },
            "_tokens_used": 100
        }

        with patch("src.engine.classifier.llm_client.complete") as mock_complete:
            mock_complete.return_value = mock_result
            
            result = classifier.classify(sample_classify_request)
            
            assert result.classification == "PROMISE_TO_PAY"
            assert result.extracted_data is not None
            assert result.extracted_data.promise_amount == 1500
            assert result.extracted_data.promise_date == date(2024, 1, 20)

    def test_classify_dispute_email(self, classifier, sample_classify_request):
        """Test classification of dispute email."""
        sample_classify_request.email.body = "I never received the goods for invoice #12345. This charge is incorrect."
        
        mock_result = {
            "classification": "DISPUTE",
            "confidence": 0.88,
            "reasoning": "Customer claims goods not received and disputes charge",
            "extracted_data": {
                "dispute_reason": "goods_not_received"
            },
            "_tokens_used": 100
        }

        with patch("src.engine.classifier.llm_client.complete") as mock_complete:
            mock_complete.return_value = mock_result
            
            result = classifier.classify(sample_classify_request)
            
            assert result.classification == "DISPUTE"
            assert result.extracted_data.dispute_reason == "goods_not_received"

    def test_classify_unsubscribe_email(self, classifier, sample_classify_request):
        """Test classification of unsubscribe request."""
        sample_classify_request.email.body = "Please remove me from your mailing list. I do not wish to receive further emails."
        
        mock_result = {
            "classification": "UNSUBSCRIBE",
            "confidence": 0.97,
            "reasoning": "Customer explicitly requests removal from mailing list",
            "extracted_data": None,
            "_tokens_used": 100
        }

        with patch("src.engine.classifier.llm_client.complete") as mock_complete:
            mock_complete.return_value = mock_result
            
            result = classifier.classify(sample_classify_request)
            
            assert result.classification == "UNSUBSCRIBE"
            assert result.confidence > 0.9

    def test_classify_handles_json_parse_error(self, classifier, sample_classify_request):
        """Test classifier handles malformed LLM response."""
        mock_result = {} # Missing classification

        with patch("src.engine.classifier.llm_client.complete") as mock_complete:
            mock_complete.return_value = mock_result
            
            with pytest.raises(KeyError):
                classifier.classify(sample_classify_request)

    def test_classify_out_of_office(self, classifier, sample_classify_request):
        """Test classification of out of office auto-reply."""
        sample_classify_request.email.body = "I am currently out of the office with no access to email. I will return on January 25th."
        sample_classify_request.email.subject = "Out of Office: Re: Invoice #12345"
        
        mock_result = {
            "classification": "OUT_OF_OFFICE",
            "confidence": 0.99,
            "reasoning": "Automatic out of office reply detected",
            "extracted_data": None,
            "_tokens_used": 100
        }

        with patch("src.engine.classifier.llm_client.complete") as mock_complete:
            mock_complete.return_value = mock_result
            
            result = classifier.classify(sample_classify_request)
            
            assert result.classification == "OUT_OF_OFFICE"
