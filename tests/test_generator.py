"""Unit tests for DraftGenerator."""
import pytest
from unittest.mock import MagicMock, patch

from src.engine.generator import DraftGenerator
from src.api.models.responses import GenerateDraftResponse

class TestDraftGenerator:
    """Tests for DraftGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return DraftGenerator()

    def test_generate_draft_referencing_invoices(self, generator, sample_generate_draft_request):
        """Test draft generation references specific invoices."""
        sample_generate_draft_request.tone = "firm"
        
        # Mock LLM response containing invoice numbers
        mock_result = {
            "subject": "Overdue Invoices",
            "body": "Dear Customer, Please pay invoice INV-12345 immediately. INV-12346 is also overdue.",
            "_tokens_used": 150
        }

        with patch("src.engine.generator.llm_client.complete") as mock_complete:
            mock_complete.return_value = mock_result
            
            result = generator.generate(sample_generate_draft_request)
            
            assert isinstance(result, GenerateDraftResponse)
            assert result.tone_used == "firm"
            # Verify invoices are detected in the body
            assert "INV-12345" in result.invoices_referenced
            assert "INV-12346" in result.invoices_referenced

    def test_generate_draft_different_tones(self, generator, sample_generate_draft_request):
        """Test draft generation with different tones."""
        tones = ["friendly_reminder", "professional", "urgent"]
        
        with patch("src.engine.generator.llm_client.complete") as mock_complete:
            for tone in tones:
                sample_generate_draft_request.tone = tone
                mock_complete.return_value = {
                    "subject": f"{tone} subject",
                    "body": f"Body with {tone} tone.",
                    "_tokens_used": 100
                }
                
                result = generator.generate(sample_generate_draft_request)
                
                assert result.tone_used == tone
                assert result.body == f"Body with {tone} tone."
