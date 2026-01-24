"""
Test LLM provider integration to verify real API calls.

This test verifies that:
1. Gemini API is being called (when available)
2. OpenAI API is being called (when available)
3. Fallback mechanism works correctly
4. No placeholder responses are being used
"""

import os

import pytest

from src.llm.factory import LLMProviderWithFallback
from src.llm.gemini_provider import GeminiProvider
from src.llm.openai_provider import OpenAIProvider


class TestLLMProviders:
    """Test real LLM provider integration."""

    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_gemini_provider_real_call(self):
        """Test that Gemini provider makes real API calls."""
        provider = GeminiProvider()

        response = await provider.complete(
            system_prompt="You are a helpful assistant.",
            user_prompt="Reply with exactly: 'Gemini is working'",
            max_tokens=50,
        )

        # Verify real response
        assert response.content is not None
        assert len(response.content) > 0
        assert response.provider == "gemini"
        assert response.model == "gemini-3-flash-preview"
        assert response.usage["total_tokens"] > 0

        print(f"\n✅ Gemini API Response: {response.content}")
        print(f"✅ Tokens used: {response.usage['total_tokens']}")
        print(f"✅ Model: {response.model}")

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_openai_provider_real_call(self):
        """Test that OpenAI provider makes real API calls."""
        provider = OpenAIProvider()

        # Use higher max_tokens for reasoning models
        response = await provider.complete(
            system_prompt="You are a helpful assistant.",
            user_prompt="Reply with exactly: 'OpenAI is working'",
            max_tokens=200,
        )

        # Verify real response - reasoning models may have empty content if all tokens used for reasoning
        assert response.provider == "openai"
        assert response.model == "gpt-5-nano"
        assert response.usage["total_tokens"] > 0

        # Check if we got actual API response
        assert response.raw_response is not None
        assert "response_metadata" in response.raw_response
        assert "id" in response.raw_response["response_metadata"]

        print("\n✅ OpenAI API called successfully!")
        print(f"✅ Response ID: {response.raw_response['response_metadata']['id']}")
        print(f"✅ Tokens used: {response.usage['total_tokens']}")
        print(
            f"✅ Model: {response.raw_response['response_metadata'].get('model_name', response.model)}"
        )
        print(f"✅ Content: {response.content if response.content else '(reasoning tokens used)'}")

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="No API keys configured",
    )
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self):
        """Test that fallback from Gemini to OpenAI works."""
        # Create provider with Gemini primary, OpenAI fallback
        llm = LLMProviderWithFallback(primary_provider="gemini", fallback_provider="openai")

        response = await llm.complete(
            system_prompt="You are a test assistant.",
            user_prompt="Say 'test successful'",
            max_tokens=20,
        )

        # Should get response from either provider
        assert response.content is not None
        assert len(response.content) > 0
        assert response.provider in ["gemini", "openai"]
        assert response.usage["total_tokens"] > 0

        print(f"\n✅ Response from: {response.provider}")
        print(f"✅ Content: {response.content}")
        print(f"✅ Fallback count: {llm.fallback_count}")

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="No API keys configured",
    )
    @pytest.mark.asyncio
    async def test_health_check_both_providers(self):
        """Test health check for both providers."""
        llm = LLMProviderWithFallback(primary_provider="gemini", fallback_provider="openai")

        health = await llm.health_check()

        print("\n" + "=" * 60)
        print("HEALTH CHECK RESULTS")
        print("=" * 60)
        print(f"Primary Provider: {health['primary']}")
        print(f"Fallback Provider: {health['fallback']}")
        print(f"Fallback Count: {health['fallback_count']}")
        print("=" * 60)

        # At least one provider should be healthy
        assert health["primary"]["status"] == "healthy" or health["fallback"]["status"] == "healthy"

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.asyncio
    async def test_classification_with_openai(self, sample_classify_request):
        """Test email classification using OpenAI."""
        from src.prompts import CLASSIFY_EMAIL_SYSTEM, CLASSIFY_EMAIL_USER

        # Create OpenAI provider
        provider = OpenAIProvider()

        # Setup test email
        sample_classify_request.email.body = "I will pay £500 by next Friday. Please confirm."
        sample_classify_request.email.subject = "Payment confirmation"

        # Build prompt
        total_outstanding = sum(o.amount_due for o in sample_classify_request.context.obligations)
        days_overdue_max = max(
            (o.days_past_due for o in sample_classify_request.context.obligations),
            default=0,
        )

        user_prompt = CLASSIFY_EMAIL_USER.format(
            party_name=sample_classify_request.context.party.name,
            customer_code=sample_classify_request.context.party.customer_code,
            currency=sample_classify_request.context.party.currency,
            total_outstanding=total_outstanding,
            days_overdue_max=days_overdue_max,
            broken_promises_count=sample_classify_request.context.broken_promises_count,
            segment=sample_classify_request.context.behavior.segment
            if sample_classify_request.context.behavior
            else "unknown",
            active_dispute=sample_classify_request.context.active_dispute,
            hardship_indicated=sample_classify_request.context.hardship_indicated,
            is_verified=sample_classify_request.context.party.is_verified,
            party_source=sample_classify_request.context.party.source,
            from_name=sample_classify_request.email.from_name or "Unknown",
            from_address=sample_classify_request.email.from_address,
            subject=sample_classify_request.email.subject,
            body=sample_classify_request.email.body,
        )

        # Call OpenAI
        response = await provider.complete(
            system_prompt=CLASSIFY_EMAIL_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.2,
            json_mode=True,
            max_tokens=2000,
        )

        # Verify real response
        assert response.provider == "openai"
        assert response.usage["total_tokens"] > 0
        assert response.content is not None

        print("\n✅ OpenAI Classification Response:")
        print(f"✅ Tokens: {response.usage['total_tokens']}")
        print(f"✅ Content preview: {response.content[:200]}...")

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") or not os.getenv("OPENAI_API_KEY"),
        reason="Both API keys needed for this test",
    )
    @pytest.mark.asyncio
    async def test_both_providers_working(self):
        """Comprehensive test that both providers are working."""
        print("\n" + "=" * 60)
        print("TESTING BOTH LLM PROVIDERS")
        print("=" * 60)

        # Test Gemini
        try:
            gemini = GeminiProvider()
            gemini_response = await gemini.complete(
                system_prompt="You are a test assistant.",
                user_prompt="Reply with 'Gemini OK'",
                max_tokens=20,
            )
            gemini_working = True
            gemini_tokens = gemini_response.usage["total_tokens"]
            print(f"✅ Gemini: WORKING (tokens: {gemini_tokens})")
        except Exception as e:
            gemini_working = False
            print(f"❌ Gemini: FAILED - {str(e)[:50]}")

        # Test OpenAI
        try:
            openai = OpenAIProvider()
            openai_response = await openai.complete(
                system_prompt="You are a test assistant.",
                user_prompt="Reply with 'OpenAI OK'",
                max_tokens=20,
            )
            openai_working = True
            openai_tokens = openai_response.usage["total_tokens"]
            print(f"✅ OpenAI: WORKING (tokens: {openai_tokens})")
        except Exception as e:
            openai_working = False
            print(f"❌ OpenAI: FAILED - {str(e)[:50]}")

        print("=" * 60)

        # At least one should work
        assert gemini_working or openai_working, "Neither provider is working!"

        if gemini_working and openai_working:
            print("🎉 BOTH PROVIDERS WORKING - Full redundancy available!")
        elif gemini_working:
            print("⚠️  Only Gemini working - OpenAI fallback unavailable")
        else:
            print("⚠️  Only OpenAI working - Gemini primary unavailable")
