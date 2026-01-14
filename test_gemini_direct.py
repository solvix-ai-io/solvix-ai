#!/usr/bin/env python
"""Direct test of Gemini 3 Flash integration."""
import requests
import json

AI_ENGINE_URL = "http://localhost:8001"

def test_health():
    """Test health endpoint shows Gemini 3 Flash."""
    print("\n=== Test 1: Health Check ===")
    response = requests.get(f"{AI_ENGINE_URL}/health")
    health = response.json()

    print(f"Status: {health['status']}")
    print(f"Provider: {health['provider']}")
    print(f"Model: {health['model']}")
    print(f"Fallback Provider: {health.get('fallback_provider')}")
    print(f"Fallback Model: {health.get('fallback_model')}")

    assert health['provider'] == 'gemini', f"Expected gemini, got {health['provider']}"
    assert health['model'] == 'gemini-3-flash-preview', f"Expected gemini-3-flash-preview, got {health['model']}"

    print("✅ Health check passed - Gemini 3 Flash is configured correctly!")
    return True

def test_classify():
    """Test classification with Gemini 3 Flash."""
    print("\n=== Test 2: Email Classification with Gemini 3 Flash ===")

    payload = {
        "context": {
            "organization_id": 1,
            "tenant_id": 1,
            "party": {
                "party_id": "test-001",
                "customer_code": "TEST001",
                "name": "Test Company Ltd",
                "currency": "GBP"
            },
            "behavior": {},
            "obligations": [{
                "invoice_number": "INV-001",
                "original_amount": 1000.00,
                "amount_due": 1000.00,
                "due_date": "2025-12-30",
                "days_past_due": 15,
                "state": "open"
            }],
            "communication": {
                "touch_count": 1
            },
            "broken_promises_count": 0
        },
        "email": {
            "subject": "Payment next week",
            "body": "Hi, I will pay the invoice on Friday this week. Thanks!",
            "from_address": "accounts@testco.com",
            "from_name": "John Doe"
        }
    }

    response = requests.post(
        f"{AI_ENGINE_URL}/classify",
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        print(f"❌ Classification failed with status {response.status_code}")
        print(f"Response: {response.text}")
        return False

    result = response.json()

    print(f"Classification: {result.get('classification')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Reasoning: {result.get('reasoning')}")
    print(f"Tokens used: {result.get('tokens_used')}")

    # Verify LLM was actually called
    assert result.get('tokens_used', 0) > 0, "LLM was not invoked (tokens_used = 0)"

    print(f"✅ Classification passed - Gemini 3 Flash processed the request!")
    print(f"   Used {result.get('tokens_used')} tokens")

    return True

def test_generate_draft():
    """Test draft generation with Gemini 3 Flash."""
    print("\n=== Test 3: Draft Generation with Gemini 3 Flash ===")

    payload = {
        "context": {
            "organization_id": 1,
            "tenant_id": 1,
            "party": {
                "party_id": "test-002",
                "customer_code": "ACM001",
                "name": "ACME Corporation",
                "currency": "USD"
            },
            "behavior": {
                "segment": "MEDIUM_RISK",
                "on_time_rate": 0.75,
                "avg_days_to_pay": 35.5
            },
            "obligations": [
                {
                    "invoice_number": "INV-2025-001",
                    "original_amount": 5000.00,
                    "amount_due": 5000.00,
                    "due_date": "2025-11-15",
                    "days_past_due": 60,
                    "state": "open"
                },
                {
                    "invoice_number": "INV-2025-002",
                    "original_amount": 2500.00,
                    "amount_due": 2500.00,
                    "due_date": "2025-12-01",
                    "days_past_due": 44,
                    "state": "open"
                }
            ],
            "communication": {
                "touch_count": 2,
                "last_touch_at": "2025-12-20T10:00:00Z",
                "last_tone_used": "professional",
                "last_response_type": None,
                "case_state": "ACTIVE"
            },
            "broken_promises_count": 0
        },
        "preferences": {
            "tone": "professional",
            "objective": "Request payment within 7 days",
            "brand_tone": "professional",
            "custom_instructions": None
        }
    }

    response = requests.post(
        f"{AI_ENGINE_URL}/generate",
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        print(f"❌ Draft generation failed with status {response.status_code}")
        print(f"Response: {response.text}")
        return False

    result = response.json()

    print(f"Subject: {result.get('subject')}")
    print(f"Body preview: {result.get('body')[:200]}...")
    print(f"Tokens used: {result.get('tokens_used')}")

    # Verify LLM was actually called
    assert result.get('tokens_used', 0) > 0, "LLM was not invoked (tokens_used = 0)"

    print(f"✅ Draft generation passed - Gemini 3 Flash generated the email!")
    print(f"   Used {result.get('tokens_used')} tokens")

    return True

if __name__ == "__main__":
    print("="*60)
    print("Gemini 3 Flash Integration Test")
    print("="*60)

    try:
        # Test 1: Health check
        if not test_health():
            print("\n❌ Health check failed, stopping tests")
            exit(1)

        # Test 2: Classification
        if not test_classify():
            print("\n❌ Classification test failed")
            exit(1)

        # Test 3: Draft generation
        if not test_generate_draft():
            print("\n❌ Draft generation test failed")
            exit(1)

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED - Gemini 3 Flash is working!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
