# Solvix AI Engine

Stateless AI service for the Solvix debt collection platform. Provides email classification, response draft generation, and gate evaluation for automated collections workflows.

## Features

- **Email Classification**: Classify inbound customer emails into categories (HARDSHIP, DISPUTE, PROMISE_TO_PAY, etc.)
- **Draft Generation**: Generate contextual response drafts with appropriate tone
- **Gate Evaluation**: Evaluate compliance gates before outbound actions (touch cap, cooling off, etc.)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Solvix Backend │────▶│  Solvix AI Engine │────▶│   OpenAI    │
│   (Django)      │◀────│   (FastAPI)       │◀────│  Model via  │
│                 │     │                  │     │  OPENAI_MODEL│
└─────────────────┘     └──────────────────┘     └─────────────┘
```

The AI Engine is stateless - it receives all context via HTTP requests and does not access the database directly.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classify` | POST | Classify inbound email |
| `/generate-draft` | POST | Generate response draft |
| `/evaluate-gates` | POST | Evaluate compliance gates |

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Local Development

```bash
# Clone and setup
cd solvix-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the server
uvicorn src.main:app --reload --port 8001
```

### Docker

```bash
# Build and run
docker-compose up --build

# Or just build
docker build -t solvix-ai .
docker run -p 8001:8001 --env-file .env solvix-ai
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | Model to use | `gpt-4o` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Classification Categories

| Category | Description |
|----------|-------------|
| `INSOLVENCY` | Bankruptcy, administration, liquidation |
| `DISPUTE` | Invoice dispute, goods/services issue |
| `ALREADY_PAID` | Claims payment already made |
| `UNSUBSCRIBE` | Requests to stop contact |
| `HOSTILE` | Aggressive or threatening language |
| `PROMISE_TO_PAY` | Commits to specific payment |
| `HARDSHIP` | Financial difficulty |
| `PLAN_REQUEST` | Requests payment plan |
| `REDIRECT` | Directs to another person |
| `REQUEST_INFO` | Asks for more information |
| `OUT_OF_OFFICE` | Auto-reply |
| `COOPERATIVE` | Positive engagement |
| `UNCLEAR` | Cannot determine intent |

## Draft Tones

- `friendly_reminder` - Light, first touch
- `professional` - Standard business tone
- `firm` - Escalated, more serious
- `final_notice` - Last warning before action
- `concerned_inquiry` - Empathetic, for hardship

## Gate Types

| Gate | Description |
|------|-------------|
| `touch_cap` | Maximum contacts per period |
| `cooling_off` | Minimum days between touches |
| `dispute_active` | Block if dispute pending |
| `hardship` | Special handling required |
| `unsubscribe` | Contact opted out |
| `escalation_appropriate` | Valid escalation path |

## Testing

### Unit tests (no OpenAI calls)

The default unit tests **mock** the LLM layer (e.g. patching `llm_client.complete`) so they are fast, deterministic, and do **not** call OpenAI.

```bash
# If your venv is activated and pytest is on PATH
pytest tests/

# Or run via the repo venv directly
./venv/bin/pytest tests/
```

### Live integration tests (real OpenAI calls)

`tests/test_live_integration.py` makes **real network calls** to OpenAI via `src/llm/client.py`. These tests:
- require `OPENAI_API_KEY`
- may incur cost and take longer
- validate that we are not just returning dummy responses

```bash
# Run only the live tests
./venv/bin/pytest tests/test_live_integration.py

# To see evidence of real HTTP requests (debug logs)
./venv/bin/pytest tests/test_live_integration.py -s --log-cli-level=DEBUG
```

## Project Structure

```
solvix-ai/
├── src/
│   ├── api/
│   │   ├── models/          # Pydantic request/response models
│   │   │   ├── requests.py
│   │   │   └── responses.py
│   │   └── routes/          # FastAPI route handlers
│   │       ├── classify.py
│   │       ├── generate.py
│   │       ├── gates.py
│   │       └── health.py
│   ├── config/
│   │   └── settings.py      # Pydantic settings
│   ├── engine/              # Core AI logic
│   │   ├── classifier.py
│   │   ├── generator.py
│   │   └── gate_evaluator.py
│   ├── llm/
│   │   ├── client.py        # OpenAI client wrapper
│   │   └── prompts.py       # System/user prompts
│   └── main.py              # FastAPI app entrypoint
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_api.py
│   ├── test_classifier.py
│   ├── test_generator.py
│   └── test_gate_evaluator.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Integration with Solvix Backend

The Django backend integrates via `services/ai_engine.py`:

```python
from services.ai_engine import AIEngineClient

async with AIEngineClient() as client:
    # Classify email
    result = await client.classify_email(email_content, context)
    
    # Generate draft
    draft = await client.generate_draft(context, classification, tone)
    
    # Check gates
    gates = await client.evaluate_gates(context, action)
```

## License

Proprietary - Solvix
