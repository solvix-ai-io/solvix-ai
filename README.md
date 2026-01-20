# Solvix AI Engine

Stateless AI service for the Solvix debt collection platform. Provides email classification, response draft generation, and gate evaluation for automated collections workflows.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://github.com/astral-sh/uv)

> **📚 Documentation Hub:** For comprehensive platform documentation, see the [Solvix docs](../Solvix/docs/) directory, including [Codebase Analysis](../Solvix/docs/CODEBASE_ANALYSIS.md), [Cross-Repo Integration](../Solvix/docs/architecture/CROSS_REPO_INTEGRATION.md), and [Development Guide](../Solvix/docs/DEVELOPMENT_GUIDE.md).

---

## Features

- **Email Classification**: Classify inbound customer emails into categories (HARDSHIP, DISPUTE, PROMISE_TO_PAY, etc.)
- **Draft Generation**: Generate contextual response drafts with appropriate tone
- **Gate Evaluation**: Evaluate compliance gates before outbound actions (touch cap, cooling off, etc.)
- **Dual LLM Support**: Primary Gemini, fallback to OpenAI

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Solvix Backend │────▶│  Solvix AI Engine │────▶│   Gemini    │
│   (Django)      │◀────│   (FastAPI)       │◀────│  (Primary)  │
│                 │     │   Port 8001       │     │   OpenAI    │
└─────────────────┘     └──────────────────┘     │  (Fallback) │
                                                  └─────────────┘
```

The AI Engine is stateless - it receives all context via HTTP requests and does not access the database directly.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classify` | POST | Classify inbound email |
| `/generate-draft` | POST | Generate response draft |
| `/evaluate-gates` | POST | Evaluate compliance gates |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Google API key (Gemini) or OpenAI API key

### Local Development (uv - Recommended)

```bash
# Clone and setup
cd solvix-ai

# Install dependencies with uv
make install

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY or OPENAI_API_KEY

# Run the server with auto-reload
make dev
# API: http://localhost:8001
# Health: http://localhost:8001/health
```

### Docker

```bash
# Build and run
make docker-build
make docker-run

# Or with docker-compose
make docker-up
make docker-logs   # View logs
make docker-down   # Stop
```

---

## Makefile Commands

```bash
# Setup
make install          # Install dependencies (uv)
make pre-commit-install  # Install pre-commit hooks

# Development
make run              # Run server
make dev              # Run with auto-reload

# Testing
make test             # Run unit tests (mocked, no API calls)
make test-cov         # Run with coverage report
make test-live        # Run live integration tests (requires API key)

# Code Quality
make lint             # Run ruff linter
make format           # Format code
make clean            # Remove cache files

# Docker
make docker-build     # Build image
make docker-run       # Run container
make docker-up        # Start with docker-compose
make docker-down      # Stop docker-compose
```

---

## Configuration

Environment variables (in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Primary LLM |
| `OPENAI_API_KEY` | OpenAI API key | Fallback LLM |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4o` |
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

## Case Context

The `CaseContext` model (in `src/api/models/requests.py`) provides context for AI decisions:

| Field | Type | Description |
|-------|------|-------------|
| `party_id` | `str` | Debtor party identifier |
| `obligations` | `list` | Outstanding obligations |
| `promise_grace_days` | `int` | Days before promise is broken (default: 3) |
| ... | ... | Other context fields |

The `promise_grace_days` value is resolved by the backend:
1. Party-level override (if set)
2. Organization default
3. System default (3 days)

## Testing

### Unit tests (no OpenAI calls)

The default unit tests **mock** the LLM layer (e.g. patching `llm_client.complete`) so they are fast, deterministic, and do **not** call OpenAI.

```bash
# If your venv is activated and pytest is on PATH
pytest tests/

# Or run via the repo venv directly
./venv/bin/pytest tests/
```

### Test Suite

| Test File | Coverage |
| --------- | -------- |
| `test_api.py` | API endpoint routing and response formats |
| `test_classifier.py` | Email classification with all 13 categories |
| `test_generator.py` | Draft generation with 5 tone types |
| `test_gate_evaluator.py` | 6 gate evaluations (touch_cap, cooling_off, etc.) |
| `test_live_integration.py` | Real OpenAI integration (requires API key) |

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

### Docker Connectivity

The Solvix backend runs inside Docker and needs to connect to the AI Engine. Since the AI Engine runs on the host machine (not in Docker), use:

**macOS / Windows (Docker Desktop):**

```bash
# In Solvix/.env
AI_ENGINE_URL=http://host.docker.internal:8001
```

**Linux:**

```bash
# In Solvix/.env
AI_ENGINE_URL=http://172.17.0.1:8001
# Or use the host's actual IP address
```

The special hostname `host.docker.internal` resolves to the host machine from within Docker containers.

### Running with Solvix Backend

1. **Start the AI Engine** (runs on host):

   ```bash
   cd solvix-ai
   source venv/bin/activate
   uvicorn src.main:app --reload --port 8001
   ```

2. **Start Solvix Backend** (runs in Docker):

   ```bash
   cd Solvix
   make dev-backend
   ```

3. **Verify connectivity**:

   ```bash
   # From inside the Docker container
   docker exec solvix_web curl http://host.docker.internal:8001/health
   ```

## License

Proprietary - Solvix
