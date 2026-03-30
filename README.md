# UniLLM

Lightweight LLM client wrapper with provider normalization, caching, and observability. Routes requests through [LiteLLM](https://github.com/BerriAI/litellm) with a unified `endpoint` format (`model@provider`) and automatic provider-specific preprocessing.

## System Architecture

UniLLM is the LLM abstraction layer in a multi-repository system:

```
         User (Console/Phone/SMS/Email)
                      │
    ┌─────────────────┴──────────────────┐
    │           Communication            │
    │    (Webhooks, Voice, SMS, Email)   │
    └────┬───────────────────────────────┘
         │
    ┌────┴────┐    ┌─────────┐    ┌─────────┐
    │  Unity  │    │  Unify  │    │Orchestra│
    │ (Brain) │───▶│  (SDK)  │───▶│  (API)  │
    │         │    │         │    │  (DB)   │
    └────┬────┘    └────┬────┘    └────┬────┘
         │              ▲              ▲
         │              │              │
         │    ┌─────────┴─┐       ┌────┴───────┐
         └───▶│  UniLLM   │       │  Console   │
              │ (LLM API) │       │(Interfaces)│
              └───────────┘       └────────────┘
```

**This repo (UniLLM)** handles all LLM inference for Unity. It normalizes requests across providers (OpenAI, Anthropic, Vertex AI, etc.), provides response caching for test determinism, and integrates with Unify for query logging.

Related repositories:
- [Unity](https://github.com/unifyai/unity) — AI assistant brain (primary consumer)
- [Unify](https://github.com/unifyai/unify) — Python SDK for logging and persistence
- [Orchestra](https://github.com/unifyai/orchestra) — Backend API and database

## Installation

```bash
pip install git+https://github.com/unifyai/unillm.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add unillm --git https://github.com/unifyai/unillm.git
```

## Configuration

### API Keys

Set API keys for the providers you want to use:

```bash
export OPENAI_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
# ... other provider keys
```

### Google Cloud / Vertex AI

For Vertex AI models (Gemini, Claude on Vertex, etc.), authenticate using Google Cloud Application Default Credentials:

```bash
# One-time setup: authenticate with your Google Cloud account
gcloud auth application-default login

# Set your GCP project and location
export VERTEXAI_PROJECT=<your-project-id>
export VERTEXAI_LOCATION=<your-location>  # e.g., us-central1, europe-west1
```

Alternatively, use a service account JSON file:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## Basic Usage

```python
import unillm

# Sync client
client = unillm.Unify("gpt-4o@openai")
response = client.generate(
    messages=[{"role": "user", "content": "Hello!"}]
)

# Async client
async_client = unillm.AsyncUnify("claude-sonnet-4-20250514@anthropic")
response = await async_client.generate(
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Features

### Unified Endpoint Format

All models use a consistent `model@provider` format:

```python
client = unillm.Unify("gpt-4o@openai")
client = unillm.Unify("claude-sonnet-4-20250514@anthropic")
client = unillm.Unify("gemini-2.0-flash@vertexai")
```

### Provider-Specific Preprocessing

Automatic handling of provider quirks (message format normalization, parameter translation, etc.) before requests are sent.

### Response Caching

Built-in caching to avoid redundant LLM calls:

```python
client = unillm.Unify("gpt-4o@openai", cache=True)

# Cache modes
client.generate(..., cache="read")       # Read from cache only if available
client.generate(..., cache="write")      # Write to cache only
client.generate(..., cache="both")       # Read and write
client.generate(..., cache="read-only")  # Must be in cache, else error
```

### Cache Event Capture

Track cache hit/miss status for observability:

```python
from unillm import capture_cache_events

with capture_cache_events() as events:
    client.generate(messages=[...])

print(events[0]["cache_status"])  # "hit" or "miss"
```

### Streaming

```python
client = unillm.Unify("gpt-4o@openai", stream=True)
for chunk in client.generate(messages=[...]):
    print(chunk, end="")
```

### Stateful Conversations

```python
client = unillm.Unify("gpt-4o@openai", stateful=True)
client.generate(user_message="What is 2+2?")
client.generate(user_message="And what is that times 3?")  # Maintains history
```

### Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {...}}
    }
}]
response = client.generate(messages=[...], tools=tools)
```

### Response Formats

```python
from pydantic import BaseModel

class Answer(BaseModel):
    value: int
    explanation: str

response = client.generate(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    response_format=Answer
)
```

## Observability

### Console & File Logging

Terminal and file logging are independently controlled:

```bash
# Terminal (console) output (default: true)
export UNILLM_TERMINAL_LOG=true

# File-based traces (independent of terminal)
export UNILLM_LOG_DIR=/path/to/logs
```

When `UNILLM_LOG_DIR` is set, structured log files are written:
- During call (cache enabled): `{base}.cache_pending.txt`
- During call (cache disabled): `{base}.pending.txt`
- After completion: `{base}.cache_hit.txt` or `{base}.cache_miss.txt` (cache enabled), or `{base}.txt` (cache disabled)

Pending files remain as evidence if an LLM call hangs or crashes.

### OpenTelemetry Tracing

```bash
# Enable OTel tracing
export UNILLM_OTEL=true

# OTLP endpoint (optional)
export UNILLM_OTEL_ENDPOINT=http://localhost:4317

# File-based span export (optional)
export UNILLM_OTEL_LOG_DIR=/path/to/traces
```

LLM calls create OTel spans that can be correlated with parent application spans and propagated to child services.

### Privacy Note

When `UNILLM_LOG_DIR` is set, full request and response payloads are written to disk. This includes user messages, tool arguments, and model responses — which may contain PII or sensitive data. Be mindful of this when enabling file logging in production environments.

## Supported Providers

- OpenAI
- Anthropic
- Vertex AI (Google)
- Bedrock (AWS)
- DeepSeek
- Groq
- Mistral
- Replicate
- Together AI
- xAI

## Project Structure

```
unillm/
├── __init__.py              # Public API exports
├── settings.py              # Configuration via env vars
├── helpers.py               # Utility functions
├── costs.py                 # Provider cost computation
├── cost_tracker.py          # Per-call cost event capture
├── tokens.py                # Token counting and context window utilities
├── cache_events.py          # Cache hit/miss event capture
├── llm_events.py            # LLM event hooks for observability
├── limit_hooks.py           # Spending limit check callbacks
├── logger.py                # File logging and OTel tracing
├── clients/                 # LLM client implementations
│   ├── base.py              # Base client class
│   ├── uni_llm.py           # Unify (sync) and AsyncUnify clients
│   ├── provider_preprocessing.py   # Provider-specific request normalization
│   ├── provider_postprocessing.py  # Response validation and retries
│   └── shared_session.py    # Shared aiohttp session management
├── caching/                 # Response caching system
│   ├── base_cache.py        # Abstract cache backend
│   ├── local_cache.py       # File-based NDJSON cache
│   └── local_separate_cache.py  # Split read/write cache (for CI)
├── endpoints/               # Provider-specific model mappings
│   ├── openai.py
│   ├── anthropic.py
│   └── ...
└── types/                   # Type definitions
    ├── cache.py
    ├── prompt.py
    └── prompt_caching.py
```

## Local Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. UniLLM depends on [unify](https://github.com/unifyai/unify) via a local path (`../unify`), so both repos must be cloned as siblings:

```
parent/
├── unillm/   # this repo
└── unify/    # https://github.com/unifyai/unify
```

### Setup

```bash
git clone https://github.com/unifyai/unillm.git
git clone https://github.com/unifyai/unify.git
cd unillm
uv sync
```

### Running Tests

Tests require at minimum an `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` set in your environment (or a `.env` file). With a populated `.cache.ndjson`, cached LLM responses are replayed — so tests run fast and deterministically without making real LLM calls.

```bash
uv run pytest tests/ -v
```

`UNIFY_KEY` is optional. If set, credit deduction runs against the [Unify API](https://unify.ai). If unset, credit deduction silently warns and tests continue normally.

### Running Tests in CI

**Tests are opt-in to reduce GitHub Actions costs.** Tests only run when explicitly requested:

- **Commit message**: Include `[run-tests]` in your commit message
- **PR title**: Include `[run-tests]` in your pull request title
- **Manual trigger**: Use the "Run workflow" button in GitHub Actions

Examples:
```bash
# Run tests on this commit
git commit -m "Fix caching logic [run-tests]"

# No tests (default)
git commit -m "Update README"
```

Note: The `black` formatting check always runs on every push.

Some CI steps (local Orchestra deployment, GCP authentication) are internal infrastructure for the Unify team and are automatically skipped on external forks.

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit` (Black, isort, autoflake). If a commit fails due to auto-formatting, re-run the commit.

```bash
pre-commit install
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
