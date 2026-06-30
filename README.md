# UniLLM

Lightweight LLM access layer with provider normalization, caching, and observability. It uses a unified `endpoint` format (`model@provider`) so application code can switch providers without rewriting call sites, while still allowing developers to choose the model provider or compatible endpoint they want to use.

## What layer is this?

UniLLM is the model-access layer in the wider Unify stack:

```
         User (Console/Phone/SMS/Email)
                      │
    ┌─────────────────┴──────────────────┐
    │           Communication            │
    │    (Webhooks, Voice, SMS, Email)   │
    └────┬───────────────────────────────┘
         │
    ┌────┴────┐    ┌─────────┐    ┌─────────┐
    │  Unify  │    │  Unify  │    │Orchestra│
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

**This repo (UniLLM)** handles LLM inference for Unify. It normalizes requests across providers (OpenAI, Anthropic, Vertex AI, etc.), provides response caching for test determinism, and can integrate with Unify for logging and billing context.

If you're here from the Unify quickstart, this is the layer that talks to model providers. OpenAI and Anthropic are the simplest documented paths, but the point of UniLLM is that the provider choice is yours, including other supported providers and compatible local endpoints.

Related repositories:
- [Unify](https://github.com/unifyai/unify) — AI assistant brain (primary consumer)
- [unisdk](https://github.com/unifyai/unisdk) — Python SDK for logging and persistence
- [Orchestra](https://github.com/unifyai/orchestra) — Backend API and database

## Installation

Install from Git:

```bash
pip install "unillm @ git+https://github.com/unifyai/unillm.git"
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add "unillm @ git+https://github.com/unifyai/unillm.git"
```

The import name is `unillm`:

```python
import unillm
```

## Configuration

### API Keys

Set credentials for whichever providers you want to use:

```bash
export OPENAI_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
# ... other provider keys
```

You do not need a Unify account for basic inference. `UNIFY_KEY` only matters for optional logging, credit, and observability features that integrate with the wider Unify stack.

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

> `unillm` is the LLM access layer, not a runnable system. To run the whole
> product locally (Orchestra + Unify + Console), use **`unify stack up`** from
> the [unify repo](https://github.com/unifyai/unify). The steps below are for
> developing this library itself.

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. The persistence/logging dependency (`unisdk`) is resolved from a sibling checkout via `[tool.uv.sources]`, so clone it alongside this repo before syncing.

### Setup

```bash
git clone https://github.com/unifyai/unisdk.git
git clone https://github.com/unifyai/unillm.git
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

MIT — see [LICENSE](LICENSE) for details.
