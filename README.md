# UniLLM

Lightweight LLM client wrapper with provider normalization, caching, and observability. Routes requests through [LiteLLM](https://github.com/BerriAI/litellm) with a unified `endpoint` format (`model@provider`) and automatic provider-specific preprocessing.

This package is used as a dependency by higher-level frameworks like [Unity](https://github.com/unifyai/unity), and integrates with the [Unify](https://github.com/unifyai/unify) SDK for query logging.

## Installation

```bash
pip install unillm
```

Or add to your project's dependencies pointing to this repo.

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

Controlled via environment variables:

```bash
# Master switch (default: true)
export UNILLM_LOG=true

# Enable file logging (optional)
export UNILLM_LOG_DIR=/path/to/logs
```

When `UNILLM_LOG_DIR` is set, structured log files are written:
- During call: `{timestamp}_pending.txt` (request only)
- After completion: `{timestamp}_hit.txt` or `{timestamp}_miss.txt` (request + response)

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

LLM calls create OTel spans that can be correlated with parent spans (from Unity) and propagated to child services (Unify).

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
├── __init__.py           # Public API exports
├── cache_events.py       # Cache event capture system
├── logger.py             # Logging and OTel tracing
├── settings.py           # Configuration via env vars
├── helpers.py            # Utility functions
├── clients/              # LLM client implementations
│   ├── base.py          # Base client class
│   ├── uni_llm.py       # Unify and AsyncUnify clients
│   ├── provider_preprocessing.py
│   └── shared_session.py
├── endpoints/            # Provider-specific model mappings
│   ├── openai.py
│   ├── anthropic.py
│   └── ...
└── types/                # Type definitions
    └── prompt.py
```

## Local Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Setup

```bash
uv sync
```

### Running Tests

```bash
uv run pytest tests/ -v
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit` (Black, isort, autoflake). If a commit fails due to auto-formatting, re-run the commit.

```bash
pre-commit install
```
