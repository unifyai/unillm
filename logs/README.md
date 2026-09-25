# Unillm Logging & Tracing

This document covers the logging infrastructure for unillm: LLM request/response traces and OpenTelemetry tracing.

---

## Log Directory Overview

All logs are organized under `logs/` in two subdirectories:

| Directory | Purpose | Structure | Control |
|-----------|---------|-----------|---------|
| `logs/unillm/` | Raw LLM request/response traces | `.txt` files per request | `UNILLM_LOG_DIR` (+ `UNILLM_TERMINAL_LOG` for console) |
| `logs/all/` | OpenTelemetry traces | `{trace_id}.jsonl` per trace | `UNILLM_OTEL_LOG_DIR` |

---

## Unillm Logs (`logs/unillm/`)

LLM request/response traces capture the raw I/O for each LLM call. These are invaluable for debugging prompt issues, inspecting actual payloads, and understanding cache behavior.

### Directory Structure

```
logs/unillm/
├── 142536_123456789.cache_hit.txt      # Cache hit - response from cache
├── 142537_987654321.cache_miss.txt     # Cache miss - fresh LLM call
├── 142538_111222333.cache_pending.txt  # In-progress (cache enabled, or crashed)
├── 142539_444555666.pending.txt        # In-progress (cache disabled, or crashed)
├── 142540_777888999.txt                # Completed (cache disabled)
└── ...
```

### Log File Naming

Files use compound extensions to encode cache status: `{HHMMSS}_{nanoseconds}[_{origin}].{ext}`

| Extension | Meaning |
|-----------|---------|
| `.cache_pending.txt` | Request started (cache enabled), waiting for response |
| `.pending.txt` | Request started (cache disabled), waiting for response |
| `.cache_hit.txt` | Response served from cache |
| `.cache_miss.txt` | Fresh LLM call completed |
| `.txt` | Completed with caching disabled |

### Log File Contents

Each file contains the full request and response payloads:

```
🔄 LLM request ➡️
{
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7
}

🔄 LLM response ⬅️ [cache: miss]
{
    "id": "chatcmpl-...",
    "model": "gpt-4-0613",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            }
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30
    }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UNILLM_TERMINAL_LOG` | `true` | Terminal (console) output for LLM I/O |
| `UNILLM_LOG_DIR` | `""` (disabled) | Directory for file-based traces (independent of terminal) |

**Quiet terminal, verbose files (typical production):**
```bash
export UNILLM_TERMINAL_LOG=false
export UNILLM_LOG_DIR=/path/to/logs/unillm
```

### Debugging Hung Requests

If an LLM call hangs or crashes, the `.pending.txt` or `.cache_pending.txt` file remains as evidence of the incomplete request. This is useful for:
- Identifying which requests are timing out
- Debugging network issues
- Spotting malformed requests that cause provider errors

---

## OpenTelemetry Traces (`logs/all/`)

When OTel tracing is enabled, unillm creates a span for each LLM call, parented to the host application's current span when there is one, and exports it for distributed tracing analysis.

### Directory Structure

```
logs/all/
├── 099b207f89222185695d25977be454fc.jsonl   # All spans for trace 099b207f...
├── a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6.jsonl   # All spans for trace a1b2c3d4...
└── ...
```

Files are keyed by the 32-character trace ID. When a host application exports its own spans to the same directory, they land in the same file as unillm's.

### Trace File Format (JSONL)

Each `.jsonl` file contains one JSON object per line, representing a span:

```json
{"service": "unillm", "trace_id": "099b207f...", "span_id": "a1b2c3d4", "parent_span_id": null, "name": "LLM openai/gpt-4@openrouter", "start_time": "2026-01-01T14:30:22.123Z", "end_time": "2026-01-01T14:30:25.456Z", "duration_ms": 3333, "status": "OK", "attributes": {"llm.endpoint": "openai/gpt-4@openrouter", "llm.model": "openrouter/openai/gpt-4", "llm.cache_status": "miss"}}
```

### Span Attributes

**Unillm spans** (LLM calls):

| Attribute | Description |
|-----------|-------------|
| `llm.endpoint` | The endpoint string (e.g., `openai/gpt-4@openrouter`) |
| `llm.model` | Model name |
| `llm.cache_status` | `hit` or `miss` |
| `llm.usage.prompt_tokens` | Input token count |
| `llm.usage.completion_tokens` | Output token count |
| `llm.usage.total_tokens` | Total token count |
| `llm.response_model` | Model from response (may differ from request) |

### Environment Variables

**Unillm OTEL settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `UNILLM_OTEL` | `false` | Master switch for unillm OTel tracing |
| `UNILLM_OTEL_ENDPOINT` | `""` | OTLP endpoint for remote export (e.g., Tempo, Jaeger) |
| `UNILLM_OTEL_LOG_DIR` | `""` | Directory for file-based span export |

**Enabling file-based tracing:**
```bash
export UNILLM_OTEL=true
export UNILLM_OTEL_LOG_DIR=/path/to/logs/all
```

### Parent TracerProvider Integration

When unillm runs inside a host application that has its own TracerProvider, it automatically detects and uses it. This ensures all spans share the same trace context for end-to-end correlation.

The integration flow:
1. The host creates a TracerProvider and root span
2. Unillm detects the existing provider and creates child spans
3. All spans are exported to the same destination (file or collector)

---

## Reading Trace Files

```bash
# View all spans for a trace (pretty-printed)
cat logs/all/099b207f89222185695d25977be454fc.jsonl | jq -s .

# Find slow LLM calls (>5s)
cat logs/all/*.jsonl | jq -s '[.[] | select(.duration_ms > 5000)]'

# Filter by cache status
cat logs/all/*.jsonl | jq -s '[.[] | select(.attributes["llm.cache_status"] == "miss")]'

# Sum tokens across all calls in a trace
cat logs/all/099b207f...jsonl | jq -s '[.[].attributes["llm.usage.total_tokens"] // 0] | add'
```

---

## Programmatic Configuration

Both logging systems can be configured at runtime:

```python
from unillm import configure_log_dir

# Enable file logging
configure_log_dir("/path/to/logs/unillm")

# Or via environment
import os
os.environ["UNILLM_LOG_DIR"] = "/path/to/logs/unillm"
os.environ["UNILLM_OTEL"] = "true"
os.environ["UNILLM_OTEL_LOG_DIR"] = "/path/to/logs/all"
```

---

## Console Logging

When `UNILLM_TERMINAL_LOG=true` (the default), request/response payloads are logged to the console via Python's logging system. Console output is truncated for readability (500 chars max).

To see console logs in pytest:
```bash
pytest -s tests/  # -s disables output capture
```

The logger name is `unillm`, so you can configure it via standard Python logging:

```python
import logging
logging.getLogger("unillm").setLevel(logging.DEBUG)
```
