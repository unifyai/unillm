# Unillm Logging & Tracing

This document covers the logging infrastructure for unillm: LLM request/response traces, Unify SDK HTTP traces, and OpenTelemetry tracing.

---

## Log Directory Overview

All logs are organized under `logs/` with four main subdirectories:

| Directory | Purpose | Structure | Control |
|-----------|---------|-----------|---------|
| `logs/unillm/` | Raw LLM request/response traces | `.txt` files per request | `UNILLM_LOG_DIR` (+ `UNILLM_TERMINAL_LOG` for console) |
| `logs/unisdk/` | Unify SDK HTTP traces | JSON files per request | `UNISDK_LOG_DIR` (+ `UNISDK_TERMINAL_LOG` for console) |
| `logs/orchestra/` | Orchestra API traces (server-side) | Per-request JSON with spans | `ORCHESTRA_LOG_DIR` |
| `logs/all/` | Cross-repo OpenTelemetry traces | `{trace_id}.jsonl` per trace | `*_OTEL_LOG_DIR` |

**Note:** Orchestra logs are only populated when running a local Orchestra server. The test infrastructure sets `ORCHESTRA_LOG_DIR` so that if you start a local orchestra, its traces will be captured here.

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

## Unify SDK Logs (`logs/unisdk/`)

Unify SDK HTTP traces capture all requests to the Orchestra API. These are useful for debugging API issues, inspecting request/response payloads, and correlating with server-side traces.

### Directory Structure

```
logs/unisdk/
├── 14-26-27.611_POST_projects-contexts_210ms_200_no-trace.json
├── 14-26-46.175_GET_logs_331ms_200_f124f0d3.json
├── 14-27-01.234_POST_logs_PENDING_a1b2c3d4.json
└── ...
```

### Log File Naming

Files follow the format: `{timestamp}_{METHOD}_{route}_{duration}ms_{status}_{trace_id}.json`

| Component | Example | Description |
|-----------|---------|-------------|
| `timestamp` | `14-26-46.175` | Request start time (HH-MM-SS.mmm) |
| `METHOD` | `GET`, `POST` | HTTP method |
| `route` | `logs`, `projects-contexts` | API route (normalized) |
| `duration` | `331ms`, `PENDING` | Request duration (or PENDING while in-flight) |
| `status` | `200`, `404` | HTTP status code |
| `trace_id` | `f124f0d3` | Last 8 chars of OpenTelemetry trace ID (or `no-trace`) |

### Log File Contents

Each JSON file contains the full request and response:

```json
{
  "trace_id": "099b207f89222185695d25977be454fc",
  "request": {
    "method": "GET",
    "url": "https://api.unify.ai/v0/logs",
    "headers": {"Authorization": "Bearer ..."},
    "params": {"limit": 100}
  },
  "response": {
    "status_code": 200,
    "headers": {"Content-Type": "application/json"},
    "body": [...]
  },
  "duration_ms": 331
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UNISDK_TERMINAL_LOG` | `true` | Terminal (console) output for HTTP requests |
| `UNISDK_LOG_DIR` | `""` (disabled) | Directory for file-based request traces (independent of terminal) |

**Quiet terminal, verbose files:**
```bash
export UNISDK_TERMINAL_LOG=false
export UNISDK_LOG_DIR=/path/to/logs/unisdk
```

### Trace Correlation

The `trace_id` suffix in filenames (last 8 chars) enables correlation with:
- Unillm LLM traces (same trace context)
- Orchestra server-side traces (in `logs/orchestra/`)
- OpenTelemetry spans in `logs/all/`

---

## Orchestra Logs (`logs/orchestra/`)

Orchestra logs capture server-side API request traces using OpenTelemetry. These are only populated when running a local Orchestra server for development/testing.

### Directory Structure

```
logs/orchestra/
└── 2026-01-05T22-00-00_unillmpid12345/
    └── requests/
        ├── 2026-01-05T22-00-01.123_GET_projects_45ms_200_f124f0d3.json
        ├── 2026-01-05T22-00-02.456_POST_logs_120ms_201_a1b2c3d4.json
        └── ...
```

### Log File Naming

Each request generates a JSON file:

```
{datetime}_{METHOD}_{route}_{duration}ms_{status}_{trace_id_short}.json
```

| Component | Example | Description |
|-----------|---------|-------------|
| `datetime` | `2026-01-05T22-00-01.123` | Request start time (millisecond precision) |
| `METHOD` | `GET`, `POST`, `DELETE` | HTTP method |
| `route` | `projects`, `logs` | API route |
| `duration` | `45ms`, `PENDING` | Request duration (or `PENDING` while in-flight) |
| `status` | `200`, `404` | HTTP status code |
| `trace_id_short` | `f124f0d3` | Last 8 chars of OpenTelemetry trace ID |

### Log File Contents

Each JSON file contains the full request trace with all spans:

```json
{
  "trace_id": "099b207f89222185695d25977be454fc",
  "status": "complete",
  "spans": [
    {
      "name": "GET /v0/projects",
      "span_id": "a1b2c3d4e5f6a7b8",
      "parent_span_id": null,
      "start_time": "2026-01-05T22:00:01.123Z",
      "end_time": "2026-01-05T22:00:01.168Z",
      "duration_ms": 45,
      "attributes": {
        "http.method": "GET",
        "http.route": "/v0/projects",
        "http.status_code": 200
      }
    },
    {
      "name": "SELECT projects",
      "span_id": "...",
      "parent_span_id": "a1b2c3d4e5f6a7b8",
      "attributes": { "db.statement": "SELECT ..." }
    }
  ]
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRA_LOG_DIR` | `""` (disabled) | Directory for per-request trace files |
| `ORCHESTRA_OTEL_LOG_DIR` | `""` | Directory for OTEL span export (typically `logs/all/`) |

**Note:** These are set automatically by the test infrastructure. Orchestra must be started with these environment variables for logging to work.

### When Are Orchestra Logs Created?

Orchestra logs are only created when:
1. You're running a **local** Orchestra server (not production)
2. The server was started with `ORCHESTRA_LOG_DIR` set
3. Requests are made to the local server

For production API calls, you only get client-side traces in `logs/unisdk/`.

---

## OpenTelemetry Traces (`logs/all/`)

When OTEL tracing is enabled, both unillm and the Unify SDK create spans that can be correlated with parent spans (from Unity) and exported for distributed tracing analysis.

### Directory Structure

```
logs/all/
├── 099b207f89222185695d25977be454fc.jsonl   # All spans for trace 099b207f...
├── a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6.jsonl   # All spans for trace a1b2c3d4...
└── ...
```

Files are keyed by the 32-character trace ID. When running as part of a larger system (e.g., Unity), spans from all services are aggregated into the same file.

### Trace File Format (JSONL)

Each `.jsonl` file contains one JSON object per line, representing a span:

```json
{"service": "unillm", "trace_id": "099b207f...", "span_id": "a1b2c3d4", "parent_span_id": null, "name": "LLM gpt-4@openai", "start_time": "2026-01-01T14:30:22.123Z", "end_time": "2026-01-01T14:30:25.456Z", "duration_ms": 3333, "status": "OK", "attributes": {"llm.endpoint": "gpt-4@openai", "llm.model": "gpt-4", "llm.cache_status": "miss"}}
{"service": "unify", "trace_id": "099b207f...", "span_id": "e5f6g7h8", "parent_span_id": "a1b2c3d4", "name": "POST /v0/logs", "start_time": "2026-01-01T14:30:22.500Z", "end_time": "2026-01-01T14:30:23.100Z", "duration_ms": 600, "status": "OK", "attributes": {"http.method": "POST", "http.status_code": 200}}
{"service": "orchestra", "trace_id": "099b207f...", "span_id": "i9j0k1l2", "parent_span_id": "e5f6g7h8", "name": "POST /v0/logs", "start_time": "2026-01-01T14:30:22.550Z", "end_time": "2026-01-01T14:30:23.050Z", "duration_ms": 500, "status": "OK", "attributes": {"http.method": "POST", "http.route": "/v0/logs"}}
```

### Span Attributes

**Unillm spans** (LLM calls):

| Attribute | Description |
|-----------|-------------|
| `llm.endpoint` | The endpoint string (e.g., `gpt-4@openai`) |
| `llm.model` | Model name |
| `llm.cache_status` | `hit` or `miss` |
| `llm.usage.prompt_tokens` | Input token count |
| `llm.usage.completion_tokens` | Output token count |
| `llm.usage.total_tokens` | Total token count |
| `llm.response_model` | Model from response (may differ from request) |

**Unify spans** (HTTP requests to Orchestra):

| Attribute | Description |
|-----------|-------------|
| `http.method` | HTTP method (GET, POST, etc.) |
| `http.url` | Full request URL |
| `http.status_code` | Response status code |
| `http.request.body` | Request body (JSON) |
| `http.response.body` | Response body (JSON) |

**Orchestra spans** (server-side, when running locally):

| Attribute | Description |
|-----------|-------------|
| `http.method` | HTTP method |
| `http.route` | API route pattern |
| `http.status_code` | Response status code |
| `db.statement` | SQL query (for database spans) |
| `db.operation` | Database operation type |

### Environment Variables

**Unillm OTEL settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `UNILLM_OTEL` | `false` | Master switch for unillm OTel tracing |
| `UNILLM_OTEL_ENDPOINT` | `""` | OTLP endpoint for remote export (e.g., Tempo, Jaeger) |
| `UNILLM_OTEL_LOG_DIR` | `""` | Directory for file-based span export |

**Unify SDK OTEL settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `UNISDK_OTEL` | `false` | Master switch for Unify SDK OTel tracing |
| `UNISDK_OTEL_ENDPOINT` | `""` | OTLP endpoint for remote export |
| `UNISDK_OTEL_LOG_DIR` | `""` | Directory for file-based span export |

**Orchestra OTEL settings** (server-side):

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRA_OTEL_LOG_DIR` | `""` | Directory for file-based span export |

**Enabling file-based tracing (all services):**
```bash
# Enable OTEL for all services, writing to same directory for correlation
export UNILLM_OTEL=true
export UNILLM_OTEL_LOG_DIR=/path/to/logs/all
export UNISDK_OTEL=true
export UNISDK_OTEL_LOG_DIR=/path/to/logs/all
export ORCHESTRA_OTEL_LOG_DIR=/path/to/logs/all  # Server-side
```

### Parent TracerProvider Integration

When unillm runs within a larger system (e.g., Unity), it automatically detects and uses the parent's TracerProvider. This ensures all spans share the same trace context for end-to-end correlation.

The integration flow:
1. Parent (Unity) creates a TracerProvider and root span
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
