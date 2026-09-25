# UniLLM: The LLM Abstraction Layer

UniLLM is a lightweight wrapper that normalizes LLM interactions across multiple providers (OpenAI, Anthropic, Vertex AI, Bedrock, Groq, Mistral, etc.) through a unified `model@provider` endpoint format.

## Core Features

- **Unified Endpoint Format**: `model@provider` (e.g., `openai/gpt-4o@openrouter`, `claude-sonnet-5@anthropic`)
- **Provider Preprocessing**: Automatic handling of provider quirks (message format normalization, parameter translation)
- **Response Caching**: Read/write/both modes with cache hit/miss tracking—critical for test determinism
- **Stateful Conversations**: Automatic history management
- **Tool Calling**: Unified interface across providers
- **Structured Outputs**: Pydantic model support via `response_format`
- **Observability**: File logging, OpenTelemetry tracing, cost computation

## Position in the System

Every LLM call in Unify flows through UniLLM's `AsyncUnify` client. The caching system is what makes Unify's tests fast and deterministic—cached responses replay in milliseconds rather than waiting for real LLM calls. UniLLM also handles trace propagation, enabling end-to-end observability from Unify through to the LLM provider.

## Testing Philosophy

UniLLM's caching is fundamental to Unify's test strategy. Tests use real LLM calls (never mocked), but responses are cached. This means:
- First run: Real LLM call, response cached
- Subsequent runs: Cached response replayed instantly
- Cache key = exact LLM input (prompts, tools, etc.)

## Related Repositories

- **unify-agent** (the `unify` package): the one consumer. Its async tool loops call UniLLM, and a local checkout links it as a sibling editable install (`../unillm`).
- A private evaluation harness runs unify-agent with `UNILLM_LLM_GATEWAY_URL` and `UNILLM_LLM_GATEWAY_KEY` set, so its OpenRouter calls go through the harness's tracking proxy. The gateway override is therefore in use, not leftover hosted-platform code: search the unifyai org's code for those variables before changing it.
- **orchestra**, **unisdk**, **unify-deploy** and **console** are archived along with the hosted platform they made up. UniLLM depends on none of them, and CI starts no server.

## Public Repository

`unifyai/unillm` is public, and anything committed stays in its history after it is deleted. Nothing here names a GCP project, a service account, an internal host or a customer. Code reads such values at runtime instead: `SecretManagerSource` takes the project from the service-account key's `project_id`. Docs use placeholders such as `<your-project-id>`.
