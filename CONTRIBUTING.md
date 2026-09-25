# Contributing to UniLLM

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- At least one model-provider key for the tests you plan to run

## Setup

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/unifyai/unillm.git
cd unillm
uv sync
```

2. Add provider credentials in your shell or `.env` file:

```bash
export OPENROUTER_API_KEY=...
# or
export ANTHROPIC_API_KEY=...
```

## Running tests

Run the local test suite with:

```bash
uv run pytest tests/ -v
```

With a populated `.cache.ndjson`, cached responses replay quickly and
deterministically.

### LLM cache and CI

CI replays the shared LLM cache in **read-only** mode. A cache miss fails the
test instead of calling provider APIs. The only workflows that **write** cache
entries are `llm-cache-refresh.yml` (paid refresh or seed publish) and local
runs with `UNILLM_CACHE=true`.

CI hydrates `.cache.ndjson` from the latest successful `llm-cache-refresh.yml`
artifact on the branch under test, falling back to `main`: the GitHub Actions
cache alone drops entries nobody has read for seven days.

When a change invalidates cache keys (e.g. response-format or caching logic),
refresh before merging:

1. **Local seed publish (typical)** — run tests locally with cache write enabled,
   consolidate into `.github/cache-seed/cache.ndjson`, commit to `main`, then
   dispatch `llm-cache-refresh.yml` with `publish_seed=PUBLISH_SEED_OK`.
2. **CI refresh** — dispatch `llm-cache-refresh.yml` with
   `confirm_llm_spend=LLM_SPEND_OK` and the relevant `test_path`.

After publish completes, re-run CI if it started before the artifact was
ready. See `.agents/rules/llm-cache-invalidation.md` for the full
step-by-step playbook.

## Code style

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Run the default checks manually:

```bash
uv run pre-commit run --all-files
```

## Pull requests

- Open PRs against the `main` branch.
- Keep changes focused and easy to review.
- Run the relevant tests for the area you changed.

## Questions

Open an issue with reproduction steps, the model/provider you used, and the
behavior you expected to see.
