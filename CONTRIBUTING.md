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
export OPENAI_API_KEY=...
# or
export ANTHROPIC_API_KEY=...
```

If you are iterating on a sibling checkout of `unify` as well, override the
published dependency with an editable install:

```bash
uv pip install -e ../unify
```

## Running tests

Run the local test suite with:

```bash
uv run pytest tests/ -v
```

With a populated `.cache.ndjson`, cached responses replay quickly and
deterministically. Some CI paths that rely on managed infrastructure are
maintainer-controlled and skipped on external forks.

### LLM cache and CI

CI replays the shared LLM cache in **read-only** mode. A cache miss fails the
test instead of calling provider APIs. The only workflows that **write** cache
entries are `llm-cache-refresh.yml` (paid refresh or seed publish) and local
runs with `UNILLM_CACHE=true`.

**`staging → main` promotion PRs** hydrate `.cache.ndjson` from the latest
successful `llm-cache-refresh.yml` artifact on `staging` (the GitHub Actions
cache alone is branch-scoped and insufficient for promotion pytest).

When a change invalidates cache keys (e.g. response-format or caching logic),
refresh before merging:

1. **Local seed publish (typical)** — run tests locally with cache write enabled,
   consolidate into `.github/cache-seed/cache.ndjson`, commit to `staging`, then
   dispatch `llm-cache-refresh.yml` with `publish_seed=PUBLISH_SEED_OK`.
2. **CI refresh** — dispatch `llm-cache-refresh.yml` with
   `confirm_llm_spend=LLM_SPEND_OK` and the relevant `test_path`.

After publish completes, re-run promotion PR CI if it started before the artifact
was ready. See `.agents/rules/llm-cache-invalidation.md` for the full
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

- Open PRs against the `staging` branch.
- Keep changes focused and easy to review.
- Run the relevant tests for the area you changed.

## Questions

Open an issue with reproduction steps, the model/provider you used, and the
behavior you expected to see.
