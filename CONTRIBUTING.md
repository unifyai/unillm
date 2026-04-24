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
