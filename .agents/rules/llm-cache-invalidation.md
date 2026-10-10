---
description: Refresh or publish the LLM response cache when cache keys change or CI pytest fails on cache misses
---

# LLM Cache Invalidation & CI Hydration

UniLLM tests replay LLM responses from `.cache.ndjson`. Normal CI is **read-only** (`UNILLM_CACHE=read-only`, `UNILLM_CACHE_BACKEND=local_separate`): a cache miss fails loudly instead of calling paid APIs.

Backends index `.cache.ndjson` via a `.cache.ndjson.idx` sidecar (sha256 of
each key → byte offset) so concurrent local sessions do not each hold the
full multi-GB key set in RAM. After consolidating or manually editing the
NDJSON file, delete `*.ndjson.idx` (consolidate scripts do this
automatically) so the next process rebuilds a matching index.

## When the cache must be refreshed

Refresh (or re-seed) the cache when a change alters cache keys or LLM payloads, for example:

- Edits to `unillm/caching/_caching.py` or response-format preprocessing
- New/changed test prompts, tools, or `response_format` handling
- New model endpoints exercised by tests

Symptom in CI: `pytest` fails with `Failed to get cache for function chat.completions.create ... from cache at None`.

## Where CI gets the cache

`tests.yml` restores `.cache.ndjson` from the GitHub Actions cache, then **hydrates** it from the latest successful **`llm-cache-refresh.yml`** run on the branch under test (fallback: `main`), downloading the `llm-cache-ndjson` artifact. That artifact is the source of truth: the Actions cache drops an entry nobody has read for seven days, and an entry saved on a branch other than `main` is visible only on that branch.

The artifact expires 90 days after its run. If CI starts missing everything, check when the last publish on `main` succeeded, and publish again (Path A, steps 4–5) if it is that old:

```bash
gh run list --repo unifyai/unillm --workflow llm-cache-refresh.yml --branch main --status success --limit 1
```

## Path A — Local seed publish (preferred when keys change)

Use when CI cache refresh misses entries (common for OpenRouter-routed models) or you already have a populated local cache.

1. **Populate locally** (repo root, provider keys in `.env` or shell):

```bash
export UNILLM_CACHE=true UNILLM_CACHE_BACKEND=local_separate
uv sync --group dev
uv run pytest --timeout=600 -p no:warnings tests/test_clients   # or the failing paths
```

2. **Consolidate** existing cache + new writes:

```bash
mkdir -p cache-artifacts/current
cp .cache_write.ndjson cache-artifacts/current/.cache_write.ndjson
python3 .github/scripts/consolidate_cache.py --artifacts-dir cache-artifacts
cp .cache.ndjson .github/cache-seed/cache.ndjson
```

3. **Commit** `.github/cache-seed/cache.ndjson` to **`main`** (tracked name avoids `.gitignore` on `.cache.ndjson`).

4. **Publish** the artifact on **`main`**:

```bash
gh workflow run llm-cache-refresh.yml --repo unifyai/unillm --ref main \
  -f confirm_llm_spend=skip -f publish_seed=PUBLISH_SEED_OK
```

5. **Wait** for that workflow to finish successfully, then **re-run** CI (or push an empty commit). If pytest hydrated before publish completed, it will still use a stale artifact — re-run failed jobs after publish.

## Path B — Paid CI cache refresh

Dispatch `llm-cache-refresh.yml` on **`main`** with real LLM spend:

```bash
gh workflow run llm-cache-refresh.yml --repo unifyai/unillm --ref main \
  -f test_path=tests/test_clients -f confirm_llm_spend=LLM_SPEND_OK
```

The refresh reads `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` from the `unillm-llm-cache-refresh` environment, which holds neither since its dead copies were deleted on 10 October 2026. Set both from Secret Manager before dispatching, as the team service account:

```bash
for k in ANTHROPIC_API_KEY OPENROUTER_API_KEY; do
  CLOUDSDK_ACTIVE_CONFIG_NAME=automation gcloud secrets versions access latest --secret "$k" --project saas-368716 \
    | gh secret set "$k" --repo unifyai/unillm --env unillm-llm-cache-refresh
done
```

Tests may still fail assertions while writing cache; the workflow uses `set +e` so misses are captured. Prefer Path A when refresh runs consistently miss OpenRouter-routed entries.

## Finding why a request missed

A read-only run's misses are its `CacheMissError` messages (`Failed to get cache for function ... with kwargs ...`) in the test log. `.cache_write.ndjson` is not a list of misses: the `local_separate` backend promotes every hit into it, so it holds everything the run served. To find the part of a missed request that drifted, parse the nearest stored key with `parse_raw_key`, pass both requests' kwargs through `canonical_kw` (both in `unillm/caching/canonical.py`) and diff the results.

## When the LLM Cache Check fails but pytest passes

The `LLM Cache Check` job counts the `.cache_miss.txt` files in the pytest job's `logs-<run id>` artifact, under `unillm/`. In read-only CI a real miss raises `CacheMissError`, is logged as `.cache_error.txt` and fails pytest, so a `.cache_miss.txt` beside a green pytest is a call answered without the cache. Either a test passes `cache=True`, which overrides `UNILLM_CACHE` and sends a miss to the paid provider, or a test patches `_get_cache` and fakes the provider. Read the miss files: a faked one carries the fake's reply (`gen-fake`, `msg_fake`). A faked call must write no log file. Send it through `captured_requests()` in `tests/test_clients/fake_transport.py`, which writes none, or patch `unillm.clients.uni_llm.write_request_pending` to return `None`, as the tests that mock `litellm.completion` do. The two misses that failed the check from 7 to 10 October 2026 came from a test that patched `_get_cache` inside the fake transport before it wrote no logs.

## Verification

- Published artifact should contain hundreds of entries (check workflow logs: `Consolidated entry count: N` in the refresh run, or `LLM cache ready: N entries` in the `tests.yml` hydrate step).
- Locally, `uv run pytest` with read-only cache should pass before publishing.
