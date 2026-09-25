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

Requires the `unillm-llm-cache-refresh` environment secrets. Tests may still fail assertions while writing cache; the workflow uses `set +e` so misses are captured. Prefer Path A when refresh runs consistently miss OpenRouter-routed entries.

## Finding why a request missed

A read-only run's misses are its `CacheMissError` messages (`Failed to get cache for function ... with kwargs ...`) in the test log. `.cache_write.ndjson` is not a list of misses: the `local_separate` backend promotes every hit into it, so it holds everything the run served. To find the part of a missed request that drifted, parse the nearest stored key with `parse_raw_key`, pass both requests' kwargs through `canonical_kw` (both in `unillm/caching/canonical.py`) and diff the results.

## Verification

- Published artifact should contain hundreds of entries (check workflow logs: `Consolidated entry count: N` in the refresh run, or `LLM cache ready: N entries` in the `tests.yml` hydrate step).
- Locally, `uv run pytest` with read-only cache should pass before publishing.
