<!--
    GENERATED FILE - DO NOT EDIT DIRECTLY.

    Regenerate from the repo's root with build_agents_md.py from
    unifyai/global-agent-rules: python3 ../global-agent-rules/build_agents_md.py
    where it is a clone beside the repo, or
    python3 .agents/global-rules/build_agents_md.py where it is a submodule.

    Edit the sources instead:
      .agents/repo.md       this repo's overview and always-on guidance
      .agents/rules/*.md    this repo's own rules
      .agents/shared.txt    which shared rules this repo includes
      rules/*.md in unifyai/global-agent-rules, the rules shared across
      all unifyai repos
-->

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

---

# Repository rules

# Calling LLMs

- **Prompt caching on every call, as far as it goes.** A call that repeats
  an earlier call's opening reads it from the provider's prompt cache, at
  the longest-lived setting the provider offers, since a cache read costs a
  tenth of fresh input on Anthropic's models. How UniLLM places Claude's
  breakpoints, how a caller marks more and how to warm a prefix before
  fanning out are in the README's Prompt Caching section, which a change to
  that code keeps true. A call path that repeats a prefix and shows no
  cache reads in its responses' cached-token counts is a bug.
- **OpenAI only through OpenRouter.** Every OpenAI chat call goes through
  OpenRouter with `OPENROUTER_API_KEY`, as `openai/<model-id>@openrouter`.
  UniLLM registers no native OpenAI provider, so `<model-id>@openai` is not
  found: treat any surviving `@openai` string as a bug, and give env
  defaults and `.env.example` the `@openrouter` form. Chat and text
  generation never use `OPENAI_API_KEY` or a native `openai.OpenAI()`
  client.
- **Anthropic models keep OpenRouter's default routing.** Where an
  OpenRouter account carries its own Anthropic keys (bring-your-own-key),
  OpenRouter serves a Claude call from them first. A `provider.order` or
  `only` naming Amazon Bedrock, Google Vertex or any other host skips them
  and pays list price.

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
  CLOUDSDK_ACTIVE_CONFIG_NAME=automation gcloud secrets versions access latest --secret "$k" \
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

---

# Shared rules

These apply across every unifyai repo. Edit them in the `unifyai/global-agent-rules`, not here.

# Aggressive Refactoring, No Backward Compatibility

The code is a prototype in rapid development: clean, minimal and correct
matters more than stability or caution.

- **No backward compatibility** unless it is asked for. Change an interface
  and update every call site at once, with no adapters, aliases or optional
  parameters to soften it. A new pattern replaces the old one everywhere.
- **Rewrite, don't staple.** When requirements change, rewrite the code as
  if the new requirements had been the first ones, rather than adding logic
  on top. Delete what is no longer the best way; never comment it out or
  keep it "just in case".
- **No defensive coding.** No try/except except around a specific, expected,
  recoverable error such as a network timeout, and no null or type checks
  the logic does not need. Fail loud and fast.
- **Simplify first.** Before adding a feature, ask whether simplifying the
  code would make it natural, and remove redundant code, unused variables
  and needless abstraction as you go.

# Timeless Comments

Comments and docstrings describe the code as it is now, never its history or
the conversation that produced it.

- Never mark code "new", "updated", "added" or "refactored"; git records that.
- Never cite the chat or the request ("as the user asked"). Give the
  technical why ("retries with backoff because the API drops connections"),
  never the conversational one.
- Say why tricky code exists or how it works, not that it is an
  implementation of something.

If a test is failing, we should **never** add test-specific information or shortcuts to production code as a hack to get the test passing. No details about specific test cases should ever make their way into production code—no special-case branches, no hardcoded values that match test inputs, no conditional logic that only exists to satisfy a test.

All fixes must be fully general and **much** broader than the specific failing test. We do **not** want to overfit production code to a specific set of tests.

# Your Knowledge Is Months Old

A model knows the world as it stood when its training data ended, and it
works months later. A model whose knowledge ends in June 2026 and drafts
in November knows nothing of the five months between, and nothing in what
it knows shows it the gap. It fills the gap with what it knew then, as
confidently as it says anything else. A smarter model makes the same
mistake. Only looking it up prevents it.

- **Look up, on the day, whatever changes within months.** That covers the
  newest or best models, tools, versions and methods; a field's state of
  the art and the baselines it is measured against; prices; who holds which
  job; what a company offers; and which rules are in force. Find each one
  that day, with a search or in the repo, and use memory only to decide
  what to search for. When the work presents a choice as current, such as
  the models a paper will use or the baselines it compares with, make the
  choice only after searching for anything newer or better.
- **Re-check what is carried over.** A fact or a choice taken from another
  draft, page or session is checked again on the day against its primary
  source, never inherited. Another draft is not a source.
- **A check asks two questions of every claim:** is it true, and is it
  still the right thing to say today? A claim fails if it is true but
  something newer has overtaken it, such as a model that exists but is no
  longer the one to name.

Why: in November 2026 an agent named two open models from 2025 as the
ones a project would use, when newer ones had been out for months. The
agent named them from memory, and its check confirmed only that each
model existed. Later work copied the list, citing the earlier work
rather than a primary source, and a reviewing agent with the same
training cutoff saw nothing wrong. To anyone who follows the field,
year-old models presented as current show that whoever chose them has
not kept up.

# Git Safety

- **Pull before editing.** Run `git pull --rebase origin main` once per repo
  per session before changing files, so you never work on a stale tree. A
  rebase refuses while anyone's change sits uncommitted in the tree; then
  `git pull --no-rebase --ff-only origin main` moves the clone on past it,
  refusing only if the new commits change the same file (`--no-rebase` keeps
  it a fast-forward where git is set to rebase). If both refuse, say so and
  why, even when you carry on in a worktree of your own: whatever reads the
  clone in place reads it stale until it moves on. After a rejected push and
  a rebase, re-read any file you are about to edit: the rebase changed it on
  disk.
- **Never `--autostash` or `git stash` in a checkout other agents share.**
  It takes their uncommitted edits along with yours, and when they do not
  reapply cleanly it leaves their files conflicted and staged, which blocks
  every commit there. If a rebase is refused over work that is not yours,
  rebase in a worktree of your own (`git worktree add`).
- **Commit explicit paths:** `git commit <files> -m "…"`, never a bare
  `git commit`. Agents working in parallel share one index, so a bare commit
  takes in whatever another agent staged. A new file needs `git add <file>`
  first, then the same explicit commit.
- **Push to `main` as the work lands,** without being asked:
  `git push origin HEAD:main`, which works from `main` and from a worktree's
  own branch alike. Never push a worktree's branch itself, since nobody
  merges it. If the push is rejected, `git pull --rebase origin main`,
  re-read what you are about to edit, and push again.
- **Never force-push** unless the user asks and understands that it rewrites
  history other people have pulled.
- Start a branch only when a change has to sit apart from `main`, and merge
  and delete it as soon as it lands.

# Python Formatting & Pre-commit

Every Python repo that includes this rule enforces formatting with **black**
(plus `isort`/`autoflake` where configured), and CI rejects unformatted code.
A missing local hook or a drifting Black target/Python version is the single
most common avoidable CI failure. This rule keeps local and CI identical so
it stops blocking us — for Cursor, Claude Code, Codex, and humans alike.

## Single source of truth: the locked `lint` group

The formatters are ordinary, locked dependencies — not a version hardcoded in
the pre-commit hook or in CI YAML.

- Each repo declares its formatters in a dedicated **`lint` dependency
  group**, pinned and committed to `uv.lock`. Every first-party Python repo
  is uv-managed with a repo-local `.venv` — there are no poetry repos.
  The `dev` group includes `lint` so a normal sync gives developers everything:
  `[dependency-groups]` → `lint = ["black==X", "isort>=…", "autoflake>=…"]`,
  and `dev = [ …, {include-group = "lint"} ]`.
- **Both** the pre-commit hook and CI run that **same locked** tool via uv —
  never a separate pin:
  - pre-commit hook: `entry: uv run black`, `language: system`
    (no `additional_dependencies`).
  - CI: `uv sync --only-group lint --no-install-project --frozen` then
    `uv run --no-sync black --check .` on **Python 3.12**.
- Pin Black's language target in every Python repo so local Mac Pythons and
  CI 3.12 cannot disagree (Black 26+ defaults toward newer targets):

```toml
[tool.black]
target-version = ["py312"]
```

- Never introduce a second black version or a parallel invocation anywhere
  (CI YAML, Dockerfiles, docs, ad-hoc `pip install black`, hook
  `additional_dependencies`). The locked `lint` group is authoritative; this
  rule deliberately does not restate the number — look it up in the repo's
  `pyproject.toml` / lockfile so guidance can never drift from reality.
- Keep the version **in lockstep across all the Python repos**: a bump is one
  coordinated change per repo (the `lint` pin + lockfile) applied to every
  repo so they don't diverge.

## Committed hooks (required once per clone / worktree)

`.git/hooks/` is a per-checkout artifact. Fresh clones, Cursor/Codex/Claude
worktrees, and cloud agents start with **no** hooks, which is why unformatted
code reaches CI.

Each Python repo commits `.githooks/pre-commit`. Enable it with the shared
helper (idempotent, tool-agnostic):

```bash
python3 .agents/global-rules/ensure_git_hooks.py
```

That sets local `core.hooksPath=.githooks`. Do this before the first commit
in any new clone or worktree. Coding agents (Cursor, Claude Code, Codex)
MUST run it at session start when working in a checkout that has
`.pre-commit-config.yaml`.

`pre-commit install` alone is no longer enough — it writes into `.git/hooks/`,
which worktrees and new clones miss. Prefer `ensure_git_hooks.py`.

## Before you commit (required)

1. Ensure committed hooks are wired (above).
2. Let the hook run on `git commit`, or run it explicitly on what you changed:

```bash
pre-commit run --files <changed-files>   # or: pre-commit run --all-files
```

3. Never bypass hooks: do not use `git commit -n` / `--no-verify`.

On newly wrapped code, `black` and `add-trailing-comma` each rewrite the
other's output once, so the hooks can fail twice before they pass. Re-stage
and run them again until they pass.

## Formatting across multiple repos

When juggling several repos, do not invoke a globally-installed `black` —
versions drift between machines and repos and produce diffs CI rejects.
Always format through the repo's pinned tooling, which uses that repo's
locked version:

```bash
pre-commit run black --all-files          # or: uv run black .
```

## The hook is the gate

Work lands on `main` by direct push, so the committed git hook is what keeps
unformatted code off `main`. CI runs `black` on every push, which shows a
slip at once, but only after it has landed.

## Why this matters

Without committed `.githooks` + `ensure_git_hooks.py`, the first place
formatting is checked is CI — which then burns agent turns on mundane
reformats. Pinning Black's target and CI Python to 3.12 removes the
"works on my Mac, fails in Actions" class of failures.
