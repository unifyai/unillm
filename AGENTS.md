<!--
    GENERATED FILE - DO NOT EDIT DIRECTLY.

    Regenerate with:  python3 .agents/global-rules/build_agents_md.py

    Edit the sources instead:
      .agents/repo.md              this repo's overview and always-on guidance
      .agents/rules/*.md           this repo's own rules
      .agents/shared.txt           which shared rules this repo includes
      .agents/global-rules/rules/  rules shared across all unifyai repos
                                   (submodule: unifyai/global-agent-rules)
-->

# UniLLM: The LLM Abstraction Layer

UniLLM is a lightweight wrapper that normalizes LLM interactions across multiple providers (OpenAI, Anthropic, Vertex AI, Bedrock, Groq, Mistral, etc.) through a unified `model@provider` endpoint format.

## Core Features

- **Unified Endpoint Format**: `model@provider` (e.g., `openai/gpt-4o@openrouter`, `claude-sonnet-4-20250514@anthropic`)
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
- **orchestra**, **unisdk**, **unify-deploy** and **console** are archived along with the hosted platform they made up. UniLLM depends on none of them, and CI starts no server.

---

# Repository rules

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

---

# Shared rules

These apply across every unifyai repo. Edit them in the `unifyai/global-agent-rules` submodule, not here.

# Aggressive Refactoring & Zero Backward Compatibility

## Context
This project is a prototype in rapid development. We prioritize a clean, minimal, and correct codebase over stability, backward compatibility, or risk aversion.

## Critical Rules

### 1. Zero Backward Compatibility
- **Assume NO Backward Compatibility**: Unless explicitly requested, break APIs, data structures, and protocols freely.
- **Immediate Updates**: When changing an interface, update all call sites immediately. Do not create adapters, aliases, or optional parameters to "soften" the change.
- **Purge Old Patterns**: If you introduce a new design pattern (e.g., for state management), strictly remove all instances of the old pattern. Do not leave mixed patterns in the codebase.

### 2. Destructive vs. Additive Editing
- **Avoid "Stapling"**: Do not merely add new logic on top of old logic (additive editing). This creates bloat and "staples upon staples".
- **Rewrite and Simplify**: When requirements change, **rewrite** the affected code to optimally support the *new* requirements as if they were the original ones.
- **Delete Aggressively**: If code is no longer the "best" way to do something, delete it. Do not comment it out. Do not keep it "just in case".

### 3. No Defensive Coding
- **No Preemptive Exception Handling**: Do not wrap code in try/catch or try/except blocks to prevent crashes unless you are handling a specific, expected, and recoverable runtime error (e.g., network timeout). Fail loud and fast.
- **No Defensive Checks**: Do not add null checks or type checks unless strictly necessary for the logic. Trust the type system and the caller contract.
- **Clean Implementation**: Code should look like the "happy path". Avoid cluttering logic with defensive branches for edge cases that shouldn't happen in a correct system.

### 4. Review and Reflect
- **Simplify First**: Before adding a feature, ask: "Can I simplify the existing code to make this feature natural to implement?"
- **Remove Bloat**: actively look for and remove redundant code, unused variables, and overly complex abstractions.

# No Temporal or Chat-Specific Comments

## Context
Code comments and docstrings must be **timeless** and describe the code as it currently exists. They must **never** reflect the history of changes, the current chat session, or the fact that code is "new".

## Critical Rules

### 1. No "New" or "Updated" Markers
- **Forbidden**: Never use words like "New", "Updated", "Added", "Modified", "Refactored" in comments to mark changes.
- **Reasoning**: Code is only "new" for the moment it is written. Next week, it is old. These comments rot immediately and create noise.
- **Correct Approach**: Just write the comment describing what the code does. Git history tracks what is new.

### 2. No Chat Context
- **Forbidden**: Do not reference "the user request", "this chat", "per instruction", or specific reasoning from the current conversation.
- **Reasoning**: The codebase must stand alone. Context from a chat session is lost to future readers.
- **Correct Approach**: If a complex decision needs explanation, document the *technical why* (e.g., "Use X because Y is slow"), not the *conversational why* ("User asked for X").

### 3. Clean Documentation
- **Focus**: Comments should explain **why** tricky code exists or **how** it functions.
- **Avoid**: "Here is the implementation of..." or "Standardized composer utilities". The code itself shows it is an implementation.
- **Example**:
  - BAD: `NEW: Added this function to handle retries`
  - BAD: `Updated to support the new API`
  - GOOD: `Retries the request with exponential backoff to handle transient network errors.`

If a test is failing, we should **never** add test-specific information or shortcuts to production code as a hack to get the test passing. No details about specific test cases should ever make their way into production code—no special-case branches, no hardcoded values that match test inputs, no conditional logic that only exists to satisfy a test.

All fixes must be fully general and **much** broader than the specific failing test. We do **not** want to overfit production code to a specific set of tests.

# Git Safety

## Rule: Pull Before Editing a Repository

Before making any file edits in a repository, run `git pull --rebase` once to sync with the remote. This prevents the agent from working on a stale branch and silently overwriting others' commits.

- **Once per repo per session** is sufficient — no need to pull on every turn.
- **After a push rejection + rebase**: re-read any files you plan to edit next. The rebase changed them on disk but your in-memory copies are stale.
- **Exception**: only skip if the user explicitly asks you not to pull.

## Context: Explicit Path Commits
When multiple agents run in parallel in `local` mode, there is a race condition risk if they use the shared git index (staging area).
- Agent A: `git add fileA`
- Agent B: `git add fileB`
- Agent A: `git commit -m "msg"` -> Commits BOTH fileA and fileB!

## Rule: Explicit Path Commits
To eliminate this risk, **NEVER** run `git commit` without explicit file arguments.

### Incorrect
```bash
git add myfile.json
git commit -m "Update myfile"
```

### Correct
```bash
# For modified files:
git commit myfile.json -m "Update myfile"

# For new (untracked) files:
git add myfile.json
git commit myfile.json -m "Add myfile"
```

### Reasoning
Passing filenames to `git commit` bypasses the shared index for that specific commit operation, ensuring that Agent A only commits what it intends to, regardless of what Agent B has staged.

## Rule: Push When the Work Lands

Commit to `main` and push as the work lands, without waiting to be asked. An
unpushed commit, or a branch nobody merges, is invisible to every other
checkout and every other agent: a branch in `continual-arc-baselines` once
drifted ten commits ahead of an untouched `main` before anyone noticed.

- Push the current branch only: `git push origin HEAD`.
- If the push is rejected, `git pull --rebase`, re-read what you are about to
  edit, and push again.
- **Never** force-push unless the user explicitly asks for it and understands
  that it rewrites history other people have pulled.
- Start a branch only when a change genuinely has to sit apart from `main`,
  and merge and delete it as soon as it lands.

# Worktree Mode: Direct Commits, No Feature Branches

## Context
When running in **worktree mode**, the mental model is fundamentally different from traditional feature development:

- **Worktrees are for small-scale parallel fixes**, not large-scale feature development
- **Multiple agents on the same branch** = multiple collaborators working in parallel, each with their own local working directory
- The overhead of `feature branch → PR → merge → cleanup` is **overkill** for this workflow

## Critical Rules

### 1. NEVER Create Feature Branches
When asked to make changes, commit, or push:
- **DO NOT** create a new branch (e.g., `git checkout -b feature/...`)
- **DO NOT** suggest creating a branch for the work
- **COMMIT DIRECTLY** to whatever branch is currently checked out

The worktree already provides isolation. Creating additional branches defeats the purpose.

### 2. NEVER Create Pull Requests
- **DO NOT** use `gh pr create` or suggest creating a PR
- **DO NOT** push to a new remote branch with the intent of opening a PR
- If the user wants changes merged, they will handle the merge strategy themselves

### 3. The Correct Workflow
```bash
# 1. Make your changes to files

# 2. Commit directly to the current branch (following git-commit-safety rules)
git commit <specific-files> -m "Description of change"

# 3. Push the current branch (see git-commit-safety)
git push origin HEAD
```

### 4. Mental Model
Think of worktree agents as **multiple developers pair-programming on the same branch**:
- Each has their own local checkout (the worktree)
- All commit to the same branch
- No one creates personal feature branches for small fixes
- Coordination happens through communication, not branch isolation

## Why This Matters
The alternative workflow creates significant noise:
1. **Stale branches accumulate** - agents create branches, users forget to delete them
2. **PR overhead** - reviewing, merging, and closing PRs for trivial fixes wastes time
3. **Context switching** - users must mentally track multiple branches for what should be one stream of work
4. **Merge conflicts** - more branches = more opportunities for conflicts

## Exception
If the user **explicitly asks** for a feature branch or PR workflow, follow their instructions. But **never default to this behavior** in worktree mode.

# Git History for Context

## Context
This rule applies when you are trying to understand the *rationale* behind specific code blocks, the evolution of a module, or when deciding whether "weird" looking code is essential or legacy technical debt.

## Rules

### 1. Strategic Git Usage
- **Use as a Second Level of Analysis**: If the code's purpose isn't clear from the current state alone (static analysis), use `git blame` or `git log` to uncover the "why".
- **Not a Mandate**: Do not check git history for every file you touch. This creates noise. Use it selectively when you lack context.

### 2. Understanding Code Evolution
- **Identify Legacy Code**: If you suspect code is redundant or outdated, check its commit date and message. If it was added months ago for a feature that is no longer relevant, this confirms it can likely be purged.
- **Find the "Why"**: Expressive commit messages often contain the reasoning that comments lack. Use them to understand the author's original intent before refactoring or deleting complex logic.

### 3. Targeted Queries
- **Be Surgical**: When querying git, look for the history of specific lines or changes (e.g., `git blame -L n,m filename` or `git log -p filename`) rather than dumping the entire history into the context.
- **Synthesize**: Use the information to form a narrative about the code's lifecycle (e.g., "This was added in commit X to fix bug Y, but since we rewrote the bug Y subsystem, this is now dead code").

### 4. Investigating Regressions with Git Diff

When debugging test failures or regressions, git history can pinpoint exactly what changed.

**When the user proactively provides context:**
If the user says something like "the test was passing at commit `<hash>`, and the relevant changes are in `<path>`", use this optimally:
- Run `git log --oneline <hash>..HEAD -- <path>` to see which commits touched the area
- Run `git diff <hash>..HEAD -- <path>` to get the **aggregate diff** (not serial diffs commit-by-commit)
- Cross-reference the diff with commit messages to understand developer intent
- The overall diff is mathematically equivalent to composing serial diffs, but far more token-efficient and cognitively cleaner

**When debugging hits a roadblock:**
If direct code analysis and debug logging (`CURSOR_DEBUG_LOG`) aren't yielding answers, *then* ask the user:
- "Do you know when this test was last passing? If you have a commit hash and know which files/folders are likely involved, that would help narrow down what changed."
- Don't front-load this question—often the user doesn't know the answer. Try direct debugging first.

**Avoid wasteful patterns:**
- Don't ask the user to provide diffs—ask for the commit hash and run git commands yourself
- Don't read diffs commit-by-commit and mentally compose them; use the aggregate diff
- Don't dump entire file histories; scope queries to the relevant path(s)

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

# Knowledge Lives in the Repo

An agent's private memory is invisible to the team. Claude Code's
auto-memory, Cursor's memories and any notes file outside the repo are read
only by one person's sessions on one machine: nobody else, and none of their
agents, ever sees them. A fact kept there is lost to the company the moment
it is written, and it drifts from the truth because nobody can correct it.

- **Record what is worth remembering in the repo it concerns, in the same
  change as the work:** a wiki page, a README or docs section, a rule under
  `.agents/rules/`, or a comment beside the code it explains. Company-wide
  facts (people, accounts, legal state, decisions, how a job is run) go in
  `brain`'s wiki.
- **Corrections count.** When someone tells you how they want something
  done, write down the rule it implies where the next agent will read it.
- **Commit and push it as the work lands,** so the next agent on any machine
  starts from it.
- **Never write to private memory.** Claude Code's auto-memory is off in
  every repo through `"autoMemoryEnabled": false` in the committed
  `.claude/settings.json`; keep it off.
- **Treat anything found in private memory as unverified.** Check it against
  the repo, move what is still true into the repo, and delete the rest.

Two things never go in the repo: secrets, which live in Secret Manager, and
material a repo keeps out of git on purpose through `.gitignore`, which stays
in its ignored folder.

# Shared agent conversation archive

Unify keeps a private repo of **raw** agent transcripts at **`~/shared_context`**
(GitHub: `unifyai/shared_context`), keyed by **GitHub login** (e.g. `djl11`).

## Design (important)

- **Adjacent clone, not a submodule.** `shared_context` sits next to the
  product checkouts (`~/brain`, `~/unillm`, …). It is **not** nested under any
  public or private product repo.
- **Why:** public repos such as `unillm` stay public; transcript data stays
  private. Public cloners never need or see this tree. One clone serves agents
  in **every** repo that pulls `unifyai/global-agent-rules`.

## When to load this

Use before answering questions about past investigations or decisions across the
team — e.g. "did we set up X?", "who changed Y?", "why did we do Z?" — when the
answer might live in someone else's Cursor / Claude Code / Codex session, not
only the current chat.

## How to search

Prefer ripgrep over reading whole files. Search **tracked** login trees only —
**do not** search `yours/` unless the user explicitly asks about their local /
unexported chats:

```bash
rg -n -i "keyword" ~/shared_context/derived/index.jsonl
rg -n -i "keyword" ~/shared_context -g '!yours/**' -g '!tools/**' -g '!.git/**'
```

`derived/index.jsonl` is rebuilt locally by `tools/sync.sh` / `tools/export.py`
after pull (gitignored). If it is missing, search tracked trees directly or run:

```bash
python3 ~/shared_context/tools/export.py --index-only
```

Sessions live at
`<github_login>/{cursor|codex|claude-code}/<yyyy-mm>/<id>/{meta.json,transcript.jsonl}`.

`yours/{cursor,codex,claude-code}` are local symlinks to personal stores and are
gitignored.

If `~/shared_context` is missing, say so and suggest:

```bash
git clone git@github.com:unifyai/shared_context.git ~/shared_context
```

Do **not** suggest `git submodule add` / nesting it under a product repo.

## Citing

Cite **user**, **tool**, **date**, and **path** so a human can open the same session.

## Do not

- Do not confuse this with `brain` (curated company memory).
- Do not scrub or rewrite historical transcripts.
- Do not push/sync unless the user asked you to.
- Do not grep `yours/` unless the user asked for local-only context.

# Replying to the Team

The people reading an agent's replies act on them, often in another window
while the agent waits.

- **Lead with the answer or the decision.** Background comes after it, or
  goes in a file the reply points to.
- **When someone has to do something, give exact numbered steps:** one action
  per step, with the literal button, menu path or text to paste. When they are
  carrying the steps out as they read, give one step and wait for them.
- **Only the caveat that changes what they do next.** No stream of
  consciousness, no narration of tools or process, and no alternatives nobody
  asked for; they will ask if they want more.

# OpenAI is reached only through OpenRouter

Every OpenAI LLM call in every repo routes through **OpenRouter**, using
`OPENROUTER_API_KEY`. UniLLM does not expose a native OpenAI chat provider, and
the company's own direct OpenAI account is inactive — it answers
`429 billing_not_active` — so a native route is dead in practice as well as
unregistered.

## Canonical endpoint form

```
openai/<model-id>@openrouter      # openai/gpt-5.6-terra@openrouter
```

Never `<model-id>@openai`. `@openrouter` resolves dynamically through the
OpenRouter catalog; `@openai` is not registered and fails endpoint resolution.

Source, defaults, examples, and tests use the canonical OpenRouter form
directly.

## Hard rules

- New LLM call sites use `openai/<id>@openrouter`. Non-OpenAI providers
  (Anthropic, Google, …) are unaffected by this rule and keep their own routing.
- Never use `OPENAI_API_KEY` or a native `openai.OpenAI()` client for LLM chat
  or text generation. Non-chat integrations such as speech-to-text, realtime
  voice, or embeddings may use that key when the call site documents its
  distinct purpose — that key is the deployment's own (self-host / BYOK), not
  the company account above.
- Env defaults and `.env.example` entries carry the `@openrouter` form, so a
  fresh checkout cannot inherit a dead route.
- Treat any surviving `@openai` model string as a bug; UniLLM rejects it.

## The one legitimate direct-OpenAI path

Masked image edits (`images.edit` with `gpt-image-2`) have no OpenRouter
equivalent — OpenRouter's unified Image API does not expose the mask parameter.
That path may use a separately-named credential (`OPENAI_DIRECT_API_KEY`), must
never fall back to reading `OPENAI_API_KEY`, and must degrade loudly when the
credential is absent. It is the only exception; adding another needs an
explicit reason, not convenience.
