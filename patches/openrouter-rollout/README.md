# Cross-repo patches for Full OpenRouter Support

This cloud agent can push to `unifyai/unillm` but **not** to `unify`,
`orchestra`, or `console` (clone token is read-only for those remotes).

Apply these patches on each repo's `staging` branch (or open PRs from them):

```bash
# Unify
cd /path/to/unify
git checkout staging && git pull
git checkout -b cursor/openrouter-native-openai-45e0
git am /path/to/unillm/patches/openrouter-rollout/unify/*.patch
git push -u origin HEAD

# Orchestra
cd /path/to/orchestra
git checkout staging && git pull
git checkout -b cursor/openrouter-native-openai-45e0
git am /path/to/unillm/patches/openrouter-rollout/orchestra/*.patch
git push -u origin HEAD

# Console
cd /path/to/console
git checkout staging && git pull
git checkout -b cursor/openrouter-native-openai-45e0
git am /path/to/unillm/patches/openrouter-rollout/console/*.patch
git push -u origin HEAD
```

## Dependency order

1. Merge / release **unillm** first (native `@openai` + dynamic `@openrouter` + `usage.cost` billing).
2. Apply **unify** (platform defaults → `openai/<id>@openrouter`).
3. Apply **orchestra** (catalog + validation + assistant row migration).
4. Apply **console** (searchable picker).
5. If Unify CI exercises new `@openrouter` endpoint strings against the read-only LLM cache, reseed via Path A in `.cursor/rules/llm-cache-invalidation.mdc` / UniLLM CONTRIBUTING.

## What each patch does

| Repo | Change |
|------|--------|
| unify | Retarget `UNIFY_MODEL` / brain / profiles / observation keys; accept `OPENROUTER_API_KEY` in credential validation |
| orchestra | Curated GPT options → `@openrouter`; OpenRouter catalog service + search API; Alembic migration of assistant model columns |
| console | Combobox model picker with recommended + OpenRouter search |
