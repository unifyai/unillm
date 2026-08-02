#!/usr/bin/env bash
set -euo pipefail

# Keep staging->main release gates aligned with .github/workflows/tests.yml.
# The required "pytest" context comes from the pytest-required job, which runs
# unconditionally on pull_request/workflow_dispatch and reports an explicit
# pass or fail there. The suite itself is pytest-run.
#
# pytest-required must never be gated on should-run-tests (or similar) again.
# GitHub counts a skipped required check as satisfied, so a job that publishes
# this context conditionally on whether tests ran gives an implicit pass on
# every commit where they don't -- and because a release PR shares its head
# SHA with pushes to staging, that stale pass satisfies this ruleset. The
# identical shape in orchestra let four releases merge with a failing suite.
#
# It also must not run on plain push: that only reintroduces the same bug in
# fail-closed form, attaching a failing run to a push's SHA that a later
# passing pull_request run on the same SHA can never supersede (branch
# protection blocks on any matching-name run, not just the latest). Scoping
# to pull_request/workflow_dispatch leaves plain pushes with no run at all
# for this context -- pending, not a stale pass or a false block.

REPO="${REPO:-unifyai/unillm}"
RULESET_ID="${RULESET_ID:-11528525}"

echo "Updating ${REPO} Staging->Main ruleset (${RULESET_ID})..."
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${REPO}/rulesets/${RULESET_ID}" \
  --input - <<'EOF'
{
  "name": "Staging->Main",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "exclude": [],
      "include": ["~DEFAULT_BRANCH"]
    }
  },
  "bypass_actors": [],
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {"type": "creation"},
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": true,
        "required_reviewers": [],
        "require_code_owner_review": false,
        "dismissal_restriction": {
          "enabled": false,
          "allowed_actors": []
        },
        "require_last_push_approval": false,
        "required_review_thread_resolution": false,
        "allowed_merge_methods": ["merge"]
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          {"context": "pytest", "integration_id": 15368},
          {"context": "black", "integration_id": 15368},
          {"context": "staging-source", "integration_id": 15368}
        ]
      }
    }
  ]
}
EOF

echo "Updating ${REPO} main branch protection required checks..."
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${REPO}/branches/main/protection" \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      {"context": "pytest", "app_id": 15368},
      {"context": "black", "app_id": 15368},
      {"context": "staging-source", "app_id": 15368}
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

echo "Release gates:"
gh api "repos/${REPO}/rulesets/${RULESET_ID}" \
  --jq '.rules[] | select(.type=="required_status_checks") | .parameters'
gh api "repos/${REPO}/branches/main/protection/required_status_checks" \
  --jq '{strict, contexts}'
