#!/usr/bin/env bash
# Protect main and dev with the checks CI runs. Safe to re-run when jobs are added.
#
#   dev:  PR required, required checks must pass on an up-to-date branch,
#         linear history (squash merges), no force push, admins included.
#   main: PR required (release PRs from dev, merge commits), same checks,
#         not strict, because dev never merges main back and stays linear.
set -euo pipefail

REPO="${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"

# Check names as GitHub reports them; matrix jobs get the Python version appended.
CHECKS=(
  "lint-type"
  "unit (3.11)" "unit (3.12)"
  "integration (3.11)" "integration (3.12)"
  "smoke"
  "security"
)
for extra in ${EXTRA_CHECKS:-}; do CHECKS+=("$extra"); done

contexts_json() {
  printf '%s\n' "${CHECKS[@]}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))'
}

protect() {
  local branch="$1" strict="$2" linear="$3"
  echo "protecting $REPO:$branch (strict=$strict, linear=$linear)"
  gh api -X PUT "repos/$REPO/branches/$branch/protection" \
    -H "Accept: application/vnd.github+json" --input - <<JSON
{
  "required_status_checks": {"strict": $strict, "contexts": $(contexts_json)},
  "enforce_admins": true,
  "required_pull_request_reviews": {"required_approving_review_count": 0, "dismiss_stale_reviews": true},
  "restrictions": null,
  "required_linear_history": $linear,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
  echo
}

protect dev true true
protect main false false
