#!/bin/bash

set -euo pipefail

release_script="scripts/release.sh"

require_line() {
    local pattern=$1
    if ! grep -Fq -- "$pattern" "$release_script"; then
        printf 'missing release workflow step: %s\n' "$pattern" >&2
        exit 1
    fi
}

reject_line() {
    local pattern=$1
    if grep -Fq -- "$pattern" "$release_script"; then
        printf 'obsolete release workflow step remains: %s\n' "$pattern" >&2
        exit 1
    fi
}

bash -n "$release_script"
require_line 'local release_branch="release/v$version"'
require_line 'git push --set-upstream origin "$release_branch"'
require_line 'pr_url=$(gh pr create \'
require_line 'gh pr checks "$pr_url" --watch --fail-fast'
require_line 'gh pr merge "$pr_url" --squash --delete-branch'
require_line 'check_release_github_work "$NEXT_VERSION"'
require_line 'gh issue list --repo "$repo" --state open'
require_line 'gh pr list --repo "$repo" --state open'
require_line 'Release-scoped GitHub issues or pull requests remain open.'
require_line 'git pull --ff-only origin main'
require_line 'git tag -a "v$version" -m "Release v$version"'
reject_line 'git push origin main'

pr_line=$(grep -nF 'pr_url=$(gh pr create \' "$release_script" | cut -d: -f1)
merge_line=$(grep -nF 'gh pr merge "$pr_url"' "$release_script" | cut -d: -f1)
tag_line=$(grep -nF 'git tag -a "v$version"' "$release_script" | cut -d: -f1)

if (( pr_line >= merge_line || merge_line >= tag_line )); then
    printf 'release PR must be created and merged before the release tag\n' >&2
    exit 1
fi

printf 'release workflow checks passed\n'
