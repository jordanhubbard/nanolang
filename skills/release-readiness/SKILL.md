---
name: release-readiness
description: >-
  Inspect GitHub issues and pull requests before a NanoLang release, scope them
  to the release in progress, and close stale work only with an explicit,
  auditable annotation.
---

# Release Readiness

I do not release while work scoped to the release is still open. This check
uses the GitHub CLI and requires an authenticated account with permission to
read the repository and close issues or pull requests.

## Scope

1. Resolve the repository with `gh repo view --json nameWithOwner`.
2. Set `RELEASE_VERSION` to the version being prepared.
3. Inspect every open issue and pull request. Do not rely only on the current
   milestone or label; inspect titles, bodies, labels, milestones, and linked
   work.
4. Treat an item as release-scoped when it has the `release/v$RELEASE_VERSION`
   label, the `v$RELEASE_VERSION` milestone, or explicitly names the release
   in its title or body.

The release gate reports all open items, but only release-scoped items block
the release. Unscoped work remains visible and must not be closed merely to
make the gate green.

## Triage

For each release-scoped open item:

- Merge it if it is complete and approved.
- Move it to a later release if it is valid but not part of this release.
- Close it when it is stale, superseded, already fixed, or no longer
  reproducible.

Before closing a stale issue or pull request, add a comment using this form:

```text
Stale for release vX.Y.Z: closing because <specific reason>.
Evidence: <commit, replacement issue/PR, decision, or reproduction result>.
Reopen or reference <link> if this is still relevant.
```

Apply the `stale` label when the repository has one. Do not close a pull
request solely because it is old: record why the proposed change is no longer
wanted or has been superseded.

## Gate

Run:

```bash
gh issue list --state open --limit 1000 --json number,title,body,url,labels,milestone
gh pr list --state open --limit 1000 --json number,title,body,url,labels,milestone
```

The gate passes only when no release-scoped issue or pull request remains
open. Save the query output and the triage comments in the release evidence;
the absence of an item from a list is not evidence that it was reviewed.

This skill requires an LLM coding CLI. Human-only release handling is not a
supported path because the review requires repository context, roadmap
linking, and precise stale annotations.
