---
name: github-issue-tracking
description: >-
  Track all NanoLang work in GitHub Issues using gh. Use when finding work,
  filing follow-ups, coordinating ownership, recording evidence or closing work.
---

# GitHub issue tracking

I track all new and resumed project work in [GitHub Issues](https://github.com/jordanhubbard/nanolang/issues).
GitHub Issues is my task ledger; `docs/ROADMAP.md` remains my ordered product
contract and evidence index. I do not use MAC, bd/beads, or local TODO files as
an alternative task ledger. Historical MAC IDs remain provenance only.

```bash
gh issue list --repo jordanhubbard/nanolang --state open
gh issue view <number> --repo jordanhubbard/nanolang --comments
gh issue edit <number> --repo jordanhubbard/nanolang --add-assignee @me
gh issue create --repo jordanhubbard/nanolang --title "<title>" --body-file /tmp/issue.md
gh issue comment <number> --repo jordanhubbard/nanolang --body-file /tmp/update.md
gh issue close <number> --repo jordanhubbard/nanolang --reason completed
```

I search before creating an issue, record branch/file ownership in an issue
comment, and link commits, PRs and test evidence before closing it. Assignment
is not an atomic worker lease; I coordinate overlapping work explicitly. When
resuming an old MAC task, I create or reuse a GitHub issue and link the old ID.
If GitHub is unavailable, I preserve my work and report the unfiled issue; I do
not fall back to MAC or claim that tracking succeeded.

## Issue lifecycle

- I put the problem, intended outcome, acceptance criteria and dependencies in
  the issue body. I use `--body-file` for multiline text and literal shell syntax.
- I link the issue from the relevant roadmap row and PR. Roadmap checkboxes do
  not replace issues; issues do not replace the roadmap's acceptance contract.
- I record current branch, source pin, file ownership and tests in comments.
- I close an issue only when its complete acceptance criteria are verified.
  Partial checkpoints stay open. I reopen it if the same failure returns.
- I retain historical MAC evidence and IDs without rewriting old records or
  mass-importing inaccessible tasks. Work resumed now gets a GitHub issue.

## Automation

`python3 scripts/autogithub.py --tests` and `--examples` run the existing Make
recipes and report failures in this repository's GitHub Issues. Summary mode
keys issues by branch and job; per-failure mode keys by title/fingerprint.
`--dry-run` makes no GitHub calls. Missing `gh`, authentication or network access
never hides the test result. Log files remain under `.test_output/`.
The optional `stdlib/mac.nano` client and its API examples are external-service
compatibility code, not this project's task workflow.
