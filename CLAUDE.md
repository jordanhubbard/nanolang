I follow `AGENTS.md` and the portable skills in `skills/`. GitHub Issues is my
only project task ledger.

## Issue Tracking

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

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

1. **File follow-up issues** via `gh issue create --body-file <path>` **and add matching
   `docs/ROADMAP.md` checkboxes** for defects discovered this session
2. **Run quality gates** (if code changed) — tests, linters, builds
3. **Update issue status** with an evidence comment; close it with `gh issue close <number>` only when all acceptance criteria pass
4. **PUSH TO REMOTE** — MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** — clear stashes, prune remote branches
6. **Verify** — all changes committed AND pushed
7. **Hand off** — context for the next session
