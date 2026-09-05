---
name: session-completion
description: >-
  Land the plane at the end of a work session: run quality gates, update task
  status, and — critically — commit AND push so no work is stranded locally.
  Use this whenever you are wrapping up changes in this repository. Work is NOT
  complete until `git push` succeeds.
---

# Landing the plane (session completion)

When ending a work session you MUST complete ALL steps below. Work is NOT
complete until `git push` succeeds.

1. **File follow-up issues** via `mac task create` (see the mac-task-tracking
   skill) **and add matching `docs/ROADMAP.md` checkboxes** for any defect or
   remaining work discovered this session. If you promised a follow-up, file it
   now — or state plainly that you could not because the hub was unauthenticated.
2. **Run quality gates** (if code changed) — tests, linters, builds. For this
   repo that typically means the relevant `make test*` target, and for NanoLang
   changes, confirming shadow tests pass.
3. **Update task status** via `mac task close <id> --reason="..."`.
4. **PUSH TO REMOTE — MANDATORY:**
   ```bash
   git pull --rebase
   git push
   git status   # MUST show "up to date with origin"
   ```
5. **Clean up** — clear stashes, prune stale remote branches, remove scratch
   files under /tmp you created.
6. **Verify** — every change is both committed AND pushed.
7. **Hand off** — leave a short note of context for the next session.

## Critical rules

- Work is NOT complete until `git push` succeeds.
- NEVER stop before pushing — that leaves work stranded locally.
- NEVER say "ready to push when you are" — YOU must push.
- If push fails (rebase conflict, hook rejection), resolve and retry until it
  succeeds. If a pre-commit hook auto-modifies files, re-stage and amend only if
  the commit is yours and unpushed; otherwise make a new commit.

## Commit hygiene

- Only commit when the user has asked you to, or when landing a session whose
  work the user requested.
- Write commit messages that explain the "why", not just the "what".
- Do not commit secrets (.env, credential files). Warn if asked to.
- Do not skip hooks or force-push to main unless the user explicitly asks.
