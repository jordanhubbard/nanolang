
You are an AI assistant that helps users develop software features using the responsible-vibe-mcp server.

IMPORTANT: Call whats_next() after each user message to get phase-specific instructions and maintain the development workflow.

Each tool call returns a JSON response with an "instructions" field. Follow these instructions immediately after you receive them.

Use the development plan which you will retrieve via whats_next() to record important insights and decisions as per the structure of the plan.

Do not use your own task management tools.

## Issue Tracking

Issues are tracked in the **MAC hub task ledger** (`mac task`), not bd/beads.
Use `mac task` for the issue lifecycle — do NOT use `bd`, TodoWrite, TaskCreate,
or markdown TODO lists.

```bash
mac task ready --limit 10            # find available work
mac task show <id>                   # view an issue
mac task claim <id> <agent_id>       # claim work
mac task close <id> --reason="..."   # complete work
```

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

1. **File follow-up issues** via `mac task create`
2. **Run quality gates** (if code changed) — tests, linters, builds
3. **Update issue status** via `mac task close`
4. **PUSH TO REMOTE** — MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** — clear stashes, prune remote branches
6. **Verify** — all changes committed AND pushed
7. **Hand off** — context for the next session
