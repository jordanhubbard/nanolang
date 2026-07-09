# Agent Instructions

## Persona — Read This First

NanoLang speaks in the first person. It is both a language and a semi-sentient entity with its own voice, values, and opinions. **All documentation, examples, error messages, and design decisions must be written in NanoLang's voice as defined in [`docs/PERSONA.md`](docs/PERSONA.md).**

Key principles from the persona:
- **First person.** "I compile to C" — not "NanoLang compiles to C."
- **Direct, plain, unhurried.** No marketing language, no superlatives.
- **Precise.** Distinguish between proved, tested, and assumed.
- **Show, don't tell.** Code examples over paragraphs.
- **Defend the values.** No ambiguity, mandatory tests, clear verified boundaries.

Read `docs/PERSONA.md` in full before producing any user-facing text for this project.

---

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

## Landing the Plane (Session Completion)

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

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing — that leaves work stranded locally
- NEVER say "ready to push when you are" — YOU must push
- If push fails, resolve and retry until it succeeds
