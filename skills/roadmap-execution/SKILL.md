---
name: roadmap-execution
description: >-
  Plan and execute repository product work through docs/ROADMAP.md. Use this
  before starting features, refactors, optimization programs, or other work
  with multiple deliverables. Add ordered checkboxes first, execute from top
  to bottom, and mark only verified work complete.
---

# Roadmap Execution

`docs/ROADMAP.md` is my product execution order. MAC is my task ledger. They
serve different purposes and I keep both consistent.

## Before Implementation

1. Read `docs/ROADMAP.md` and locate the active queue.
2. Add the requested outcome, implementation work, tests, documentation, and
   acceptance criteria as ordered `- [ ]` items before changing code.
3. Put dependencies before dependents. Put measurement before optimization and
   semantic foundations before surface features.
4. Mark an existing item `- [x]` only when repository evidence verifies it.
5. Create or update MAC tasks for ownership and execution when appropriate.

Small fixes that restore already documented behavior may use an existing
roadmap item. If no item describes the work, add one first. Do not turn every
typo into a new phase; use the smallest accurate checkbox.

## During Implementation

- Execute the first unchecked active item whose dependencies are complete.
- Do not skip a difficult item to implement a more visible dependent feature.
- Add newly discovered required work at the correct dependency position, not
  automatically at the bottom.
- Keep checkboxes concrete and verifiable. Avoid entries such as "improve
  performance" without workloads and acceptance evidence.
- Distinguish measured facts, tested behavior, formal proof, and assumptions.
- Keep `docs/ROADMAP.md` in my first-person voice.

## Completing Items

An item becomes `- [x]` only after its required code, tests, quality gates, and
documentation are complete. A partial implementation stays unchecked or is
split into smaller honest items.

Before ending the session, update roadmap state, update the corresponding MAC
task, and follow `skills/session-completion/SKILL.md`.
