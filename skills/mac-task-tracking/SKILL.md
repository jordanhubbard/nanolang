---
name: mac-task-tracking
description: >-
  Track issues and work items for this repository using the MAC hub task ledger
  (`mac task`). Use this whenever you need to find available work, file a
  follow-up issue, claim a task, or close one. NanoLang uses `mac task` — NOT
  bd/beads, NOT TodoWrite/TaskCreate, NOT markdown TODO lists.
---

# MAC task tracking

Issues for this repository live in the MAC hub task ledger. The canonical CLI is
`mac task`. Do not use `bd`/beads (removed), and do not invent markdown TODO
files as a substitute.

## The lifecycle

```bash
mac task ready --limit 10            # find available work + eligible agents
mac task show <id>                   # view one task (add --json for full record)
mac task claim <id> <agent_id>       # atomically claim a task for an agent
mac task close <id> --reason="..."   # complete a task (cancellation needs a reason)
```

## Filing a task

```bash
mac task create "<title>" --description "..." --priority <0-4> --kind <code|report>
```

- `--kind code` (default) expects a repository change; `--kind report` is for an
  investigation/answer/triage that is satisfied by a written finding.
- For descriptions that are awkward to quote on the shell (newlines, quotes, log
  tails), write the body to a file and pass `--description-file <path>` (or `-`
  for stdin). This avoids shell-quoting hazards.
- Priority is an integer, lower = more urgent on this hub.

## Reading tasks programmatically

`mac task list --json` returns a JSON array of task objects. The stable fields
are: `id`, `title`, `description`, `state`, `priority` (int), `project`,
`created_at`, `updated_at`. States are `open`, `waiting`, `blocked`, `claimed`,
`running`, `needs_review`, `needs_input`, `stopped`, `reviewing`, and the
terminal `completed` / `failed` / `cancelled`. By default the list shows only
active (non-terminal) tasks; pass `--all-states` to include terminal ones.

From NanoLang, use `stdlib/mac.nano` (Task struct + `mac_list`, `mac_ready`,
`mac_create`, `mac_close`, `mac_stats`, `assert_with_task`). It shells out to the
`mac` CLI and degrades gracefully (empty results) when the CLI is missing or the
hub is unauthenticated.

## Graceful degradation

`mac` needs a hub token. When it is missing you will see `missing bearer token`
on stderr and commands return no data. That is expected in unauthenticated
environments (CI without a token, fresh clones): tooling that files tasks should
treat this as "skip filing" and continue, never as a hard failure. The
`scripts/automac.py` wrapper and `stdlib/mac.nano` both follow this rule.

## Do not

- Do not use `bd`, `beads`, or the old `.beads/` directory — that tool is gone.
- Do not use TodoWrite / TaskCreate or markdown TODO lists as the issue tracker.
- Do not silently drop a follow-up you promised to file; file it as a `mac task`
  (or state clearly that you could not because auth was unavailable).
- A MAC task is not a substitute for `docs/ROADMAP.md`. File the task **and**
  add a concrete roadmap checkbox in dependency order (see
  `skills/roadmap-execution/SKILL.md`).
