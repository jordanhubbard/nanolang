# Nanolang Project Rules for AI Assistants

## Beads Workflow (Primary Project Management)

### Session Start Protocol
**CRITICAL**: At the start of EVERY session, run:
```bash
bd ready --json
```
This shows all open issues/tasks ready to be worked on. Use this to understand current project priorities and work items.

### Issue Management
**USE BEADS FOR ALL PLANNING** instead of creating markdown files:

- **Creating Issues**: `bd create --title "..." --description "..." --priority [0-4] --type [feature|bug|task|chore] --labels "..."`
- **Viewing Issues**: `bd ready --json` (ready to work) or `bd list --json` (all issues)
- **Updating Issues**: `bd update <id> --status [open|in_progress|blocked|done] --notes "..."`
- **Closing Issues**: `bd close <id> --reason "..."`
- **Adding Dependencies**: `bd link <child-id> --depends-on <parent-id>` or `bd link <id> --blocks <blocked-id>`

### Issue Types
- **feature**: New functionality
- **bug**: Defects or errors
- **task**: Work items (refactoring, cleanup, documentation)
- **chore**: Maintenance (dependencies, build system)
- **epic**: Large initiatives that span multiple issues

### Priority Levels
- **P0 (0)**: Critical - Drop everything
- **P1 (1)**: High - Next up
- **P2 (2)**: Medium - Soon
- **P3 (3)**: Low - Backlog
- **P4 (4)**: Nice to have

### Labels
Use descriptive labels: `transpiler`, `module-system`, `metadata`, `testing`, `memory-safety`, `performance`, etc.

### Workflow
1. Start session: `bd ready --json` to see what's ready
2. Pick an issue or create new: `bd create ...`
3. Start work: `bd update <id> --status in_progress`
4. During work: Update with `--notes` as you make progress
5. Complete: `bd close <id> --reason "Brief completion summary"`
6. Create follow-ups if needed: `bd create ... --depends-on <parent-id>`

## File Organization Rules

### Planning Documentation Location
**RULE**: DO NOT create planning markdown files. Use beads issues instead.

#### Allowed Top-Level Files ONLY:
- `README.md`
- `CONTRIBUTING.md`
- `CONTRIBUTORS.md`
- `LICENSE`
- `MEMORY.md`
- `SPEC_AUDIT.md`

#### Files That Belong in `planning/` (Legacy - Prefer Beads):
- Existing design documents (don't delete, but create new ones as beads)
- Implementation plans (create as beads going forward)
- Status reports (use bead updates instead)

#### Rationale:
- Beads provides structured issue tracking with dependencies, priorities, and status
- No need to manually manage markdown files
- Better visibility and searchability
- Integrated with git workflow via `.beads/issues.jsonl`

## Cross-Tool Integration

- **Factory Droid**: Uses this file as primary rules source
- **Cursor AI**: Create `.cursorrules` in project root referencing beads workflow
- **Claude**: This file is loaded at session start
