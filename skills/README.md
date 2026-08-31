# Skills

Tool-neutral [Agent Skills](https://code.claude.com/docs/en/skills) for this
repository. Each subdirectory holds a `SKILL.md` with YAML frontmatter
(`name`, `description`) plus instructions. Any coding agent that supports the
`SKILL.md` convention can discover and load these — they are intentionally NOT
locked to a single CLI (`.claude/`, `.codex/`, etc.).

If your agent does not auto-discover skills, read the relevant `SKILL.md`
directly before working:

| Skill | When to use it |
| --- | --- |
| [`reading-nanolang`](reading-nanolang/SKILL.md) | Reading, understanding, or writing any `.nano` source (prefix calls, mandatory shadow tests, explicit types, the first-person voice). |
| [`roadmap-execution`](roadmap-execution/SKILL.md) | Planning or executing multi-deliverable product work through the ordered checklist in `docs/ROADMAP.md`. |
| [`mac-task-tracking`](mac-task-tracking/SKILL.md) | Finding work, filing follow-ups, claiming or closing issues. We use `mac task`, not bd/beads. |
| [`session-completion`](session-completion/SKILL.md) | Wrapping up a session: quality gates, task status, and pushing so no work is stranded. |
| [`release-readiness`](release-readiness/SKILL.md) | Reviewing GitHub issues and PRs in release scope and annotating stale work before release. |
| [`nanovm-opcode-debugging`](nanovm-opcode-debugging/SKILL.md) | Enabling and reading optional NanoVM opcode, stack, value, and FFI traces. |

These skills are the portable source of truth. Per-tool config files
(`.claude/`, `.codex/`, `.cursor/`, `.factory/`) should point at them rather
than duplicating or contradicting them.
