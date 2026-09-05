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
| [`roadmap-execution`](roadmap-execution/SKILL.md) | Planning or executing multi-deliverable product work through the ordered checklist in `docs/ROADMAP.md`. Discovered bugs always get a checkbox in the same session. |
| [`mac-task-tracking`](mac-task-tracking/SKILL.md) | Finding work, filing follow-ups, claiming or closing issues. We use `mac task`, not bd/beads. |
| [`session-completion`](session-completion/SKILL.md) | Wrapping up a session: quality gates, task status, and pushing so no work is stranded. |
| [`releasing`](releasing/SKILL.md) | **Cutting a release.** The ordered gates and the index of the skills that own them. Start here; it points at the rest. |
| [`releasing/documentation`](releasing/documentation/SKILL.md) | Bringing README, CONTRIBUTING, the user guide and the roadmap current before the tag. |
| [`releasing/presentation`](releasing/presentation/SKILL.md) | Bringing the deck and narrative current before the tag, and publishing them after it. |
| [`release-readiness`](release-readiness/SKILL.md) | Reviewing GitHub issues and PRs in release scope and annotating stale work before release. |
| [`nanovm-opcode-debugging`](nanovm-opcode-debugging/SKILL.md) | Enabling and reading optional NanoVM opcode, stack, value, and FFI traces. |

Skills nest. `releasing/` is the rule set and index for a release; its
subdirectories own one concern each and carry the Python helper that does the
mechanical part of it. Shared logic lives in `releasing/lib/` rather than being
copied into each checker -- if you are writing a second implementation of a
question one of them already asks, it belongs there.

These skills are the portable source of truth. Per-tool config files
(`.claude/`, `.codex/`, `.cursor/`, `.factory/`) should point at them rather
than duplicating or contradicting them.
