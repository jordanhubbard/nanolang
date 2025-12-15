---
description: "Enforces proper file organization for planning documentation - all planning .md files must go in planning/ directory"
alwaysApply: true
---

# File Organization Rules

**CRITICAL RULE**: All planning-related markdown files MUST be created in the `planning/` directory, never at the top level.

## Allowed Top-Level Files ONLY:
- `README.md`
- `CONTRIBUTING.md`
- `CONTRIBUTORS.md`
- `LICENSE`
- `MEMORY.md`
- `SPEC_AUDIT.md`

## Files That Belong in `planning/`:
- Design documents (e.g., `GC_DESIGN.md`, `GENERICS_DESIGN.md`)
- Implementation plans (e.g., `IMPLEMENTATION_ROADMAP.md`)
- Status reports (e.g., `SELF_HOST_STATUS.md`, `PHASE2_COMPLETE.md`)
- Roadmaps and strategies (e.g., `BOOTSTRAP_STRATEGY.md`)
- Architecture decisions
- Project audits and assessments
- TODO lists and task tracking
- Any other planning or project management documentation

## Enforcement Mechanisms:
1. **AI Level**: This rule + Factory droid (`nanolang-planning-directory-enforcer`)
2. **Git Level**: Pre-commit hook (see `scripts/pre-commit-planning-enforcer`)

## Rationale:
Keeps the repository root clean and navigable by segregating project documentation from essential top-level files.

---

**For complete details, see**: @.factory/PROJECT_RULES.md
