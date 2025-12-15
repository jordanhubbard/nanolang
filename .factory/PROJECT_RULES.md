# Nanolang Project Rules for AI Assistants

## File Organization Rules

### Planning Documentation Location
**CRITICAL RULE**: All planning-related markdown files MUST be created in the `planning/` directory.

#### Allowed Top-Level Files ONLY:
- `README.md`
- `CONTRIBUTING.md`
- `CONTRIBUTORS.md`
- `LICENSE`
- `MEMORY.md`
- `SPEC_AUDIT.md`

#### Files That Belong in `planning/`:
- Design documents (e.g., `GC_DESIGN.md`, `GENERICS_DESIGN.md`)
- Implementation plans (e.g., `IMPLEMENTATION_ROADMAP.md`)
- Status reports (e.g., `SELF_HOST_STATUS.md`, `PHASE2_COMPLETE.md`)
- Roadmaps and strategies (e.g., `BOOTSTRAP_STRATEGY.md`)
- Architecture decisions
- Project audits and assessments
- TODO lists and task tracking
- Any other planning or project management documentation

#### Enforcement:
1. **AI Level**: Custom droid (`nanolang-planning-directory-enforcer`) prevents creation of planning files at top level
2. **Git Level**: Pre-commit hook automatically moves misplaced planning `.md` files to `planning/` and blocks the commit for review
   - Hook source: `scripts/pre-commit-planning-enforcer`
   - Already installed in `.git/hooks/pre-commit`
   - To reinstall: `cp scripts/pre-commit-planning-enforcer .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`

#### Rationale:
Keeps the repository root clean and navigable by segregating project documentation from essential top-level files.
