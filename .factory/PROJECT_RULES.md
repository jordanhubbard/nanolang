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

## Dual Implementation Requirement ⚠️ CRITICAL ⚠️

### Every Language Feature Must Be Implemented TWICE

**RULE**: ALL new language features, syntax additions, or compiler capabilities MUST be implemented in BOTH:
1. **C Reference Compiler** (`src/`)
2. **Self-Hosted NanoLang Compiler** (`src_nano/`)

### Why This Matters
- **Self-Hosting Parity**: The NanoLang compiler must be able to compile itself
- **Reference Implementation**: C version is the definitive specification
- **Validation**: Each implementation validates the other
- **Maintenance**: Changes must be synchronized across both

### Components Requiring Dual Implementation

For each new feature, expect changes in **both** implementations:

| Component | C Implementation | NanoLang Implementation |
|-----------|------------------|-------------------------|
| **Lexer** | `src/lexer.c` | `src_nano/compiler/lexer.nano` |
| **Parser** | `src/parser_iterative.c` | `src_nano/compiler/parser.nano` |
| **Type System** | `src/typechecker.c` | `src_nano/compiler/typecheck.nano` |
| **Code Generator** | `src/transpiler_iterative_v3_twopass.c` | `src_nano/compiler/transpiler.nano` |

### Cost Analysis for New Features

Before proposing new syntax or language features, consider:

**Example: Adding Rust-style `[Type; size]` array syntax**
- Lexer changes: `;` token in array context (2 implementations)
- Parser changes: New grammar rules (2 implementations)
- Type system: Fixed-size array types (2 implementations)
- Code gen: Initialization logic (2 implementations)
- **Total: 8 substantial changes + testing**

**Question to ask:** Is the syntax sugar worth 2× the implementation and maintenance cost?

### Guidelines for Feature Proposals

1. **Justify Complexity**: New features must provide significant value
2. **Consider Alternatives**: Can existing syntax solve the problem?
3. **Implementation Cost**: Estimate dual-implementation effort
4. **Breaking Changes**: Will this break self-hosting?
5. **Test Coverage**: Shadow tests required in both implementations

### Current Constraints

These design constraints reflect the dual-implementation reality:

✅ **Simple Designs Win**
- Prefer library functions over new syntax
- Keep grammar minimal and regular
- Avoid complex type inference
- Favor explicit over implicit

❌ **Avoid These**
- Syntax sugar that requires parser changes
- Complex type system features
- Inference requiring sophisticated algorithms
- Features that complicate code generation

### For AI Assistants

**Before proposing a language feature:**
1. Estimate implementation effort × 2
2. Check if existing features can solve the problem
3. Consider helper functions or library additions instead
4. If language change is necessary, create a bead with full analysis

**When implementing a language feature:**
1. Implement in C first (reference implementation)
2. Test thoroughly with bootstrap process
3. Implement in NanoLang (self-hosted)
4. Verify self-hosting still works: `make bootstrap`
5. Both implementations must pass `make test`

## Shadow Test Policy ⚠️ MANDATORY ⚠️

### Critical Design Principle
**Shadow tests are MANDATORY for ALL NanoLang code. This is non-negotiable.**

### Coverage Requirements
Shadow tests must be included for:
- ✅ **ALL functions in core library code**
- ✅ **ALL functions in application code**
- ✅ **ALL functions in example code**
- ✅ **ALL utility and helper functions**
- ✅ **ALL demonstration programs**
- ❌ **ONLY EXCEPTION**: `extern` functions (C FFI)

### Why This Matters
1. **Correctness**: Functions are validated at compile time
2. **Documentation**: Tests show expected behavior
3. **LLM Training**: Examples teach proper practices
4. **Self-Hosting**: Compiler validates itself

### Current State
Many examples show "missing shadow test" warnings. **These are NOT false positives** - they represent technical debt that should be addressed.

### Action Required
When you see:
```
Warning: Function 'foo' is missing a shadow test
```

**This means**: Add a shadow test for `foo`. Do not ignore these warnings.

### For AI Assistants
When generating ANY NanoLang code (including examples), ALWAYS include shadow tests for every function. This is part of the language's core design, not optional.

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
