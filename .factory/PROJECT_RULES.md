# Nanolang Project Rules for AI Assistants

## Task Tracking (Primary Project Management)

Issues are tracked in the **MAC hub task ledger** (`mac task`), not bd/beads.
Use `mac task` for the issue lifecycle — do NOT use `bd`, TodoWrite, TaskCreate,
or markdown TODO lists.

### Session Start Protocol
**CRITICAL**: At the start of EVERY session, run:
```bash
mac task ready --limit 10
```
This shows all open issues/tasks ready to be worked on. Use this to understand current project priorities and work items.

### Issue Management
**USE `mac task` FOR ALL PLANNING** instead of creating markdown files:

- **Creating Issues**: `mac task create "<title>" --description "..." --priority [0-4] --kind [code|report]`
- **Viewing Issues**: `mac task ready --limit 10` (ready to work) or `mac task show <id>` (one issue)
- **Claiming Work**: `mac task claim <id> <agent_id>`
- **Closing Issues**: `mac task close <id> --reason "..."`

For content that is awkward to quote on the shell, use `--description-file <path>`
(or `-` for stdin).

### Priority Levels
- **P0 (0)**: Critical - Drop everything
- **P1 (1)**: High - Next up
- **P2 (2)**: Medium - Soon
- **P3 (3)**: Low - Backlog
- **P4 (4)**: Nice to have

### Workflow
1. Start session: `mac task ready --limit 10` to see what's ready
2. Pick an issue or create new: `mac task create "..."`
3. Claim work: `mac task claim <id> <agent_id>`
4. Complete: `mac task close <id> --reason "Brief completion summary"`
5. Create follow-ups if needed: `mac task create "..."`

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

## Build Timeout Requirements ⚠️ CRITICAL ⚠️

### Infinite Loop Detection

**RULE**: ALL compilation commands MUST include timeout detection to catch infinite loops.

### The Problem

The NanoLang compiler can enter infinite loops during:
- Lexer tokenization errors
- Parser error recovery
- Type inference cycles
- Code generation recursion

These loops print the same error repeatedly and appear to succeed if you only check truncated output.

### Required Build Pattern

```bash
# WRONG - can hang forever
./bin/nanoc file.nano -o output

# WRONG - truncating output hides the loop
./bin/nanoc file.nano 2>&1 | head -20

# CORRECT - with timeout detection
perl -e 'alarm 30; exec @ARGV' ./bin/nanoc file.nano -o output

# OR check for repeated errors
./bin/nanoc file.nano 2>&1 | tee /tmp/build.log
if grep -q "Error.*Error.*Error" /tmp/build.log; then
    echo "ERROR: Infinite loop detected in compilation"
    exit 1
fi
```

### For AI Assistants

When running ANY compilation:
1. **Set a timeout**: 30 seconds for simple files, 60s for complex
2. **Check for loops**: Look for repeated identical error lines
3. **Verify success**: Check exit code AND output for errors
4. **Never use head/tail**: These hide infinite loops

**Example Safe Build:**
```bash
# Compile with timeout and loop detection
COMPILE_OUTPUT=$(perl -e 'alarm 30; exec @ARGV' \
    ./bin/nanoc examples/file.nano -o bin/output 2>&1) || {
    echo "ERROR: Compilation failed or timed out"
    echo "$COMPILE_OUTPUT" | head -20
    exit 1
}

# Check for infinite loops (same error repeated)
if echo "$COMPILE_OUTPUT" | grep -E "^Error.*column.*Unexpected" | \
   uniq -c | grep -q " [3-9][0-9]* "; then
    echo "ERROR: Infinite loop detected"
    exit 1
fi

# Verify binary was created
if [ ! -f bin/output ]; then
    echo "ERROR: Binary not created despite exit 0"
    exit 1
fi
```

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
**RULE**: DO NOT create planning markdown files. Use `mac task` issues instead.

#### Allowed Top-Level Files ONLY:
- `README.md`
- `CONTRIBUTING.md`
- `CONTRIBUTORS.md`
- `LICENSE`
- `MEMORY.md`
- `SPEC_AUDIT.md`

#### Files That Belong in `planning/` (Legacy - Prefer `mac task`):
- Existing design documents (don't delete, but create new ones as `mac task` issues)
- Implementation plans (create as `mac task` issues going forward)
- Status reports (use `mac task` updates instead)

#### Rationale:
- The MAC hub task ledger provides structured issue tracking with dependencies, priorities, and status
- No need to manually manage markdown files
- Better visibility and searchability
- Integrated with the shared team workflow

## Cross-Tool Integration

- **Factory Droid**: Uses this file as primary rules source
- **Cursor AI**: `.cursorrules` in the project root references the `mac task` workflow
- **Claude**: This file is loaded at session start
