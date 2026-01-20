# NanoLang Agent Rules

## Module MVP Requirements

Every module MUST include a minimal, standalone MVP program and snippet:

- `modules/<name>/mvp.nano` - runnable smoke test
- `modules/<name>/mvp.md` - snippet wrapper for the user guide

The MVP should:

- compile without user input
- avoid long-running loops
- include shadow tests for all functions it defines
- use `unsafe { ... }` for extern calls

## Documentation Integration

Module MVP snippets are appended to the Modules chapter during HTML build.
The user guide does not compile module MVP snippets during doc generation.

## Testing & Build Targets

- `make test` runs user guide snippets that are not module MVPs
- `make module-mvp` compiles each module MVP as a smoke test
- `make examples` should not be required to compile module MVPs

## Future Module Checklist

When adding a new module, include:

- `mvp.nano` and `mvp.md`
- at least one example under `examples/`
- a `module.manifest.json` entry that references the example
# AGENTS.md - LLM Agent Training Protocol

> **For LLM Agents:** This is your single canonical onboarding document. Follow this protocol exactly.

---

## What is NanoLang?

NanoLang is a **compiled systems programming language** designed specifically for LLM code generation. It transpiles to C for native performance.

**Key Properties:**
- **LLM-First:** Exactly ONE canonical way to write each construct
- **Prefix Notation:** All operations use `(f x y)` form - no infix operators
- **Explicit Types:** Always annotate types, minimal inference
- **Shadow Tests:** Every function has compile-time tests (mandatory)
- **C Interop:** Full FFI for calling C libraries

**Output:** Native binaries with zero runtime overhead.

---

## Canonical Style Rules (CRITICAL)

### 1. Only Write Code in Canonical Form

**There is EXACTLY ONE way to write each operation.**

Read these FIRST:
- `docs/CANONICAL_STYLE.md` - The One True Way™
- `docs/LLM_CORE_SUBSET.md` - 50-primitive core subset

**Quick Rules:**
```nano
# ✅ ALWAYS DO THIS
(+ a b)                    # Prefix notation
(cond ((< x 0) -1) (else 0))  # Expressions use cond
(+ "hello" " world")       # String concatenation
(array_get arr 0)          # Array access

# ❌ NEVER DO THIS
a + b                      # No infix operators
if/else for expressions    # Use cond instead
(str_concat "a" "b")       # Deprecated, use + instead
arr[0]                     # No subscript syntax
```

### 2. Run Shadow Tests

**Every function MUST have shadow tests. No exceptions** (except `extern` FFI functions).

```nano
fn factorial(n: int) -> int {
    # implementation
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 5) 120)
}
```

### 3. Never Invent Syntax

**If unsure about syntax, search documentation:**
1. Check `docs/CANONICAL_STYLE.md`
2. Check `docs/LLM_CORE_SUBSET.md`
3. Search `modules/index.json` for capabilities
4. Look at examples in the target module

**Never guess. Never invent. Always verify.**

---

## The LLM Core Subset

**Start here. Use ONLY these features unless user asks for more.**

### Core Types (6)
`int`, `float`, `string`, `bool`, `array<T>`, `void`

### Core Operations (~30)
- Math: `(+ a b)`, `(- a b)`, `(* a b)`, `(/ a b)`, `(% a b)`
- Compare: `(== a b)`, `(!= a b)`, `(< a b)`, `(> a b)`
- Logic: `(and a b)`, `(or a b)`, `(not x)`
- String: `(+ s1 s2)`, `(str_length s)`, `(str_equals s1 s2)`
- Array: `(array_new size val)`, `(array_get arr i)`, `(array_set arr i val)`
- I/O: `(println s)`, `(read_line)`

### Core Control Flow
```nano
# Expressions (returns value)
(cond
    ((test1) result1)
    ((test2) result2)
    (else default)
)

# Statements (side effects)
if (condition) { body } else { alternative }

# Loops
while (condition) { body }
for (let i: int = 0) (< i 10) (set i (+ i 1)) { body }
```

**Rule:** Use only core subset unless user explicitly requests advanced features (structs, enums, generics, etc.).

---

## Debugging & Instrumentation Tools

### When Code Doesn't Work: The 3-Layer Strategy

#### Layer 1: Structured Logging (stdlib/log.nano)
Use when you need to see what's happening at runtime.

```nano
from "stdlib/log.nano" import log_info, log_debug, log_error

fn process_data(x: int) -> int {
    (log_debug "processor" (+ "Input: " (int_to_string x)))
    
    if (< x 0) {
        (log_error "processor" "Negative input detected")
        return -1
    }
    
    let result: int = (* x 2)
    (log_info "processor" (+ "Result: " (int_to_string result)))
    return result
}
```

**6 log levels:** TRACE, DEBUG, INFO (default), WARN, ERROR, FATAL

**See:** `stdlib/README.md`, `docs/DEBUGGING_GUIDE.md`, `examples/logging_levels_demo.nano`

#### Layer 2: Shadow Tests (Mandatory)
Every function MUST have shadow tests. No exceptions.

When shadow tests fail, you'll see:
```
========================================
Shadow Test Assertion Failed
========================================
Location: line 9, column 5
Expression type: operator call with 2 arguments

Context: Variables in current scope:
  x: int = 5
  y: int = 7
  result: int = 12
========================================
```

**Enhanced context shows variable values automatically.**

#### Layer 3: Coverage & Tracing (stdlib/coverage.nano)
Use when you need to verify code paths or find untested areas.

```nano
from "stdlib/coverage.nano" import coverage_init, coverage_record, coverage_report

fn my_function(x: int) -> int {
    (coverage_record "my_file.nano" 10 5)
    return (* x 2)
}

fn main() -> int {
    (coverage_init)
    let result: int = (my_function 5)
    (coverage_report)  # Shows which lines executed
    return 0
}
```

**See:** `stdlib/README.md`, `examples/coverage_demo.nano`

### Error Recovery Protocol

When compilation fails:

1. **Read the error message carefully** - Line, column, type
2. **Check error code** (if using JSON diagnostics)
3. **Make MINIMAL fix** - One error at a time
4. **Recompile and iterate**

**Structured JSON errors** available via C API (see `src/json_diagnostics.h`)

**Complete workflow:** `docs/SELF_VALIDATING_CODE_GENERATION.md`

### Compiler Inspection Flags

- Use `-fshow-intermediate-code` to print generated C to stdout.
- Prefer this over `--keep-c` when you only need to inspect output.

---

## Module Selection Protocol

### When Given a User Goal:

#### Step 1: Extract Keywords and Capabilities
From the user's prompt, identify:
- **IO Surfaces:** terminal, window, files, network, audio?
- **Capabilities (verbs):** render_window, parse_json, query_data?
- **Keywords:** game, database, web server, graphics?

#### Step 2: Search modules/index.json
```bash
# Look for modules matching capabilities/keywords
# Each module has:
#   - use_when: rules for when to pick it
#   - capabilities: action verbs
#   - keywords: goal words
#   - examples: canonical programs
```

#### Step 3: Pick ONE Module
- Choose the **best single module** for the task
- Don't mix modules unless user explicitly asks
- Prefer `stable` over `experimental`
- Check `avoid_when` to rule out wrong fits

#### Step 4: Follow Module's Examples First
- Copy the **closest example** from the module
- Adapt it to the user's specific goal
- Use the module's established patterns

**Example:**
```
User: "Make a bouncing ball animation"
→ Keywords: animation, window, render
→ IO Surfaces: window
→ Capabilities: render_2d, create_window
→ Search index → SDL module matches
→ Use examples/sdl_particles.nano as template
```

---

## Deterministic Dependency Protocol

### Prefer Lockfiles
- If `module.lock` exists, use it (reproducible builds)
- Never install deps without logging what changed

### Use Dry-Run
```bash
# Before installing, show what would happen
./bin/nanoc --dry-run file.nano
# → Would install: sdl2 via brew
```

### Log Installs
- Always output: "Installing X via Y"
- Show versions being installed
- Show any files being modified

**Rule:** Builds must be reproducible and auditable.

---

## Examples-First Development

### Protocol:
1. **Search for closest example** in target module
2. **Copy it as starting point**
3. **Adapt to user's goal**
4. **Add shadow tests immediately**
5. **Compile and iterate**

### Example:
```
User: "Make a snake game"
→ Module: ncurses (terminal UI + keyboard)
→ Closest example: examples/ncurses_snake.nano
→ Copy, adapt, test
```

**Never write from scratch if an example exists.**

---

## Error Repair Loop

### Protocol:
```
1. Compile: ./bin/nanoc file.nano -o bin/output
2. Read error message carefully
3. Make MINIMAL fix (one thing at a time)
4. Rerun tests
5. Repeat until tests pass
```

### Common Errors:
- **"Undefined function X"** → Check module imports
- **"Type mismatch"** → Verify all type annotations
- **"Missing shadow test"** → Add shadow block
- **Infinite loop** → Use timeout: `perl -e 'alarm 30; exec @ARGV' ./bin/nanoc ...`

**Never make large changes. One error = one minimal fix.**

---

## The Agent Algorithm

**When given a task, follow these 8 steps exactly:**

### 1. Restate Goal in 1 Sentence
```
"User wants: [concise goal]"
```

### 2. Identify Required IO Surfaces
Check which the program needs:
- `terminal` - Text output, cursor control
- `window` - Graphical display
- `files` - Read/write disk
- `network` - HTTP, sockets
- `audio` - Sound playback

### 3. Identify Required Capabilities (Verbs)
What actions must the program perform?
- `render_2d`, `handle_input`, `query_data`, `parse_json`, etc.

### 4. Search modules/index.json
Find modules matching:
- Capabilities from step 3
- Keywords from user prompt
- IO surfaces from step 2

### 5. Choose Best Single Module
Apply rules:
- Check `use_when` - does it match?
- Check `avoid_when` - any red flags?
- Check `stability` - prefer `stable`
- Check `examples` - are there relevant ones?

### 6. Write NanoLang in Canonical Syntax Only
- Use `docs/CANONICAL_STYLE.md` as reference
- Use only `docs/LLM_CORE_SUBSET.md` features (unless user asks for advanced)
- Follow module's example patterns
- Never invent syntax

### 7. Add Shadow Tests Immediately
```nano
fn my_function(x: int) -> int {
    return (* x 2)
}

shadow my_function {
    assert (== (my_function 5) 10)
    assert (== (my_function 0) 0)
    assert (== (my_function -3) -6)
}
```

### 8. Iterate Until Tests Pass
```bash
perl -e 'alarm 30; exec @ARGV' ./bin/nanoc file.nano -o bin/output
./bin/output  # Run shadow tests
# → Fix errors → Recompile → Repeat
```

---

## Complete Example Walkthrough

**User Request:** "Make a program that displays a bouncing ball in a window"

### Step 1: Restate Goal
"User wants: Display animated bouncing ball in graphical window"

### Step 2: IO Surfaces
- `window` ✓ (graphical display)
- `terminal` ✗ (not needed)
- `files` ✗ (not needed)

### Step 3: Capabilities
- `create_window`
- `render_2d`
- `draw_shapes` (circle for ball)
- Render loop with timing

### Step 4: Search modules/index.json
```json
{
  "name": "sdl",
  "use_when": [
    "User wants a graphical window, real-time animation, sprites",
    "User says 'game', 'render loop', 'frame', 'FPS', 'sprite', '2D graphics'"
  ],
  "capabilities": ["create_window", "render_2d", "draw_shapes"],
  "io_surfaces": ["window"]
}
```
**Match! SDL is the right module.**

### Step 5: Choose Module
- SDL matches all requirements ✓
- `stability: "stable"` ✓
- Examples exist: `examples/sdl_particles.nano` (similar physics) ✓

### Step 6: Write Canonical Code
```nano
from "modules/sdl/sdl.nano" import init, create_window, destroy_window

fn main() -> int {
    (init)
    let window: Window = (create_window "Bouncing Ball" 800 600)
    
    let mut x: float = 400.0
    let mut y: float = 300.0
    let mut vx: float = 2.0
    let mut vy: float = 2.0
    
    while (is_running) {
        # Update physics
        set x (+ x vx)
        set y (+ y vy)
        
        # Bounce off walls
        if (or (< x 0.0) (> x 800.0)) {
            set vx (* vx -1.0)
        }
        if (or (< y 0.0) (> y 600.0)) {
            set vy (* vy -1.0)
        }
        
        # Render
        (clear_screen window)
        (draw_circle window x y 20)
        (present window)
        (delay 16)
    }
    
    (destroy_window window)
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

### Step 7: Shadow Tests Added ✓
(See `shadow main` block above)

### Step 8: Iterate
```bash
perl -e 'alarm 30; exec @ARGV' ./bin/nanoc bouncing_ball.nano -o bin/bouncing_ball
# → Compiles successfully
./bin/bouncing_ball
# → Tests pass, program runs
```

**Done! ✓**

---

## Critical Reminders

1. **ONE canonical form per operation** - Never deviate
2. **Shadow tests are MANDATORY** - No exceptions
3. **Examples first** - Copy closest, adapt
4. **Core subset first** - Advanced features only on request
5. **Module selection is critical** - Follow the 8-step algorithm
6. **Never invent syntax** - Verify in docs
7. **Minimal fixes** - One error at a time
8. **Deterministic builds** - Log all installs

---

## Quick Reference Card

```
User request
    ↓
[1] Restate goal
[2] Identify IO surfaces (terminal/window/files/network/audio)
[3] Identify capabilities (verbs)
[4] Search modules/index.json
[5] Pick best module (check use_when, avoid_when, examples)
[6] Write canonical NanoLang (prefix notation, core subset)
[7] Add shadow tests
[8] Iterate (compile → fix → test)
    ↓
Working program ✓
```

---

## For Training

**An LLM agent has mastered NanoLang when:**
1. ✅ Always uses canonical syntax (never invents)
2. ✅ Always adds shadow tests (no prompting needed)
3. ✅ Follows 8-step algorithm for every task
4. ✅ Picks correct module on first try (via index search)
5. ✅ Uses core subset by default (advanced on request)
6. ✅ Copies examples rather than writing from scratch
7. ✅ Makes minimal fixes (no large refactors per error)
8. ✅ Produces reproducible builds (logs deps)

---

## Resources

### Core Documentation (Read These First)
- `AGENTS.md` **(this file)** - Your onboarding protocol
- `docs/CANONICAL_STYLE.md` - Syntax reference (the ONE true way)
- `docs/LLM_CORE_SUBSET.md` - 50-primitive core subset
- `MEMORY.md` - Patterns and idioms
- `spec.json` - Formal language specification

### Debugging & Validation (Critical for Self-Correction)
- `docs/DEBUGGING_GUIDE.md` - 3-layer debugging strategy
- `docs/SELF_VALIDATING_CODE_GENERATION.md` - Complete error recovery workflow
- `docs/PROPERTY_TESTING_GUIDE.md` - Property-based testing reference
- `stdlib/README.md` - Standard library index (log, coverage, etc.)

### Standard Library
- `stdlib/log.nano` - Structured logging (6 levels, categories)
- `stdlib/coverage.nano` - Coverage, timing, tracing APIs
- `stdlib/regex.nano` - Regular expressions
- `stdlib/StringBuilder.nano` - Efficient string building
- `docs/STDLIB.md` - Built-in functions reference

### Modules & Examples
- `modules/index.json` - Searchable module index (SDL, ncurses, etc.)
- `examples/` - Working demonstration programs
- `docs/MODULE_SYSTEM.md` - Module creation guide

### Quick Reference Card

| Task | Resource |
|------|----------|
| **Syntax questions** | `docs/CANONICAL_STYLE.md` |
| **Code not working** | `docs/DEBUGGING_GUIDE.md` → use `stdlib/log.nano` |
| **Shadow test failure** | Read variable context in error output |
| **Error recovery** | `docs/SELF_VALIDATING_CODE_GENERATION.md` |
| **Property tests** | `docs/PROPERTY_TESTING_GUIDE.md` |
| **Coverage tracking** | `stdlib/coverage.nano` |
| **Find a module** | `modules/index.json` |
| **Built-in functions** | `docs/STDLIB.md` |
| **Examples** | `examples/` directory |

**Start here. Follow the algorithm. Write canonical code. Use debugging tools when stuck.**

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
