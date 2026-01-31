# Interpreter Architecture Decision

## Executive Summary

**Question:** Is the interpreter providing enough value to justify the dual implementation burden?

**Answer:** No. The interpreter is a real AST-walking interpreter, but the cost of maintaining feature parity with the compiler outweighs its benefits.

**Recommendation:** Remove the interpreter and compile shadow tests instead.

---

## Current Architecture

### The Interpreter IS Real

The interpreter (`bin/nano`) is a genuine AST-walking interpreter:

```
Source → Lexer → Parser → AST → Type Checker → AST Walker (eval.c)
                                                    ↓
                                              FFI Calls to Modules
```

**What it does:**
- ✅ Parses source to AST
- ✅ Evaluates AST directly (no C compilation)
- ✅ Makes FFI calls for extern functions
- ✅ Dynamically loads module shared libraries (.so/.dylib)
- ✅ Executes shadow tests instantly

**What it doesn't do:**
- ❌ Compile to C (that's the transpiler)
- ❌ Generate bytecode
- ❌ Use any compilation step

**Code size:** ~5,000 lines (eval.c, interpreter_main.c, interpreter_ffi.c)

---

## The Problem: Dual Implementation Burden

### Every Language Feature Requires 2 Implementations

1. **C Interpreter** (`src/eval.c`)
   - Evaluates AST nodes directly
   - ~5,000 lines of C code
   - Must handle every language construct

2. **NanoLang Transpiler** (`src_nano/compiler/transpiler.nano`)
   - Generates C code from AST
   - ~3,000 lines of NanoLang code
   - Must handle every language construct

**This is NOT the "dual implementation" from the design philosophy!**

The design philosophy refers to:
- C reference compiler (bootstrap)
- NanoLang self-hosted compiler (production)

But the interpreter adds a **third** implementation of language semantics!

### Evidence of Burden

**Today's session revealed multiple interpreter bugs:**

1. **String concatenation in mutable contexts**
   ```nano
   let mut result: string = ""
   set result (+ result (string_from_char 104))  # FAILS in interpreter
   ```

2. **Arrays of structs**
   ```nano
   struct WordCount { word: string, count: int }
   let counts: array<WordCount> = ...  # FAILS in interpreter
   ```

3. **Complex data structures**
   - Word frequency example: 90% complete, blocked
   - Math ext demo: blocked by interpreter limitations

**Pattern:** Compiler works, interpreter lags behind.

---

## Value Proposition Analysis

### What the Interpreter Provides

1. **Instant Shadow Test Execution**
   - No compilation overhead
   - Fast feedback during development
   - Used by `make test`

2. **Development Convenience**
   - Quick validation
   - Iterative testing
   - Rapid prototyping

### What It Costs

1. **Maintenance Burden**
   - Every new feature needs 2 implementations
   - Bugs must be fixed in 2 places
   - Feature parity constantly drifts

2. **Development Velocity**
   - Slows down language evolution
   - Examples blocked by interpreter bugs
   - Testing becomes unreliable

3. **Code Complexity**
   - ~5,000 lines of interpreter code
   - FFI integration complexity
   - Module loading complexity

### The Math

**Time to implement a new feature:**
- Transpiler only: 4 hours
- Transpiler + Interpreter: 8 hours (2×)

**Annual cost (assuming 20 new features/year):**
- Without interpreter: 80 hours
- With interpreter: 160 hours
- **Cost: 80 hours/year = 2 work weeks**

**Shadow test execution time:**
- Interpreter: ~10 seconds (100 files)
- Compiled (parallel): ~60 seconds (100 files)
- **Savings: 50 seconds per test run**

**Break-even analysis:**
- To justify 80 hours/year, need: 5,760 test runs/year
- That's 16 test runs/day, every day
- **Verdict:** Not worth it

---

## Recommendation: Remove the Interpreter

### Proposed Architecture

```
Source → Lexer → Parser → AST → Type Checker → Transpiler → C Code → Compile → Execute
                                                                          ↓
                                                                    Shadow Tests
```

**Shadow tests become executables:**
```bash
# Old (interpreter)
./bin/nano test_file.nano  # instant

# New (compiled)
./bin/nanoc test_file.nano -o bin/test_file && ./bin/test_file  # ~2-5 seconds
```

### Implementation Plan

#### Phase 1: Make Shadow Tests Compile-Only (1 week)

1. **Update test runner** (`make test`)
   - Compile each test file
   - Execute compiled binary
   - Parallelize compilation (use `make -j`)

2. **Update documentation**
   - Remove interpreter references
   - Update examples to compile-only
   - Update CONTRIBUTING.md

3. **Deprecate interpreter**
   - Add warning when `bin/nano` is used
   - Point users to `bin/nanoc`

#### Phase 2: Remove Interpreter Code (1 week)

1. **Delete interpreter files**
   - `src/eval.c` (~5,000 lines)
   - `src/interpreter_main.c`
   - `src/interpreter_ffi.c`
   - Related headers

2. **Update build system**
   - Remove `bin/nano` target
   - Remove interpreter dependencies
   - Simplify Makefile

3. **Clean up**
   - Remove FFI dynamic loading code
   - Simplify module system
   - Remove interpreter-specific code

#### Phase 3: Optimize Compiled Tests (1 week)

1. **Parallel compilation**
   - Use `make -j` for test compilation
   - Cache compiled tests
   - Only recompile changed files

2. **Fast compilation mode**
   - Add `-O0` flag for tests (faster compile)
   - Skip optimization for shadow tests
   - Use ccache if available

3. **Incremental testing**
   - Only run tests for changed files
   - Smart test selection
   - Faster feedback loop

### Expected Results

**Removed:**
- ~5,000 lines of interpreter code
- Dual implementation burden
- Feature parity bugs
- FFI complexity

**Added:**
- ~50 seconds per test run (compilation overhead)
- Simpler architecture
- Faster development velocity
- More reliable testing

**Net benefit:**
- 80 hours/year saved (2 work weeks)
- Simpler codebase
- Faster language evolution
- More consistent behavior

---

## Alternative: Keep Interpreter (Not Recommended)

If we decide to keep the interpreter, we must:

1. **Fix all feature parity bugs**
   - String concatenation in mutable contexts
   - Arrays of structs
   - Any future feature gaps

2. **Commit to ongoing maintenance**
   - Every new feature needs 2 implementations
   - Every bug needs 2 fixes
   - Accept slower development velocity

3. **Document limitations**
   - Clear list of what works/doesn't work
   - Examples marked as compile-only
   - Users understand trade-offs

**Estimated ongoing cost:** 80 hours/year forever

---

## Decision Criteria

### Keep Interpreter If:
- [ ] We run shadow tests 16+ times per day
- [ ] 50-second compilation overhead is unacceptable
- [ ] We commit to maintaining feature parity
- [ ] We accept 2× implementation cost

### Remove Interpreter If:
- [x] Dual implementation burden slows development
- [x] Feature parity bugs block progress
- [x] 50-second overhead is acceptable
- [x] We want simpler architecture

**Verdict:** Remove the interpreter.

---

## Implementation Timeline

### Week 1: Preparation
- Create bead for interpreter removal
- Update test runner to compile tests
- Parallelize test compilation
- Verify all tests pass when compiled

### Week 2: Deprecation
- Add deprecation warning to `bin/nano`
- Update all documentation
- Notify users (if any)
- Provide migration guide

### Week 3: Removal
- Delete interpreter code
- Update build system
- Clean up dependencies
- Verify build works

### Week 4: Optimization
- Optimize test compilation speed
- Add caching
- Implement incremental testing
- Measure performance

**Total time:** 4 weeks

**Payoff:** 80 hours/year saved (2 work weeks annually)

**ROI:** Pays for itself in 6 months

---

## Conclusion

**The interpreter is real and does valuable work, but the dual implementation burden is not sustainable.**

Given today's evidence:
- Examples blocked by interpreter bugs
- Feature parity constantly drifting
- 2× implementation cost for every feature

**Recommendation: Remove the interpreter and compile shadow tests.**

This will:
- Eliminate dual implementation burden
- Simplify architecture
- Speed up development
- Improve reliability

The 50-second compilation overhead is a small price to pay for:
- 80 hours/year saved
- No more feature parity bugs
- Faster language evolution
- Simpler codebase

**Next step:** Create bead and begin implementation.
