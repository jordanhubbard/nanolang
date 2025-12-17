# Compiler vs Interpreter Parity Analysis

**Date:** 2025-12-17  
**Issue:** nanolang-695

---

## Executive Summary

**CRITICAL GAP IDENTIFIED:** The interpreter (`bin/nano`) does **NOT** support shadow tests, while the compiler (`bin/nanoc`) does. This is the primary feature parity gap.

### Test Results

| Mode | Shadow Test Support | Test Pass Rate |
|------|---------------------|----------------|
| **Compiler** | ✅ YES | 74/74 (100%) |
| **Interpreter** | ❌ NO | 0/65 shadow tests (0%)* |

\* _Tests execute successfully but don't run shadow test validation_

---

## Detailed Findings

### 1. Shadow Test Support

#### Compiler Implementation (`src/main.c`)
```c
/* Phase 5: Shadow-Test Execution */
if (!run_shadow_tests(program, env)) {
    fprintf(stderr, "Shadow tests failed\n");
    return 1;
}
printf("✓ Shadow tests passed\n");
```

**Status:** ✅ Fully implemented and working

#### Interpreter Implementation (`src/interpreter_main.c`)
```c
/* Phase 5: Interpret */
if (!run_program(program, env)) {
    fprintf(stderr, "Interpretation failed\n");
    return 1;
}
/* NO SHADOW TEST EXECUTION */
```

**Status:** ❌ Shadow tests are never executed

### 2. Impact Assessment

**Shadow tests are used extensively:**
- All 74 test files use shadow tests for validation
- Shadow tests are the primary testing mechanism in nanolang
- Without shadow tests, the interpreter cannot verify correctness

**Example test file structure:**
```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)  /* ← THIS NEVER RUNS IN INTERPRETER */
}
```

### 3. Why This Matters

| Aspect | Impact |
|--------|--------|
| **Testing** | Cannot run test suite with interpreter |
| **Development** | Developers can't use interpreter for TDD |
| **Education** | Learners can't see instant feedback |
| **Parity** | Major behavioral difference between modes |

---

## Root Cause

The interpreter was implemented as a **direct execution mode** for running programs, while the compiler was designed to also serve as a **testing tool** via transpilation and shadow test execution.

The `run_shadow_tests()` function exists in `src/eval.c` but is only called by the compiler, not the interpreter.

---

## Recommended Solutions

### Option 1: Add Shadow Test Phase to Interpreter (RECOMMENDED)

**Complexity:** Low  
**Impact:** Solves the parity issue completely

**Implementation:**
```c
/* In src/interpreter_main.c, after Phase 5: Interpret */

/* Phase 6: Shadow Tests (if present) */
if (!run_shadow_tests(program, env)) {
    fprintf(stderr, "Shadow tests failed\n");
    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    free(source);
    gc_shutdown();
    return 1;
}
printf("All shadow tests passed!\n");

/* Phase 7: Call main() or specified function */
// ... existing code ...
```

**Benefits:**
- ✅ Achieves 100% parity with compiler
- ✅ Enables test suite to run in interpreter mode
- ✅ Uses existing `run_shadow_tests()` infrastructure
- ✅ Minimal code changes required

**Drawbacks:**
- Shadow tests run before main() (different from compiler which runs them after typechecking but before transpilation)

### Option 2: Add `--test` Flag to Interpreter

**Complexity:** Low  
**Impact:** Provides opt-in testing mode

**Implementation:**
- Add `--test` flag to interpreter options
- Only run shadow tests when flag is present
- Maintains backward compatibility

**Benefits:**
- ✅ Preserves current behavior for production use
- ✅ Enables testing when needed
- ✅ Clear separation of concerns

**Drawbacks:**
- ⚠️ Doesn't achieve automatic parity
- ⚠️ Users must remember to use `--test` flag

### Option 3: Document as Intentional Difference

**Complexity:** None  
**Impact:** Accepts the status quo

**Benefits:**
- ✅ No code changes needed
- ✅ Clear documentation of difference

**Drawbacks:**
- ❌ Perpetuates the parity gap
- ❌ Limits interpreter usefulness for testing
- ❌ Confusing for users

---

## Other Parity Observations

### Features That Work in Both Modes

The following features work identically in both compiler and interpreter:
- ✅ Basic types (int, float, bool, string)
- ✅ Functions and function calls
- ✅ Control flow (if, while, for, match)
- ✅ Structs and enums
- ✅ Unions (including generic unions)
- ✅ Arrays (static and dynamic)
- ✅ Module system and imports
- ✅ First-class functions
- ✅ Tuples

### Execution Model Differences

| Aspect | Compiler | Interpreter |
|--------|----------|-------------|
| **Execution** | Transpile to C → compile → run | Direct AST execution |
| **Performance** | Native C speed | Slower (interpreted) |
| **Startup time** | Slow (C compilation) | Fast (immediate) |
| **Error detection** | Compile-time + runtime | Runtime only |
| **Shadow tests** | ✅ Executed | ❌ Not executed |

---

## Recommendations

### Immediate Actions (P0)

1. **Implement Option 1** - Add shadow test phase to interpreter
2. **Verify test suite** - Run all 74 tests with modified interpreter
3. **Update documentation** - Document parity achievement

### Short-term Actions (P1)

1. **Add integration test** - Ensure both modes remain in parity
2. **Update CI/CD** - Run tests in both modes
3. **Benchmark** - Compare performance characteristics

### Long-term Actions (P2)

1. **Unified test runner** - Single script that tests both modes
2. **Parity dashboard** - Track feature support across modes
3. **Performance optimization** - Improve interpreter speed

---

## Testing Verification

### Verification Steps

Once shadow tests are added to the interpreter:

```bash
# Run test suite with compiler (baseline)
./tests/run_all_tests.sh
# Expected: 74/74 passing

# Run test suite with interpreter (after fix)
./tests/run_all_tests_interpreter.sh
# Expected: 74/74 passing

# Compare results
diff <(./tests/run_all_tests.sh 2>&1) <(./tests/run_all_tests_interpreter.sh 2>&1)
# Expected: No significant differences
```

---

## Conclusion

The primary parity gap between compiler and interpreter is the **absence of shadow test execution in the interpreter**. This is a straightforward fix that requires adding a single phase to the interpreter's execution pipeline.

**Recommendation:** Implement Option 1 (add shadow test phase to interpreter) to achieve 100% feature parity.

**Estimated Effort:** 1-2 hours  
**Risk Level:** Low (uses existing infrastructure)  
**Impact:** High (enables full test suite on both modes)

