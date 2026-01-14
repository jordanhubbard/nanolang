# Known Issues in NanoLang Compiler

This document tracks known bugs and limitations in the NanoLang compiler that need to be addressed.

## Critical Bugs

### 1. Expression Statements Not Validated (Parser/Typechecker Gap)

**Severity:** High  
**Component:** Parser/Typechecker  
**Discovered:** 2026-01-14

**Description:**
The compiler allows standalone expressions as statements without validating that they have side effects or are meaningful. This means syntactically invalid code can compile successfully.

**Example - Should FAIL but compiles with only warnings:**
```nano
fn main() -> int {
    let x: int = 5
    let y: int = 10
    
    # Invalid: just identifiers doing nothing
    x y
    
    return 0
}
```

**What happens:**
- NanoLang compiler: Compiles with C compiler warnings about "unused value"
- Should be: **Compilation error** - standalone expressions without side effects are not valid statements

**Root Cause:**
The parser allows any expression to be used as a statement without validation. The typechecker doesn't verify that expression statements:
1. Are function calls (which may have side effects)
2. Are assignments/mutations
3. Have any meaningful purpose

**Impact:**
- LLM-generated code with syntax errors (like deleted function names) can compile
- Makes debugging harder because invalid code silently compiles
- Reduces compiler's ability to catch programmer errors
- False sense of correctness when code "compiles successfully"

**Files Involved:**
- `src/parser.c` line ~2550: "Try to parse as expression statement"
- `src/typechecker.c` line ~2919: "Expression statement" handling
- `src/transpiler.c`: Expression statement transpilation

**Potential Fix:**
Add validation in typechecker for expression statements:
1. If expression is a function call -> allow (may have side effects)
2. If expression is assignment/set -> allow (mutation)
3. If expression is pure identifier/literal -> **ERROR** "Expression statement has no effect"
4. For other operators, could warn or error

**Related Issues:**
- Self-hosted typechecker also has gaps (see tests/selfhost/test_function_arg_type_errors.nano)
- Need comprehensive expression statement validation rules

**Priority:** High - this masks real bugs and reduces code quality

---

## Self-Hosted Compiler Gaps

### 2. Self-Hosted Typechecker Misses Function Argument Type Errors

**Severity:** Medium  
**Component:** Self-hosted typechecker (src_nano/)  
**Status:** Test disabled in CI

**Description:**
The self-hosted compiler's typechecker doesn't properly validate function argument types, allowing type mismatches that the C reference compiler correctly rejects.

**Example:**
```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

fn main() -> int {
    let label: string = "oops"
    return (add label 10)  # Should fail: passing string to int parameter
}
```

**What happens:**
- C reference compiler: Correctly rejects with type error
- Self-hosted compiler: Compiles (incorrectly)

**Workaround:**
Test `test_function_arg_type_errors.nano` disabled in CI until fixed

**Priority:** Medium - self-hosted compiler is not primary yet

---

## Documentation

Last updated: 2026-01-14
Maintainer: NanoLang Team
