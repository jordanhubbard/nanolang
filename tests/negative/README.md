# Negative Tests

This directory contains **negative test cases** - programs that should **fail to compile**.

These tests verify that the NanoLang compiler correctly **rejects** invalid code and provides helpful error messages.

## Purpose

Negative tests ensure:
- Type errors are caught at compile time
- Syntax errors are detected and reported clearly
- Undefined variables/functions are flagged
- Return type mismatches are caught
- Immutability rules are enforced
- Missing shadow tests are required

## Running Tests

```bash
cd tests/negative
./run_negative_tests.sh
```

**Expected Output:** All tests should PASS, meaning the compiler correctly rejected the invalid code.

## Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **type_errors/** | Type mismatches | Verify type system catches errors |
| **syntax_errors/** | Invalid syntax | Verify parser rejects bad syntax |
| **undefined_vars/** | Undefined symbols | Verify scope checking works |
| **mutability_errors/** | Const violations | Verify immutability rules enforced |
| **return_errors/** | Missing returns | Verify control flow analysis |
| **missing_shadows/** | No shadow tests | Verify shadow tests are mandatory |
| **builtin_collision/** | Redefining builtins | Verify stdlib protection |
| **duplicate_functions/** | Name collisions | Verify uniqueness checking |
| **scope_errors/** | Scope violations | Verify variable scoping |
| **struct_errors/** | Invalid struct ops | Verify struct field checking |

## Writing Negative Tests

1. Create a `.nano` file in the appropriate category directory
2. Add a comment explaining the expected error:
   ```nano
   # Expected error: Type mismatch
   # Cannot add int and string
   ```
3. Write code that should fail to compile
4. Include shadow tests (even though they won't run)
5. Run `./run_negative_tests.sh` to verify the test fails as expected

### Example

```nano
# Expected error: Type mismatch
# Cannot add int and string

fn bad_add(x: int, y: string) -> int {
    return (+ x y)  # Error: incompatible types
}

shadow bad_add {
    assert (== (bad_add 5 "hello") 5)
}

fn main() -> int {
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

## Test Results

Current status: **20 negative tests** - All passing ✅

The compiler correctly rejects all invalid code in these tests.

---

**See Also:**
- [ERROR_MESSAGES.md](../../docs/ERROR_MESSAGES.md) - Error message guide
- [FEATURE_COVERAGE.md](../FEATURE_COVERAGE.md) - Positive test coverage
- [run_all_tests.sh](../run_all_tests.sh) - Full test suite

**Last Updated:** January 25, 2026
