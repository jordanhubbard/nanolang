# Extern Functions in Shadow Tests

## Current Behavior

Shadow tests that use extern functions are **SKIPPED** when running in interpreter mode.

### Why?

- The interpreter (`bin/nano`) directly evaluates the AST
- Extern functions require compiled C code and linking
- The interpreter cannot dynamically link and call extern functions

### What About the Compiler?

When you compile with `bin/nanoc`, shadow tests are run via the **interpreter** before transpilation. This means:

```
nanoc input.nano -o output
  ↓
1. Lex & Parse ✅
2. Type Check ✅  
3. Run Shadow Tests (via interpreter) ← extern functions SKIPPED here
4. Transpile to C ✅
5. Compile C (with extern functions) ✅
```

## Workarounds

### Option 1: Separate Test File (Recommended)

Instead of shadow tests, create a separate test file that compiles:

```nanolang
/* my_module.nano */
extern fn external_func(x: int) -> int

fn my_wrapper(x: int) -> int {
    return (external_func x)
}

/* No shadow test here - uses extern */
```

```nanolang
/* test_my_module.nano */
import "my_module.nano"

fn main() -> int {
    let result: int = (my_wrapper 42)
    assert (== result expected_value)
    return 0
}
```

Then compile and run:
```bash
nanoc test_my_module.nano -o test && ./test
```

### Option 2: Mock in Shadow Tests

Create a non-extern version for testing:

```nanolang
extern fn real_func(x: int) -> int

/* Mock for testing */
fn mock_func(x: int) -> int {
    return (* x 2)  /* Simulated behavior */
}

fn my_logic(x: int, func: fn(int) -> int) -> int {
    return (+ (func x) 10)
}

shadow my_logic {
    /* Test with mock instead of extern */
    let result: int = (my_logic 5 mock_func)
    assert (== result 20)  /* (5 * 2) + 10 = 20 */
}
```

### Option 3: Conditional Shadow Tests

Use a flag to enable/disable extern-dependent tests:

```nanolang
let ENABLE_EXTERN_TESTS: bool = false

extern fn external_func(x: int) -> int

fn my_func(x: int) -> int {
    if ENABLE_EXTERN_TESTS {
        return (external_func x)
    } else {
        return x  /* Fallback */
    }
}

shadow my_func {
    /* This will work since we use fallback */
    let result: int = (my_func 42)
    assert (== result 42)
}
```

## Future Enhancement

**Proposed**: Compile and run extern shadow tests automatically

When a shadow test uses extern functions:
1. Extract the shadow test code
2. Transpile to C (with dependencies)
3. Compile with appropriate linking flags
4. Run the compiled test
5. Report results

This would enable full testing without workarounds.

**Implementation Complexity**: Medium
**Timeline**: Phase 3 or later

## Current Status

- ✅ Shadow tests work for all non-extern code
- ✅ Extern functions work in compiled code
- ⚠️  Shadow tests with extern functions are skipped
- ⬜ Automatic compilation of extern shadow tests (planned)

## Examples

### ✅ Works - No Extern

```nanolang
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)  /* ✓ Runs in interpreter */
}
```

### ⚠️ Skipped - Uses Extern

```nanolang
extern fn sqrt(x: float) -> float

fn distance(x: float, y: float) -> float {
    return (sqrt (+ (* x x) (* y y)))
}

shadow distance {
    let d: float = (distance 3.0 4.0)
    /* SKIPPED - uses extern sqrt */
    assert (== d 5.0)
}
```

### ✅ Workaround - Separate Test

```nanolang
/* distance.nano */
extern fn sqrt(x: float) -> float

fn distance(x: float, y: float) -> float {
    return (sqrt (+ (* x x) (* y y)))
}
```

```nanolang
/* test_distance.nano - compiles and runs */
import "distance.nano"

fn main() -> int {
    let d: float = (distance 3.0 4.0)
    assert (== d 5.0)
    (println "✓ Distance test passed")
    return 0
}
```

```bash
$ nanoc test_distance.nano -o test -lm
$ ./test
✓ Distance test passed
```

## Best Practices

1. **Prefer Pure Functions**: Write logic that doesn't depend on extern functions when possible
2. **Separate Concerns**: Keep extern calls isolated in wrapper functions
3. **Mock for Testing**: Create test versions that don't use extern
4. **Integration Tests**: Use separate compiled test files for extern functionality
5. **Document**: Note which functions can't be shadow tested

## Summary

Shadow tests are a great feature for inline testing, but they have limitations with extern functions due to the interpreter architecture. Use the workarounds above, or wait for Phase 3 when we plan to add automatic compilation for extern shadow tests.
