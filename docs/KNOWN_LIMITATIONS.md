# Known Limitations

## Function Variables (First-Class Functions)

**Status:** Partial Support

### What Works ✅
- Functions as parameters: `fn apply(f: fn(int)->int, x: int) -> int`
- Passing functions to other functions: `(apply increment 5)`
- Functions returning function references: `fn get_func() -> fn(int)->int`

### What Doesn't Work ❌
- **Function variables**: Storing functions in variables and calling them causes memory corruption
  ```nano
  let f: fn(int) -> int = increment  # Variable assignment works
  let result: int = (f 10)            # Calling through variable CRASHES
  ```

### Technical Details
- **Root Cause:** Memory management issue in function value cleanup
- **Symptom:** Segfault when returning from a function that has a function variable in scope
- **Location:** `eval.c` function scope cleanup and `env.c` free_environment
- **Impact:** Function variables cannot be used reliably

### Workaround
Use function parameters and returns instead of variables:
```nano
# DON'T DO THIS (crashes):
fn bad_example() -> int {
    let f: fn(int) -> int = increment
    return (f 10)  # CRASH!
}

# DO THIS (works):
fn good_example(f: fn(int) -> int) -> int {
    return (f 10)  # OK - function as parameter
}

fn call_it() -> int {
    return (good_example increment)  # OK - direct function reference
}
```

### Future Fix
This requires redesigning how function values are stored and cleaned up to avoid:
1. Double-free of function_name string
2. Double-free of FunctionSignature
3. Potential stack corruption during cleanup

The issue is complex and requires careful analysis of Value copy semantics and ownership.

## Top-Level Constants with Uppercase Names in Conditionals

**Status:** FIXED in parser.c

This was causing issues but has been resolved by improving the struct literal detection heuristic.
