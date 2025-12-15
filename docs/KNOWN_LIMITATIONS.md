# Known Limitations

## Function Variables (First-Class Functions)

**Status:** ✅ FULLY SUPPORTED (Fixed in commits d5dceb2 and c12704f)

### What Works ✅
- Functions as parameters: `fn apply(f: fn(int)->int, x: int) -> int`
- Passing functions to other functions: `(apply increment 5)`
- Functions returning function references: `fn get_func() -> fn(int)->int`
- **Function variables**: Storing functions in variables and calling them
  ```nano
  let f: fn(int) -> int = increment  # Variable assignment works ✅
  let result: int = (f 10)            # Calling through variable works! ✅
  ```
- Calling function-typed parameters: `fn filter(predicate: fn(int)->bool, n: int) -> bool { return (predicate n) }`
- Conditional function selection: `fn select(use_add: bool) -> fn(int,int)->int { if use_add { return add } else { return mul } }`

### Fixed Issues
1. **Memory Management (d5dceb2)**: Fixed function value cleanup in `env_set_var()`, `free_environment()`, and `eval_function()` to properly use `free_function_signature()` instead of incomplete manual cleanup
2. **Type Checking (c12704f)**: Added `fn_sig` field to `TypeInfo` and stored function signatures when adding function parameters to environment, allowing proper return type inference when calling function parameters

### Technical Details
The fixes addressed:
- Proper cleanup of function_name and signature in all paths (no more double-free)
- Complete signature cleanup including param_struct_names and nested signatures
- Storage of function signatures in TypeInfo for function-typed parameters
- Return type retrieval from stored signatures during type checking

All first-class function features are now fully functional and tested!

## Top-Level Constants with Uppercase Names in Conditionals

**Status:** FIXED in parser.c

This was causing issues but has been resolved by improving the struct literal detection heuristic.
