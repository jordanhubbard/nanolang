# Requirements to Make dep_locator.nano Work

## Current Status: Type Checker Issues

The `dep_locator.nano` implementation is **syntactically correct** but fails during **type checking** with these errors:

```
Error at line 56, column 9: Type mismatch in let statement
Error at line 80, column 9: Type mismatch in let statement
```

## Root Cause Analysis

### The Failing Code

```nano
fn find_include_dir(prefixes: array<string>, lib_name: string) -> string {
    let mut i: int = 0
    while (< i (array_length prefixes)) {
        let prefix: string = (at prefixes i)  // ← ERROR at line 56
        let temp1: string = (str_concat prefix "/include/")
        // ...
    }
}
```

### What's Happening

The type checker cannot infer that `(at prefixes i)` returns `string` when `prefixes` is `array<string>`.

### Type Checker Investigation

From `src/typechecker.c`:

```c
/* at(arr: array<T>, index: int) -> T */
func.name = "at";
func.params = NULL;
func.param_count = 2;
func.return_type = TYPE_UNKNOWN;  /* Will be determined by array element type */
```

The type checker has **special handling** for `at()`:

```c
if (strcmp(expr->as.call.name, "at") == 0) {
    /* at(array, index) returns the element type of the array */
    if (expr->as.call.arg_count >= 1) {
        ASTNode *array_arg = expr->as.call.args[0];
        
        /* Check if it's an array literal - get element type from it */
        if (array_arg->type == AST_ARRAY_LITERAL && ...) {
            Type elem_type = check_expression(...);
            return elem_type;
        }
        
        /* Check if it's a variable - look up its element type */
        if (array_arg->type == AST_IDENTIFIER) {
            // ... lookup logic ...
        }
    }
}
```

**The Problem:** The type inference for `at()` works for:
- ✅ Array literals: `(at [1, 2, 3] 0)` → returns `int`
- ✅ Local variables: Works when array is a local variable
- ❌ **Function parameters**: **Fails** when array is a function parameter

The type checker isn't properly extracting the element type from function parameter type annotations like `prefixes: array<string>`.

## What Needs to Be Fixed

### Fix #1: Function Parameter Type Extraction in `at()` Type Checking

**File:** `src/typechecker.c`

**Current Issue:** When the first argument to `at()` is an identifier that refers to a function parameter, the type checker needs to:

1. Look up the identifier in the environment
2. Check if it's a function parameter
3. Extract its type (e.g., `array<string>`)
4. Parse the generic type to get the element type (e.g., `string`)
5. Return that element type

**Required Changes:**

```c
// In check_expression() for AT function calls
if (strcmp(expr->as.call.name, "at") == 0) {
    if (expr->as.call.arg_count >= 1) {
        ASTNode *array_arg = expr->as.call.args[0];
        
        if (array_arg->type == AST_IDENTIFIER) {
            // Look up the variable/parameter
            VariableInfo *var_info = env_lookup_variable(env, array_arg->as.identifier);
            
            if (var_info && var_info->type.base_type == TYPE_ARRAY) {
                // Extract element type from array<T>
                if (var_info->type.array_element_type) {
                    return *var_info->type.array_element_type;
                }
            }
        }
    }
    
    return (Type){.base_type = TYPE_UNKNOWN};
}
```

### Fix #2: Improve Generic Type Representation

**File:** `src/nanolang.h`

**Current Issue:** The `Type` struct needs better support for generic types:

```c
typedef struct Type {
    TypeKind base_type;
    struct Type *array_element_type;  // For array<T>
    // ... other fields ...
} Type;
```

**Required:** Ensure that when parsing `array<string>`, the `array_element_type` pointer is properly set to a `Type` with `base_type = TYPE_STRING`.

### Fix #3: Environment Variable Lookup Enhancement

**File:** `src/env.c`

**Required:** Ensure that `env_lookup_variable()` returns complete type information including generic parameters for function parameters.

## Alternative Workarounds (If Type System Fixes Are Too Complex)

### Workaround #1: Use Array Literals in Scope

Instead of passing arrays as parameters, construct them inline:

```nano
fn find_include_dir(lib_name: string) -> string {
    let prefixes: array<string> = ["/usr", "/usr/local", "/opt/homebrew"]
    let mut i: int = 0
    while (< i (array_length prefixes)) {
        let prefix: string = (at prefixes i)  // Should work!
        // ...
    }
}
```

This would work because the type checker handles array literals correctly.

### Workaround #2: Add String Interpolation

Instead of nested `str_concat`, add string interpolation:

```nano
let include_path: string = "${prefix}/include/${lib_name}"
```

This would eliminate the need for complex nested expressions.

### Workaround #3: Use Index-Based Access

If the language supports it, add array indexing syntax:

```nano
let prefix: string = prefixes[i]  // Instead of (at prefixes i)
```

## Testing the Fix

Once the type checker is fixed, test with:

```bash
# Should compile and run without errors
DEP_LOCATOR_NAME=SDL2 ./bin/nano modules/tools/dep_locator.nano --call main

# Should output valid JSON
{
  "name": "SDL2",
  "found": true,
  "origin": "heuristic",
  "include_dirs": ["/opt/homebrew/include/SDL2"],
  "library_dirs": ["/opt/homebrew/lib"],
  "libraries": ["SDL2"]
}
```

## Priority Assessment

### High Priority (Blocking Many Use Cases)
- **Generic type inference for function parameters** - This affects ANY function that takes `array<T>` as a parameter and uses `at()` to access elements
- Many real-world programs need this pattern

### Medium Priority (Nice to Have)
- **String interpolation** - Makes string building much cleaner
- **Array index syntax** - More familiar to programmers from other languages

### Low Priority (Workarounds Available)
- The current `dep_locator.sh` shell version works perfectly
- Can wait until type system is more mature

## Estimated Complexity

### Type Checker Fix (Recommended)
**Complexity:** Medium
**Time Estimate:** 2-4 hours
**Benefit:** Fixes a whole class of programs, not just dep_locator

**Steps:**
1. Add proper generic type tracking in `Type` struct
2. Update `env_lookup_variable()` to include full type info
3. Enhance `at()` type checking to handle function parameters
4. Add test cases for `at()` with function parameter arrays
5. Verify doesn't break existing code

### Alternative: String Interpolation
**Complexity:** Medium-High
**Time Estimate:** 4-8 hours
**Benefit:** Much cleaner code overall

Would require:
1. Lexer changes to recognize `${...}` syntax
2. Parser changes to build interpolation AST nodes
3. Type checker support
4. Transpiler support (generate `str_concat` calls)
5. Interpreter support

## Recommendation

**Short Term:** Keep using `dep_locator.sh` (working perfectly)

**Medium Term:** Fix the type checker's generic type inference for function parameters
- This is a fundamental capability the language needs
- Will enable many more programs to work correctly
- Relatively contained change to typechecker.c

**Long Term:** Add string interpolation and array index syntax
- Makes the language more ergonomic
- Reduces cognitive load for new users
- Standard features in modern languages

## Related Issues

This same type inference issue likely affects:
- Any function taking `array<T>` and using `at()`
- Possibly other generic types in the future (e.g., `Option<T>`, `Result<T, E>`)
- Method calls on generic types

Fixing this will improve the overall type system robustness.

## Conclusion

**To make dep_locator.nano work:**

**Minimum Requirement:**
- Fix type inference for `(at array<T> index)` when array is a function parameter

**Full Solution:**
- Fix generic type inference across the type checker
- Ensure function parameter types properly track element types
- Test with various generic type scenarios

**Current Status:**
- dep_locator.sh works perfectly ✅
- dep_locator.nano blocked by type system limitation ❌
- Not blocking progress - shell version is production-ready ✅


