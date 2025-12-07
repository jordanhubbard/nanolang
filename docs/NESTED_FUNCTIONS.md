# Nested Functions Design Document

## Status: NOT YET IMPLEMENTED

This document describes the design for nested function support in nanolang.

## Overview

Nested functions would allow functions to return other functions, enabling:
- Higher-order functions (map, filter, reduce)
- Currying and partial application
- Function factories and closures

## Proposed Syntax

```nano
# Function that returns a function
fn make_adder(x: int) -> fn(int) -> int {
    fn adder(y: int) -> int {
        return (+ x y)  # Captures x from outer scope
    }
    return adder
}

# Usage
fn main() -> int {
    let add5: fn(int) -> int = (make_adder 5)
    let result: int = (add5 10)  # result = 15
    return result
}
```

## Implementation Requirements

### 1. Parser Changes

**Current State:**
- Parser recognizes `fn(int) -> int` function type syntax
- Rejects nested function return types with error message

**Needed Changes:**
```c
// In parse_type_with_element() around line 203
case TOKEN_FN: {
    // ... existing code ...
    
    // Allow recursive function signature parsing
    if (return_type == TYPE_FUNCTION) {
        // Parse nested function signature
        fn_sig->return_fn_sig = parse_function_signature_recursive(p);
    }
}
```

### 2. Type System Changes

**Current State:**
- `TYPE_FUNCTION` exists for function types
- `FunctionSignature` struct can store `return_fn_sig`
- Not fully utilized for nested functions

**Needed Changes:**
- Extend type checker to validate nested function signatures
- Support function type matching with nested returns
- Handle closure capture analysis

### 3. AST Representation

**Current State:**
```c
typedef struct FunctionSignature {
    Type *param_types;
    int param_count;
    char **param_struct_names;
    Type return_type;
    char *return_struct_name;
    struct FunctionSignature *return_fn_sig;  // Already exists!
} FunctionSignature;
```

**Needed Changes:**
- Add closure capture information to Function struct
- Track which variables are captured from outer scope

### 4. Runtime Representation

**Needed:**
- Closure struct to hold function pointer + captured variables
- GC type for closures (`GC_TYPE_CLOSURE` already exists!)

```c
typedef struct {
    void *function_ptr;  // Pointer to generated C function
    void **captured_vars;  // Array of captured variable pointers
    int capture_count;
} Closure;
```

### 5. Transpiler Changes

**Major Work Needed:**
- Generate closure structs in C
- Transform nested functions into separate C functions
- Pass captured variables as hidden parameters
- Generate trampoline functions for closures

**Example Transformation:**

nanolang:
```nano
fn make_adder(x: int) -> fn(int) -> int {
    fn adder(y: int) -> int {
        return (+ x y)
    }
    return adder
}
```

Generated C:
```c
// Inner function with captured variable as parameter
static int64_t adder_impl(int64_t y, int64_t captured_x) {
    return y + captured_x;
}

// Closure struct
typedef struct {
    int64_t (*fn)(int64_t, int64_t);
    int64_t captured_x;
} Closure_adder;

// Outer function returns closure
static Closure_adder* make_adder(int64_t x) {
    Closure_adder *closure = gc_alloc(sizeof(Closure_adder), GC_TYPE_CLOSURE);
    closure->fn = adder_impl;
    closure->captured_x = x;
    return closure;
}

// Wrapper to call closure
static int64_t call_closure_adder(Closure_adder *closure, int64_t y) {
    return closure->fn(y, closure->captured_x);
}
```

### 6. Interpreter Changes

**Needed:**
- Support for returning function values
- Closure creation at runtime
- Variable capture from parent scope

## Challenges

1. **Closure Lifetime**: Captured variables must outlive the creating function
   - Solution: GC-managed closures

2. **Type Complexity**: Nested function signatures can be complex
   - Solution: Recursive signature matching

3. **C Code Generation**: C doesn't have closures natively
   - Solution: Generate struct + function pointer pattern

4. **Performance**: Closure calls have overhead
   - Solution: Inline when possible, optimize hot paths

## Workarounds (Current)

Until nested functions are implemented, use these patterns:

### Function Pointers (Limited)

```nano
# Can pass function names, but not create closures
fn apply(f: fn(int) -> int, x: int) -> int {
    return (f x)
}

fn double(x: int) -> int {
    return (* x 2)
}

fn main() -> int {
    return (apply double 5)  # Returns 10
}
```

### Struct-Based State

```nano
# Simulate closure with struct
struct Adder {
    value: int
}

fn add_to_struct(adder: Adder, x: int) -> int {
    return (+ adder.value x)
}

fn main() -> int {
    let adder: Adder = Adder { value: 5 }
    return (add_to_struct adder 10)  # Returns 15
}
```

## Estimated Implementation Effort

| Component | Complexity | Estimated Time |
|-----------|------------|----------------|
| Parser | Medium | 4-8 hours |
| Type System | High | 8-16 hours |
| Interpreter | Medium | 6-12 hours |
| Transpiler | Very High | 20-40 hours |
| Testing | Medium | 8-16 hours |
| **Total** | **High** | **46-92 hours** |

## Test Cases (Future)

```nano
# Test 1: Simple closure
fn test_simple_closure() -> int {
    fn make_adder(x: int) -> fn(int) -> int {
        fn adder(y: int) -> int {
            return (+ x y)
        }
        return adder
    }
    let add5: fn(int) -> int = (make_adder 5)
    return (add5 10)  # Should return 15
}

# Test 2: Multiple captures
fn test_multiple_captures() -> int {
    fn make_calculator(a: int, b: int) -> fn(int) -> int {
        fn calculate(c: int) -> int {
            return (+ (+ a b) c)
        }
        return calculate
    }
    let calc: fn(int) -> int = (make_calculator 3 4)
    return (calc 5)  # Should return 12
}

# Test 3: Nested closures (2 levels)
fn test_nested_closures() -> int {
    fn level1(x: int) -> fn(int) -> fn(int) -> int {
        fn level2(y: int) -> fn(int) -> int {
            fn level3(z: int) -> int {
                return (+ (+ x y) z)
            }
            return level3
        }
        return level2
    }
    let f1: fn(int) -> fn(int) -> int = (level1 1)
    let f2: fn(int) -> int = (f1 2)
    return (f2 3)  # Should return 6
}
```

## Related Features

- **First-class functions**: Already partially supported
- **Lambda expressions**: Could be added alongside closures
- **Partial application**: Natural extension of closures
- **Higher-order functions**: Enabled by nested functions

## Priority

**Medium Priority** - This is a valuable feature but not critical:

- ✅ Many languages work without nested function returns
- ✅ Workarounds exist (structs, function pointers)
- ✅ Implementation is complex (40-90 hours)
- ❌ Would enable functional programming patterns
- ❌ Would improve code expressiveness

## Alternatives

1. **Keep current approach**: Functions can be passed but not returned
2. **Add lambda expressions first**: Simpler than full nested functions
3. **Focus on other features**: Arrays, lists, structs more critical

## Conclusion

Nested functions would be a valuable addition to nanolang, but the implementation effort is substantial. Given the current workarounds and other priorities, this feature should be implemented after:

1. ✅ Nested arrays (DONE!)
2. Nested lists
3. Better error messages
4. Standard library expansion

When implemented, nested functions would bring nanolang closer to languages like JavaScript, Python, and Rust in terms of functional programming support.
