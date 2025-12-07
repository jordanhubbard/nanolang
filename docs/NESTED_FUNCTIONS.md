# Nested Functions Design Document

## Status: PARTIALLY IMPLEMENTED (Function Pointers ✅, Closures ❌)

This document describes the design for nested function support in nanolang.

## What's Implemented Now ✅

**Function Pointers** are fully functional in both interpreter and compiled modes!

- ✅ Functions can be returned from other functions
- ✅ Functions can be stored in variables  
- ✅ Functions can be passed as arguments
- ✅ Higher-order functions work (map, filter-style)
- ✅ Function factories work
- ❌ Closures with variable capture (not yet implemented)

See [Function Pointers](#function-pointers-implemented) section below for examples.

## Function Pointers (Implemented)

Function pointers allow passing, storing, and returning functions as values. This enables functional programming patterns without requiring closures.

### Working Examples

```nano
# Example 1: Higher-order function
fn double(x: int) -> int {
    return (* x 2)
}

fn apply_twice(f: fn(int) -> int, x: int) -> int {
    let result1: int = (f x)
    let result2: int = (f result1)
    return result2
}

fn main() -> int {
    let result: int = (apply_twice double 5)
    return result  # Returns 20
}
```

```nano
# Example 2: Function factory
fn double(x: int) -> int {
    return (* x 2)
}

fn triple(x: int) -> int {
    return (* x 3)
}

fn make_multiplier(factor: int) -> fn(int) -> int {
    if (== factor 2) {
        return double
    } else {
        return triple
    }
}

fn main() -> int {
    let mult: fn(int) -> int = (make_multiplier 2)
    let result: int = (mult 7)
    return result  # Returns 14
}
```

```nano
# Example 3: Storing functions in variables
fn add_one(x: int) -> int {
    return (+ x 1)
}

fn main() -> int {
    let f: fn(int) -> int = add_one
    let result: int = (f 10)
    return result  # Returns 11
}
```

### Use Cases

Function pointers enable many functional programming patterns:

1. **Higher-Order Functions**: `apply_twice`, `map`, `filter`, `fold`
2. **Function Factories**: Return different functions based on conditions
3. **Strategy Pattern**: Pass different algorithms as function arguments
4. **Callbacks**: Store functions to be called later
5. **Function Composition**: Chain operations by passing functions

### Limitations

**No Variable Capture**: Functions cannot access variables from their defining scope:

```nano
# ❌ This doesn't work - closures not implemented
fn make_adder(x: int) -> fn(int) -> int {
    # This would require capturing 'x'
    fn adder(y: int) -> int {
        return (+ x y)  # ❌ Cannot capture 'x'
    }
    return adder
}
```

**Workaround**: Use pre-defined functions or pass values explicitly:

```nano
# ✅ This works - no capture needed
fn double(x: int) -> int {
    return (* x 2)
}

fn make_multiplier(factor: int) -> fn(int) -> int {
    if (== factor 2) {
        return double  # Return pre-defined function
    }
    # ...
}
```

## Overview

Nested functions would allow functions to return other functions, enabling:
- ✅ Higher-order functions (map, filter, reduce) - **IMPLEMENTED via function pointers**
- ❌ Currying and partial application - **Requires closures**
- ✅ Function factories - **IMPLEMENTED via function pointers**
- ❌ Closures with variable capture - **NOT YET IMPLEMENTED**

## Proposed Syntax (For Full Closures)

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

**Low Priority for Full Closures** - Function pointers are now implemented, closures less critical:

- ✅ Function pointers already enable most functional programming patterns
- ✅ Higher-order functions work without closures
- ✅ Function factories work without closures
- ✅ Workarounds exist for most use cases
- ❌ Closures would enable variable capture
- ❌ Implementation is complex (40-90 hours for closures)

## Alternatives

1. ~~**Keep current approach**: Functions can be passed but not returned~~ ✅ **DONE! Function pointers work!**
2. **Add lambda expressions**: Simpler syntax for inline functions
3. **Focus on other features**: Many functional patterns already work

## Conclusion

**Function pointers are now fully functional!** This enables:
- ✅ Higher-order functions
- ✅ Function factories
- ✅ Callbacks and strategy patterns
- ✅ Function composition

**Full closures with variable capture** remain unimplemented, but they're less critical now that function pointers work. The effort required (40-90 hours) may not be justified given the workarounds available.

Nanolang now has feature parity with Go, C, and C++ for function pointers, and is approaching the functional capabilities of Rust, TypeScript, and Python (minus closures).

Closure implementation should be considered after:

1. ✅ Nested arrays (DONE!)
2. ✅ Function pointers (DONE!)
3. Nested lists  
4. Better error messages
5. Standard library expansion
6. Lambda expression syntax

If closures are never implemented, nanolang still provides excellent functional programming support through function pointers!
