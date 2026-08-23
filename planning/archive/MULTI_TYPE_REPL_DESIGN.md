# Multi-Type REPL Design

## Challenge

The current `vars_repl.nano` only supports integer expressions because:
1. The `wrap_with_context` function hardcodes the result type as `int`
2. The result display uses `int_to_string`
3. No way to determine the expression's result type without parsing

## Constraint

Without full parser support or dynamic typing, we cannot automatically infer the type of an arbitrary expression like `(+ x y)` where `x` and `y` might be floats.

## Proposed Solution

### Option A: Type-Specific Evaluation Functions

Create separate evaluation contexts for each type:

```nano
nano> let x: int = 42
Defined: x (int)

nano> eval_int (+ x 10)
=> 52

nano> let y: float = 3.14
Defined: y (float)

nano> eval_float (* y 2.0)
=> 6.28
```

**Pros:**
- Simple to implement
- Clear type separation
- No parsing needed

**Cons:**
- Requires explicit type specification by user
- Not a natural REPL experience

### Option B: Type Annotation in Expression

Require type annotation for expressions:

```nano
nano> (+ x 10) : int
=> 52

nano> (* y 2.0) : float
=> 6.28
```

**Pros:**
- More natural than eval_int/eval_float
- Clear type specification

**Cons:**
- Still requires user to specify type
- Need to parse the annotation

### Option C: Hybrid - Smart Default

Default to int, but allow explicit type wrapping:

```nano
nano> (+ 2 3)        # Defaults to int
=> 5

nano> (as_float (* 2.0 3.14))
=> 6.28

nano> (as_string (+ "Hello, " "World"))
=> "Hello, World"
```

**Pros:**
- Backwards compatible with current REPL
- Explicit when needed
- Can wrap expressions

**Cons:**
- Not true multi-type support
- Awkward syntax

### Option D: Multiple Result Wrappers (Recommended)

Generate wrapper functions for each type:

```nano
# In wrapped code:
fn main() -> int {
    let x: int = 42
    let y: float = 3.14

    # Try each type wrapper
    let _int_result: int = EXPR
    (println (+ "(int) " (int_to_string _int_result)))

    let _float_result: float = EXPR
    (println (+ "(float) " (float_to_string _float_result)))

    return 0
}
```

Run all wrappers, show whichever doesn't type-error.

**Pros:**
- Automatic type detection
- Natural REPL experience
- No user annotations needed

**Cons:**
- Multiple compilations (slow)
- Error messages might be confusing
- Doesn't work if expression is polymorphic

## Recommended Implementation

**For MVP:** Implement Option C (Hybrid)
- Keep current int-defaulting behavior
- Add `as_float`, `as_string`, `as_bool` wrapper functions
- Document the limitation

**For Future:** Implement Option D after adding:
- Better error recovery in compiler
- Faster compilation (caching)
- Type inference API

## Implementation Plan for Option C

### Step 1: Add Type Wrapper Functions

```nano
# Evaluate as float
fn eval_float_expr(preamble: string, expr: string) -> int {
    # Wrap as: let _result: float = EXPR
    # Print using float_to_string
}

# Evaluate as string
fn eval_string_expr(preamble: string, expr: string) -> int {
    # Wrap as: let _result: string = EXPR
    # Print using println directly
}

# Evaluate as bool
fn eval_bool_expr(preamble: string, expr: string) -> int {
    # Wrap as: let _result: bool = EXPR
    # Print using bool_to_string or conditional
}
```

### Step 2: Add Commands

```nano
nano> :float (* 3.14 2.0)
=> 6.28

nano> :string (+ "Hello, " "World")
=> "Hello, World"

nano> :bool (> 5 3)
=> true
```

### Step 3: Support Typed Variables

Variables already support all types via let statements:

```nano
nano> let pi: float = 3.14159
Defined: pi (float)

nano> let name: string = "Alice"
Defined: name (string)
```

The preamble will inject these correctly typed.

## Example Session

```nano
$ ./bin/multi_type_repl

NanoLang REPL (Multi-Type Support)
=====================================
Commands: :int EXPR, :float EXPR, :string EXPR, :bool EXPR
          :vars, :clear, :quit

nano> let x: int = 42
Defined: x (int)

nano> let y: float = 3.14
Defined: y (float)

nano> let name: string = "Bob"
Defined: name (string)

nano> :vars
Defined variables: x (int), y (float), name (string)

nano> (+ x 10)           # Default: int
=> 52

nano> :float (* y 2.0)
=> 6.28

nano> :string (+ "Hello, " name)
=> "Hello, Bob"

nano> :bool (> x 40)
=> true
```

## Files to Create

- `examples/language/multi_type_repl.nano` - REPL with multi-type support

## Estimated Effort

- 2-3 hours for Option C implementation
- Will demonstrate Task #4 capabilities
- Document limitations for future work
