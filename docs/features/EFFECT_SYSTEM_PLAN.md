# Effect System - Implementation Plan

## Goal

Track function purity and side effects at the type level (IO monad style).

## Problem Statement

No way to distinguish pure from impure functions:

```nano
fn pure_math(x: int) -> int {
    return (* x x)  // Pure: no side effects
}

fn impure_io(x: int) -> int {
    (println x)  // Impure: I/O side effect!
    return x
}

// Both have same type: fn(int) -> int
// Caller can't know if function does I/O!
```

## Proposed Solution

Add effect annotations to track purity:

```nano
fn pure_math(x: int) -> int @pure {
    return (* x x)  // OK: no effects
}

fn impure_io(x: int) -> int @io {
    (println x)  // OK: @io allows I/O
    return x
}

fn needs_pure(f: fn(int) -> int @pure) -> int {
    return (f 42)  // Only accepts pure functions
}

// ERROR: Can't pass @io function where @pure expected
let result: int = (needs_pure impure_io)
```

## Effect Hierarchy

```
@pure      // No effects at all
@read      // Reads external state (files, network)
@write     // Writes external state  
@io        // Both read and write
@unsafe    // Can crash, undefined behavior
```

**Subtyping:** `@pure <: @read <: @io`, `@pure <: @write <: @io`

## Design Options

### Option A: Annotation-Based (Simpler)

```nano
// Explicit annotations
fn map(arr: array<int>, f: fn(int) -> int @pure) -> array<int> @pure {
    // Implementation must be pure
}

// Type error if calling impure function
let result: array<int> = (map numbers impure_io)  // ERROR
```

### Option B: Inferred Effects (Advanced)

```nano
// Compiler infers effects
fn compute(x: int) -> int {  // Inferred: @pure
    return (+ x 1)
}

fn log_and_compute(x: int) -> int {  // Inferred: @io
    (println x)
    return (+ x 1)
}
```

### Option C: Monadic (Haskell-style)

```nano
// IO monad separates pure and impure
fn pure_compute(x: int) -> int {  // Pure
    return (+ x 1)
}

fn io_action(x: int) -> IO<int> {  // Returns IO action
    do IO {
        (println x)
        return (+ x 1)
    }
}

fn main() -> int {
    let result: int = (run_io (io_action 42))  // Execute IO
    return result
}
```

## Implementation Strategy

### Phase 1: Effect Annotations (20 hours)

1. **Add effect keywords:** `@pure`, `@io`, `@read`, `@write`, `@unsafe`
2. **Parse annotations:** Extend function syntax
3. **Store in AST:** Add effect field to FunctionSignature
4. **Type check:** Verify function body matches annotation

```nano
// Lexer: Add tokens
TOKEN_AT_PURE
TOKEN_AT_IO
TOKEN_AT_READ
TOKEN_AT_WRITE
TOKEN_AT_UNSAFE

// Parser: Parse effect annotations
fn parse_effect_annotation() -> Effect

// TypeChecker: Verify effect compliance
fn check_function_effect(fn_sig: FunctionSignature, body: ASTNode) -> bool
```

### Phase 2: Effect Inference (25 hours)

1. **Infer effects:** Bottom-up from expressions
2. **Propagate through calls:** Track transitive effects
3. **Default to safe:** Assume @pure unless proven otherwise

```nano
// Inference rules
pure_expr → @pure
(println x) → @io
(+ a b) → @pure
(f x) where f: @io → @io  // Transitivity
```

### Phase 3: Effect Polymorphism (20 hours)

```nano
// Effect-polymorphic map
fn map<E>(arr: array<int>, f: fn(int) -> int @E) -> array<int> @E {
    // Effect E propagates through
}

// Specialized to pure
let pure_result: array<int> = (map nums pure_fn)  // @pure

// Specialized to io  
let io_result: array<int> = (map nums io_fn)  // @io
```

### Phase 4: Effect Handlers (40 hours)

Advanced: Algebraic effects and handlers

```nano
effect State<S> {
    get: () -> S
    set: (S) -> ()
}

fn counter() -> int with State<int> {
    let current: int = (effect.get)
    (effect.set (+ current 1))
    return current
}

fn run_with_state<S, A>(init: S, action: () -> A with State<S>) -> A {
    // Handler implementation
}
```

## Breaking Changes

**Major breaking change:** All existing functions need effect annotations.

**Migration Strategy:**

1. **Phase 1:** Warnings only (--warn-effects)
2. **Phase 2:** Default to @io (permissive)
3. **Phase 3:** Require explicit annotations (strict)

```nano
// Auto-migration tool
nanoc --infer-effects old.nano > new.nano

// Before
fn compute(x: int) -> int {
    return (+ x 1)
}

// After (inferred)
fn compute(x: int) -> int @pure {
    return (+ x 1)
}
```

## Standard Library Impact

**All stdlib functions need effects:**

```nano
// Math (pure)
fn add(a: int, b: int) -> int @pure { ... }
fn sqrt(x: float) -> float @pure { ... }

// I/O (io)
fn println(s: string) -> void @io { ... }
fn read_file(path: string) -> string @io { ... }

// Unsafe (unsafe)
extern fn malloc(size: int) -> ptr @unsafe
```

## Performance Considerations

1. **Effect checking is compile-time:** Zero runtime overhead
2. **Monomorphization:** Effect-polymorphic functions duplicated per effect
3. **Optimization:** Pure functions can be memoized, reordered

## Integration with Existing Features

### With Checked Arithmetic

```nano
fn safe_add(a: int, b: int) -> Result<int, string> @pure {
    return (checked_add a b)  // Pure: no I/O
}
```

### With Unsafe Blocks

```nano
fn ffi_call() -> int @unsafe {
    unsafe {
        return (external_c_function)  // Marked unsafe
    }
}
```

### With Totality Checker

```nano
fn total_and_pure(x: int) -> int @pure @total {
    return (+ x 1)  // Provably total AND pure
}
```

## Estimated Effort

- **Phase 1 (Annotations):** 20 hours
- **Phase 2 (Inference):** 25 hours  
- **Phase 3 (Polymorphism):** 20 hours
- **Phase 4 (Handlers):** 40 hours
- **Stdlib Migration:** 10 hours
- **Documentation:** 5 hours
- **Testing:** 10 hours
- **TOTAL:** ~130 hours (3+ weeks)

**Dual-Impl Overhead:** 1.8x (C + NanoLang typecheckers)

## References

- Haskell IO Monad: https://wiki.haskell.org/IO_inside
- Koka Effects: https://koka-lang.github.io/koka/doc/book.html
- OCaml Effects: https://ocaml.org/manual/effects.html
- Eff Language: https://www.eff-lang.org/
- Frank: https://arxiv.org/abs/1611.09259

## Status

🟡 **PLANNED** - Requires major type system extension

**Recommendation:** 
- Too large for immediate implementation
- Consider lightweight @pure annotation first (Phase 1 only)
- Full effect system is a v2.0 feature

**Alternatives:**
- Document pure/impure in comments
- Use naming conventions (io_read_file vs pure_compute)
- Wait for community demand

Related: nanolang-yq92

