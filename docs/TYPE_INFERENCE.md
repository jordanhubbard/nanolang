# Type Inference in NanoLang

## Overview

NanoLang is primarily an **explicitly-typed language** where most type annotations are mandatory. However, the compiler does perform **limited type inference** in specific contexts to reduce verbosity while maintaining clarity.

This document explains exactly what the compiler can and cannot infer, helping you understand when type annotations are required.

## Quick Reference

| Situation | Inference | Example |
|-----------|-----------|---------|
| Variable declarations | ❌ **Required** | `let x: int = 42` |
| Function parameters | ❌ **Required** | `fn f(x: int)` |
| Function return types | ❌ **Required** | `fn f() -> int` |
| Array element types | ✅ **Inferred from literal** | `let arr: array<int> = [1, 2, 3]` |
| Empty array types | ⚠️ **Context-dependent** | Requires type annotation |
| Struct literal names | ✅ **Inferred from context** | See below |
| Generic type parameters | ❌ **Required** | `let list: List<int> = ...` |

## What MUST Be Explicitly Annotated

### 1. Variable Type Annotations

**Rule:** ALL variable declarations must include a type annotation.

```nano
# ✅ CORRECT
let x: int = 42
let name: string = "Alice"
let flag: bool = true
let pi: float = 3.14

# ❌ ERROR: Missing type annotation
let x = 42          # Compiler error
let name = "Alice"  # Compiler error
```

**Why?** Explicit types improve code clarity and make it easier for both humans and LLMs to understand code without context.

### 2. Function Parameter Types

**Rule:** ALL function parameters must have type annotations.

```nano
# ✅ CORRECT
fn add(a: int, b: int) -> int {
    return (+ a b)
}

# ❌ ERROR: Missing parameter types
fn add(a, b) -> int {  # Compiler error
    return (+ a b)
}
```

### 3. Function Return Types

**Rule:** ALL functions must declare their return type, even `void`.

```nano
# ✅ CORRECT
fn get_value() -> int {
    return 42
}

fn print_message() -> void {
    (println "Hello")
}

# ❌ ERROR: Missing return type
fn get_value() {  # Compiler error
    return 42
}
```

**Exception:** `shadow` test blocks don't need return types (they're always `void`).

### 4. Generic Type Parameters

**Rule:** Generic types like `List<T>` and `HashMap<K,V>` require explicit type parameters.

```nano
# ✅ CORRECT
let numbers: List<int> = (List_int_new)
let mapping: HashMap<string, int> = (map_new)

# ❌ ERROR: Missing type parameters
let numbers: List = (List_int_new)     # Compiler error
let mapping: HashMap = (map_new)       # Compiler error
```

**Why?** NanoLang uses **monomorphization** (compile-time specialization). The compiler needs to know which concrete types to generate code for.

## What CAN Be Inferred

### 1. Array Element Types (From Literals)

When you provide an array literal with elements, the compiler can infer the element type from the literal itself.

```nano
# Type annotation specifies array<int>, but could be inferred from [1, 2, 3]
let numbers: array<int> = [1, 2, 3]

# This works because the literal [1, 2, 3] clearly contains ints
let values: array<int> = [10, 20, 30]

# The literal determines element type
let names: array<string> = ["Alice", "Bob"]
```

**Important:** You still need the type annotation on the variable (`array<int>`), but the compiler verifies it matches the literal.

### 2. Empty Array Types (Context-Dependent)

Empty arrays `[]` require type information from context:

```nano
# ✅ CORRECT: Type specified in variable annotation
let empty: array<int> = []

# ✅ CORRECT: Type known from function parameter
fn process_numbers(nums: array<int>) -> void {
    # ...
}
# Calling with empty array - type inferred from parameter
(process_numbers [])

# ❌ ERROR: No context for empty array type
let unknown = []  # Would error anyway (no variable type)
```

### 3. Struct Literal Names (From Context)

When the compiler knows what struct type is expected, you can use **anonymous struct literals** by omitting the struct name.

**Important:** This feature exists in the typechecker but may have limited practical use since variable declarations always require explicit types. It's most useful for:

1. Function return values
2. Nested struct initialization
3. Function call arguments

```nano
struct Point {
    x: int,
    y: int
}

# Explicit struct name (always works)
fn create_point() -> Point {
    return Point { x: 10, y: 20 }
}

# In theory, could be anonymous if return type provides context
# (Current syntax requires struct name, but typechecker supports inference)
```

**Current Status:** While the typechecker has logic for anonymous struct literals (line 2100-2104 in `typechecker.c`), the parser and syntax currently require explicit struct names. This is documented for completeness.

## Type Inference Examples

### Example 1: Array Type Inference

```nano
fn process_data() -> void {
    # Element type inferred from literal
    let scores: array<int> = [95, 87, 92, 88]
    
    # Empty array requires explicit type
    let mut results: array<int> = []
    
    # Type matches literal
    let names: array<string> = ["Alice", "Bob", "Charlie"]
}

shadow process_data {
    (process_data)
    assert true
}
```

### Example 2: Generic Type Parameters

```nano
# HashMap requires explicit K,V parameters
fn create_lookup() -> HashMap<string, int> {
    let map: HashMap<string, int> = (map_new)
    (map_insert map "answer" 42)
    return map
}

shadow create_lookup {
    let m: HashMap<string, int> = (create_lookup)
    assert true
}

# List requires explicit T parameter
fn create_list() -> List<int> {
    let list: List<int> = (List_int_new)
    (List_int_push list 1)
    (List_int_push list 2)
    return list
}

shadow create_list {
    let l: List<int> = (create_list)
    assert true
}
```

### Example 3: No Variable Type Inference

```nano
# ❌ These all fail - variables need explicit types
# let x = 42
# let name = "Alice"
# let flag = (> 5 3)

# ✅ Correct versions
let x: int = 42
let name: string = "Alice"
let flag: bool = (> 5 3)
```

## Common Mistakes

### Mistake 1: Omitting Variable Types

```nano
# ❌ ERROR
let count = 0

# ✅ CORRECT
let count: int = 0
```

**Error Message:**
```
Error: Variable declaration requires explicit type annotation
```

### Mistake 2: Omitting Function Parameter Types

```nano
# ❌ ERROR
fn double(x) -> int {
    return (* x 2)
}

# ✅ CORRECT
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
}
```

### Mistake 3: Missing Generic Type Parameters

```nano
# ❌ ERROR
let numbers: List = (List_int_new)

# ✅ CORRECT
let numbers: List<int> = (List_int_new)
```

**Error Message:**
```
Error: Generic type List requires type parameters
```

### Mistake 4: Empty Array Without Type

```nano
# ❌ ERROR
let mut results = []

# ✅ CORRECT
let mut results: array<int> = []
```

## Design Philosophy

### Why So Explicit?

NanoLang's explicit typing philosophy serves several goals:

1. **LLM-Friendly:** AI systems can generate correct code without complex inference rules
2. **Readability:** Code is self-documenting; types are always visible
3. **Simplicity:** Fewer inference rules mean less cognitive load
4. **Error Messages:** When types are explicit, error messages are clearer
5. **Compile Speed:** Less type inference = faster compilation

### Comparison with Other Languages

| Language | Variable Types | Parameter Types | Return Types |
|----------|---------------|-----------------|--------------|
| **NanoLang** | Explicit | Explicit | Explicit |
| C | Explicit | Explicit | Explicit |
| Go | Can omit (`:=`) | Explicit | Explicit |
| Rust | Can omit | Can omit | Can omit |
| TypeScript | Can omit | Can omit | Can infer |
| Python | Optional | Optional | Optional |

NanoLang is closer to C in its explicitness, but simpler than Rust or TypeScript's inference.

## Future Considerations

The type inference system is intentionally minimal. Potential future additions could include:

1. **Local variable inference:** `let x = 42  # infer int`
2. **Return type inference:** `fn f() { return 42 }  # infer -> int`
3. **Generic function parameters:** `fn identity<T>(x: T) -> T`

However, these are **not currently planned** as they conflict with the design goal of explicitness.

## Best Practices

### 1. Always Annotate Variables

Even when you think the type is "obvious," include the annotation:

```nano
# ✅ Good - clear and explicit
let message: string = "Hello"
let count: int = 0
let ready: bool = false

# ❌ Bad - would be an error anyway
# let message = "Hello"
```

### 2. Use Descriptive Types for Generics

```nano
# ✅ Good - type parameters are clear
let user_ids: List<int> = (List_int_new)
let user_map: HashMap<string, int> = (map_new)

# ❌ Confusing - what are the types?
# let data: List = ...  # Won't compile anyway
```

### 3. Leverage Function Signatures

Function signatures serve as documentation:

```nano
# ✅ Good - signature tells you everything
fn calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float {
    # Implementation
    return 0.0
}

shadow calculate_distance {
    let dist: float = (calculate_distance 0 0 3 4)
    assert true
}
```

## Summary

**NanoLang's Type Inference Philosophy:**

- **Default:** Explicit type annotations required
- **Exception:** Limited inference in specific contexts (array literals, struct literals)
- **Goal:** Maximize clarity and LLM-friendliness

**Key Takeaway:** When in doubt, add the type annotation. NanoLang values explicitness over brevity.

---

**See Also:**
- [Type System Specification](SPECIFICATION.md#3-types)
- [Generics Deep Dive](GENERICS_DEEP_DIVE.md) - Monomorphization explained
- [Error Messages Guide](ERROR_MESSAGES.md) - Understanding type errors
- [Getting Started](GETTING_STARTED.md) - Basic type usage

**Last Updated:** January 25, 2026
