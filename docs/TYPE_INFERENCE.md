# Type Inference

## Overview

I am primarily an explicitly-typed language. I require you to provide most type annotations. I perform limited type inference in specific contexts to reduce verbosity. I do this only where it does not sacrifice clarity.

This document explains what I can and cannot infer. It will help you understand when I require you to be explicit.

## Quick Reference

| Situation | Inference | Example |
|-----------|-----------|---------|
| Variable declarations | Required | `let x: int = 42` |
| Function parameters | Required | `fn f(x: int)` |
| Function return types | Required | `fn f() -> int` |
| Array element types | Inferred from literal | `let arr: array<int> = [1, 2, 3]` |
| Empty array types | Context-dependent | Requires type annotation |
| Struct literal names | Inferred from context | See below |
| Generic type parameters | Required | `let list: List<int> = ...` |

## What I Require You To Annotate

### 1. Variable Type Annotations

I require all variable declarations to include a type annotation.

```nano
# CORRECT
let x: int = 42
let name: string = "Alice"
let flag: bool = true
let pi: float = 3.14

# ERROR: Missing type annotation
let x = 42          # I will not compile this
let name = "Alice"  # I will not compile this
```

Explicit types improve code clarity. They make it easier for humans and machines to understand code without searching for context.

### 2. Function Parameter Types

I require all function parameters to have type annotations.

```nano
# CORRECT
fn add(a: int, b: int) -> int {
    return (+ a b)
}

# ERROR: Missing parameter types
fn add(a, b) -> int {  # I will not compile this
    return (+ a b)
}
```

### 3. Function Return Types

I require all functions to declare their return type. This includes functions that return `void`.

```nano
# CORRECT
fn get_value() -> int {
    return 42
}

fn print_message() -> void {
    (println "Hello")
}

# ERROR: Missing return type
fn get_value() {  # I will not compile this
    return 42
}
```

I do not require return types for `shadow` test blocks. They are always `void`.

### 4. Generic Type Parameters

I require explicit type parameters for generic types like `List<T>` and `HashMap<K,V>`.

```nano
# CORRECT
let numbers: List<int> = (List_int_new)
let mapping: HashMap<string, int> = (map_new)

# ERROR: Missing type parameters
let numbers: List = (List_int_new)     # I will not compile this
let mapping: HashMap = (map_new)       # I will not compile this
```

I use monomorphization to specialize code at compile time. I must know the concrete types to generate the correct implementation.

## What I Can Infer

### 1. Array Element Types (From Literals)

When you provide an array literal with elements, I infer the element type from the literal itself.

```nano
# I verify that [1, 2, 3] matches array<int>
let numbers: array<int> = [1, 2, 3]

# I determine the element type from the literal contents
let values: array<int> = [10, 20, 30]
let names: array<string> = ["Alice", "Bob"]
```

You must still provide the type annotation on the variable. I check that it matches the literal.

### 2. Empty Array Types (Context-Dependent)

Empty arrays `[]` require type information from the context.

```nano
# CORRECT: I use the variable annotation
let empty: array<int> = []

# CORRECT: I use the function parameter type
fn process_numbers(nums: array<int>) -> void {
    # ...
}
# I infer the type from the parameter when you call me
(process_numbers [])

# ERROR: I have no context for this empty array type
let unknown = []  # I will not compile this
```

### 3. Struct Literal Names (From Context)

When I know which struct type is expected, you can use anonymous struct literals by omitting the struct name.

This feature is present in my typechecker. It is useful for:

1. Function return values
2. Nested struct initialization
3. Function call arguments

```nano
struct Point {
    x: int,
    y: int
}

# I accept the explicit struct name
fn create_point() -> Point {
    return Point { x: 10, y: 20 }
}
```

My typechecker has logic for anonymous struct literals. My parser and syntax currently require explicit struct names. I document this for completeness.

## Type Inference Examples

### Example 1: Array Type Inference

```nano
fn process_data() -> void {
    # I infer the element type from the literal
    let scores: array<int> = [95, 87, 92, 88]
    
    # I require an explicit type for empty arrays
    let mut results: array<int> = []
    
    # I ensure the type matches the literal
    let names: array<string> = ["Alice", "Bob", "Charlie"]
}

shadow process_data {
    (process_data)
    assert true
}
```

### Example 2: Generic Type Parameters

```nano
# I require explicit K,V parameters for HashMap
fn create_lookup() -> HashMap<string, int> {
    let map: HashMap<string, int> = (map_new)
    (map_insert map "answer" 42)
    return map
}

shadow create_lookup {
    let m: HashMap<string, int> = (create_lookup)
    assert true
}

# I require an explicit T parameter for List
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

### Example 3: Variable Type Requirements

```nano
# I will not compile these variable declarations
# let x = 42
# let name = "Alice"
# let flag = (> 5 3)

# I require these versions
let x: int = 42
let name: string = "Alice"
let flag: bool = (> 5 3)
```

## Common Mistakes

### Mistake 1: Omitting Variable Types

```nano
# ERROR
let count = 0

# CORRECT
let count: int = 0
```

I will produce this error:
`Error: Variable declaration requires explicit type annotation`

### Mistake 2: Omitting Function Parameter Types

```nano
# ERROR
fn double(x) -> int {
    return (* x 2)
}

# CORRECT
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
}
```

### Mistake 3: Missing Generic Type Parameters

```nano
# ERROR
let numbers: List = (List_int_new)

# CORRECT
let numbers: List<int> = (List_int_new)
```

I will produce this error:
`Error: Generic type List requires type parameters`

### Mistake 4: Empty Array Without Type

```nano
# ERROR
let mut results = []

# CORRECT
let mut results: array<int> = []
```

## My Design Philosophy

### Why I Am Explicit

My explicit typing philosophy serves these goals:

1. **Machine-Friendly.** AI systems generate correct code without complex inference rules.
2. **Readability.** Code is self-documenting. Types are always visible.
3. **Simplicity.** Fewer inference rules mean less cognitive load for you.
4. **Clarity.** Error messages are clearer when types are explicit.
5. **Speed.** Less type inference results in faster compilation.

### Comparison With Other Languages

| Language | Variable Types | Parameter Types | Return Types |
|----------|---------------|-----------------|--------------|
| **NanoLang** | Explicit | Explicit | Explicit |
| C | Explicit | Explicit | Explicit |
| Go | Can omit (`:=`) | Explicit | Explicit |
| Rust | Can omit | Can omit | Can omit |
| TypeScript | Can omit | Can omit | Can infer |
| Python | Optional | Optional | Optional |

I am closer to C in my explicitness. I am simpler than Rust or TypeScript.

## Future Considerations

My type inference system is intentionally minimal. I could add these features:

1. **Local variable inference.** `let x = 42`
2. **Return type inference.** `fn f() { return 42 }`
3. **Generic function parameters.** `fn identity<T>(x: T) -> T`

I do not currently plan to add these features. They conflict with my goal of explicitness.

## Best Practices

### 1. Always Annotate Variables

Include the annotation even when you think the type is obvious.

```nano
# Good. I find this clear and explicit.
let message: string = "Hello"
let count: int = 0
let ready: bool = false
```

### 2. Use Descriptive Types for Generics

```nano
# Good. Your type parameters are clear to me.
let user_ids: List<int> = (List_int_new)
let user_map: HashMap<string, int> = (map_new)
```

### 3. Use Function Signatures

Your function signatures serve as documentation for me and for you.

```nano
# Good. This signature tells me everything I need to know.
fn calculate_distance(x1: int, y1: int, x2: int, y2: int) -> float {
    return 0.0
}

shadow calculate_distance {
    let dist: float = (calculate_distance 0 0 3 4)
    assert true
}
```

## Summary

I follow these principles for type inference:

- **Default.** I require explicit type annotations.
- **Exception.** I allow limited inference for array and struct literals.
- **Goal.** I maximize clarity and machine-friendliness.

When you are in doubt, add the type annotation. I value explicitness over brevity.

---

**See Also:**
- [Type System Specification](SPECIFICATION.md#3-types)
- [Generics Deep Dive](GENERICS_DEEP_DIVE.md)
- [Error Messages Guide](ERROR_MESSAGES.md)
- [Getting Started](GETTING_STARTED.md)

**Last Updated:** February 20, 2026

