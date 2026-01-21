# Chapter 3: Variables & Bindings

**Learn how NanoLang handles variables, mutability, and scope.**

NanoLang distinguishes between immutable bindings (`let`) and mutable variables (`let mut`). This chapter explains when to use each and how scope works.

## 3.1 let Bindings (Immutable by Default)

In NanoLang, variables are **immutable by default**. Once you bind a value to a name with `let`, you cannot change it.

### Creating Bindings

```nano
fn binding_example() -> int {
    let x: int = 42
    let name: string = "Alice"
    let is_valid: bool = true
    return x
}

shadow binding_example {
    assert (== (binding_example) 42)
}
```

**Syntax:** `let identifier: type = value`

### Immutability Benefits

**Why immutable by default?**

1. **Easier to reason about** - The value never changes
2. **Prevents accidental modification** - Compiler catches mistakes
3. **Better for concurrent code** - No race conditions
4. **Matches functional programming style** - Values flow through transformations

### Attempting to Modify

This won't compile:

```nano
# ❌ This is an error:
# fn try_to_modify() -> int {
#     let x: int = 10
#     set x 20  # Error: x is immutable
#     return x
# }
```

The compiler will reject this because `x` is immutable.

### Examples of Immutable Bindings

```nano
fn calculate_circle_area(radius: float) -> float {
    let pi: float = 3.14159
    let radius_squared: float = (* radius radius)
    let area: float = (* pi radius_squared)
    return area
}

shadow calculate_circle_area {
    let area: float = (calculate_circle_area 5.0)
    assert (and (> area 78.5) (< area 78.6))
}
```

Each binding (`pi`, `radius_squared`, `area`) is immutable. They're created once and never change.

### Multiple Bindings

You can create many bindings in sequence:

```nano
fn multi_binding_example(x: int) -> int {
    let a: int = (+ x 1)
    let b: int = (* a 2)
    let c: int = (- b 3)
    return c
}

shadow multi_binding_example {
    assert (== (multi_binding_example 5) 9)
    # (5 + 1) * 2 - 3 = 6 * 2 - 3 = 12 - 3 = 9
}
```

## 3.2 mut Variables (When You Need Mutation)

When you need to change a value, declare it as mutable with `let mut`.

### Mutable Variables

```nano
fn counter_example() -> int {
    let mut count: int = 0
    set count (+ count 1)
    set count (+ count 1)
    set count (+ count 1)
    return count
}

shadow counter_example {
    assert (== (counter_example) 3)
}
```

**Syntax:** `let mut identifier: type = initial_value`

### When to Use mut

Use `mut` when:
- Implementing loops with counters
- Accumulating results
- Building data structures incrementally
- You genuinely need to update a value

**Prefer immutable by default.** Only use `mut` when you have a good reason.

### Mutable Example: Sum

```nano
fn sum_to_n(n: int) -> int {
    let mut sum: int = 0
    let mut i: int = 1
    while (<= i n) {
        set sum (+ sum i)
        set i (+ i 1)
    }
    return sum
}

shadow sum_to_n {
    assert (== (sum_to_n 5) 15)  # 1+2+3+4+5=15
    assert (== (sum_to_n 10) 55)
    assert (== (sum_to_n 0) 0)
}
```

### Mutable Example: Finding Maximum

```nano
fn find_max(arr: array<int>) -> int {
    let mut max: int = (array_get arr 0)
    let mut i: int = 1
    let len: int = (array_length arr)
    
    while (< i len) {
        let current: int = (array_get arr i)
        if (> current max) {
            set max current
        }
        set i (+ i 1)
    }
    return max
}

shadow find_max {
    assert (== (find_max [1, 5, 3, 9, 2]) 9)
    assert (== (find_max [10]) 10)
    assert (== (find_max [-5, -2, -10]) -2)
}
```

## 3.3 set Statements

The `set` statement updates mutable variables.

### Basic set Usage

```nano
fn set_example() -> int {
    let mut x: int = 10
    set x 20        # Update x to 20
    set x (+ x 5)   # Update x to 25
    return x
}

shadow set_example {
    assert (== (set_example) 25)
}
```

**Syntax:** `set identifier new_value`

⚠️ **Watch Out:** You can only `set` variables declared with `let mut`.

### Updating Based on Current Value

Common pattern: update a variable based on its current value.

```nano
fn increment_example() -> int {
    let mut counter: int = 0
    set counter (+ counter 1)  # counter = counter + 1
    set counter (+ counter 1)
    set counter (+ counter 1)
    return counter
}

shadow increment_example {
    assert (== (increment_example) 3)
}
```

### Multiple Updates

```nano
fn accumulate_example(n: int) -> int {
    let mut result: int = 0
    let mut i: int = 0
    
    while (< i n) {
        set result (+ result (* i 2))
        set i (+ i 1)
    }
    return result
}

shadow accumulate_example {
    assert (== (accumulate_example 5) 20)
    # 0*2 + 1*2 + 2*2 + 3*2 + 4*2 = 0+2+4+6+8 = 20
}
```

### Common Patterns

**Accumulator pattern:**

```nano
fn sum_array(arr: array<int>) -> int {
    let mut sum: int = 0
    let mut i: int = 0
    while (< i (array_length arr)) {
        set sum (+ sum (array_get arr i))
        set i (+ i 1)
    }
    return sum
}

shadow sum_array {
    assert (== (sum_array [1, 2, 3, 4, 5]) 15)
}
```

**Counter pattern:**

```nano
fn count_evens(arr: array<int>) -> int {
    let mut count: int = 0
    let mut i: int = 0
    while (< i (array_length arr)) {
        if (== (% (array_get arr i) 2) 0) {
            set count (+ count 1)
        }
        set i (+ i 1)
    }
    return count
}

shadow count_evens {
    assert (== (count_evens [1, 2, 3, 4, 5, 6]) 3)
}
```

## 3.4 Scope & Shadowing

Variables have **lexical scope**: they're only visible within the block where they're defined.

### Block Scope

```nano
fn scope_example() -> int {
    let x: int = 10
    if true {
        let y: int = 20     # y only visible in this block
        let z: int = (+ x y) # Can access x from outer scope
    }
    # y and z are not visible here
    return x
}

shadow scope_example {
    assert (== (scope_example) 10)
}
```

### Nested Scopes

```nano
fn nested_scope() -> int {
    let x: int = 1
    if true {
        let x: int = 2      # This shadows outer x
        if true {
            let x: int = 3  # This shadows both outer x's
            # Here x is 3
        }
        # Here x is 2
    }
    # Here x is 1
    return x
}

shadow nested_scope {
    assert (== (nested_scope) 1)
}
```

### Shadowing (Redeclaring Variables)

You can declare a new variable with the same name as an existing one. This is called **shadowing**.

```nano
fn shadowing_example() -> int {
    let x: int = 5
    let x: int = (+ x 1)  # Shadows outer x, new x = 6
    let x: int = (* x 2)  # Shadows previous x, new x = 12
    return x
}

shadow shadowing_example {
    assert (== (shadowing_example) 12)
}
```

**What's happening:**
1. First `x` is bound to `5`
2. Second `x` is bound to `6` (computed from first `x`)
3. Third `x` is bound to `12` (computed from second `x`)

### Shadowing vs Mutation

**Shadowing creates a NEW binding:**

```nano
fn shadow_demo() -> string {
    let x: int = 42
    let x: string = "hello"  # Different type, new binding
    return x
}

shadow shadow_demo {
    assert (== (shadow_demo) "hello")
}
```

**Mutation updates EXISTING variable:**

```nano
fn mutation_demo() -> int {
    let mut x: int = 42
    set x 100  # Same variable, updated value
    # Can't change type with set
    return x
}

shadow mutation_demo {
    assert (== (mutation_demo) 100)
}
```

### When to Use Shadowing

**Use shadowing for:**
- Transforming a value through multiple steps
- Changing the type of a value
- Avoiding name pollution

```nano
fn transform_example(input: string) -> int {
    let input: string = (+ input "!")     # Add exclamation
    let input: int = (str_length input)   # Convert to length
    let input: int = (* input 2)          # Double it
    return input
}

shadow transform_example {
    assert (== (transform_example "hi") 6)  # "hi!" = 3 chars, * 2 = 6
}
```

### Function Parameters and Scope

Function parameters create bindings in the function's scope:

```nano
fn parameter_scope(x: int, y: int) -> int {
    # x and y are immutable bindings
    let sum: int = (+ x y)
    # Can shadow parameters if needed
    let x: int = (* x 2)
    return (+ x sum)
}

shadow parameter_scope {
    assert (== (parameter_scope 3 4) 13)
    # sum = 3+4 = 7, new x = 3*2 = 6, return 6+7 = 13
}
```

### Complete Example: Fibonacci

```nano
fn fibonacci(n: int) -> int {
    if (<= n 1) {
        return n
    }
    
    let mut prev: int = 0
    let mut curr: int = 1
    let mut i: int = 2
    
    while (<= i n) {
        let next: int = (+ prev curr)
        set prev curr
        set curr next
        set i (+ i 1)
    }
    
    return curr
}

shadow fibonacci {
    assert (== (fibonacci 0) 0)
    assert (== (fibonacci 1) 1)
    assert (== (fibonacci 5) 5)
    assert (== (fibonacci 10) 55)
}
```

### Best Practices

**1. Prefer immutable (let) over mutable (let mut)**

```nano
# ✅ Good: Immutable transformation
fn good_transform(x: int) -> int {
    let doubled: int = (* x 2)
    let plus_ten: int = (+ doubled 10)
    return plus_ten
}

# ❌ Less good: Unnecessary mutation
fn less_good_transform(x: int) -> int {
    let mut result: int = x
    set result (* result 2)
    set result (+ result 10)
    return result
}
```

**2. Use descriptive names**

```nano
# ✅ Good: Clear names
fn calculate_total(price: float, quantity: int) -> float {
    let price_per_item: float = price
    let item_count: float = (int_to_float quantity)
    let total: float = (* price_per_item item_count)
    return total
}

# ❌ Bad: Unclear names
fn calc(p: float, q: int) -> float {
    let x: float = p
    let y: float = (int_to_float q)
    let z: float = (* x y)
    return z
}
```

**3. Limit scope of mutable variables**

```nano
# ✅ Good: Narrow scope
fn good_scope() -> int {
    let result: int = (cond
        (true (
            let mut temp: int = 0
            set temp 5
            temp
        ))
        (else 0)
    )
    return result
}
```

### Summary

In this chapter, you learned:
- ✅ `let` creates immutable bindings (default, preferred)
- ✅ `let mut` creates mutable variables (use sparingly)
- ✅ `set` updates mutable variables
- ✅ Variables have lexical scope
- ✅ Shadowing creates new bindings with the same name

### Practice Exercises

```nano
# 1. Compute factorial using mut
fn factorial(n: int) -> int {
    let mut result: int = 1
    let mut i: int = 1
    while (<= i n) {
        set result (* result i)
        set i (+ i 1)
    }
    return result
}

shadow factorial {
    assert (== (factorial 5) 120)
    assert (== (factorial 0) 1)
}

# 2. Count positive numbers in array
fn count_positive(arr: array<int>) -> int {
    let mut count: int = 0
    let mut i: int = 0
    while (< i (array_length arr)) {
        if (> (array_get arr i) 0) {
            set count (+ count 1)
        }
        set i (+ i 1)
    }
    return count
}

shadow count_positive {
    assert (== (count_positive [1, -2, 3, -4, 5]) 3)
}

# 3. Use shadowing to transform a value
fn transform_by_shadowing(x: int) -> int {
    let x: int = (+ x 10)
    let x: int = (* x 2)
    let x: int = (- x 5)
    return x
}

shadow transform_by_shadowing {
    assert (== (transform_by_shadowing 5) 25)
    # (5+10)*2-5 = 15*2-5 = 30-5 = 25
}
```

---

**Previous:** [Chapter 2: Basic Syntax & Types](02_syntax_types.md)  
**Next:** [Chapter 4: Functions](04_functions.md)
