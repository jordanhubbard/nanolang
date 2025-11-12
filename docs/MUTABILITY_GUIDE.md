# Variable Mutability in nanolang

## Summary

nanolang uses **explicit mutability** for safety and clarity.

## Syntax

```nano
let x: int = 5           # Immutable (default, safe)
let mut y: int = 10      # Mutable (explicit, clear intent)
```

## Rules

### Immutable Variables (Default)

```nano
let x: int = 5
# x can be read but NOT modified
# set x 10  # ❌ Error: Cannot assign to immutable variable 'x'
```

**Benefits:**
- Safe by default
- Prevents accidental modification
- Makes code easier to reason about
- Compiler can optimize better

### Mutable Variables (Explicit)

```nano
let mut counter: int = 0
set counter (+ counter 1)  # ✅ OK: counter is mutable
```

**When to use `mut`:**
- Loop counters
- Accumulator variables  
- Any variable that changes over time
- Algorithm state that evolves

## Examples

### ❌ Wrong (Immutable by default)

```nano
fn sum_to_n(n: int) -> int {
    let sum: int = 0
    let i: int = 1
    
    while (<= i n) {
        set sum (+ sum i)  # ❌ Error: sum is immutable
        set i (+ i 1)      # ❌ Error: i is immutable
    }
    
    return sum
}
```

### ✅ Correct (Explicit mut)

```nano
fn sum_to_n(n: int) -> int {
    let mut sum: int = 0      # ✅ Mutable
    let mut i: int = 1        # ✅ Mutable
    
    while (<= i n) {
        set sum (+ sum i)     # ✅ OK
        set i (+ i 1)         # ✅ OK
    }
    
    return sum
}
```

### Mixed (Some mutable, some not)

```nano
fn calculate(n: int, multiplier: int) -> int {
    let fixed_value: int = 100          # Immutable (never changes)
    let mut result: int = 0             # Mutable (accumulator)
    let mut i: int = 0                  # Mutable (counter)
    
    while (< i n) {
        set result (+ result (* i multiplier))
        set i (+ i 1)
    }
    
    return (+ result fixed_value)
}
```

## Philosophy

nanolang follows the **principle of least privilege**:

1. **Immutable by default** - Variables cannot change unless explicitly marked
2. **Explicit mutability** - `mut` keyword makes intent clear
3. **Compiler enforced** - Type checker prevents mistakes

This design is inspired by Rust and provides:
- **Safety** - Prevents accidental modification
- **Clarity** - Readers know which variables change
- **Optimization** - Compiler knows what can't change

## Common Patterns

### Loop Counter

```nano
let mut i: int = 0
while (< i 10) {
    (println i)
    set i (+ i 1)
}
```

### Accumulator

```nano
let mut sum: int = 0
let mut product: int = 1

for i in (range 1 11) {
    set sum (+ sum i)
    set product (* product i)
}
```

### State Machine

```nano
let mut state: int = 0
let mut running: bool = true

while running {
    if (== state 0) {
        set state 1
    } else {
        if (== state 1) {
            set state 2
        } else {
            set running false
        }
    }
}
```

### Conditional Update

```nano
let mut max: int = 0

for x in (range 1 100) {
    if (> x max) {
        set max x
    } else {
        # max stays the same
    }
}
```

## Type Checker Errors

### Error: Cannot assign to immutable variable

```nano
let x: int = 5
set x 10  # ❌ Error: Cannot assign to immutable variable 'x'
```

**Solution:** Add `mut` keyword

```nano
let mut x: int = 5
set x 10  # ✅ OK
```

### Error: Undefined variable

```nano
set x 10  # ❌ Error: Undefined variable 'x'
```

**Solution:** Declare before using

```nano
let mut x: int = 0
set x 10  # ✅ OK
```

## Comparison with Other Languages

### Rust

```rust
let x = 5;           // immutable
let mut y = 10;      // mutable
```

nanolang uses the **same syntax as Rust**.

### JavaScript

```javascript
const x = 5;         // immutable (cannot reassign)
let y = 10;          // mutable
var z = 15;          // mutable (legacy)
```

### Python

```python
x = 5                // mutable (everything is mutable)
```

Python has no immutability by default. nanolang is safer.

### C

```c
const int x = 5;     // immutable
int y = 10;          // mutable
```

C uses `const` keyword. nanolang uses `mut` for mutability instead.

## Best Practices

1. **Default to immutable** - Use plain `let` unless you need to change the value
2. **Add `mut` when needed** - Only make variables mutable if they'll be modified
3. **Group declarations** - Put all `let mut` declarations together at the start
4. **Comment intent** - Explain WHY a variable needs to be mutable

```nano
fn process_data(n: int) -> int {
    # Constants (never change)
    let max_iterations: int = 1000
    let threshold: float = 0.001
    
    # Mutable state (updated in loop)
    let mut result: int = 0
    let mut error: float = 1.0
    let mut iteration: int = 0
    
    # Algorithm...
    while (and (< iteration max_iterations) (> error threshold)) {
        set result (calculate result)
        set error (compute_error result)
        set iteration (+ iteration 1)
    }
    
    return result
}
```

## Summary

✅ **DO:**
- Use `let` for values that never change
- Use `let mut` for values that will be modified
- Make mutability explicit and intentional

❌ **DON'T:**
- Use `let mut` for everything "just in case"
- Modify variables unnecessarily
- Change variable types (not allowed in nanolang)

---

**Remember:** Immutable by default, mutable when needed. This makes nanolang code safe, clear, and correct.

