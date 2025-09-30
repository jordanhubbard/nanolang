# nanolang Quick Reference

A one-page reference for nanolang syntax and features.

## Basic Syntax

### Comments
```nano
# Single-line comment only
```

### Function Definition
```nano
fn name(param: type) -> return_type {
    # body
}

shadow name {
    # tests
}
```

### Variable Declaration
```nano
let x: int = 42              # Immutable
let mut y: int = 0           # Mutable
```

### Variable Assignment
```nano
set variable_name new_value
```

## Types

| Type     | Description           | Example        |
|----------|-----------------------|----------------|
| `int`    | 64-bit integer        | `42`, `-17`    |
| `float`  | 64-bit float          | `3.14`, `-0.5` |
| `bool`   | Boolean               | `true`, `false`|
| `string` | UTF-8 text            | `"hello"`      |
| `void`   | No value (return only)| -              |

## Operators (Prefix Notation)

### Arithmetic
```nano
(+ a b)    # a + b
(- a b)    # a - b
(* a b)    # a * b
(/ a b)    # a / b
(% a b)    # a % b
```

### Comparison
```nano
(== a b)   # a == b
(!= a b)   # a != b
(< a b)    # a < b
(<= a b)   # a <= b
(> a b)    # a > b
(>= a b)   # a >= b
```

### Logical
```nano
(and a b)  # a && b
(or a b)   # a || b
(not a)    # !a
```

## Control Flow

### If Expression
```nano
if condition {
    # then branch
} else {
    # else branch
}
# Both branches required
```

### While Loop
```nano
while condition {
    # body
}
```

### For Loop
```nano
for var in (range start end) {
    # body
}
```

### Return
```nano
return expression
```

## Built-in Functions

```nano
print value        # Print to stdout
assert condition   # Assert (for shadow-tests)
range start end    # Generate range [start, end)
```

## Shadow-Tests

Every function must have a shadow-test:

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -1 1) 0)
}
```

## Keywords

```
fn       let      mut      set      if       else
while    for      in       return   assert   shadow
int      float    bool     string   void
true     false    print    and      or       not
range
```

## Common Patterns

### Function with Multiple Parameters
```nano
fn max(a: int, b: int) -> int {
    if (> a b) {
        return a
    } else {
        return b
    }
}

shadow max {
    assert (== (max 5 3) 5)
    assert (== (max 3 5) 5)
}
```

### Recursive Function
```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    } else {
        return (* n (factorial (- n 1)))
    }
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 5) 120)
}
```

### Loop with Mutable Variable
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
    assert (== (sum_to_n 5) 15)
    assert (== (sum_to_n 10) 55)
}
```

### Boolean Function
```nano
fn is_even(n: int) -> bool {
    return (== (% n 2) 0)
}

shadow is_even {
    assert (== (is_even 4) true)
    assert (== (is_even 5) false)
}
```

### Nested Operations
```nano
# Calculate: (a + b) * (c - d)
let result: int = (* (+ a b) (- c d))

# Calculate: (x > 0) && (x < 10)
if (and (> x 0) (< x 10)) {
    # x is between 0 and 10
}
```

## Program Structure

```nano
# 1. Define helper functions
fn helper1(x: int) -> int {
    return (* x 2)
}

shadow helper1 {
    assert (== (helper1 5) 10)
}

# 2. Define more functions
fn helper2(x: int) -> int {
    return (+ (helper1 x) 1)
}

shadow helper2 {
    assert (== (helper2 5) 11)
}

# 3. Define main
fn main() -> int {
    print (helper2 10)
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

## Common Mistakes

❌ **Infix notation**
```nano
let x: int = a + b  # WRONG
```

✅ **Prefix notation**
```nano
let x: int = (+ a b)  # CORRECT
```

---

❌ **Missing type**
```nano
let x = 42  # WRONG
```

✅ **Explicit type**
```nano
let x: int = 42  # CORRECT
```

---

❌ **No shadow-test**
```nano
fn double(x: int) -> int {
    return (* x 2)
}
# WRONG: Missing shadow test
```

✅ **With shadow-test**
```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
}
# CORRECT
```

---

❌ **Mutating immutable**
```nano
let x: int = 10
set x 20  # WRONG
```

✅ **Declare as mutable**
```nano
let mut x: int = 10
set x 20  # CORRECT
```

---

❌ **If without else**
```nano
if (> x 0) {
    return 1
}
# WRONG: Missing else
```

✅ **Complete if/else**
```nano
if (> x 0) {
    return 1
} else {
    return 0
}
# CORRECT
```

## Tips

1. **Think prefix**: Convert math to prefix before writing
   - `a + b * c` → `(+ a (* b c))`
   - `x == 5 && y < 10` → `(and (== x 5) (< y 10))`

2. **Test edge cases**: Always test 0, negatives, boundaries

3. **Keep functions small**: Easier to test and understand

4. **Use descriptive names**: `calculate_total` not `calc`

5. **Immutable first**: Only use `mut` when necessary

## Complete Example

```nano
# Check if a number is prime
fn is_prime(n: int) -> bool {
    if (< n 2) {
        return false
    }
    let mut i: int = 2
    while (< i n) {
        if (== (% n i) 0) {
            return false
        }
        set i (+ i 1)
    }
    return true
}

shadow is_prime {
    assert (== (is_prime 2) true)
    assert (== (is_prime 4) false)
    assert (== (is_prime 17) true)
}

fn main() -> int {
    for n in (range 1 20) {
        if (is_prime n) {
            print n
        }
    }
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

## Resources

- Full guide: [GETTING_STARTED.md](GETTING_STARTED.md)
- Language spec: [SPECIFICATION.md](SPECIFICATION.md)
- Testing guide: [SHADOW_TESTS.md](SHADOW_TESTS.md)
- Examples: [examples/](examples/)

---

**nanolang** - Minimal, LLM-friendly, test-driven programming
