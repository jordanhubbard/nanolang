# nanolang Quick Reference

A one-page reference for nanolang syntax and features.

## Basic Syntax

### Comments
```nano
# Single-line comment
/* Multi-line comment */
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

### Composite Types

**Structs:**
```nano
struct Point { x: int, y: int }
let p: Point = Point { x: 10, y: 20 }
let x: int = p.x
```

**Enums:**
```nano
enum Status { Pending = 0, Active = 1 }
let s: int = Status.Active
```

**Unions (Tagged Unions):**
```nano
union Result {
    Ok { value: int },
    Error { code: int }
}

let r: Result = Result.Ok { value: 42 }

match r {
    Ok(v) => (println "Success"),
    Error(e) => (println "Failed")
}
```

**Generics (Monomorphized):**
```nano
# Built-in generic: List<T>
let numbers: List<int> = (List_int_new)
(List_int_push numbers 42)
(List_int_push numbers 17)

let len: int = (List_int_length numbers)
let first: int = (List_int_get numbers 0)

# Generic with user-defined types
struct Point { x: int, y: int }
let points: List<Point> = (List_Point_new)
```

**First-Class Functions:**
```nano
# Function type: fn(param_types) -> return_type
fn double(x: int) -> int {
    return (* x 2)
}

# Assign function to variable
let f: fn(int) -> int = double
let result: int = (f 5)  # result = 10

# Function as parameter
fn apply_op(op: fn(int) -> int, x: int) -> int {
    return (op x)
}

let y: int = (apply_op double 7)  # y = 14

# Function as return value
fn get_doubler() -> fn(int) -> int {
    return double
}

let op: fn(int) -> int = (get_doubler)
```

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

## Standard Library (37 Functions)

### Core I/O (3)
```nano
(print value)      # Print without newline
(println value)    # Print with newline
(assert condition) # Runtime assertion
```

### Math Operations (11)
```nano
(abs x)            # Absolute value
(min a b)          # Minimum of two values
(max a b)          # Maximum of two values
(sqrt x)           # Square root
(pow base exp)     # Power function
(floor x)          # Round down
(ceil x)           # Round up
(round x)          # Round to nearest
(sin x)            # Sine (radians)
(cos x)            # Cosine (radians)
(tan x)            # Tangent (radians)
```

### String Operations (18)

**Basic String Operations (5):**
```nano
(str_length s)           # Get string length
(str_concat s1 s2)       # Concatenate strings
(str_substring s pos len)# Extract substring
(str_contains s sub)     # Check if contains substring
(str_equals s1 s2)       # String equality
```

**Character Access (2):**
```nano
(char_at s index)        # Get ASCII value at index (0-based)
(string_from_char code)  # Create string from ASCII value
```

**Character Classification (6):**
```nano
(is_digit c)             # Check if '0'-'9'
(is_alpha c)             # Check if a-z, A-Z
(is_alnum c)             # Check if digit or letter
(is_whitespace c)        # Check if space/tab/newline
(is_upper c)             # Check if A-Z
(is_lower c)             # Check if a-z
```

**Type Conversions (5):**
```nano
(int_to_string n)        # Convert int to string
(string_to_int s)        # Parse string to int
(digit_value c)          # Convert '5' -> 5
(char_to_lower c)        # Convert 'A' -> 'a'
(char_to_upper c)        # Convert 'a' -> 'A'
```

### Array Operations (4)
```nano
(at arr index)           # Get element (bounds-checked)
(array_length arr)       # Get array length
(array_new size default) # Create new array
(array_set arr idx val)  # Set element (bounds-checked)
```

### OS/System (3)
```nano
(getcwd)           # Get current directory
(getenv name)      # Get environment variable
(range start end)  # Range iterator (for loops only)
```

📖 **Full documentation:** See [`docs/STDLIB.md`](STDLIB.md) for complete reference with examples and type signatures.

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
range    module   pub      from     import   use
struct   enum     union    extern   as
```

**Namespace Keywords (NEW!):**
- `module` - Declare module namespace
- `pub` - Make symbol public (exported)
- `from` - Selective import syntax
- `use` - Re-export symbols

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

### Basic Program
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

### Module with Namespace (NEW!)
```nano
/* Declare module namespace */
module my_app

/* Import with selective symbols */
from "modules/std/io/stdio.nano" import fopen, fclose

/* Public API - exported to other modules */
pub fn process_file(path: string) -> bool {
    let file: int = (fopen path "rb")
    if (== file 0) {
        return false
    } else {
        (fclose file)
        return true
    }
}

/* Private helper - module-only */
fn validate_path(path: string) -> bool {
    return true  # Simplified
}

/* Public struct */
pub struct Config {
    enabled: bool,
    timeout: int
}

/* Main entry point */
pub fn main() -> int {
    let cfg: Config = Config { enabled: true, timeout: 30 }
    return 0
}
```

**Key Namespace Features:**
- `module name` - Declares namespace
- `pub fn` - Public function (exported)
- `fn` (no pub) - Private function (module-only)
- `from "path" import symbols` - Selective imports
- See `docs/NAMESPACE_USAGE.md` for complete guide

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
