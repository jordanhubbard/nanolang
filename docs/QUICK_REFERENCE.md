# My Syntax at a Glance

I am a set of convictions expressed as syntax. This is how I am written.

## Basic Syntax

### Comments
```nano
# Single-line comment
/* Multi-line comment */
```

### Function Definition
I require a shadow test for every function. If you do not provide one, I will not compile.
```nano
fn name(param: type) -> return_type {
    # body
}

shadow name {
    # tests
}
```

### Variable Declaration
My variables are immutable by default. Use `mut` if you intend to change a value.
```nano
let x: int = 42              # Immutable
let mut y: int = 0           # Mutable
```

### Variable Assignment
```nano
set variable_name new_value
```

## Types

I am statically typed. I do not guess what you mean.

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
I use `match` to ensure you handle every case.
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
I resolve generics at compile time.
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
I treat functions as values. They have specific types.
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

## Operators (Prefix and Infix)

I support both prefix `(+ a b)` and infix `a + b` notation for all operators. I evaluate everything from left to right.

### Arithmetic
```nano
# Prefix              # Infix
(+ a b)               a + b
(- a b)               a - b
(* a b)               a * b
(/ a b)               a / b
(% a b)               a % b
```

### Comparison
```nano
# Prefix              # Infix
(== a b)              a == b
(!= a b)              a != b
(< a b)               a < b
(<= a b)              a <= b
(> a b)               a > b
(>= a b)              a >= b
```

### Logical
```nano
# Prefix              # Infix
(and a b)             a and b
(or a b)              a or b
(not a)               not a
```

### Precedence

All my infix operators have equal precedence. I evaluate them left-to-right. I do not use PEMDAS. Use parentheses if you need a different order.

```nano
a * (b + c)           # Parentheses required for non-left-to-right evaluation
2 + 3 * 4             # Evaluates as (2 + 3) * 4 = 20
```

Unary `not` and `-` do not require parentheses: `not flag`, `-x`.

## Control Flow

### If Expression
I require both branches.
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

## Standard Library (72 Functions)

I provide 72 built-in functions. This list covers the most common ones. See `docs/STDLIB.md` and `spec.json` for the rest.

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

I have documented the complete set in `docs/STDLIB.md`.

## Shadow-Tests

I demand a shadow test for every function. If you write a function, you must say at least one true thing about what it does.

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

**Namespace Keywords:**
- `module` - I use this to declare a namespace.
- `pub` - I use this to export a symbol.
- `from` - I use this for selective imports.
- `use` - I use this to re-export symbols.

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
let result: int = (* (+ a b) (- c d))       # Prefix
let result: int = (a + b) * (c - d)          # Infix (parens for grouping)

# Calculate: (x > 0) && (x < 10)
if (and (> x 0) (< x 10)) {                 # Prefix
    # x is between 0 and 10
}
if x > 0 and x < 10 {                       # Infix
    # same thing - equal precedence, left-to-right
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
    (println (helper2 10))
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

### Module with Namespace
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

I explain namespaces in detail in `docs/NAMESPACE_USAGE.md`.

## Compilation

### C Backend (default)
I transpile to C for native performance.
```bash
nanoc program.nano -o program       # Compile to native binary
nanoc program.nano --keep-c -o prog # Keep generated C source
```

### NanoISA VM Backend
I can also compile to my own virtual machine.
```bash
nano_virt program.nano --run              # Compile + run in VM
nano_virt program.nano -o program         # Native binary (embeds VM)
nano_virt program.nano --emit-nvm -o p.nvm  # Raw bytecode
nano_vm p.nvm                             # Execute bytecode
nano_vm --isolate-ffi p.nvm              # FFI in separate process
nano_vm --daemon p.nvm                    # Run via VM daemon
```

### Build Targets
```bash
make build        # Build C compiler (nanoc)
make vm           # Build VM backend (nano_virt, nano_vm, nano_cop, nano_vmd)
make test         # Run tests with C backend
make test-vm      # Run tests with VM backend
make install      # Install all binaries
```

## Common Mistakes

I support both prefix and infix notation for operators.
```nano
let x: int = (+ a b)  # Prefix - I accept this
let x: int = a + b    # Infix - I also accept this
```

I do not use PEMDAS. I evaluate left-to-right.
```nano
let x: int = 2 + 3 * 4  # I evaluate this as (2+3)*4 = 20, NOT 2+(3*4)
```

I expect you to use parentheses if you need a specific order.
```nano
let x: int = 2 + (3 * 4)  # Explicit grouping: 14
```

I will refuse to compile if you omit the type.
```nano
let x = 42  # I will refuse this
```

I require explicit types.
```nano
let x: int = 42  # This is what I expect
```

I will not compile a function without a shadow test.
```nano
fn double(x: int) -> int {
    return (* x 2)
}
# I will refuse this: Missing shadow test
```

I require a test block.
```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
}
# I accept this
```

I do not allow mutation of immutable variables.
```nano
let x: int = 10
set x 20  # I will refuse this
```

I require the `mut` keyword for variables you intend to change.
```nano
let mut x: int = 10
set x 20  # I accept this
```

I require an `else` branch for every `if`.
```nano
if (> x 0) {
    return 1
}
# I will refuse this: Missing else
```

I expect a complete expression.
```nano
if (> x 0) {
    return 1
} else {
    return 0
}
# I accept this
```

## Tips

1. I evaluate infix operators from left to right. Use parentheses or prefix notation to be clear.
   - Infix: `a + (b * c)` or prefix: `(+ a (* b c))`
   - Infix: `x == 5 and y < 10` or prefix: `(and (== x 5) (< y 10))`

2. I expect you to test edge cases. Always test 0, negatives, and boundaries in your shadow tests.

3. I find small functions easier to verify and understand.

4. I prefer descriptive names like `calculate_total` over `calc`.

5. I recommend using immutable variables by default. Only use `mut` when you have no other choice.

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
            (print n)
        }
    }
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

## Interactive REPL

I include a REPL for interactive development.

### Build & Launch
```bash
./bin/nanoc examples/language/full_repl.nano -o bin/repl
./bin/repl
```

### Features
- I maintain persistent variables: `let x: int = 42`
- I accept function definitions: `fn double(x: int) -> int { return (* x 2) }`
- I support multi-line input with continuation prompts (`....>`)
- I allow type-specific evaluation: `:int`, `:float`, `:string`, `:bool`
- I provide session management: `:vars`, `:funcs`, `:imports`, `:clear`, `:quit`

### Example Session
```nano
nano> let x: int = 42
Defined: x

nano> fn double(n: int) -> int {
....>     return (* n 2)
....> }
Defined: double(n: int) -> int

nano> (double x)
=> 84

nano> :float (* 3.14 2.0)
=> 6.28

nano> :vars
Defined variables: x

nano> :funcs
Defined functions: double(n: int) -> int
```

### REPL Commands

| Command | Description |
|---------|-------------|
| `:vars` | List variables |
| `:funcs` | List functions |
| `:imports` | List imports |
| `:clear` | Clear session |
| `:quit` | Exit (or Ctrl-D) |

I designed this for learning, prototyping, and quick calculations. See `examples/language/full_repl.nano` for how I implemented it.

## Resources

- I have a full guide in `docs/GETTING_STARTED.md`.
- I have a detailed specification in `docs/SPECIFICATION.md`.
- I explain my testing requirements in `docs/SHADOW_TESTS.md`.
- I have many examples in the `examples/` directory.

---

I say what I mean, I prove what I claim, and I compile myself.
