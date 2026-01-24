# Appendix B: Quick Reference

**Single-page cheat sheet for NanoLang syntax and built-in functions.**

## B.1 Operator Precedence

NanoLang uses prefix notation, so there's no operator precedence ambiguity. All operators are function-like:

```nano
(+ a b)      # Addition
(- a b)      # Subtraction
(* a b)      # Multiplication
(/ a b)      # Division
(% a b)      # Modulo
```

For nested operations, parentheses make order explicit:

```nano
(+ (* a b) (/ c d))    # (a*b) + (c/d)
(* (+ a 1) (- b 2))    # (a+1) * (b-2)
```

### Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal | `(== x 5)` |
| `!=` | Not equal | `(!= x 5)` |
| `<` | Less than | `(< x 5)` |
| `>` | Greater than | `(> x 5)` |
| `<=` | Less than or equal | `(<= x 5)` |
| `>=` | Greater than or equal | `(>= x 5)` |

### Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `and` | Logical AND (short-circuit) | `(and cond1 cond2)` |
| `or` | Logical OR (short-circuit) | `(or cond1 cond2)` |
| `not` | Logical NOT | `(not condition)` |

## B.2 Built-in Functions

### Arithmetic

| Function | Signature | Description |
|----------|-----------|-------------|
| `+` | `(+ a b) -> int\|float` | Addition |
| `-` | `(- a b) -> int\|float` | Subtraction |
| `*` | `(* a b) -> int\|float` | Multiplication |
| `/` | `(/ a b) -> int\|float` | Division |
| `%` | `(% a b) -> int` | Modulo (integers only) |
| `abs` | `abs(int\|float) -> int\|float` | Absolute value |
| `min` | `min(a, b) -> int\|float` | Minimum of two values |
| `max` | `max(a, b) -> int\|float` | Maximum of two values |
| `sqrt` | `sqrt(float) -> float` | Square root |
| `pow` | `pow(base, exp) -> float` | Power |
| `floor` | `floor(float) -> float` | Round down |
| `ceil` | `ceil(float) -> float` | Round up |
| `round` | `round(float) -> float` | Round to nearest |
| `sin` | `sin(float) -> float` | Sine (radians) |
| `cos` | `cos(float) -> float` | Cosine (radians) |
| `tan` | `tan(float) -> float` | Tangent (radians) |

### String Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `+` | `(+ s1 s2) -> string` | Concatenation (preferred) |
| `str_length` | `str_length(string) -> int` | String length |
| `str_substring` | `str_substring(s, start, len) -> string` | Extract substring |
| `str_contains` | `str_contains(s, substr) -> bool` | Check for substring |
| `==` | `(== s1 s2) -> bool` | String equality (preferred) |
| `str_equals` | `str_equals(s1, s2) -> bool` | String equality |
| `char_at` | `char_at(s, index) -> int` | Get ASCII value at index |
| `string_from_char` | `string_from_char(int) -> string` | Create string from ASCII |
| `is_digit` | `is_digit(int) -> bool` | Check if character is digit |
| `is_alpha` | `is_alpha(int) -> bool` | Check if character is letter |
| `is_alnum` | `is_alnum(int) -> bool` | Check if alphanumeric |
| `is_whitespace` | `is_whitespace(int) -> bool` | Check if whitespace |
| `is_upper` | `is_upper(int) -> bool` | Check if uppercase |
| `is_lower` | `is_lower(int) -> bool` | Check if lowercase |
| `char_to_lower` | `char_to_lower(int) -> int` | Convert to lowercase |
| `char_to_upper` | `char_to_upper(int) -> int` | Convert to uppercase |

### Array Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `array_new` | `array_new(size, default) -> array<T>` | Create new array |
| `array_get` | `array_get(arr, index) -> T` | Get element (bounds-checked) |
| `at` | `at(arr, index) -> T` | Get element (alias) |
| `array_set` | `array_set(arr, i, val) -> array<T>` | Set element (returns new array) |
| `array_length` | `array_length(arr) -> int` | Get array length |
| `array_push` | `array_push(arr, elem) -> array<T>` | Append element |
| `array_pop` | `array_pop(arr) -> T` | Remove and return last element |
| `array_remove_at` | `array_remove_at(arr, i) -> array<T>` | Remove element at index |
| `filter` | `filter(arr, fn) -> array<T>` | Keep elements matching predicate |
| `map` | `map(arr, fn) -> array<T>` | Transform each element |
| `reduce` | `reduce(arr, init, fn) -> A` | Fold array into single value |

### Type Conversions

| Function | Signature | Description |
|----------|-----------|-------------|
| `int_to_string` | `int_to_string(int) -> string` | Convert int to string |
| `string_to_int` | `string_to_int(string) -> int` | Parse string to int |
| `digit_value` | `digit_value(int) -> int` | Convert '5' char to 5 int |

### I/O Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `print` | `print(any) -> void` | Print without newline |
| `println` | `println(any) -> void` | Print with newline |
| `assert` | `assert(bool) -> void` | Runtime assertion |
| `read_line` | `read_line() -> string` | Read line from stdin |

## B.3 Standard Library Overview

### Core Utilities

Always available without imports:

- **I/O**: `print`, `println`, `assert`
- **Math**: `+`, `-`, `*`, `/`, `%`, `abs`, `min`, `max`, `sqrt`, `pow`
- **Trig**: `sin`, `cos`, `tan`, `floor`, `ceil`, `round`
- **Strings**: `str_length`, `str_substring`, `char_at`, `+` (concat)
- **Arrays**: `array_new`, `array_get`, `array_set`, `array_push`
- **OS**: `getcwd`, `getenv`, `range`

### Collections

```nano
# Dynamic arrays (built-in)
let arr: array<int> = [1, 2, 3]
set arr (array_push arr 4)

# Generic lists
let nums: List<int> = (List_int_new)
(List_int_push nums 42)
let val: int = (List_int_get nums 0)
```

### Filesystem

Import from `modules/std/fs.nano`:

- `read(path) -> string` - Read file contents
- `write(path, content) -> void` - Write to file
- `exists(path) -> bool` - Check if file exists
- `walkdir(path) -> array<string>` - List directory contents

### System

Import from `modules/std/env.nano` or `modules/std/process.nano`:

- `getcwd() -> string` - Get current directory
- `getenv(name) -> string` - Get environment variable
- `run(command) -> int` - Execute shell command

## B.4 Module Index

### Text Processing

| Module | Purpose |
|--------|---------|
| `modules/std/log/log.nano` | Structured logging with levels |
| `modules/std/collections/StringBuilder.nano` | Efficient string building |
| `stdlib/regex.nano` | Regular expressions |

### Data Formats

| Module | Purpose |
|--------|---------|
| `modules/std/json/json.nano` | JSON parsing and generation |
| `modules/sqlite/sqlite.nano` | SQLite database access |

### Web & Networking

| Module | Purpose |
|--------|---------|
| `modules/curl/curl.nano` | HTTP client requests |
| `modules/http_server/http_server.nano` | HTTP server |
| `modules/uv/uv.nano` | Async I/O (libuv) |

### Graphics

| Module | Purpose |
|--------|---------|
| `modules/sdl/sdl.nano` | 2D graphics, windows, events |
| `modules/sdl_mixer/sdl_mixer.nano` | Audio playback |
| `modules/sdl_ttf/sdl_ttf.nano` | TrueType font rendering |
| `modules/sdl_image/sdl_image.nano` | Image loading |
| `modules/glfw/glfw.nano` | OpenGL windows and input |
| `modules/opengl/opengl.nano` | 3D graphics |
| `modules/ncurses/ncurses.nano` | Terminal UI |

### Testing

| Module | Purpose |
|--------|---------|
| `stdlib/coverage.nano` | Code coverage tracking |
| `modules/proptest/proptest.nano` | Property-based testing |

## B.5 Control Flow Cheat Sheet

### Variables

```nano
let x: int = 42                    # Immutable
let mut counter: int = 0           # Mutable
set counter (+ counter 1)          # Assignment
```

### Conditionals

```nano
# Expression (returns value) - use cond
let sign: int = (cond
    ((< x 0) -1)
    ((> x 0) 1)
    (else 0)
)

# Statement (side effects) - use if/else
if (< x 0) {
    (println "negative")
} else {
    (println "non-negative")
}
```

### Loops

```nano
# While loop
while (< i 10) {
    set i (+ i 1)
}

# For loop
for (let i: int = 0) (< i 10) (set i (+ i 1)) {
    (println (int_to_string i))
}
```

### Functions

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
}
```

### Pattern Matching

```nano
match result {
    Ok(v) => { (println v.value) }
    Err(e) => { (println e.error) }
}
```

## B.6 Types Quick Reference

### Primitive Types

| Type | Size | Description |
|------|------|-------------|
| `int` | 64-bit | Signed integer |
| `float` | 64-bit | IEEE 754 double |
| `bool` | 1-bit | `true` or `false` |
| `string` | ptr | UTF-8 text |
| `void` | 0 | No value (return type only) |

### Composite Types

```nano
# Struct
struct Point { x: int, y: int }
let p: Point = Point { x: 10, y: 20 }

# Enum
enum Status { Idle = 0, Running = 1, Done = 2 }
let s: Status = Status.Running

# Union (tagged union)
union Result { Ok { value: int }, Error { msg: string } }

# Tuple
let coord: (int, int) = (10, 20)
let x: int = coord.0

# Array
let arr: array<int> = [1, 2, 3]

# Function type
let f: fn(int) -> int = double
```

---

**Previous:** [Appendix A: Examples Gallery](a_examples_gallery.md)  
**Next:** [Appendix C: Troubleshooting Guide](c_troubleshooting.md)
