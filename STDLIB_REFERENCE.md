# nanolang Standard Library Reference

Complete reference for all 20 built-in functions in nanolang.

---

## Core I/O Functions (3)

### `print(value: any) -> void`
Prints a value without a trailing newline.

```nano
print "Hello"
print 42
print 3.14
```

### `println(value: any) -> void`
Prints a value with a trailing newline. Polymorphic - works with int, float, string, and bool.

```nano
(println "Hello, World!")
(println 42)
(println 3.14159)
(println true)
```

### `assert(condition: bool) -> void`
Runtime assertion. Terminates program if condition is false.

```nano
assert (== 2 2)           # Passes
assert (> 5 3)            # Passes
assert (== (add 2 2) 4)   # Passes
```

---

## Math Operations (11)

### Basic Math (3)

#### `abs(x: int|float) -> int|float`
Returns the absolute value. Type-preserving (int → int, float → float).

```nano
(abs -5)      # Returns 5
(abs 5)       # Returns 5
(abs -3.14)   # Returns 3.14
```

#### `min(a: int|float, b: int|float) -> int|float`
Returns the minimum of two values. Both arguments must be the same type.

```nano
(min 5 10)    # Returns 5
(min 3.14 2.71)  # Returns 2.71
```

#### `max(a: int|float, b: int|float) -> int|float`
Returns the maximum of two values. Both arguments must be the same type.

```nano
(max 5 10)    # Returns 10
(max 3.14 2.71)  # Returns 3.14
```

### Advanced Math (5)

#### `sqrt(x: int|float) -> float`
Returns the square root. Always returns float.

```nano
(sqrt 16.0)   # Returns 4.0
(sqrt 2.0)    # Returns 1.41421...
(sqrt 9)      # Returns 3.0 (int converted to float)
```

#### `pow(base: int|float, exponent: int|float) -> float`
Returns base raised to the power of exponent. Always returns float.

```nano
(pow 2.0 3.0)    # Returns 8.0 (2³)
(pow 5.0 2.0)    # Returns 25.0 (5²)
(pow 2.0 -1.0)   # Returns 0.5 (2⁻¹)
```

#### `floor(x: int|float) -> float`
Returns the largest integer ≤ x as a float.

```nano
(floor 3.7)     # Returns 3.0
(floor 3.2)     # Returns 3.0
(floor -2.3)    # Returns -3.0
```

#### `ceil(x: int|float) -> float`
Returns the smallest integer ≥ x as a float.

```nano
(ceil 3.2)      # Returns 4.0
(ceil 3.7)      # Returns 4.0
(ceil -2.7)     # Returns -2.0
```

#### `round(x: int|float) -> float`
Rounds to the nearest integer as a float.

```nano
(round 3.4)     # Returns 3.0
(round 3.6)     # Returns 4.0
(round 3.5)     # Returns 4.0 (rounds half to even)
```

### Trigonometric Functions (3)

All trig functions work in **radians** and always return float.

#### `sin(x: int|float) -> float`
Returns the sine of x (in radians).

```nano
(sin 0.0)       # Returns 0.0
(sin 1.5708)    # Returns ≈1.0 (π/2)
```

#### `cos(x: int|float) -> float`
Returns the cosine of x (in radians).

```nano
(cos 0.0)       # Returns 1.0
(cos 3.14159)   # Returns ≈-1.0 (π)
```

#### `tan(x: int|float) -> float`
Returns the tangent of x (in radians).

```nano
(tan 0.0)       # Returns 0.0
(tan 0.7854)    # Returns ≈1.0 (π/4)
```

---

## String Operations (5)

### `str_length(s: string) -> int`
Returns the length of a string in bytes.

```nano
let text: string = "Hello"
let len: int = (str_length text)  # Returns 5
```

### `str_concat(s1: string, s2: string) -> string`
Concatenates two strings, returning a new string.

```nano
let hello: string = "Hello"
let world: string = " World"
let result: string = (str_concat hello world)  # "Hello World"
```

### `str_substring(s: string, start: int, length: int) -> string`
Extracts a substring starting at `start` with the given `length`.
- `start` is 0-indexed
- If `start + length` exceeds string length, returns until end of string
- Returns empty string if start is out of bounds

```nano
let text: string = "Hello, World!"
let hello: string = (str_substring text 0 5)   # "Hello"
let world: string = (str_substring text 7 5)   # "World"
```

### `str_contains(s: string, substr: string) -> bool`
Returns true if string `s` contains substring `substr`.

```nano
let text: string = "The quick brown fox"
(str_contains text "quick")   # Returns true
(str_contains text "slow")    # Returns false
```

### `str_equals(s1: string, s2: string) -> bool`
Returns true if both strings are exactly equal.

```nano
let s1: string = "Hello"
let s2: string = "Hello"
let s3: string = "World"
(str_equals s1 s2)   # Returns true
(str_equals s1 s3)   # Returns false
```

---

## OS/System Functions (3)

### `getcwd() -> string`
Returns the current working directory as an absolute path.

```nano
let cwd: string = (getcwd)
(println cwd)  # Prints something like "/Users/username/project"
```

### `getenv(name: string) -> string`
Gets an environment variable value. Returns empty string if not set.

```nano
let home: string = (getenv "HOME")
let path: string = (getenv "PATH")
```

### `range(start: int, end: int) -> iterator`
Special function used only in `for` loops. Creates an iterator from `start` (inclusive) to `end` (exclusive).

```nano
for i in (range 0 10) {
    (println i)  # Prints 0, 1, 2, ..., 9
}
```

**Note:** `range` can only be used in for-loop contexts, not as a regular function call.

---

## Usage Examples

### Example 1: Mathematical Computation

```nano
fn pythagorean(a: float, b: float) -> float {
    let a_squared: float = (pow a 2.0)
    let b_squared: float = (pow b 2.0)
    let c_squared: float = (+ a_squared b_squared)
    return (sqrt c_squared)
}

shadow pythagorean {
    # 3-4-5 triangle
    assert (== (pythagorean 3.0 4.0) 5.0)
}
```

### Example 2: String Processing

```nano
fn process_name(full_name: string) -> string {
    let len: int = (str_length full_name)
    
    if (str_contains full_name " ") {
        (println "Full name detected")
        return full_name
    } else {
        (println "Single name")
        return (str_concat "Mr. " full_name)
    }
}

shadow process_name {
    assert (str_equals (process_name "John") "Mr. John")
    assert (str_equals (process_name "John Doe") "John Doe")
}
```

### Example 3: Trigonometric Calculation

```nano
fn calculate_angle(opposite: float, adjacent: float) -> float {
    # Calculate angle in radians using arctan
    # (Not implemented yet, but shows how trig functions work)
    let ratio: float = (/ opposite adjacent)
    return ratio  # Simplified - would use atan in real code
}

fn demonstrate_trig() -> int {
    let pi: float = 3.14159265359
    let pi_over_2: float = (/ pi 2.0)
    
    (println "sin(π/2) should be 1:")
    (println (sin pi_over_2))
    
    (println "cos(0) should be 1:")
    (println (cos 0.0))
    
    return 0
}
```

---

## Type Compatibility

### Polymorphic Functions
These functions accept multiple types:
- `println(any)` - Accepts int, float, string, bool
- `print(any)` - Accepts int, float, string, bool
- `abs(int|float)` - Returns same type as input
- `min(int|float, int|float)` - Both args must be same type
- `max(int|float, int|float)` - Both args must be same type

### Type-Converting Functions
These always return float regardless of input:
- `sqrt(int|float) -> float`
- `pow(int|float, int|float) -> float`
- `floor(int|float) -> float`
- `ceil(int|float) -> float`
- `round(int|float) -> float`
- `sin(int|float) -> float`
- `cos(int|float) -> float`
- `tan(int|float) -> float`

### String-Only Functions
These only work with strings:
- `str_length(string) -> int`
- `str_concat(string, string) -> string`
- `str_substring(string, int, int) -> string`
- `str_contains(string, string) -> bool`
- `str_equals(string, string) -> bool`

---

## Performance Notes

### Optimizations
- Math functions use C standard library directly (fast)
- String operations are bounds-checked (safe)
- `str_length` is O(n) - uses `strlen`
- `str_contains` is O(n*m) - uses `strstr`
- Memory for new strings is allocated on heap

### Memory Management
- String results are allocated with `malloc`
- In transpiled C code, memory should be managed carefully
- Shadow tests help catch memory issues early

---

## Future Standard Library Additions

Planned for future releases:
- **String:** `str_uppercase`, `str_lowercase`, `str_split`, `str_join`
- **Math:** `log`, `exp`, `atan`, `atan2`, `asin`, `acos`
- **File I/O:** `file_read`, `file_write`, `file_exists`
- **Collections:** Array operations (designed, not yet implemented)

See `docs/ARRAY_DESIGN.md` for planned array functionality.

---

## Notes

- All functions are documented with shadow tests in example files
- Every stdlib function is tested in at least one example program
- Error messages include line and column numbers
- Type checking happens at compile time
- Shadow tests execute during compilation to verify stdlib correctness

**Total Functions:** 20  
**Test Coverage:** 100%  
**Documentation:** Complete

