# Chapter 9: Core Utilities

**Essential built-in functions for I/O, math, strings, and type conversions.**

This chapter covers NanoLang's standard library functions that are always available without imports. These are the building blocks for all NanoLang programs.

## 9.1 Input & Output

Basic functions for displaying output and debugging.

### print & println

```nano
fn io_basics() -> void {
    (print "Hello ")
    (print "World")  # No newline
    (println "!")    # With newline
    
    (println 42)
    (println 3.14)
    (println true)
}

shadow io_basics {
    (io_basics)
}
```

**Key differences:**
- `print` - No trailing newline
- `println` - Adds newline after output

**Polymorphic:** Both functions work with `int`, `float`, `string`, and `bool`.

### assert

Runtime assertion for validating conditions:

```nano
fn validate_input(x: int) -> int {
    assert (> x 0)  # Terminates if false
    return (* x 2)
}

shadow validate_input {
    assert (== (validate_input 5) 10)
    # (validate_input -1)  # Would terminate program
}
```

**When to use:**
- Validate preconditions
- Check invariants
- Catch logic errors early

⚠️ **Warning:** Assertions terminate the program immediately. Use for unrecoverable errors only.

## 9.2 Math Operations

### Basic Math Functions

```nano
fn math_basics() -> int {
    let a: int = (abs -5)      # Returns 5
    let b: int = (min 10 20)   # Returns 10
    let c: int = (max 10 20)   # Returns 20
    return (+ a (+ b c))
}

shadow math_basics {
    assert (== (math_basics) 35)  # 5 + 10 + 20
}
```

**Type-preserving:**
- `abs(int) -> int` and `abs(float) -> float`
- `min` and `max` require both args to be same type

### Advanced Math Functions

```nano
fn advanced_math() -> bool {
    let root: float = (sqrt 16.0)       # 4.0
    let power: float = (pow 2.0 3.0)    # 8.0
    let down: float = (floor 3.7)       # 3.0
    let up: float = (ceil 3.2)          # 4.0
    let nearest: float = (round 3.5)    # 4.0
    
    return (and (== root 4.0) (== power 8.0))
}

shadow advanced_math {
    assert (advanced_math)
}
```

**Return types:**
- All advanced math functions return `float`
- Input can be `int` or `float` (auto-converted)

### Trigonometric Functions

All trig functions use **radians**:

```nano
fn trig_example() -> bool {
    let pi: float = 3.14159265359
    let half_pi: float = (/ pi 2.0)
    
    let sine: float = (sin 0.0)           # 0.0
    let cosine: float = (cos 0.0)         # 1.0
    let tangent: float = (tan 0.0)        # 0.0
    
    let sine_90: float = (sin half_pi)    # ≈1.0
    let cosine_180: float = (cos pi)      # ≈-1.0
    
    return (and (< sine 0.1) (> cosine 0.9))
}

shadow trig_example {
    assert (trig_example)
}
```

**Practical example: Distance formula**

```nano
fn distance(x1: float, y1: float, x2: float, y2: float) -> float {
    let dx: float = (- x2 x1)
    let dy: float = (- y2 y1)
    let dx2: float = (pow dx 2.0)
    let dy2: float = (pow dy 2.0)
    return (sqrt (+ dx2 dy2))
}

shadow distance {
    let dist: float = (distance 0.0 0.0 3.0 4.0)
    assert (and (> dist 4.9) (< dist 5.1))
}
```

## 9.3 String Operations

### Basic String Functions

```nano
fn string_basics() -> bool {
    let text: string = "Hello, World!"
    
    let len: int = (str_length text)                # 13
    let contains: bool = (str_contains text "World") # true
    let equals: bool = (str_equals text "Hello")    # false
    
    return (and contains (== len 13))
}

shadow string_basics {
    assert (string_basics)
}
```

### String Concatenation

```nano
fn build_greeting(name: string) -> string {
    let prefix: string = "Hello, "
    let suffix: string = "!"
    return (+ prefix (+ name suffix))
}

shadow build_greeting {
    assert (== (build_greeting "Alice") "Hello, Alice!")
}
```

💡 **Pro Tip:** Use `+` for string concatenation. `str_concat` is deprecated.

### Substrings

```nano
fn extract_parts(text: string) -> bool {
    # str_substring(string, start, length)
    let first_word: string = (str_substring text 0 5)
    let last_word: string = (str_substring text 7 5)
    
    return (and 
        (== first_word "Hello")
        (== last_word "World")
    )
}

shadow extract_parts {
    assert (extract_parts "Hello, World!")
}
```

**Bounds behavior:**
- If `start + length` exceeds string length, returns until end
- If `start` is out of bounds, returns empty string

## 9.4 Character Operations

### Character Access

```nano
fn char_access_demo() -> bool {
    let text: string = "ABC"
    
    let a: int = (char_at text 0)  # 65 (ASCII 'A')
    let b: int = (char_at text 1)  # 66 (ASCII 'B')
    let c: int = (char_at text 2)  # 67 (ASCII 'C')
    
    return (and (== a 65) (and (== b 66) (== c 67)))
}

shadow char_access_demo {
    assert (char_access_demo)
}
```

**Building strings from characters:**

```nano
fn build_from_chars() -> string {
    let a: string = (string_from_char 65)  # "A"
    let b: string = (string_from_char 66)  # "B"
    let c: string = (string_from_char 67)  # "C"
    return (+ a (+ b c))
}

shadow build_from_chars {
    assert (== (build_from_chars) "ABC")
}
```

### Character Classification

```nano
fn classify_char(c: int) -> string {
    if (is_digit c) {
        return "digit"
    }
    if (is_alpha c) {
        return "letter"
    }
    if (is_whitespace c) {
        return "whitespace"
    }
    return "other"
}

shadow classify_char {
    assert (== (classify_char 48) "digit")        # '0'
    assert (== (classify_char 65) "letter")       # 'A'
    assert (== (classify_char 32) "whitespace")   # ' '
    assert (== (classify_char 33) "other")        # '!'
}
```

**Available classifiers:**
- `is_digit(c)` - True for '0'-'9'
- `is_alpha(c)` - True for a-z, A-Z
- `is_alnum(c)` - True for digit or letter
- `is_whitespace(c)` - True for space, tab, newline
- `is_upper(c)` - True for A-Z
- `is_lower(c)` - True for a-z

### Case Conversion

```nano
fn case_conversion_demo() -> bool {
    let upper_a: int = 65   # 'A'
    let lower_a: int = 97   # 'a'
    
    let to_lower: int = (char_to_lower upper_a)  # 97
    let to_upper: int = (char_to_upper lower_a)  # 65
    
    return (and (== to_lower lower_a) (== to_upper upper_a))
}

shadow case_conversion_demo {
    assert (case_conversion_demo)
}
```

**Converting full strings:**

```nano
fn to_uppercase(s: string) -> string {
    let len: int = (str_length s)
    let mut result: string = ""
    let mut i: int = 0
    
    while (< i len) {
        let c: int = (char_at s i)
        let upper: int = (char_to_upper c)
        let char_str: string = (string_from_char upper)
        set result (+ result char_str)
        set i (+ i 1)
    }
    
    return result
}

shadow to_uppercase {
    assert (== (to_uppercase "hello") "HELLO")
    assert (== (to_uppercase "World") "WORLD")
}
```

## 9.5 Type Conversions

### Integer Conversions

```nano
fn int_conversions() -> bool {
    # int to string
    let s1: string = (int_to_string 42)     # "42"
    let s2: string = (int_to_string -100)   # "-100"
    let s3: string = (int_to_string 0)      # "0"
    
    # string to int
    let n1: int = (string_to_int "42")      # 42
    let n2: int = (string_to_int "-100")    # -100
    let n3: int = (string_to_int "bad")     # 0 (parse error)
    
    return (and (== n1 42) (== n2 -100))
}

shadow int_conversions {
    assert (int_conversions)
}
```

⚠️ **Warning:** `string_to_int` returns `0` on parse failure. Always validate input first.

### Float Conversions

```nano
fn float_conversions() -> bool {
    # float to string
    let s1: string = (float_to_string 3.14)    # "3.14"
    let s2: string = (float_to_string -2.5)    # "-2.5"
    
    # string to float
    let f1: float = (string_to_float "3.14")   # 3.14
    let f2: float = (string_to_float "-2.5")   # -2.5
    let f3: float = (string_to_float "bad")    # 0.0 (parse error)
    
    return (and (> f1 3.0) (< f1 3.2))
}

shadow float_conversions {
    assert (float_conversions)
}
```

### Digit Value Extraction

```nano
fn digit_value_demo() -> int {
    let zero: int = (digit_value 48)   # 0 ('0' -> 0)
    let five: int = (digit_value 53)   # 5 ('5' -> 5)
    let nine: int = (digit_value 57)   # 9 ('9' -> 9)
    let bad: int = (digit_value 65)    # -1 ('A' is not digit)
    
    return (+ zero (+ five nine))
}

shadow digit_value_demo {
    assert (== (digit_value_demo) 14)  # 0 + 5 + 9
}
```

**Use case: Parsing multi-digit numbers**

```nano
fn parse_number(s: string) -> int {
    let len: int = (str_length s)
    let mut result: int = 0
    let mut i: int = 0
    
    while (< i len) {
        let c: int = (char_at s i)
        if (is_digit c) {
            let digit: int = (digit_value c)
            set result (+ (* result 10) digit)
        }
        set i (+ i 1)
    }
    
    return result
}

shadow parse_number {
    assert (== (parse_number "42") 42)
    assert (== (parse_number "1234") 1234)
    assert (== (parse_number "007") 7)
}
```

## 9.6 Practical Examples

### Example 1: Word Counter

```nano
fn count_words(text: string) -> int {
    let len: int = (str_length text)
    let mut count: int = 0
    let mut in_word: bool = false
    let mut i: int = 0
    
    while (< i len) {
        let c: int = (char_at text i)
        if (is_whitespace c) {
            set in_word false
        } else {
            if (not in_word) {
                set count (+ count 1)
                set in_word true
            }
        }
        set i (+ i 1)
    }
    
    return count
}

shadow count_words {
    assert (== (count_words "Hello World") 2)
    assert (== (count_words "The quick brown fox") 4)
    assert (== (count_words "   spaces   ") 1)
    assert (== (count_words "") 0)
}
```

### Example 2: Number Validator

```nano
fn is_valid_number(s: string) -> bool {
    let len: int = (str_length s)
    if (== len 0) {
        return false
    }
    
    let mut i: int = 0
    let first: int = (char_at s 0)
    
    # Handle optional minus sign
    if (== first 45) {  # '-'
        set i 1
        if (== len 1) {
            return false  # Just "-" is not valid
        }
    }
    
    # All remaining chars must be digits
    while (< i len) {
        let c: int = (char_at s i)
        if (not (is_digit c)) {
            return false
        }
        set i (+ i 1)
    }
    
    return true
}

shadow is_valid_number {
    assert (is_valid_number "42")
    assert (is_valid_number "-100")
    assert (is_valid_number "0")
    assert (not (is_valid_number "abc"))
    assert (not (is_valid_number "12.34"))
    assert (not (is_valid_number "-"))
    assert (not (is_valid_number ""))
}
```

### Example 3: Simple Calculator

```nano
fn calculate(operator: string, a: int, b: int) -> int {
    if (== operator "+") {
        return (+ a b)
    }
    if (== operator "-") {
        return (- a b)
    }
    if (== operator "*") {
        return (* a b)
    }
    if (== operator "/") {
        assert (!= b 0)  # Prevent division by zero
        return (/ a b)
    }
    return 0  # Unknown operator
}

shadow calculate {
    assert (== (calculate "+" 10 5) 15)
    assert (== (calculate "-" 10 5) 5)
    assert (== (calculate "*" 10 5) 50)
    assert (== (calculate "/" 10 5) 2)
}
```

### Example 4: String Sanitizer

```nano
fn sanitize_identifier(s: string) -> string {
    let len: int = (str_length s)
    let mut result: string = ""
    let mut i: int = 0
    
    while (< i len) {
        let c: int = (char_at s i)
        if (or (is_alnum c) (== c 95)) {  # 95 is '_'
            let char_str: string = (string_from_char c)
            set result (+ result char_str)
        }
        set i (+ i 1)
    }
    
    return result
}

shadow sanitize_identifier {
    assert (== (sanitize_identifier "hello_world") "hello_world")
    assert (== (sanitize_identifier "foo-bar") "foobar")
    assert (== (sanitize_identifier "test123") "test123")
    assert (== (sanitize_identifier "bad@name!") "badname")
}
```

### Example 5: Pythagorean Theorem

```nano
fn hypotenuse(a: float, b: float) -> float {
    let a2: float = (pow a 2.0)
    let b2: float = (pow b 2.0)
    return (sqrt (+ a2 b2))
}

shadow hypotenuse {
    let h: float = (hypotenuse 3.0 4.0)
    assert (and (> h 4.9) (< h 5.1))
    
    let h2: float = (hypotenuse 5.0 12.0)
    assert (and (> h2 12.9) (< h2 13.1))
}
```

## 9.7 Best Practices

### ✅ DO

**1. Use type conversions for formatting:**

```nano
fn format_score(name: string, score: int) -> string {
    return (+ name (+ ": " (int_to_string score)))
}

shadow format_score {
    assert (== (format_score "Alice" 100) "Alice: 100")
}
```

**2. Validate before converting:**

```nano
fn safe_string_to_int(s: string) -> int {
    if (is_valid_number s) {
        return (string_to_int s)
    }
    return 0
}

shadow safe_string_to_int {
    assert (== (safe_string_to_int "42") 42)
    assert (== (safe_string_to_int "bad") 0)
}
```

**3. Use character classifiers for parsing:**

```nano
fn skip_whitespace(s: string, start: int) -> int {
    let len: int = (str_length s)
    let mut pos: int = start
    
    while (and (< pos len) (is_whitespace (char_at s pos))) {
        set pos (+ pos 1)
    }
    
    return pos
}

shadow skip_whitespace {
    assert (== (skip_whitespace "   hello" 0) 3)
    assert (== (skip_whitespace "hello" 0) 0)
}
```

### ❌ DON'T

**1. Don't use magic numbers for characters:**

```nano
# ❌ Bad: What is 65?
if (== c 65) { ... }

# ✅ Good: Clear intent
let upper_a: int = 65  # 'A'
if (== c upper_a) { ... }
```

**2. Don't ignore parse errors:**

```nano
# ❌ Bad: Returns 0 on error
let n: int = (string_to_int user_input)

# ✅ Good: Validate first
if (is_valid_number user_input) {
    let n: int = (string_to_int user_input)
} else {
    (println "Invalid number")
}
```

**3. Don't use deprecated functions:**

```nano
# ❌ Bad: Deprecated
let result: string = (str_concat a b)

# ✅ Good: Use + operator
let result: string = (+ a b)
```

## Summary

In this chapter, you learned:
- ✅ I/O functions: `print`, `println`, `assert`
- ✅ Math operations: basic, advanced, and trigonometric
- ✅ String operations: length, contains, substring
- ✅ Character operations: access, classification, case conversion
- ✅ Type conversions: int/float/string conversions
- ✅ Practical examples: word counter, validator, calculator

### Quick Reference

| Category | Functions |
|----------|-----------|
| **I/O** | `print`, `println`, `assert` |
| **Basic Math** | `abs`, `min`, `max` |
| **Advanced Math** | `sqrt`, `pow`, `floor`, `ceil`, `round` |
| **Trig** | `sin`, `cos`, `tan` (radians) |
| **Strings** | `str_length`, `str_contains`, `str_equals`, `str_substring` |
| **Characters** | `char_at`, `string_from_char`, `is_digit`, `is_alpha`, `is_alnum` |
| **Case** | `char_to_lower`, `char_to_upper`, `is_upper`, `is_lower` |
| **Conversions** | `int_to_string`, `string_to_int`, `float_to_string`, `string_to_float` |

---

**Previous:** [Chapter 8: Modules & Imports](../part1_fundamentals/08_modules.md)  
**Next:** [Chapter 10: Collections Library](10_collections_library.md)
