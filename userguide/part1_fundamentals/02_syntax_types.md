# Chapter 2: Basic Syntax & Types

**Master NanoLang's core syntax and fundamental types.**

This chapter covers NanoLang's distinctive syntax and basic type system. By the end, you'll understand how operators and function calls work, and how to work with numbers, strings, booleans, and types.

## 2.1 Operator Notation

NanoLang supports **two notations** for binary operators: **prefix** and **infix**. Function calls always use prefix notation.

### Why Two Notations?

Prefix notation like `(+ a b)` eliminates ambiguity and is ideal for LLM code generation -- there is exactly one way to parse any expression. Infix notation like `a + b` is more natural for humans reading and writing code. NanoLang supports both so you can use whichever is clearest for the situation.

### Prefix Notation (Operator First)

```nano
(+ 1 2)        # Addition
(* 3 4)        # Multiplication
(< x 10)       # Comparison
(println "hi") # Function call (always prefix)
```

### Infix Notation (Operator Between)

```nano
1 + 2          # Addition
3 * 4          # Multiplication
x < 10         # Comparison
```

### Function Calls Are Always Prefix

```nano
(println "hi")       # Correct
(str_length name)    # Correct
```

### Nesting and Grouping

**Prefix nesting works inside-out:**

```nano
(+ (* 2 3) 4)
# First: (* 2 3) = 6
# Then: (+ 6 4) = 10
```

**Infix uses parentheses for grouping:**

```nano
2 * 3 + 4       # Evaluates left-to-right: (2 * 3) + 4 = 10
2 * (3 + 4)     # Parentheses override: 2 * 7 = 14
```

**Important:** All infix operators have **equal precedence** and evaluate **left-to-right** (no PEMDAS). Use parentheses to control grouping.

### Reading Expressions

Let's practice reading expressions in both notations:

```nano
fn calculate_example() -> int {
    return (+ (* 2 3) (- 10 5))   # prefix style
    # equivalent infix: 2 * 3 + (10 - 5)
}

shadow calculate_example {
    assert (== (calculate_example) 11)
}
```

Read it step by step:
1. `(* 2 3)` or `2 * 3` --> `6`
2. `(- 10 5)` or `10 - 5` --> `5`
3. `(+ 6 5)` or `6 + 5` --> `11`

### All Binary Operators

These operators work in both prefix and infix notation:

```nano
# Arithmetic
(+ a b)   or   a + b      # Add
(- a b)   or   a - b      # Subtract
(* a b)   or   a * b      # Multiply
(/ a b)   or   a / b      # Divide
(% a b)   or   a % b      # Modulo

# Comparison
(== a b)  or   a == b     # Equal
(!= a b)  or   a != b     # Not equal
(< a b)   or   a < b      # Less than
(> a b)   or   a > b      # Greater than
(<= a b)  or   a <= b     # Less than or equal
(>= a b)  or   a >= b     # Greater than or equal

# Logical
(and a b) or   a and b    # Logical AND
(or a b)  or   a or b     # Logical OR

# Unary (no infix form)
(not x)   or   not x      # Logical NOT
-x                         # Negation

# String operations
(+ s1 s2) or   s1 + s2    # Concatenate strings
```

### Practice Example

```nano
fn is_even(n: int) -> bool {
    return (== (% n 2) 0)
}

shadow is_even {
    assert (is_even 4)
    assert (not (is_even 5))
    assert (is_even 0)
}

fn main() -> int {
    if (is_even 42) {
        (println "42 is even")
    }
    return 0
}

shadow main { assert true }
```

## 2.2 Numbers & Arithmetic

NanoLang has two numeric types: `int` and `float`.

### Integer Types

Integers are whole numbers (positive, negative, or zero).

```nano
fn integer_examples() -> int {
    let positive: int = 42
    let negative: int = -17
    let zero: int = 0
    return (+ positive negative)
}

shadow integer_examples {
    assert (== (integer_examples) 25)
}
```

**Integer range:** Typically -2,147,483,648 to 2,147,483,647 (32-bit signed).

### Floating-Point Numbers

Floats represent decimal numbers.

```nano
fn float_examples() -> float {
    let pi: float = 3.14159
    let negative: float = -2.5
    let zero: float = 0.0
    return (+ pi negative)
}

shadow float_examples {
    let result: float = (float_examples)
    assert (and (> result 0.6) (< result 0.7))
}
```

⚠️ **Watch Out:** Floating-point arithmetic has rounding errors. Use range checks in tests, not exact equality.

### Arithmetic Operations

```nano
fn arithmetic_demo(a: int, b: int) -> int {
    let sum: int = (+ a b)
    let diff: int = (- a b)
    let product: int = (* a b)
    let quotient: int = (/ a b)
    let remainder: int = (% a b)
    return sum
}

shadow arithmetic_demo {
    assert (== (+ 5 3) 8)
    assert (== (- 5 3) 2)
    assert (== (* 5 3) 15)
    assert (== (/ 10 3) 3)    # Integer division
    assert (== (% 10 3) 1)    # Remainder
}
```

### Division Behavior

**Integer division truncates:**

```nano
fn division_examples() -> void {
    assert (== (/ 10 3) 3)     # Not 3.333...
    assert (== (/ 7 2) 3)      # Not 3.5
    assert (== (/ -7 2) -3)    # Rounds toward zero
}

shadow division_examples {
    (division_examples)
}
```

**Float division preserves decimals:**

```nano
fn float_division() -> float {
    return (/ 10.0 3.0)
}

shadow float_division {
    let result: float = (float_division)
    assert (and (> result 3.3) (< result 3.4))
}
```

### Common Pitfalls

❌ **Don't mix int and float without conversion:**

```nano
# This won't compile:
# let x: int = 5
# let y: float = 3.0
# let result: float = (+ x y)  # Type error!
```

✅ **Convert explicitly:**

```nano
fn convert_and_add(x: int, y: float) -> float {
    return (+ (int_to_float x) y)
}

shadow convert_and_add {
    assert (== (convert_and_add 5 3.0) 8.0)
}
```

## 2.3 Strings & Characters

Strings represent text. They're immutable sequences of characters.

### String Literals

```nano
fn string_examples() -> string {
    let greeting: string = "Hello"
    let empty: string = ""
    let multiline: string = "Line 1
Line 2
Line 3"
    return greeting
}

shadow string_examples {
    assert (== (string_examples) "Hello")
}
```

### String Operations

**Concatenation:**

```nano
fn concat_example(first: string, last: string) -> string {
    return (+ (+ first " ") last)
}

shadow concat_example {
    assert (== (concat_example "John" "Doe") "John Doe")
}
```

**String length:**

```nano
fn length_example(s: string) -> int {
    return (str_length s)
}

shadow length_example {
    assert (== (length_example "hello") 5)
    assert (== (length_example "") 0)
}
```

**String equality:**

```nano
fn equality_example(a: string, b: string) -> bool {
    return (== a b)
}

shadow equality_example {
    assert (equality_example "hello" "hello")
    assert (not (equality_example "hello" "Hello"))
}
```

⚠️ **Watch Out:** String comparison is case-sensitive.

### Escape Sequences

```nano
fn escape_sequences() -> string {
    let newline: string = "Line 1\nLine 2"
    let tab: string = "Col 1\tCol 2"
    let quote: string = "She said \"Hello\""
    let backslash: string = "Path: C:\\Users"
    return newline
}

shadow escape_sequences {
    assert (== (str_length (escape_sequences)) 15)
}
```

Common escapes:
- `\n` - Newline
- `\t` - Tab
- `\"` - Double quote
- `\\` - Backslash

### Character Handling

Access individual characters by index (0-based):

```nano
fn get_first_char(s: string) -> int {
    return (char_at s 0)
}

shadow get_first_char {
    assert (== (get_first_char "Hello") 72)  # ASCII code for 'H'
}
```

💡 **Pro Tip:** `char_at` returns the ASCII code as an `int`.

## 2.4 Booleans & Comparisons

Booleans represent true/false values.

### Boolean Values

```nano
fn boolean_examples() -> bool {
    let yes: bool = true
    let no: bool = false
    return yes
}

shadow boolean_examples {
    assert (boolean_examples)
}
```

### Comparison Operators

```nano
fn comparison_examples(x: int, y: int) -> bool {
    let equal: bool = (== x y)
    let not_equal: bool = (!= x y)
    let less: bool = (< x y)
    let greater: bool = (> x y)
    let less_equal: bool = (<= x y)
    let greater_equal: bool = (>= x y)
    return equal
}

shadow comparison_examples {
    assert (== 5 5)
    assert (!= 5 3)
    assert (< 3 5)
    assert (> 5 3)
    assert (<= 5 5)
    assert (>= 5 5)
}
```

### Logical Operations

```nano
fn logical_examples(a: bool, b: bool) -> bool {
    let both: bool = (and a b)
    let either: bool = (or a b)
    let opposite: bool = (not a)
    return both
}

shadow logical_examples {
    assert (and true true)
    assert (not (and true false))
    assert (or true false)
    assert (not (or false false))
    assert (== (not true) false)
}
```

### Short-Circuit Evaluation

**`and` stops at first false:**

```nano
fn and_shortcircuit(x: int) -> bool {
    return (and (> x 0) (< x 10))
}

shadow and_shortcircuit {
    assert (and_shortcircuit 5)
    assert (not (and_shortcircuit 15))
    assert (not (and_shortcircuit -5))
}
```

**`or` stops at first true:**

```nano
fn or_shortcircuit(x: int) -> bool {
    return (or (== x 0) (> x 100))
}

shadow or_shortcircuit {
    assert (or_shortcircuit 0)
    assert (or_shortcircuit 200)
    assert (not (or_shortcircuit 50))
}
```

### Combining Comparisons

```nano
fn is_valid_age(age: int) -> bool {
    return (and (>= age 0) (<= age 120))
}

shadow is_valid_age {
    assert (is_valid_age 25)
    assert (is_valid_age 0)
    assert (is_valid_age 120)
    assert (not (is_valid_age -1))
    assert (not (is_valid_age 150))
}
```

## 2.5 Type Annotations

NanoLang requires **explicit type annotations** for all variables and function parameters.

### Why Explicit Types?

Explicit types eliminate ambiguity and make code generation by LLMs more reliable. The compiler always knows what type you intend.

### Variable Type Annotations

```nano
fn type_annotation_examples() -> int {
    let x: int = 42
    let y: float = 3.14
    let name: string = "Alice"
    let is_valid: bool = true
    return x
}

shadow type_annotation_examples {
    assert (== (type_annotation_examples) 42)
}
```

**Rule:** Every `let` binding must have a type annotation.

❌ **This doesn't compile:**
```nano
let x = 42  # Missing type annotation
```

✅ **This compiles:**
```nano
let x: int = 42
```

### Function Type Annotations

Functions must annotate:
1. Parameter types
2. Return type

```nano
fn add_numbers(a: int, b: int) -> int {
    return (+ a b)
}

shadow add_numbers {
    assert (== (add_numbers 2 3) 5)
}
```

### Void Return Type

Functions that don't return a value use `void`:

```nano
fn print_greeting(name: string) -> void {
    (println (+ "Hello, " name))
}

shadow print_greeting {
    (print_greeting "World")
}
```

⚠️ **Watch Out:** `void` functions still need a return type annotation.

### Type Inference (Limited)

NanoLang has **minimal type inference**. In most cases, you must annotate types explicitly.

**Where inference works:**

```nano
fn infer_from_return() -> int {
    return 42  # Return type inferred from function signature
}

shadow infer_from_return {
    assert (== (infer_from_return) 42)
}
```

**Where inference doesn't work:**

```nano
# This won't compile:
# fn needs_annotation(x) -> int {  # Parameter type missing
#     return (+ x 1)
# }
```

### Type Conversion Functions

Convert between types explicitly:

```nano
fn type_conversions() -> void {
    let i: int = 42
    let f: float = (int_to_float i)
    let s: string = (int_to_string i)
    
    assert (== f 42.0)
    assert (== s "42")
}

shadow type_conversions {
    (type_conversions)
}
```

Common conversions:
- `int_to_float(int) -> float`
- `int_to_string(int) -> string`
- `float_to_int(float) -> int`
- `float_to_string(float) -> string`
- `string_to_int(string) -> int`

### Complete Example

```nano
fn calculate_average(a: int, b: int, c: int) -> float {
    let sum: int = (+ (+ a b) c)
    let sum_float: float = (int_to_float sum)
    let count: float = 3.0
    return (/ sum_float count)
}

shadow calculate_average {
    let avg: float = (calculate_average 10 20 30)
    assert (and (> avg 19.9) (< avg 20.1))
}

fn main() -> int {
    let average: float = (calculate_average 5 10 15)
    (println (+ "Average: " (float_to_string average)))
    return 0
}

shadow main { assert true }
```

### Summary

In this chapter, you learned:
- ✅ Operator notation: prefix `(operator arg1 arg2)` or infix `arg1 operator arg2`
- ✅ Numbers: `int` and `float`
- ✅ Strings: Immutable text with operations
- ✅ Booleans: `true` and `false` with logical operators
- ✅ Type annotations: Always explicit, never inferred

### Practice Exercises

Try writing these functions (solutions in comments):

```nano
# 1. Write a function that checks if a number is positive
fn is_positive(n: int) -> bool {
    return (> n 0)
}

shadow is_positive {
    assert (is_positive 5)
    assert (not (is_positive -3))
    assert (not (is_positive 0))
}

# 2. Write a function that returns the absolute value
fn absolute(n: int) -> int {
    return (cond
        ((< n 0) (- 0 n))
        (else n)
    )
}

shadow absolute {
    assert (== (absolute -5) 5)
    assert (== (absolute 5) 5)
    assert (== (absolute 0) 0)
}

# 3. Write a function that concatenates three strings
fn concat_three(a: string, b: string, c: string) -> string {
    return (+ (+ a b) c)
}

shadow concat_three {
    assert (== (concat_three "a" "b" "c") "abc")
}
```

---

**Previous:** [Chapter 1: Getting Started](01_getting_started.html)  
**Next:** [Chapter 3: Variables & Bindings](03_variables.html)
