# Canonical syntax rules

NanoLang is intentionally strict: there is **exactly ONE canonical way** to write each construct. This makes code generation by LLMs more reliable and code review easier.

> **For the complete style guide** see [Canonical Style Guide](https://github.com/jordanhubbard/nanolang/blob/main/docs/CANONICAL_STYLE.md) and [LLM Core Subset](https://github.com/jordanhubbard/nanolang/blob/main/docs/LLM_CORE_SUBSET.md).

## Why Canonical Syntax?

When LLMs see multiple equivalent forms, they guess wrong ~50% of the time. NanoLang eliminates ambiguity:

| ❌ Avoided | ✅ Canonical |
|-----------|-------------|
| `a + b` | `(+ a b)` |
| `arr[i]` | `(at arr i)` |
| `str1 + str2` | `(+ str1 str2)` |
| `if x > 0 then ...` | `if (> x 0) { ... }` |
| `x == y ? a : b` | `(cond ((== x y) a) (else b))` |

## Prefix operators

All arithmetic, comparison, and logical operators use prefix notation:

<!--nl-snippet {"name":"ug_canonical_prefix","check":true}-->
```nano
fn distance_1d(a: int, b: int) -> int {
    return (cond
        ((> a b) (- a b))
        (else (- b a))
    )
}

shadow distance_1d {
    assert (== (distance_1d 10 3) 7)
    assert (== (distance_1d 3 10) 7)
    assert (== (distance_1d 5 5) 0)
}

fn main() -> int {
    assert (== (distance_1d -5 5) 10)
    return 0
}

shadow main { assert true }
```

### Arithmetic

```nano
(+ a b)      # Addition
(- a b)      # Subtraction
(* a b)      # Multiplication
(/ a b)      # Division
(% a b)      # Modulo (integers only)
```

### Comparison

```nano
(== a b)     # Equal
(!= a b)     # Not equal
(< a b)      # Less than
(> a b)      # Greater than
(<= a b)     # Less than or equal
(>= a b)     # Greater than or equal
```

### Logical

```nano
(and a b)    # Logical AND (short-circuit)
(or a b)     # Logical OR (short-circuit)
(not a)      # Logical NOT
```

## String operations

String concatenation uses the `+` operator in prefix form:

<!--nl-snippet {"name":"ug_canonical_strings","check":true}-->
```nano
fn greet(name: string) -> string {
    return (+ "Hello, " (+ name "!"))
}

shadow greet {
    assert (== (greet "World") "Hello, World!")
    assert (== (greet "NanoLang") "Hello, NanoLang!")
}

fn main() -> int {
    (println (greet "User"))
    return 0
}

shadow main { assert true }
```

**Key rules:**
- ✅ Use `(+ s1 s2)` for concatenation
- ❌ Never use `str_concat` (deprecated)
- ✅ Use `(== s1 s2)` for string equality
- ✅ Use `str_equals` only if you need it explicitly

## Array access

Arrays use function-style access, not subscript notation:

<!--nl-snippet {"name":"ug_canonical_arrays","check":true}-->
```nano
fn sum_first_two(arr: array<int>) -> int {
    let a: int = (at arr 0)
    let b: int = (at arr 1)
    return (+ a b)
}

shadow sum_first_two {
    let nums: array<int> = [10, 20, 30]
    assert (== (sum_first_two nums) 30)
}

fn main() -> int {
    let data: array<int> = [5, 10, 15]
    assert (== (sum_first_two data) 15)
    return 0
}

shadow main { assert true }
```

**Key rules:**
- ✅ Use `(at arr i)` or `(array_get arr i)` to read
- ✅ Use `(array_set arr i val)` to write
- ❌ Never use `arr[i]` subscript notation

## Conditionals

### cond for expressions (returns value)

Use `cond` when you need to return a value:

<!--nl-snippet {"name":"ug_canonical_cond","check":true}-->
```nano
fn classify(n: int) -> string {
    return (cond
        ((< n 0) "negative")
        ((== n 0) "zero")
        (else "positive")
    )
}

shadow classify {
    assert (== (classify -5) "negative")
    assert (== (classify 0) "zero")
    assert (== (classify 42) "positive")
}

fn main() -> int {
    (println (classify 10))
    return 0
}

shadow main { assert true }
```

### if/else for statements (side effects)

Use `if/else` when you're executing code for side effects:

<!--nl-snippet {"name":"ug_canonical_if","check":true}-->
```nano
fn print_sign(n: int) -> void {
    if (< n 0) {
        (println "negative")
    } else {
        if (== n 0) {
            (println "zero")
        } else {
            (println "positive")
        }
    }
}

shadow print_sign {
    (print_sign -1)
    (print_sign 0)
    (print_sign 1)
}

fn main() -> int {
    (print_sign 42)
    return 0
}

shadow main { assert true }
```

**Note:** NanoLang doesn't have `else if` - nest `if` statements in `else` blocks.

## Loops

### for loop (counted iteration)

<!--nl-snippet {"name":"ug_canonical_for","check":true}-->
```nano
fn sum_range(n: int) -> int {
    let mut total: int = 0
    for i in (range 0 n) {
        set total (+ total i)
    }
    return total
}

shadow sum_range {
    assert (== (sum_range 5) 10)  # 0+1+2+3+4
    assert (== (sum_range 0) 0)
}

fn main() -> int {
    assert (== (sum_range 10) 45)
    return 0
}

shadow main { assert true }
```

### while loop (condition-based)

<!--nl-snippet {"name":"ug_canonical_while","check":true}-->
```nano
fn count_digits(n: int) -> int {
    let mut count: int = 0
    let mut val: int = n
    if (== val 0) { return 1 }
    while (> val 0) {
        set val (/ val 10)
        set count (+ count 1)
    }
    return count
}

shadow count_digits {
    assert (== (count_digits 0) 1)
    assert (== (count_digits 5) 1)
    assert (== (count_digits 99) 2)
    assert (== (count_digits 1000) 4)
}

fn main() -> int {
    (println (int_to_string (count_digits 12345)))
    return 0
}

shadow main { assert true }
```

## Function definitions

Functions always have explicit type annotations:

<!--nl-snippet {"name":"ug_canonical_fn","check":true}-->
```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add -1 1) 0)
}

fn greet() -> void {
    (println "Hello!")
}

shadow greet {
    (greet)
}

fn main() -> int {
    (println (int_to_string (add 10 20)))
    (greet)
    return 0
}

shadow main { assert true }
```

**Key rules:**
- ✅ Always specify return type
- ✅ Always type parameters
- ✅ Use `void` for functions that return nothing
- ✅ Every function needs a shadow test

## Variable declarations

### Immutable (default)

```nano
let x: int = 42
# x cannot be changed
```

### Mutable

```nano
let mut counter: int = 0
set counter (+ counter 1)  # OK: counter is mutable
```

**Key rules:**
- ✅ Use `let` for immutable bindings (preferred)
- ✅ Use `let mut` only when mutation is needed
- ✅ Use `set` to mutate, not `=`

## Type declarations

### Structs

<!--nl-snippet {"name":"ug_canonical_struct","check":true}-->
```nano
struct Point {
    x: int
    y: int
}

fn make_point(x: int, y: int) -> Point {
    return Point { x: x, y: y }
}

shadow make_point {
    let p: Point = (make_point 3 4)
    assert (== p.x 3)
    assert (== p.y 4)
}

fn main() -> int {
    let origin: Point = Point { x: 0, y: 0 }
    (println (int_to_string origin.x))
    return 0
}

shadow main { assert true }
```

### Enums

<!--nl-snippet {"name":"ug_canonical_enum","check":true}-->
```nano
enum Color {
    Red = 0
    Green = 1
    Blue = 2
}

fn color_name(c: Color) -> string {
    return (cond
        ((== c Color.Red) "red")
        ((== c Color.Green) "green")
        (else "blue")
    )
}

shadow color_name {
    assert (== (color_name Color.Red) "red")
    assert (== (color_name Color.Green) "green")
    assert (== (color_name Color.Blue) "blue")
}

fn main() -> int {
    (println (color_name Color.Green))
    return 0
}

shadow main { assert true }
```

## Summary: The Canonical Forms

| Construct | Canonical Form |
|-----------|---------------|
| Arithmetic | `(+ a b)`, `(- a b)`, `(* a b)`, `(/ a b)` |
| Comparison | `(== a b)`, `(< a b)`, `(> a b)` |
| Logic | `(and a b)`, `(or a b)`, `(not a)` |
| Strings | `(+ s1 s2)`, `(str_length s)` |
| Arrays | `(at arr i)`, `(array_set arr i v)` |
| Expression conditional | `(cond ((test) value) (else default))` |
| Statement conditional | `if (cond) { ... } else { ... }` |
| Counted loop | `for i in (range 0 n) { ... }` |
| Condition loop | `while (cond) { ... }` |
| Function call | `(function_name arg1 arg2)` |
| Variable declaration | `let x: Type = value` |
| Mutable variable | `let mut x: Type = value` |
| Assignment | `set x new_value` |

When in doubt, use prefix notation and explicit types. There is exactly ONE way.
