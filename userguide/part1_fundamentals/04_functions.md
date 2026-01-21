# Chapter 4: Functions

**Master function definitions, parameters, and NanoLang's unique shadow testing system.**

Functions are the building blocks of NanoLang programs. This chapter covers how to define functions, work with parameters and return values, and write shadow tests.

## 4.1 Function Definitions

Functions in NanoLang follow a consistent, explicit syntax.

### Basic Function Syntax

```nano
fn function_name(param1: type1, param2: type2) -> return_type {
    # function body
    return value
}
```

**Required components:**
1. `fn` keyword
2. Function name (lowercase with underscores)
3. Parameter list with types
4. Return type annotation (` -> type`)
5. Function body in braces `{ }`
6. `return` statement

### Simple Function Example

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -5 5) 0)
}
```

### Functions Without Parameters

```nano
fn get_pi() -> float {
    return 3.14159
}

shadow get_pi {
    let pi: float = (get_pi)
    assert (and (> pi 3.14) (< pi 3.15))
}
```

⚠️ **Watch Out:** Even parameterless functions need parentheses in their call: `(get_pi)` not `get_pi`.

### Void Functions

Functions that don't return a value use `void`:

```nano
fn print_greeting(name: string) -> void {
    (println (+ "Hello, " name))
}

shadow print_greeting {
    (print_greeting "World")
}
```

**Note:** Void functions still execute code, they just don't return a value.

### Naming Conventions

Use snake_case for function names:

```nano
✅ calculate_total
✅ is_valid
✅ get_user_name

❌ calculateTotal  # camelCase - not used
❌ Calculate_Total # PascalCase - not used
```

## 4.2 Parameters & Return Types

Functions can have multiple parameters and must declare their types explicitly.

### Single Parameter

```nano
fn square(x: int) -> int {
    return (* x x)
}

shadow square {
    assert (== (square 5) 25)
    assert (== (square 0) 0)
    assert (== (square -3) 9)
}
```

### Multiple Parameters

```nano
fn calculate_area(width: float, height: float) -> float {
    return (* width height)
}

shadow calculate_area {
    assert (== (calculate_area 5.0 10.0) 50.0)
    assert (== (calculate_area 0.0 10.0) 0.0)
}
```

### Different Parameter Types

```nano
fn repeat_string(s: string, times: int) -> string {
    let mut result: string = ""
    let mut i: int = 0
    while (< i times) {
        set result (+ result s)
        set i (+ i 1)
    }
    return result
}

shadow repeat_string {
    assert (== (repeat_string "hi" 3) "hihihi")
    assert (== (repeat_string "x" 0) "")
}
```

### Return Values

The return statement exits the function and provides the result:

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
    assert (== (max 4 4) 4)
}
```

### Early Returns

You can return early from a function:

```nano
fn divide_safe(a: int, b: int) -> int {
    if (== b 0) {
        return 0  # Avoid division by zero
    }
    return (/ a b)
}

shadow divide_safe {
    assert (== (divide_safe 10 2) 5)
    assert (== (divide_safe 10 0) 0)
}
```

### Multiple Return Paths

```nano
fn sign(x: int) -> int {
    if (< x 0) {
        return -1
    }
    if (> x 0) {
        return 1
    }
    return 0
}

shadow sign {
    assert (== (sign 5) 1)
    assert (== (sign -3) -1)
    assert (== (sign 0) 0)
}
```

## 4.3 Shadow Tests (Built-in Testing)

Shadow tests are NanoLang's unique compile-time testing feature. **Every function must have a shadow test** (except `extern` functions).

### What Are Shadow Tests?

Shadow tests are code blocks that:
1. Run at compile time
2. Verify function correctness
3. Are mandatory for all functions
4. Use the `shadow` keyword

### Basic Shadow Test

```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
}
```

**Syntax:** `shadow function_name { test code }`

### Why Shadow Tests Are Mandatory

**Benefits:**
1. **Catch bugs at compile time** - Tests run before your program executes
2. **Documentation** - Tests show how to use the function
3. **Confidence** - Every function is tested
4. **No separate test framework** - Testing is built into the language

**Philosophy:** If a function isn't worth testing, it isn't worth writing.

### Writing Shadow Tests

**Rule 1: Test happy path**

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
}
```

**Rule 2: Test edge cases**

```nano
fn absolute(x: int) -> int {
    return (cond
        ((< x 0) (- 0 x))
        (else x)
    )
}

shadow absolute {
    assert (== (absolute 5) 5)     # Positive
    assert (== (absolute -5) 5)    # Negative
    assert (== (absolute 0) 0)     # Zero
}
```

**Rule 3: Test multiple scenarios**

```nano
fn is_even(n: int) -> bool {
    return (== (% n 2) 0)
}

shadow is_even {
    assert (is_even 4)         # Even positive
    assert (not (is_even 5))   # Odd positive
    assert (is_even 0)         # Zero
    assert (is_even -4)        # Even negative
    assert (not (is_even -5))  # Odd negative
}
```

### Multiple Assertions

You can have multiple assertions in a shadow block:

```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    }
    let mut result: int = 1
    let mut i: int = 2
    while (<= i n) {
        set result (* result i)
        set i (+ i 1)
    }
    return result
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 1) 1)
    assert (== (factorial 5) 120)
    assert (== (factorial 10) 3628800)
}
```

### Testing with Different Types

```nano
fn string_length_valid(s: string) -> bool {
    return (and (>= (str_length s) 1) (<= (str_length s) 100))
}

shadow string_length_valid {
    assert (string_length_valid "hello")
    assert (not (string_length_valid ""))
    assert (string_length_valid "x")
}
```

### Testing Void Functions

Void functions can have trivial shadow tests:

```nano
fn print_number(n: int) -> void {
    (println (int_to_string n))
}

shadow print_number {
    (print_number 42)
}
```

### What If Tests Fail?

If a shadow test fails, compilation stops with an error:

```nano
# This won't compile:
# fn broken_add(a: int, b: int) -> int {
#     return (* a b)  # Bug: multiplying instead of adding
# }
#
# shadow broken_add {
#     assert (== (broken_add 2 3) 5)  # FAILS: 2*3 ≠ 5
# }
```

The compiler shows you which assertion failed, helping you fix the bug before running your program.

### Shadow Test Best Practices

**1. Test boundary conditions**

```nano
fn safe_divide(a: int, b: int) -> int {
    if (== b 0) { return 0 }
    return (/ a b)
}

shadow safe_divide {
    assert (== (safe_divide 10 2) 5)    # Normal case
    assert (== (safe_divide 10 0) 0)    # Division by zero
    assert (== (safe_divide 0 5) 0)     # Zero numerator
    assert (== (safe_divide -10 2) -5)  # Negative numbers
}
```

**2. Keep tests simple and readable**

```nano
✅ Good:
shadow is_positive {
    assert (is_positive 5)
    assert (not (is_positive -3))
    assert (not (is_positive 0))
}

❌ Too complex:
shadow is_positive {
    assert (and (is_positive 5) (and (not (is_positive -3)) (not (is_positive 0))))
}
```

**3. Test realistic scenarios**

```nano
fn format_name(first: string, last: string) -> string {
    return (+ (+ first " ") last)
}

shadow format_name {
    assert (== (format_name "John" "Doe") "John Doe")
    assert (== (format_name "A" "B") "A B")
    assert (== (format_name "" "") " ")
}
```

## 4.4 Recursion by Example

Recursion is when a function calls itself. It's a powerful technique for solving problems that have recursive structure.

### Base Cases and Recursive Cases

Every recursive function needs:
1. **Base case** - When to stop recursing
2. **Recursive case** - How to break down the problem

### Simple Recursion: Factorial

```nano
fn factorial_recursive(n: int) -> int {
    if (<= n 1) {
        return 1  # Base case
    }
    return (* n (factorial_recursive (- n 1)))  # Recursive case
}

shadow factorial_recursive {
    assert (== (factorial_recursive 0) 1)
    assert (== (factorial_recursive 1) 1)
    assert (== (factorial_recursive 5) 120)
    assert (== (factorial_recursive 6) 720)
}
```

**How it works:**
- `factorial_recursive(5)` calls `factorial_recursive(4)`
- Which calls `factorial_recursive(3)`
- Which calls `factorial_recursive(2)`
- Which calls `factorial_recursive(1)`
- Base case returns 1
- Results multiply back up: 1 * 2 * 3 * 4 * 5 = 120

### Recursion: Sum of Array

```nano
fn sum_recursive(arr: array<int>, index: int) -> int {
    if (>= index (array_length arr)) {
        return 0  # Base case: past end of array
    }
    let current: int = (array_get arr index)
    let rest: int = (sum_recursive arr (+ index 1))
    return (+ current rest)
}

shadow sum_recursive {
    assert (== (sum_recursive [1, 2, 3, 4, 5] 0) 15)
    assert (== (sum_recursive [] 0) 0)
    assert (== (sum_recursive [10] 0) 10)
}
```

### Recursion: Fibonacci

```nano
fn fibonacci_recursive(n: int) -> int {
    if (<= n 1) {
        return n  # Base cases: fib(0)=0, fib(1)=1
    }
    return (+ (fibonacci_recursive (- n 1)) (fibonacci_recursive (- n 2)))
}

shadow fibonacci_recursive {
    assert (== (fibonacci_recursive 0) 0)
    assert (== (fibonacci_recursive 1) 1)
    assert (== (fibonacci_recursive 5) 5)
    assert (== (fibonacci_recursive 10) 55)
}
```

### Tail Recursion

Tail recursion is when the recursive call is the last operation:

```nano
fn sum_tail_recursive(arr: array<int>, index: int, accumulator: int) -> int {
    if (>= index (array_length arr)) {
        return accumulator
    }
    let next_acc: int = (+ accumulator (array_get arr index))
    return (sum_tail_recursive arr (+ index 1) next_acc)
}

shadow sum_tail_recursive {
    assert (== (sum_tail_recursive [1, 2, 3, 4, 5] 0 0) 15)
}
```

💡 **Pro Tip:** Tail recursion can be optimized by compilers to avoid stack growth.

### Recursion vs Iteration

**When to use recursion:**
- Problem has natural recursive structure (trees, nested data)
- Code is clearer with recursion

**When to use iteration:**
- Simple loops
- Performance is critical
- Risk of stack overflow

```nano
# Recursive version (elegant but can overflow)
fn count_down_recursive(n: int) -> void {
    if (<= n 0) { return }
    (println (int_to_string n))
    (count_down_recursive (- n 1))
}

# Iterative version (more efficient)
fn count_down_iterative(n: int) -> void {
    let mut i: int = n
    while (> i 0) {
        (println (int_to_string i))
        set i (- i 1)
    }
}

shadow count_down_recursive {
    (count_down_recursive 3)
}

shadow count_down_iterative {
    (count_down_iterative 3)
}
```

### Complete Example: Binary Search

```nano
fn binary_search(arr: array<int>, target: int, left: int, right: int) -> int {
    if (> left right) {
        return -1  # Not found
    }
    
    let mid: int = (+ left (/ (- right left) 2))
    let mid_val: int = (array_get arr mid)
    
    if (== mid_val target) {
        return mid  # Found!
    }
    
    if (< target mid_val) {
        return (binary_search arr target left (- mid 1))
    } else {
        return (binary_search arr target (+ mid 1) right)
    }
}

shadow binary_search {
    let sorted: array<int> = [1, 3, 5, 7, 9, 11, 13]
    assert (== (binary_search sorted 7 0 6) 3)
    assert (== (binary_search sorted 1 0 6) 0)
    assert (== (binary_search sorted 13 0 6) 6)
    assert (== (binary_search sorted 8 0 6) -1)
}
```

### Summary

In this chapter, you learned:
- ✅ Function syntax: parameters, return types, body
- ✅ Shadow tests are mandatory and run at compile time
- ✅ Writing comprehensive tests for functions
- ✅ Recursion: base cases and recursive cases
- ✅ When to use recursion vs iteration

### Practice Exercises

```nano
# 1. Write a function to compute power (x^n)
fn power(x: int, n: int) -> int {
    if (== n 0) {
        return 1
    }
    return (* x (power x (- n 1)))
}

shadow power {
    assert (== (power 2 3) 8)
    assert (== (power 5 0) 1)
    assert (== (power 10 2) 100)
}

# 2. Write a function to reverse a string
fn reverse_string(s: string) -> string {
    let len: int = (str_length s)
    if (== len 0) {
        return ""
    }
    let mut result: string = ""
    let mut i: int = (- len 1)
    while (>= i 0) {
        set result (+ result (string_from_char (char_at s i)))
        set i (- i 1)
    }
    return result
}

shadow reverse_string {
    assert (== (reverse_string "hello") "olleh")
    assert (== (reverse_string "") "")
    assert (== (reverse_string "a") "a")
}

# 3. Write a recursive function to find max in array
fn max_recursive(arr: array<int>, index: int, current_max: int) -> int {
    if (>= index (array_length arr)) {
        return current_max
    }
    let val: int = (array_get arr index)
    let new_max: int = (cond
        ((> val current_max) val)
        (else current_max)
    )
    return (max_recursive arr (+ index 1) new_max)
}

shadow max_recursive {
    assert (== (max_recursive [1, 5, 3, 9, 2] 1 1) 9)
    assert (== (max_recursive [10, 5, 8] 1 10) 10)
}
```

---

**Previous:** [Chapter 3: Variables & Bindings](03_variables.md)  
**Next:** [Chapter 5: Control Flow](05_control_flow.md)
