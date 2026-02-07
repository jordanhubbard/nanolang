# Chapter 5: Control Flow

**Learn how to control program flow with conditionals and loops.**

This chapter covers NanoLang's control flow constructs: conditionals (`if/else` and `cond`), loops (`while` and `for`), and common patterns.

## 5.1 if/else Expressions

The `if/else` construct is used for conditional execution of statements.

### Basic if Statement

```nano
fn check_positive(x: int) -> void {
    if (> x 0) {
        (println "Positive")
    }
}

shadow check_positive {
    (check_positive 5)
    (check_positive -3)
}
```

### if/else Statement

```nano
fn describe_number(x: int) -> void {
    if (> x 0) {
        (println "Positive")
    } else {
        (println "Not positive")
    }
}

shadow describe_number {
    (describe_number 5)
    (describe_number -3)
}
```

### if/else if/else Chain

```nano
fn classify_number(x: int) -> void {
    if x > 0 {
        (println "Positive")
    } else if x < 0 {
        (println "Negative")
    } else {
        (println "Zero")
    }
}

shadow classify_number {
    (classify_number 5)
    (classify_number -3)
    (classify_number 0)
}
```

NanoLang supports `else if` chaining for multi-way conditionals. You can also nest `if` inside `else` blocks for the same effect.

### Expressions vs Statements

**When to use if/else:**
- For side effects (printing, modifying variables)
- When you don't need a return value
- For control flow in procedures

```nano
fn example_statement(x: int) -> void {
    if (> x 10) {
        (println "Big")
        set x (* x 2)  # Side effect
    }
}

shadow example_statement {
    (example_statement 15)
}
```

## 5.2 cond Multi-way Branches

For expressions that **return a value**, use `cond` instead of `if/else`.

### Basic cond Syntax

```nano
fn sign(x: int) -> int {
    return (cond
        ((< x 0) -1)
        ((> x 0) 1)
        (else 0)
    )
}

shadow sign {
    assert (== (sign 5) 1)
    assert (== (sign -3) -1)
    assert (== (sign 0) 0)
}
```

**Syntax:** `(cond (test result) (test result) ... (else default))`

### When to Use cond

Use `cond` when:
- You need to return a value based on conditions
- You want expression-style conditionals
- You're choosing between multiple alternatives

```nano
fn grade_to_points(grade: string) -> int {
    return (cond
        ((== grade "A") 4)
        ((== grade "B") 3)
        ((== grade "C") 2)
        ((== grade "D") 1)
        (else 0)
    )
}

shadow grade_to_points {
    assert (== (grade_to_points "A") 4)
    assert (== (grade_to_points "C") 2)
    assert (== (grade_to_points "F") 0)
}
```

### Multiple Tests

Each test is evaluated in order until one matches:

```nano
fn categorize_age(age: int) -> string {
    return (cond
        ((< age 13) "child")
        ((< age 20) "teenager")
        ((< age 65) "adult")
        (else "senior")
    )
}

shadow categorize_age {
    assert (== (categorize_age 10) "child")
    assert (== (categorize_age 15) "teenager")
    assert (== (categorize_age 30) "adult")
    assert (== (categorize_age 70) "senior")
}
```

### else Clause

The `else` clause provides a default value:

```nano
fn is_vowel(c: int) -> bool {
    return (cond
        ((== c 97) true)   # 'a'
        ((== c 101) true)  # 'e'
        ((== c 105) true)  # 'i'
        ((== c 111) true)  # 'o'
        ((== c 117) true)  # 'u'
        (else false)
    )
}

shadow is_vowel {
    assert (is_vowel 97)   # 'a'
    assert (not (is_vowel 98))  # 'b'
}
```

### Pattern Matching with cond

`cond` is useful for pattern-like matching:

```nano
fn day_type(day: string) -> string {
    return (cond
        ((or (== day "Saturday") (== day "Sunday")) "weekend")
        ((== day "Friday") "almost weekend")
        (else "weekday")
    )
}

shadow day_type {
    assert (== (day_type "Monday") "weekday")
    assert (== (day_type "Friday") "almost weekend")
    assert (== (day_type "Saturday") "weekend")
}
```

### Nested cond

You can nest `cond` expressions:

```nano
fn tax_bracket(income: int, married: bool) -> float {
    return (cond
        (married (cond
            ((< income 20000) 0.10)
            ((< income 50000) 0.15)
            (else 0.25)
        ))
        (else (cond
            ((< income 10000) 0.10)
            ((< income 40000) 0.15)
            (else 0.25)
        ))
    )
}

shadow tax_bracket {
    assert (== (tax_bracket 15000 true) 0.10)
    assert (== (tax_bracket 15000 false) 0.15)
}
```

## 5.3 while Loops

The `while` loop repeats code while a condition is true.

### Basic while Loop

```nano
fn count_to_n(n: int) -> void {
    let mut i: int = 1
    while (<= i n) {
        (println (int_to_string i))
        set i (+ i 1)
    }
}

shadow count_to_n {
    (count_to_n 3)
}
```

**Syntax:** `while (condition) { body }`

### Loop Conditions

The condition is checked before each iteration:

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
    assert (== (sum_to_n 5) 15)   # 1+2+3+4+5
    assert (== (sum_to_n 0) 0)
    assert (== (sum_to_n 10) 55)
}
```

### Breaking Out of Loops

Use early returns to exit a loop:

```nano
fn find_first_negative(arr: array<int>) -> int {
    let mut i: int = 0
    while (< i (array_length arr)) {
        let val: int = (array_get arr i)
        if (< val 0) {
            return i  # Exit loop and function
        }
        set i (+ i 1)
    }
    return -1  # Not found
}

shadow find_first_negative {
    assert (== (find_first_negative [1, 2, -3, 4]) 2)
    assert (== (find_first_negative [1, 2, 3]) -1)
}
```

### Infinite Loops

Be careful: loops without a way to exit will run forever:

```nano
# ❌ Don't do this (infinite loop):
# fn infinite() -> void {
#     while true {
#         (println "Forever!")
#     }
# }
```

### Common while Loop Patterns

**Counter pattern:**

```nano
fn factorial(n: int) -> int {
    let mut result: int = 1
    let mut i: int = 1
    while (<= i n) {
        set result (* result i)
        set i (+ i 1)
    }
    return result
}

shadow factorial {
    assert (== (factorial 5) 120)
}
```

**Accumulator pattern:**

```nano
fn sum_array(arr: array<int>) -> int {
    let mut sum: int = 0
    let mut i: int = 0
    while (< i (array_length arr)) {
        set sum (+ sum (array_get arr i))
        set i (+ i 1)
    }
    return sum
}

shadow sum_array {
    assert (== (sum_array [1, 2, 3, 4, 5]) 15)
}
```

**Search pattern:**

```nano
fn contains(arr: array<int>, target: int) -> bool {
    let mut i: int = 0
    while (< i (array_length arr)) {
        if (== (array_get arr i) target) {
            return true
        }
        set i (+ i 1)
    }
    return false
}

shadow contains {
    assert (contains [1, 2, 3, 4, 5] 3)
    assert (not (contains [1, 2, 3] 9))
}
```

## 5.4 for-in-range Loops

The `for` loop uses range iteration for counting.

### Basic for Loop

```nano
fn print_numbers() -> void {
    for i in (range 0 5) {
        (println (int_to_string i))
    }
}

shadow print_numbers {
    (print_numbers)
}
```

**Syntax:** `for var in (range start end) { body }`

**Components:**
1. **var** - Loop variable name (immutable within loop)
2. **range** - Built-in function: `(range start end)`
3. **start** - First value (inclusive)
4. **end** - Last value (exclusive)
5. **body** - Code to execute each iteration

**Range behavior:**
- `(range 0 5)` yields: 0, 1, 2, 3, 4
- `(range 1 4)` yields: 1, 2, 3
- `(range 0 0)` yields nothing (no iterations)

### Range Iteration

Common pattern: iterate from 0 to n-1:

```nano
fn sum_range(n: int) -> int {
    let mut sum: int = 0
    for i in (range 0 n) {
        set sum (+ sum i)
    }
    return sum
}

shadow sum_range {
    assert (== (sum_range 5) 10)  # 0+1+2+3+4
    assert (== (sum_range 0) 0)
}
```

### Loop Variables

The loop variable is automatically declared and scoped to the loop:

```nano
fn use_loop_variable() -> int {
    let mut result: int = 0
    for i in (range 1 11) {
        set result (+ result i)
    }
    # i is not accessible here
    return result
}

shadow use_loop_variable {
    assert (== (use_loop_variable) 55)  # 1+2+3+...+10
}
```

### Counting Backwards

For descending ranges, use a while loop (range only goes forward):

```nano
fn countdown(n: int) -> void {
    let mut i: int = n
    while (> i 0) {
        (println (int_to_string i))
        set i (- i 1)
    }
}

shadow countdown {
    (countdown 3)
}
```

### Array Iteration

Iterate over array indices with range:

```nano
fn process_array(arr: array<int>) -> int {
    let mut sum: int = 0
    for i in (range 0 (array_length arr)) {
        set sum (+ sum (array_get arr i))
    }
    return sum
}

shadow process_array {
    assert (== (process_array [10, 20, 30]) 60)
}
```

### Custom Step Sizes

For non-unit steps, use while loops:

```nano
fn sum_evens(n: int) -> int {
    let mut sum: int = 0
    let mut i: int = 0
    while (< i n) {
        set sum (+ sum i)
        set i (+ i 2)  # Step by 2
    }
    return sum
}

shadow sum_evens {
    assert (== (sum_evens 10) 20)  # 0+2+4+6+8
}
```

## 5.5 Loop Patterns & Idioms

Common patterns you'll use frequently.

### Accumulation

Building up a result:

```nano
fn multiply_array(arr: array<int>) -> int {
    let mut product: int = 1
    for i in (range 0 (array_length arr)) {
        set product (* product (array_get arr i))
    }
    return product
}

shadow multiply_array {
    assert (== (multiply_array [2, 3, 4]) 24)
    assert (== (multiply_array [1, 1, 1]) 1)
}
```

### Iteration

Processing each element:

```nano
fn print_array(arr: array<int>) -> void {
    for i in (range 0 (array_length arr)) {
        (println (int_to_string (array_get arr i)))
    }
}

shadow print_array {
    (print_array [1, 2, 3])
}
```

### Filtering

Counting elements that match a condition:

```nano
fn count_positives(arr: array<int>) -> int {
    let mut count: int = 0
    for i in (range 0 (array_length arr)) {
        if (> (array_get arr i) 0) {
            set count (+ count 1)
        }
    }
    return count
}

shadow count_positives {
    assert (== (count_positives [1, -2, 3, -4, 5]) 3)
}
```

### Finding

Searching for an element:

```nano
fn find_index(arr: array<int>, target: int) -> int {
    for i in (range 0 (array_length arr)) {
        if (== (array_get arr i) target) {
            return i
        }
    }
    return -1
}

shadow find_index {
    assert (== (find_index [10, 20, 30] 20) 1)
    assert (== (find_index [10, 20, 30] 99) -1)
}
```

### Transforming

Building a new result based on each element:

```nano
fn double_sum(arr: array<int>) -> int {
    let mut sum: int = 0
    for i in (range 0 (array_length arr)) {
        let doubled: int = (* (array_get arr i) 2)
        set sum (+ sum doubled)
    }
    return sum
}

shadow double_sum {
    assert (== (double_sum [1, 2, 3]) 12)  # (1*2)+(2*2)+(3*2)
}
```

### Nested Loops

Processing multi-dimensional data:

```nano
fn multiplication_table(n: int) -> void {
    for i in (range 1 (+ n 1)) {
        for j in (range 1 (+ n 1)) {
            let product: int = (* i j)
            (println (int_to_string product))
        }
    }
}

shadow multiplication_table {
    (multiplication_table 3)
}
```

### Best Practices

**1. Choose the right loop**

```nano
# Use for when you know the iteration count
fn count_up(n: int) -> void {
    for i in (range 0 n) {
        (println (int_to_string i))
    }
}

# Use while when the condition is complex
fn read_until_negative(arr: array<int>) -> int {
    let mut i: int = 0
    while (and (< i (array_length arr)) (>= (array_get arr i) 0)) {
        set i (+ i 1)
    }
    return i
}

shadow count_up { (count_up 3) }
shadow read_until_negative {
    assert (== (read_until_negative [1, 2, -1, 4]) 2)
}
```

**2. Keep loop bodies simple**

```nano
✅ Good: Extract complex logic into functions
fn is_prime(n: int) -> bool {
    if (< n 2) { return false }
    for i in (range 2 n) {
        if (== (% n i) 0) {
            return false
        }
    }
    return true
}

fn count_primes(arr: array<int>) -> int {
    let mut count: int = 0
    for i in (range 0 (array_length arr)) {
        if (is_prime (array_get arr i)) {
            set count (+ count 1)
        }
    }
    return count
}

shadow is_prime {
    assert (is_prime 7)
    assert (not (is_prime 4))
}

shadow count_primes {
    assert (== (count_primes [2, 3, 4, 5, 6, 7]) 4)
}
```

**3. Avoid off-by-one errors**

```nano
# ✅ Correct: < for array indices
fn correct_iteration(arr: array<int>) -> int {
    let mut sum: int = 0
    for i in (range 0 (array_length arr)) {
        set sum (+ sum (array_get arr i))
    }
    return sum
}

# ❌ Wrong: <= would go past end
# fn wrong_iteration(arr: array<int>) -> int {
#     let mut sum: int = 0
#     for i in (range 0 (+ (array_length arr) 1)) {
#         set sum (+ sum (array_get arr i))  # Error on last iteration
#     }
#     return sum
# }

shadow correct_iteration {
    assert (== (correct_iteration [1, 2, 3]) 6)
}
```

### Summary

In this chapter, you learned:
- ✅ `if/else` for statements (side effects)
- ✅ `cond` for expressions (return values)
- ✅ `while` loops for condition-based iteration
- ✅ `for` loops for counted iteration
- ✅ Common loop patterns (accumulation, filtering, finding)

### Practice Exercises

```nano
# 1. Find maximum using for loop
fn find_max(arr: array<int>) -> int {
    let mut max: int = (array_get arr 0)
    for i in (range 1 (array_length arr)) {
        let val: int = (array_get arr i)
        if (> val max) {
            set max val
        }
    }
    return max
}

shadow find_max {
    assert (== (find_max [1, 5, 3, 9, 2]) 9)
}

# 2. Check if all elements are positive
fn all_positive(arr: array<int>) -> bool {
    for i in (range 0 (array_length arr)) {
        if (<= (array_get arr i) 0) {
            return false
        }
    }
    return true
}

shadow all_positive {
    assert (all_positive [1, 2, 3])
    assert (not (all_positive [1, -2, 3]))
}

# 3. Compute GCD using while loop
fn gcd(a: int, b: int) -> int {
    let mut x: int = a
    let mut y: int = b
    while (!= y 0) {
        let temp: int = y
        set y (% x y)
        set x temp
    }
    return x
}

shadow gcd {
    assert (== (gcd 48 18) 6)
    assert (== (gcd 100 50) 50)
}
```

---

**Previous:** [Chapter 4: Functions](04_functions.html)  
**Next:** [Chapter 6: Collections](06_collections.html)
