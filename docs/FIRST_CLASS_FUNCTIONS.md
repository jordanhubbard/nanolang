# My First-Class Functions

I support first-class functions. I allow you to pass functions as parameters, return them from other functions, and store them in variables. I do this without exposing raw function pointers.

## Table of Contents

1. [Overview](#overview)
2. [Function Types](#function-types)
3. [Functions as Parameters](#functions-as-parameters)
4. [Functions as Return Values](#functions-as-return-values)
5. [Function Variables](#function-variables)
6. [Common Patterns](#common-patterns)
7. [Examples](#examples)
8. [Implementation Details](#implementation-details)

## Overview

I enable functional programming patterns:
- Higher-order functions (map, filter, fold)
- Function factories
- Strategy pattern
- Callback mechanisms
- Function dispatch tables

You write clean syntax like `fn(int) -> bool`. I handle the C function pointer mechanics.

## Function Types

My function types describe the signature of a function: parameter types and return type.

### Syntax

```nano
fn(param1_type, param2_type, ...) -> return_type
```

### Examples

```nano
# Function that takes an int and returns an int
fn(int) -> int

# Function that takes two ints and returns an int
fn(int, int) -> int

# Function that takes an int and returns a bool (predicate)
fn(int) -> bool

# Function with no parameters
fn() -> void

# Function returning a function
fn(int) -> fn(int) -> int
```

## Functions as Parameters

You can pass functions to other functions to enable higher-order operations.

### Basic Example

```nano
# Higher-order function: applies f twice
fn apply_twice(x: int, f: fn(int) -> int) -> int {
    return (f (f x))
}

fn double(x: int) -> int {
    return (* x 2)
}

shadow apply_twice {
    # 5 * 2 * 2 = 20
    assert (== (apply_twice 5 double) 20)
}
```

### Multiple Function Parameters

```nano
fn combine(a: int, b: int,
           f: fn(int, int) -> int,
           g: fn(int, int) -> int) -> int {
    let x: int = (f a b)
    let y: int = (g a b)
    return (+ x y)
}

fn add(a: int, b: int) -> int {
    return (+ a b)
}

fn multiply(a: int, b: int) -> int {
    return (* a b)
}

shadow combine {
    # (5 + 3) + (5 * 3) = 8 + 15 = 23
    assert (== (combine 5 3 add multiply) 23)
}
```

### Common Higher-Order Patterns

```nano
# Map: transform each element
fn map_int(arr: array<int>, f: fn(int) -> int) -> array<int> {
    let result: array<int> = [0; (array_length arr)]
    let mut i: int = 0
    while (< i (array_length arr)) {
        set (at result i) (f (at arr i))
        set i (+ i 1)
    }
    return result
}

# Filter: keep elements matching predicate
fn filter(arr: array<int>, pred: fn(int) -> bool) -> array<int> {
    # Implementation left as exercise
    return arr
}

# Fold: reduce to single value
fn fold(arr: array<int>, init: int, f: fn(int, int) -> int) -> int {
    let mut acc: int = init
    let mut i: int = 0
    while (< i (array_length arr)) {
        set acc (f acc (at arr i))
        set i (+ i 1)
    }
    return acc
}
```

## Functions as Return Values

I allow functions to return other functions. This enables function factories and partial application.

### Function Factory Example

```nano
fn get_operation(choice: int) -> fn(int, int) -> int {
    if (== choice 0) {
        return add
    } else {
        if (== choice 1) {
            return multiply
        } else {
            return subtract
        }
    }
}

shadow get_operation {
    let add_fn: fn(int, int) -> int = (get_operation 0)
    assert (== (add_fn 10 5) 15)

    let mul_fn: fn(int, int) -> int = (get_operation 1)
    assert (== (mul_fn 10 5) 50)
}
```

### Partial Application Pattern

```nano
# Returns a function that adds 'x' to its argument
fn make_adder(x: int) -> fn(int) -> int {
    # I do not support closures yet.
    # This must be implemented differently.
    # See my limitations section.
    return add_x  # Simplified example
}
```

## Function Variables

You can store functions in variables for dynamic dispatch and code organization.

### Basic Function Variables

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

fn multiply(a: int, b: int) -> int {
    return (* a b)
}

fn main() -> int {
    # Store function in variable
    let my_op: fn(int, int) -> int = add
    let result: int = (my_op 10 20)
    (println (int_to_string result))  # Prints: 30

    # Change operation
    set my_op multiply
    let result2: int = (my_op 10 20)
    (println (int_to_string result2))  # Prints: 200

    return 0
}
```

### Calculator with Function Dispatch

```nano
fn calculator(a: int, b: int, operation: int) -> int {
    # Get the operation function
    let func: fn(int, int) -> int = (get_operation operation)

    # Call it
    let result: int = (func a b)

    return result
}

shadow calculator {
    assert (== (calculator 100 25 0) 125)  # add
    assert (== (calculator 100 25 1) 2500) # multiply
    assert (== (calculator 100 25 2) 75)   # subtract
}
```

### Function Dispatch Table Pattern

```nano
# Store multiple operations
fn process_data(data: array<int>, op_code: int) -> int {
    let operations: array<fn(int, int) -> int> = [add, multiply, subtract, divide]

    let mut result: int = (at data 0)
    let mut i: int = 1

    while (< i (array_length data)) {
        let op: fn(int, int) -> int = (at operations op_code)
        set result (op result (at data i))
        set i (+ i 1)
    }

    return result
}
```

## Common Patterns

### Strategy Pattern

```nano
fn process_numbers(x: int, y: int, strategy: fn(int, int) -> int) -> int {
    # Strategy function passed as parameter
    return (strategy x y)
}

shadow process_numbers {
    assert (== (process_numbers 12 4 subtract) 8)   # 12 - 4
    assert (== (process_numbers 7 3 add) 10)        # 7 + 3
    assert (== (process_numbers 5 6 multiply) 30)   # 5 * 6
}
```

### Callback Pattern

```nano
fn async_operation(data: int, callback: fn(int) -> void) -> void {
    # Do some processing
    let result: int = (* data 2)

    # Call the callback with result
    (callback result)
}

fn handle_result(value: int) -> void {
    (println (+ "Got result: " (int_to_string value)))
}

fn main() -> int {
    (async_operation 42 handle_result)  # Prints: "Got result: 84"
    return 0
}
```

### Predicate Pattern

```nano
fn is_positive(x: int) -> bool {
    return (> x 0)
}

fn is_even(x: int) -> bool {
    return (== (% x 2) 0)
}

fn count_matching(arr: array<int>, pred: fn(int) -> bool) -> int {
    let mut count: int = 0
    let mut i: int = 0

    while (< i (array_length arr)) {
        if (pred (at arr i)) {
            set count (+ count 1)
        }
        set i (+ i 1)
    }

    return count
}

shadow count_matching {
    let numbers: array<int> = [1, -2, 3, -4, 5]

    assert (== (count_matching numbers is_positive) 3)
    assert (== (count_matching numbers is_even) 2)
}
```

## Examples

### Complete Example: Array Utilities

```nano
# Transform each element
fn transform(arr: array<int>, f: fn(int) -> int) -> array<int> {
    let result: array<int> = [0; (array_length arr)]
    let mut i: int = 0

    while (< i (array_length arr)) {
        set (at result i) (f (at arr i))
        set i (+ i 1)
    }

    return result
}

fn square(x: int) -> int {
    return (* x x)
}

fn negate(x: int) -> int {
    return (- 0 x)
}

shadow transform {
    let numbers: array<int> = [1, 2, 3, 4, 5]

    let squares: array<int> = (transform numbers square)
    assert (== (at squares 2) 9)  # 3^2 = 9

    let negated: array<int> = (transform numbers negate)
    assert (== (at negated 2) -3)  # -3
}
```

### Complete Example: Function Composition

```nano
fn compose(f: fn(int) -> int, g: fn(int) -> int) -> fn(int) -> int {
    # Returns h where h(x) = f(g(x))
    # I do not support closures yet.
    # This is a conceptual example.
    return composed_function
}
```

## Implementation Details

### How I Work

I compile first-class functions to C function pointers.

```nano
# My code
fn apply_twice(x: int, f: fn(int) -> int) -> int {
    return (f (f x))
}
```

I produce this C code:

```c
// Generated C code
typedef int64_t (*UnaryOp_0)(int64_t);

int64_t nl_apply_twice(int64_t x, UnaryOp_0 f) {
    return f(f(x));
}
```

### Type Signatures

I generate typedefs for each unique function signature:

- `fn(int) -> int` translates to `typedef int64_t (*UnaryOp_0)(int64_t);`
- `fn(int, int) -> int` translates to `typedef int64_t (*BinaryOp_0)(int64_t, int64_t);`
- `fn(int) -> bool` translates to `typedef bool (*Predicate_0)(int64_t);`

### Function Name Mangling

When you pass a function as a value, I prefix its name with `nl_`:

```nano
let f: fn(int) -> int = double
```

This becomes:

```c
UnaryOp_0 f = nl_double;
```

When you call through a function parameter, I do not add a prefix:

```c
int64_t nl_apply_twice(int64_t x, UnaryOp_0 f) {
    return f(f(x));
}
```

### My Current Limitations

1. **No Closures.** Functions cannot capture variables from outer scopes.
2. **No Anonymous Functions.** You must define named functions.
3. **No Generic Function Types.** My function types are monomorphic.
4. **Static Dispatch Only.** I resolve all function types at compile time.

### Future Enhancements

I may add these:

- **Closures.** Capture variables from enclosing scope.
- **Lambda expressions.**
- **Generic function types.**
- **Method references.**

## See Also

- [SPECIFICATION.md](SPECIFICATION.md)
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- [ERROR_HANDLING.md](ERROR_HANDLING.md)
- [examples/language/nl_first_class_functions.nano](../examples/language/nl_first_class_functions.nano)
- [examples/language/nl_function_return_values.nano](../examples/language/nl_function_return_values.nano)
- [examples/language/nl_function_variables.nano](../examples/language/nl_function_variables.nano)
- [examples/language/nl_filter_map_fold.nano](../examples/language/nl_filter_map_fold.nano)

---

**I provide first-class functions to enable functional programming patterns. I maintain my simplicity and C-based compilation.**
