# My Language Specification v0.1

## Table of Contents

1. [Introduction](#introduction)
2. [Lexical Structure](#lexical-structure)
3. [Types](#types)
   - 3.1 [Built-in Types](#31-built-in-types)
   - 3.2 [Type Annotations](#32-type-annotations)
   - 3.3 [Type Checking](#33-type-checking)
   - 3.4 [Composite Types](#34-composite-types)
     - 3.4.1 [Structs](#341-structs)
     - 3.4.2 [Enums](#342-enums)
     - 3.4.3 [Union Types](#343-union-types)
     - 3.4.4 [Generic Types](#344-generic-types)
     - 3.4.5 [First-Class Function Types](#345-first-class-function-types)
     - 3.4.6 [HashMap<K,V>](#346-hashmapkv)
4. [Expressions](#expressions)
5. [Statements](#statements)
6. [Functions](#functions)
7. [Shadow Tests](#shadow-tests)
8. [Semantics](#semantics)
9. [Compilation Model](#compilation-model)
10. [Example Programs](#example-programs)
11. [Design Rationale](#design-rationale)
12. [Future Extensions](#future-extensions)

## 1. Introduction

I am a minimal, statically-typed programming language. I prioritize clarity and am designed for machines to write and humans to read. My design rests on these principles:

- **Unambiguity**: I provide one clear syntax for every semantic concept.
- **Explicitness**: I do not use implicit conversions or hide my behavior.
- **Testability**: I require shadow tests for every function I compile.
- **Simplicity**: I maintain a minimal feature set with clear semantics.

## 2. Lexical Structure

### 2.1 Comments

```nano
# Single-line comment
/* Multi-line comment */
```

### 2.2 Identifiers

My identifiers must start with a letter or underscore. They can be followed by letters, digits, or underscores.

```
identifier = (letter | "_") { letter | digit | "_" }
```

Examples: `x`, `my_var`, `count2`, `_internal`

### 2.3 Keywords

These are my reserved keywords. You cannot use them as identifiers.

```
fn       let      mut      set      if       else
while    for      in       return   assert   shadow
extern   int      float    bool     string   void     
true     false    print    and      or       not
array    struct   enum     union    match
```

### 2.4 Literals

**Integer Literals**: A sequence of digits. You may use a leading `-` for negative values.
```nano
42
-17
0
```

**Float Literals**: Digits separated by a decimal point.
```nano
3.14
-0.5
2.0
```

**String Literals**: UTF-8 text enclosed in double quotes.
```nano
"Hello, World!"
"nanolang"
""
```

**Boolean Literals**: 
```nano
true
false
```

### 2.5 Operators and Delimiters

```
(  )  {  }  ,  :  =  ->
+  -  *  /  %  ==  !=  <  <=  >  >=
and  or  not
```

I support both prefix notation (`(+ a b)`) and infix notation (`a + b`) for my operators. I describe this in detail in Section 4.3.

### 2.6 Whitespace

I use whitespace (spaces, tabs, newlines) to separate tokens. Beyond that, I do not find it significant.

## 3. Types

### 3.1 Built-in Types

| Type     | Description                    | Example Values    |
|----------|--------------------------------|-------------------|
| `int`    | 64-bit signed integer          | `42`, `-17`, `0`  |
| `float`  | 64-bit floating point          | `3.14`, `-0.5`    |
| `bool`   | Boolean value                  | `true`, `false`   |
| `string` | UTF-8 encoded text             | `"hello"`         |
| `void`   | Absence of value (return only) | N/A               |

### 3.2 Type Annotations

I require explicit type annotations for all variables and function parameters.

```nano
let x: int = 42
let name: string = "Alice"

fn add(a: int, b: int) -> int {
    return (+ a b)
}
```

### 3.3 Type Checking

I am statically typed. I catch all type errors at compile time.

```nano
let x: int = 42
let y: string = "hello"
# let z: int = y        # ERROR: Type mismatch
# let w: int = (+ x y)  # ERROR: Cannot add int and string
```

### 3.4 Composite Types

#### 3.4.1 Structs

I use structs to group related data.

```nano
struct Point {
    x: int,
    y: int
}

let p: Point = Point { x: 10, y: 20 }
let x_coord: int = p.x
```

#### 3.4.2 Enums

I define enums as a fixed set of named constants. I treat these constants as integers.

```nano
enum Status {
    Pending = 0,
    Active = 1,
    Complete = 2
}

let s: int = Status.Active
```

#### 3.4.3 Union Types

My union types represent a value that can be one of several variants. These are tagged unions.

```nano
union Result {
    Ok { value: int },
    Error { code: int, message: string }
}

fn divide(a: int, b: int) -> Result {
    if (== b 0) {
        return Result.Error { code: 1, message: "Division by zero" }
    } else {
        return Result.Ok { value: (/ a b) }
    }
}
```

**Union Construction:**

```nano
# Empty variant
let status: Status = Status.Ok {}

# Variant with fields
let result: Result = Result.Error { code: 1, message: "Failed" }
```

**Pattern Matching:**

I use the `match` expression to destructure unions.

```nano
match result {
    Ok(r) => (println "Success"),
    Error(e) => (println "Error")
}
```

#### 3.4.4 Generic Types

I use generic types to allow parameterization. I implement these using monomorphization. I specialize generic types at compile time for each concrete type you use.

**Built-in Generic: List<T>**

```nano
# Create typed lists
let numbers: List<int> = (List_int_new)
(List_int_push numbers 1)
(List_int_push numbers 2)
(List_int_push numbers 3)

let names: List<string> = (List_string_new)
(List_string_push names "Alice")
(List_string_push names "Bob")

# Lists with user-defined types
struct Point { x: int, y: int }
let points: List<Point> = (List_Point_new)
```

**Generic Instantiation:**

When you use a generic type like `List<int>`, I generate specialized functions for that type:
- `List_int_new()`
- `List_int_push(list, value)`
- `List_int_length(list)`
- `List_int_get(list, index)`

**Monomorphization:**

I generate a separate implementation for each concrete type you use with a generic.

```nano
let integers: List<int> = (List_int_new)     # I generate List_int functions
let strings: List<string> = (List_string_new) # I generate List_string functions
```

I generate specialized C code for each instantiation to avoid runtime overhead.

#### 3.4.5 First-Class Function Types

I treat functions as first-class values. You can pass them as parameters, return them from other functions, or assign them to variables.

**Function Type Syntax:**

```nano
fn(param_type1, param_type2) -> return_type
```

**Function Variables:**

```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
}

# Assign function to variable
let f: fn(int) -> int = double

# Call through variable
let result: int = (f 7)  # result = 14
```

**Functions as Parameters:**

```nano
fn apply_twice(op: fn(int) -> int, x: int) -> int {
    return (op (op x))
}

shadow apply_twice {
    assert (== (apply_twice double 5) 20)
}
```

**Functions as Return Values:**

```nano
fn get_operation(choice: int) -> fn(int) -> int {
    if (== choice 0) {
        return double
    } else {
        return triple
    }
}

shadow get_operation {
    let op: fn(int) -> int = (get_operation 0)
    assert (== (op 5) 10)
}
```

My function types do not expose underlying C function pointers. I treat them as opaque values.

#### 3.4.6 HashMap<K,V>

I provide a HashMap as a generic hash table. Like my List type, I use monomorphization to generate specialized implementations for each combination of key and value types.

**Supported Key Types:**
- `int`
- `string`

**Supported Value Types:**
- `int`
- `string`

**Declaration:**

```nano
let scores: HashMap<string, int> = (map_new)
let counts: HashMap<int, int> = (map_new)
```

**Operations:**

| Function | Description | Example |
|----------|-------------|---------|
| `map_new` | Create empty map | `(map_new)` |
| `map_put` | Insert/update key-value | `(map_put hm "alice" 10)` |
| `map_get` | Retrieve value by key | `(map_get hm "alice")` |
| `map_has` | Check if key exists | `(map_has hm "alice")` |
| `map_size` | Get number of entries | `(map_size hm)` |

**Example:**

```nano
fn count_words(text: string) -> HashMap<string, int> {
    let counts: HashMap<string, int> = (map_new)
    (map_put counts "hello" 1)
    (map_put counts "world" 2)
    return counts
}

shadow count_words {
    let hm: HashMap<string, int> = (count_words "test")
    assert (== (map_has hm "hello") true)
    assert (== (map_get hm "hello") 1)
}
```

**Implementation Notes:**

I support HashMap in both my interpreter and my compiled modes. In interpreter mode, I use a runtime implementation. In compiled mode, I generate specialized C code. I manage HashMap memory automatically using ARC.

**Performance:** My hash table operations are O(1) on average. I rehash automatically when the load factor exceeds 0.75.

## 4. Expressions

### 4.1 Literals

I evaluate literals to their corresponding values.

```nano
42          # int
3.14        # float
"hello"     # string
true        # bool
```

### 4.2 Variables

I evaluate identifiers to the value of the named variable.

```nano
let x: int = 42
let y: int = x  # y = 42
```

### 4.3 Operations (Prefix and Infix)

I support both prefix notation and infix notation for my binary operators.

My prefix notation uses parentheses.

```nano
(+ 2 3)              # 5
(* (+ 2 3) 4)        # 20
(== x 5)             # Comparison
(and (> x 0) (< x 10))  # Logical AND
```

My infix notation uses standard operator placement.

```nano
2 + 3                # 5
(2 + 3) * 4          # 20
x == 5               # Comparison
x > 0 and x < 10    # Logical AND
```

My infix operators are: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `and`, `or`.

**Precedence**: I assign equal precedence to all my infix operators. I evaluate them strictly from left to right. I do not use PEMDAS. You must use parentheses if you want to control grouping.

```nano
a * (b + c)          # I add before I multiply
a + b * c            # I evaluate this as (a + b) * c
```

My unary operators `not` and `-` work without parentheses.

```nano
not flag             # Logical negation
-x                   # Numeric negation
```

You can mix these notations. I maintain prefix notation for compatibility.

### 4.4 Arithmetic Operations

| Operator | Description    | Type Signature     |
|----------|----------------|--------------------|
| `+`      | Addition       | `(int, int) -> int` or `(float, float) -> float` |
| `-`      | Subtraction    | `(int, int) -> int` or `(float, float) -> float` |
| `*`      | Multiplication | `(int, int) -> int` or `(float, float) -> float` |
| `/`      | Division       | `(int, int) -> int` or `(float, float) -> float` |
| `%`      | Modulo         | `(int, int) -> int` |

### 4.5 Comparison Operations

My comparison operations all return a `bool`.

| Operator | Description      | Type Signature     |
|----------|------------------|--------------------|
| `==`     | Equal            | `(T, T) -> bool`   |
| `!=`     | Not equal        | `(T, T) -> bool`   |
| `<`      | Less than        | `(int, int) -> bool` or `(float, float) -> bool` |
| `<=`     | Less or equal    | `(int, int) -> bool` or `(float, float) -> bool` |
| `>`      | Greater than     | `(int, int) -> bool` or `(float, float) -> bool` |
| `>=`     | Greater or equal | `(int, int) -> bool` or `(float, float) -> bool` |

### 4.6 Logical Operations

| Operator | Description  | Type Signature          |
|----------|--------------|-------------------------|
| `and`    | Logical AND  | `(bool, bool) -> bool`  |
| `or`     | Logical OR   | `(bool, bool) -> bool`  |
| `not`    | Logical NOT  | `(bool) -> bool`        |

### 4.7 Function Calls

I always use prefix notation for function calls.

```nano
(add 2 3)
(multiply (add 1 2) 4)
(is_prime 17)
(println "hello")
```

### 4.8 If Expressions

I treat `if` as an expression that returns a value.

```nano
let x: int = if (> a 0) {
    42
} else {
    -1
}
```

I require both branches to return the same type. I require an `else` branch.

I support `else if` chaining for multiple conditions.

```nano
if x > 10 {
    (println "big")
} else if x > 5 {
    (println "medium")
} else {
    (println "small")
}
```

### 4.9 Evaluation Order

I evaluate expressions from left to right.

```nano
(+ (f x) (g y))  # I evaluate f(x) before g(y)
(f x) + (g y)    # I evaluate f(x) before g(y)
```

## 5. Statements

### 5.1 Variable Declaration

I use `let` to declare variables.

```nano
let x: int = 42
let mut counter: int = 0
```

I make variables immutable by default. You must use `mut` to declare a mutable variable.

### 5.2 Assignment

I only allow you to reassign mutable variables. I use `set` for this.

```nano
let mut x: int = 0
set x (+ x 1)
# let y: int = 0
# set y 1  # ERROR: y is not mutable
```

### 5.3 While Loop

```nano
while condition {
    # body
}
```

I require the condition to be a `bool` expression. I execute the loop while the condition is `true`.

```nano
let mut i: int = 0
while (< i 10) {
    print i
    set i (+ i 1)
}
```

### 5.4 For Loop

```nano
for identifier in expression {
    # body
}
```

I provide the `for` loop as a way to iterate over a range.

```nano
for i in (range 0 10) {
    print i
}

# This is equivalent to:
let mut i: int = 0
while (< i 10) {
    print i
    set i (+ i 1)
}
```

### 5.5 Return Statement

```nano
return expression
```

I use this to return a value from a function. I check that the expression type matches the return type I defined for the function.

### 5.6 Expression Statement

I allow any expression to be used as a statement.

```nano
print "hello"
(add 2 3)  # I discard the result
```

## 6. Functions

### 6.1 Function Definition

```nano
fn name(param1: type1, param2: type2) -> return_type {
    # body
}
```

I require that every function:
1. Defines explicit parameter types.
2. Defines an explicit return type.
3. Returns a value if the return type is not `void`.
4. Includes a shadow test.

### 6.2 Parameters

I pass parameters by value. I make them immutable within the function.

```nano
fn increment(x: int) -> int {
    # set x (+ x 1)  # ERROR: Parameters are immutable
    return (+ x 1)
}
```

### 6.3 Return Type

I require a return type for every function.
- If the function is not `void`, I ensure every code path returns a value.
- If the function is `void`, you can use `return` without a value or omit it.

```nano
fn get_sign(x: int) -> int {
    if (> x 0) {
        return 1
    } else {
        if (< x 0) {
            return -1
        } else {
            return 0
        }
    }
}

fn greet() -> void {
    print "Hello"
}
```

### 6.4 External Functions (FFI)

I allow you to declare external functions to call the C standard library.

```nano
extern fn function_name(param: type) -> return_type
```

These declarations have no body and do not require a shadow test. I call them using their C names. I require these functions to be safe.

```nano
# Declare external C functions
extern fn sqrt(x: float) -> float
extern fn pow(x: float, y: float) -> float
extern fn isdigit(c: int) -> int
extern fn strlen(s: string) -> int

# Use them in my code
fn hypotenuse(a: float, b: float) -> float {
    let a_sq: float = (pow a 2.0)
    let b_sq: float = (pow b 2.0)
    return (sqrt (+ a_sq b_sq))
}

shadow hypotenuse {
    assert (== (hypotenuse 3.0 4.0) 5.0)
}
```

I expect you to only expose C functions that are safe. They should use explicit lengths and avoid pointer arithmetic.

## 7. Shadow Tests

### 7.1 Purpose

I use shadow tests to ensure honesty.
- I run them during compilation.
- I stop compilation if a test fails.
- I remove them from production builds.
- I use them to document my behavior.

### 7.2 Syntax

```nano
shadow function_name {
    # test body with assertions
}
```

I require exactly one shadow test block for every function you define. You must place it after the function definition.

### 7.3 Assertions

I use `assert` within shadow tests to verify my behavior.

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -5 3) -2)
}
```

### 7.4 Assertion Semantics

I evaluate the boolean expression given to `assert`.
- If it is `true`, I continue.
- If it is `false`, I stop compilation and report the error.

### 7.5 Coverage Requirements

I expect shadow tests to cover normal cases, edge cases, and boundary conditions.

### 7.6 Execution Order

I run each shadow test immediately after I finish defining the function it tests.

## 8. Semantics

### 8.1 Static Scoping

I use static scoping. I resolve variables when I compile your code.

```nano
let x: int = 1

fn f() -> int {
    return x  # This is the global x
}

fn g() -> int {
    let x: int = 2
    return (f)  # I return 1
}

shadow f {
    assert (== (f) 1)
}

shadow g {
    assert (== (g) 1)
}
```

### 8.2 Variable Shadowing

I allow inner scopes to shadow variables from outer scopes.

```nano
let x: int = 1
{
    let x: int = 2  # I shadow the outer x
    print x         # I print 2
}
print x            # I print 1
```

### 8.3 Type Equivalence

I consider types equivalent only if they share the same name. I do not use structural typing.

### 8.4 No Implicit Conversions

I require all type conversions to be explicit.

```nano
let x: int = 42
# let y: float = x  # ERROR: I do not convert implicitly
```

### 8.5 Short-Circuit Evaluation

I use short-circuit evaluation for `and` and `or`.

```nano
(and false (expensive_computation))  # I do not call expensive_computation
(or true (expensive_computation))    # I do not call expensive_computation
```

## 9. Compilation Model

### 9.1 Phases

1. **Lexing**: I turn your source text into tokens.
2. **Parsing**: I turn those tokens into an AST.
3. **Type Checking**: I verify your types, tests, and return paths.
4. **Shadow Test Execution**: I run your tests.
5. **Transpilation**: I turn my AST into C code.
6. **C Compilation**: I use a C compiler to produce a binary.

### 9.2 Shadow Test Compilation

I extract tests during parsing and check them for correctness. I execute them while I compile. I do not include them in my final output.

### 9.3 C Transpilation

I produce clean C code.

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}
```

I turn that into:

```c
int64_t add(int64_t a, int64_t b) {
    return a + b;
}
```

### 9.4 Entry Point

I require you to define a `main` function.

```nano
fn main() -> int {
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

### 9.5 Built-in Functions

I provide built-in functions through my runtime. I currently offer 72 functions across several categories.

**Core I/O:**
- `print`, `println`
- `assert`

**Math Operations:**
- `abs`, `min`, `max`, `sqrt`, `pow`, `floor`, `ceil`, `round`, `sin`, `cos`, `tan`

**String Operations:**
- `str_length`, `str_concat`, `str_substring`, `str_contains`, `str_equals`
- `char_at`, `string_from_char`, `is_digit`, `is_alpha`, `is_alnum`, `is_whitespace`, `is_upper`, `is_lower`
- `int_to_string`, `string_to_int`, `digit_value`, `char_to_lower`, `char_to_upper`

**Array Operations:**
- `at`, `array_length`, `array_new`, `array_set`

**List Operations:**
- `list_int_*` operations for dynamic lists.

**OS Operations:**
- Functions for files, directories, paths, and system commands.

**Iteration:**
- `range`

I have documented these in [STDLIB.md](STDLIB.md).

## 10. Example Programs

### 10.1 Fibonacci

```nano
fn fib(n: int) -> int {
    if (<= n 1) {
        return n
    } else {
        return (+ (fib (- n 1)) (fib (- n 2)))
    }
}

shadow fib {
    assert (== (fib 0) 0)
    assert (== (fib 1) 1)
    assert (== (fib 2) 1)
    assert (== (fib 5) 5)
    assert (== (fib 10) 55)
}

fn main() -> int {
    print (fib 10)
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

### 10.2 Prime Number Checker

```nano
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
    assert (== (is_prime 1) false)
    assert (== (is_prime 2) true)
    assert (== (is_prime 3) true)
    assert (== (is_prime 4) false)
    assert (== (is_prime 17) true)
    assert (== (is_prime 100) false)
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

## 11. Design Rationale

### 11.1 Why Both Prefix and Infix Notation?

I support both notations to balance precision and familiarity. My prefix notation makes nesting explicit and removes ambiguity. This is helpful for machines. My infix notation is familiar to humans.

To avoid traditional precedence issues, I give all my infix operators equal precedence. I evaluate them from left to right. I require you to use parentheses if you want a different order.

I only use prefix notation for my function calls. This keeps them distinct from my operators.

### 11.2 Why Mandatory Shadow Tests?

I require tests because untested code is a claim without proof. My tests also serve as documentation and give you confidence in my correctness.

### 11.3 Why Static Typing?

I use static typing to catch errors before your code ever runs. It makes my semantics clearer and helps machines generate correct code.

### 11.4 Why C Transpilation?

I transpile to C because it is portable and efficient. It allows me to use existing tools and will eventually allow me to compile myself.

## 12. Future Extensions

**What I have implemented in v0.1:**
- Arrays (static and dynamic)
- Structs
- Enums
- Unions
- Generics (List<T> and HashMap<K,V>)
- First-class functions
- Pattern matching
- My standard library
- My C transpiler

**What I am developing:**
- Tuple types
- My self-hosted compiler

**My plans for the future:**
- A module system
- More generic types
- Async primitives
- Memory management hints
- A package manager

I will ensure that all my extensions follow my core principles. I will remain minimal and unambiguous. I will always require tests.
