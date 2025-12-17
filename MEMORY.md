# MEMORY.md - NanoLang LLM Training Reference

> **Purpose:** This file is designed specifically for Large Language Model consumption. It contains the essential knowledge needed to generate, debug, and understand NanoLang code. Pair this with `spec.json` for complete language coverage.

## Critical First Principles

### 1. ALWAYS Use Prefix Notation
```nano
# CORRECT
(+ a b)
(* (+ x 1) (- y 2))
(println "hello")

# WRONG - This is not valid nanolang!
a + b
x * y
println("hello")
```

### 2. ALWAYS Include Shadow Tests
Every function MUST have a shadow test. No exceptions.

```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
    assert (== (double -3) -6)
    assert (== (double 0) 0)
}
```

**If you forget shadow tests, compilation will fail.**

### 3. ALWAYS Use Explicit Types
```nano
# CORRECT
let x: int = 42
let name: string = "Alice"
fn add(a: int, b: int) -> int { ... }

# WRONG - Type inference does not exist
let x = 42  # ERROR!
fn add(a, b) { ... }  # ERROR!
```

## Quick Syntax Reference

### Variables
```nano
let x: int = 42                    # Immutable (default)
let mut counter: int = 0           # Mutable
set counter (+ counter 1)          # Assignment (mut only)
```

### Functions
```nano
fn name(param1: type1, param2: type2) -> return_type {
    # body
    return value
}

shadow name {
    assert (== (name arg1 arg2) expected)
}

# External C Functions
extern fn function_name(param: type) -> return_type

# Public external functions (for modules)
pub extern fn module_function(param: type) -> return_type
```

### Control Flow
```nano
# If-else (both branches required!)
if condition {
    # then branch
} else {
    # else branch
}

# While loop
while condition {
    # body
}

# For loop (range only)
for i in (range 0 10) {
    # i goes from 0 to 9
}
```

### Operators (Always Prefix!)
```nano
# Arithmetic
(+ a b)  (- a b)  (* a b)  (/ a b)  (% a b)

# Comparison
(== a b)  (!= a b)  (< a b)  (<= a b)  (> a b)  (>= a b)

# Logical
(and p q)  (or p q)  (not p)

# Nested operations
(+ (* a b) (/ c d))  # (a*b) + (c/d)
```

### Types
```nano
# Primitives
int, float, bool, string, void

# Struct
struct Point { x: int, y: int }
let p: Point = Point { x: 10, y: 20 }
let x_val: int = p.x

# Enum
enum Status { Idle = 0, Running = 1, Done = 2 }
let s: Status = Status.Running

# Union (tagged union)
union Result { 
    Ok { value: int }, 
    Error { message: string } 
}

# Generic Union (NEW!)
union Result<T, E> {
    Ok { value: T },
    Err { error: E }
}

# Using generic unions
let success: Result<int, string> = Result.Ok { value: 42 }
let failure: Result<int, string> = Result.Err { error: "failed" }

# Match on union
match result {
    Ok(r) => { (println r.value) }
    Error(e) => { (println e.message) }
}

# Array
array<int>  # Fixed type, dynamic size
let arr: array<int> = [1, 2, 3, 4]
let first: int = (at arr 0)

# Generic List
List<int>, List<string>, List<Point>
let nums: List<int> = (List_int_new)
(List_int_push nums 42)

# Function types
fn(int, int) -> int
fn(string) -> bool
fn() -> void

# Tuples
(int, string)
(float, float, float)
let coord: (int, int) = (10, 20)
let x: int = coord.0
let y: int = coord.1
```

## Common Patterns

### Pattern 1: Counter Loop
```nano
let mut i: int = 0
while (< i 10) {
    (println i)
    set i (+ i 1)
}
```

### Pattern 2: Range Loop
```nano
for i in (range 0 10) {
    (println i)
}
```

### Pattern 3: Accumulator
```nano
fn sum_list(nums: List<int>) -> int {
    let mut total: int = 0
    let len: int = (List_int_length nums)
    let mut i: int = 0
    
    while (< i len) {
        let val: int = (List_int_get nums i)
        set total (+ total val)
        set i (+ i 1)
    }
    
    return total
}
```

### Pattern 4: Recursive Function
```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    }
    return (* n (factorial (- n 1)))
}

shadow factorial {
    assert (== (factorial 5) 120)
    assert (== (factorial 0) 1)
}
```

### Pattern 5: Struct Constructor
```nano
struct Point { x: int, y: int }

fn make_point(x: int, y: int) -> Point {
    return Point { x: x, y: y }
}

shadow make_point {
    let p: Point = (make_point 5 10)
    assert (== p.x 5)
    assert (== p.y 10)
}
```

### Pattern 6: First-Class Functions
```nano
fn apply(f: fn(int) -> int, x: int) -> int {
    return (f x)
}

fn double(x: int) -> int {
    return (* x 2)
}

shadow apply {
    assert (== (apply double 5) 10)
}

# Function variables
fn get_doubler() -> fn(int) -> int {
    return double
}

fn test() -> int {
    let f: fn(int) -> int = (get_doubler)
    return (f 21)  # Returns 42
}
```

## Common Errors and Fixes

### Error 1: Missing Shadow Test
```
Warning: Function 'foo' is missing a shadow test
```
**Fix:** Add a shadow block after every function:
```nano
shadow foo {
    assert (== (foo 5) expected_value)
}
```

### Error 2: Type Mismatch
```
Error at line X: Type mismatch in let statement
Error at line X: Return type mismatch
```
**Fix:** Check that types match exactly:
```nano
# Wrong
let x: int = 3.14  # ERROR: float vs int

# Right
let x: float = 3.14
let y: int = 42
```

### Error 3: Forgetting Prefix Notation
```
Error: Unexpected token
```
**Fix:** All operations use prefix notation:
```nano
# Wrong
if x > 5 { ... }
let result = a + b

# Right
if (> x 5) { ... }
let result: int = (+ a b)
```

### Error 4: Immutability Violation
```
Error: Cannot assign to immutable variable
```
**Fix:** Declare variable as mutable:
```nano
# Wrong
let counter: int = 0
set counter (+ counter 1)  # ERROR!

# Right
let mut counter: int = 0
set counter (+ counter 1)  # OK
```

### Error 5: Missing Else Branch
```
Error: If statement requires else branch
```
**Fix:** Always include else:
```nano
# Wrong
if condition {
    do_something
}

# Right
if condition {
    do_something
} else {
    # Even if empty, else is required
    (print "")
}
```

### Error 6: Undefined Variable/Function
```
Error: Undefined variable 'x'
Error: Undefined function 'foo'
```
**Fix:** Declare before use, check spelling:
```nano
# Wrong
let y: int = (add x 5)  # x not declared!

# Right
let x: int = 10
let y: int = (add x 5)
```

## Debugging Workflow

### Step 1: Read Error Messages
Errors show line and column numbers:
```
Error at line 42, column 15: Type mismatch
```
Look at line 42, character 15 to find the issue.

### Step 2: Use the Interpreter for Testing
```bash
# Quick test without compilation
./bin/nano mycode.nano

# Trace execution
./bin/nano mycode.nano --trace-all
./bin/nano mycode.nano --trace-function=myfunc
./bin/nano mycode.nano --trace-var=counter
```

### Step 3: Check Shadow Tests
Shadow tests run at compile time. If they fail, fix them first:
```nano
shadow factorial {
    assert (== (factorial 5) 120)  # This runs during compilation!
}
```

### Step 4: Verify Types
Most errors are type mismatches. Check:
1. Variable declarations match usage
2. Function return types match return statements
3. Function call arguments match parameter types

### Step 5: Use --keep-c to See Generated Code
```bash
./bin/nanoc mycode.nano -o mycode --keep-c
# Look at mycode.c to see what was generated
```

## How to Read spec.json

The `spec.json` file is your authoritative reference. Here's how to navigate it:

### Types Section
```json
"types": {
  "primitives": [...],     # Basic types: int, float, bool, string, void
  "composite": {           # Complex types
    "array": {...},        # Dynamic arrays
    "struct": {...},       # Product types
    "enum": {...},         # Named constants
    "union": {...},        # Tagged unions (sum types)
    "generic": {...},      # List<T>
    "function": {...},     # First-class functions
    "tuple": {...}         # Multiple values
  }
}
```

### Stdlib Section
All built-in functions are documented:
```json
"stdlib": {
  "io": { "print": {...}, "println": {...}, "assert": {...} },
  "math": { "abs": {...}, "sqrt": {...}, "sin": {...}, ... },
  "string": { "str_length": {...}, "str_concat": {...}, ... },
  "array": { "at": {...}, "array_length": {...}, ... },
  "generics": { "List_new": {...}, "List_push": {...}, ... }
}
```

### Operations Section
How to use operators:
```json
"operations": {
  "arithmetic": { "add": { "operator": "+", "arity": 2, ... }, ... },
  "comparison": { "eq": { "operator": "==", ... }, ... },
  "logical": { "and": { "operator": "and", ... }, ... }
}
```

## Code Generation Best Practices

### 1. Start with Shadow Tests
Write the shadow test first, then implement:
```nano
fn process_data(x: int) -> int {
    # TODO: implement
    return x
}

shadow process_data {
    assert (== (process_data 5) 10)   # Define expected behavior
    assert (== (process_data 0) 0)
}
```

### 2. Use Descriptive Names
```nano
# Good
fn calculate_distance(p1: Point, p2: Point) -> float
let total_count: int = 0
let is_valid: bool = true

# Avoid
fn calc(p1: Point, p2: Point) -> float
let tc: int = 0
let b: bool = true
```

### 3. Keep Functions Small
Each function should do one thing:
```nano
# Good - focused functions
fn parse_input(s: string) -> int { ... }
fn validate_input(x: int) -> bool { ... }
fn process_input(x: int) -> int { ... }

# Avoid - monolithic function
fn do_everything(s: string) -> int {
    # 100 lines of mixed concerns
}
```

### 4. Use Comments for Complex Logic
```nano
fn complex_calculation(x: int) -> int {
    # Step 1: Normalize input to 0-100 range
    let normalized: int = (% x 101)
    
    # Step 2: Apply scaling factor
    let scaled: int = (* normalized 3)
    
    # Step 3: Add offset
    return (+ scaled 10)
}
```

### 5. Leverage Type Safety
Let the type system catch errors:
```nano
# Use distinct types instead of raw ints
enum UserId { Value = 0 }
enum ProductId { Value = 0 }

# Now you can't mix them up by accident
fn get_user(id: UserId) -> User { ... }
fn get_product(id: ProductId) -> Product { ... }
```

## Module System Quick Reference

### Using Modules
```nano
# Import an FFI module (looks for module.json with FFI bindings)
import sdl

# Import a standard library module
import std.result

# Use module functions with namespace prefix
fn main() -> int {
    # SDL functions available directly
    (SDL_Init 0)
    
    # stdlib functions use namespace
    let r: Result<int, string> = Result.Ok { value: 42 }
    let is_success: bool = (std.result.is_ok r)
    
    return 0
}
```

### Module Installation
Modules automatically install dependencies:
```bash
# First use of SDL module auto-runs:
# brew install sdl2  (or apt-get install libsdl2-dev)
```

### Available Modules
- **ncurses** - Terminal UI
- **sdl** - 2D graphics and input
- **sdl_mixer** - Audio playback
- **sdl_ttf** - Font rendering
- **glfw** - OpenGL windowing
- **onnx** - Neural network inference

Check `modules/` directory for full list.

## Performance Tips

### 1. Use Primitives Over Structs When Possible
```nano
# Faster
fn add_coords(x1: int, y1: int, x2: int, y2: int) -> (int, int)

# Slower (struct allocation)
fn add_points(p1: Point, p2: Point) -> Point
```

### 2. Avoid Unnecessary String Operations
```nano
# Expensive
let mut result: string = ""
for i in (range 0 1000) {
    set result (str_concat result "x")  # Allocates each time!
}

# Better: Use array/list and join once
```

### 3. Use Lists for Dynamic Data
```nano
# Good for dynamic data
let numbers: List<int> = (List_int_new)
(List_int_push numbers 42)

# Arrays are fixed-element-type but dynamic size
let arr: array<int> = [1, 2, 3]
```

### 4. Minimize Shadow Test Complexity
Shadow tests run at compile time, keep them fast:
```nano
shadow process {
    # Good - quick checks
    assert (== (process 5) 10)
    assert (== (process 0) 0)
}

shadow process {
    # Avoid - slow compile times
    let mut i: int = 0
    while (< i 10000) {
        assert (== (process i) (* i 2))
        set i (+ i 1)
    }
}
```

## Testing Philosophy

### Shadow Tests Are Mandatory
- Every function must have shadow tests
- Shadow tests run at compile time
- They verify correctness before runtime

### Shadow Test Coverage
```nano
fn divide(a: int, b: int) -> int {
    return (/ a b)
}

shadow divide {
    # Test normal cases
    assert (== (divide 10 2) 5)
    assert (== (divide 7 3) 2)
    
    # Test edge cases
    assert (== (divide 0 5) 0)
    assert (== (divide -10 2) -5)
    
    # Test negative divisor
    assert (== (divide 10 -2) -5)
}
```

### What NOT to Test in Shadow
- Don't test external I/O (SDL, files, network)
- Don't test random/non-deterministic functions
- Don't test functions that use `extern` C functions
- Keep shadow tests pure and deterministic

For complex integration testing, mark functions as using extern:
```nano
fn render_graphics() -> void {
    # Uses SDL - can't shadow test
}

shadow render_graphics {
    # Skip or keep minimal
}
```

## Idioms and Conventions

### 1. Use Struct Constructors
```nano
struct Point { x: int, y: int }

fn Point_new(x: int, y: int) -> Point {
    return Point { x: x, y: y }
}

fn Point_zero() -> Point {
    return Point { x: 0, y: 0 }
}
```

### 2. Name Generic Functions with Type Suffix
```nano
# For List<int> use:
List_int_new()
List_int_push()
List_int_get()

# For List<Point> use:
List_Point_new()
List_Point_push()
List_Point_get()
```

### 3. Use Enums for Status/State
```nano
enum GameState {
    Menu = 0,
    Playing = 1,
    Paused = 2,
    GameOver = 3
}

let state: GameState = GameState.Menu
```

### 4. Return Early for Clarity
```nano
fn validate_and_process(x: int) -> int {
    # Early return for invalid cases
    if (< x 0) {
        return -1
    }
    if (> x 100) {
        return -1
    }
    
    # Main logic for valid case
    return (* x 2)
}
```

### 5. Use Bool Functions for Predicates
```nano
fn is_positive(x: int) -> bool {
    return (> x 0)
}

fn is_even(x: int) -> bool {
    return (== (% x 2) 0)
}

fn is_valid_range(x: int, min: int, max: int) -> bool {
    return (and (>= x min) (<= x max))
}
```

## Quick Checklist for Code Generation

Before generating nanolang code, verify:

- [ ] All functions have shadow tests
- [ ] All operations use prefix notation: `(+ a b)` not `a + b`
- [ ] All variables have explicit types: `let x: int = 5`
- [ ] All if statements have else branches
- [ ] Mutable variables declared with `mut` keyword
- [ ] Function parameters have type annotations
- [ ] Function return type is specified
- [ ] Struct field access uses dot notation: `point.x`
- [ ] Function calls use prefix: `(function arg1 arg2)`
- [ ] Range loops use proper syntax: `for i in (range 0 10)`

## Advanced Features

### First-Class Functions
```nano
# Function as parameter
fn apply(f: fn(int) -> int, x: int) -> int {
    return (f x)
}

# Function as return value
fn get_incrementor() -> fn(int) -> int {
    return increment
}

# Function in variable
let operation: fn(int, int) -> int = add
let result: int = (operation 5 3)
```

### Tuples
```nano
# Tuple type and literal
let coord: (int, int) = (10, 20)
let x: int = coord.0
let y: int = coord.1

# Function returning tuple
fn divide_with_remainder(a: int, b: int) -> (int, int) {
    let quotient: int = (/ a b)
    let remainder: int = (% a b)
    return (quotient, remainder)
}

shadow divide_with_remainder {
    let result: (int, int) = (divide_with_remainder 10 3)
    assert (== result.0 3)
    assert (== result.1 1)
}
```

### Pattern Matching on Unions
```nano
union Result {
    Ok { value: int },
    Error { code: int, message: string }
}

fn process_result(r: Result) -> int {
    match r {
        Ok(data) => {
            return data.value
        }
        Error(err) => {
            (println err.message)
            return -1
        }
    }
}
```

### Standard Library Result<T, E>
NanoLang includes a standard library generic Result type for error handling:

```nano
import std.result

fn divide(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Result.Err { error: "Division by zero" }
    }
    return Result.Ok { value: (/ a b) }
}

fn main() -> int {
    let result: Result<int, string> = (divide 10 2)
    
    if (std.result.is_ok result) {
        let value: int = (std.result.unwrap result "Should not fail")
        (println value)  # Prints 5
    }
    
    return 0
}

shadow divide {
    let r1: Result<int, string> = (divide 10 2)
    assert (std.result.is_ok r1)
    assert (== (std.result.unwrap r1 "failed") 5)
    
    let r2: Result<int, string> = (divide 10 0)
    assert (std.result.is_err r2)
}
```

**Standard Library Result Functions:**
- `std.result.is_ok<T,E>(result: Result<T,E>) -> bool`
- `std.result.is_err<T,E>(result: Result<T,E>) -> bool`
- `std.result.unwrap<T,E>(result: Result<T,E>, msg: string) -> T`
- `std.result.unwrap_or<T,E>(result: Result<T,E>, default: T) -> T`
- `std.result.map<T,E,U>(result: Result<T,E>, f: fn(T) -> U) -> Result<U,E>`
- `std.result.map_err<T,E,F>(result: Result<T,E>, f: fn(E) -> F) -> Result<T,F>`

## Summary: The NanoLang Vibe

NanoLang is designed for **clarity over cleverness**:

1. **Explicit everything** - No hidden behavior, no inference, no magic
2. **Test-driven** - Shadow tests force you to think about correctness
3. **Prefix notation** - One syntax rule eliminates precedence confusion
4. **Immutable by default** - Mutability is explicit with `mut`
5. **Static typing** - Catch errors at compile time
6. **Simple but complete** - Minimalist syntax, powerful features

When generating nanolang code:
- Think "What would the simplest, clearest version look like?"
- Make types explicit
- Write shadow tests that verify correctness
- Use prefix notation for all operations
- Leverage the type system to catch errors early

---

**For complete language reference, always consult `spec.json` in the root directory.**
