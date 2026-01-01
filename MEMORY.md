# MEMORY.md - NanoLang LLM Training Reference

> **Purpose:** This file is designed specifically for Large Language Model consumption. It contains the essential knowledge needed to generate, debug, and understand NanoLang code. Pair this with `spec.json` for complete language coverage.

## 🚨 READ FIRST: LLM-First Design

**NanoLang has EXACTLY ONE canonical way to write each construct.**

**REQUIRED READING:**
1. **`docs/CANONICAL_STYLE.md`** - The One True Way™ for every operation
2. **`docs/LLM_CORE_SUBSET.md`** - The 50-primitive core to learn first

**Quick Rules:**
- ✅ **ONE syntax per operation** - No alternatives
- ✅ **Prefix notation ONLY** - `(f x y)` never `f(x, y)` or `x + y`
- ✅ **Explicit types** - Always annotate
- ✅ **Core subset first** - Advanced features only when asked
- ✅ **Use `(+ str1 str2)`** not `str_concat` (deprecated)
- ✅ **Use `(cond ...)`** for expressions, `if/else` for statements

**Why this matters:** When there's only one way to do it, LLMs can't get it wrong.

---

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

### 2. ALWAYS Include Shadow Tests ⚠️ MANDATORY ⚠️
**Every function MUST have a shadow test. This is a CORE DESIGN PRINCIPLE of NanoLang.**

**NO EXCEPTIONS: This applies to:**
- ✅ Core library functions
- ✅ Application code
- ✅ **Example code**
- ✅ **Utility functions**
- ✅ **Demo programs**
- ✅ **Test files**
- ❌ ONLY EXCEPTION: `extern` functions (C FFI)

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

**Why Shadow Tests Are Mandatory:**
1. **Correctness:** Code is tested at compile time, not runtime
2. **Documentation:** Tests show how to use functions
3. **LLM-Friendly:** Forces AI to think through edge cases
4. **Self-Hosting:** Enables the compiler to validate itself

**If you forget shadow tests, compilation will show warnings. This is intentional.**

**For LLMs:** When generating NanoLang code, ALWAYS include shadow tests for every function you write. This is not optional - it's part of the language design.

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

# External C Functions (FFI)
extern fn function_name(param: type) -> return_type

# ⚠️ Extern calls MUST be inside unsafe blocks
fn call_extern() -> int {
    let mut result: int = 0
    unsafe {
        set result (function_name arg)
    }
    return result
}

# Public external functions (for modules)
pub extern fn module_function(param: type) -> return_type

# Import module with alias
import "modules/math_helper.nano" as Math
let result: int = (Math.add 2 3)

# Import without alias
import "modules/utilities.nano"
(some_function arg)
```

### Control Flow
```nano
# If-else (both branches required!)
if condition {
    # then branch
} else {
    # else branch
}

# Cond expression (multi-branch conditional - cleaner than nested if/else)
(cond
    (condition1 value1)
    (condition2 value2)
    (condition3 value3)
    (else default_value))

# Example: Number classification
fn classify(n: int) -> string {
    return (cond
        ((< n 0) "negative")
        ((== n 0) "zero")
        ((< n 10) "small")
        (else "large"))
}

# Example: Letter grade
fn grade(score: int) -> string {
    return (cond
        ((>= score 90) "A")
        ((>= score 80) "B")
        ((>= score 70) "C")
        ((>= score 60) "D")
        (else "F"))
}

# While loop
while condition {
    # body
}

# For loop (range only)
for i in (range 0 10) {
    # i goes from 0 to 9
}

# Unsafe blocks (for FFI and unchecked operations)
unsafe {
    # Mark explicit trust boundaries
    # Used for extern function calls and unchecked ops
}
```

### Operators (Always Prefix!)
```nano
# Arithmetic
(+ a b)  (- a b)  (* a b)  (/ a b)  (% a b)

# String concatenation (+ works for strings too!)
(+ "hello" " world")  # Returns "hello world"
(+ (+ "nano" "lang") "!")  # Returns "nanolang!"

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

# Array (dynamic, garbage-collected)
array<int>  # Fixed type, dynamic size
let arr: array<int> = [1, 2, 3, 4]          # Array literal
let empty: array<int> = []                   # Empty array
let first: int = (at arr 0)                  # Access element
let mut nums: array<int> = []
set nums (array_push nums 42)                # Append element
let val: int = (array_pop nums)              # Remove last
set nums (array_remove_at nums 0)            # Remove at index

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

### Step 2: Compile and Test
```bash
# Compile to binary
./bin/nanoc mycode.nano -o mycode

# Run the binary
./mycode
```

### Step 3: Check Shadow Tests
Shadow tests compile into the binary and run at runtime. If they fail, fix them first:
```nano
shadow factorial {
    assert (== (factorial 5) 120)  # This runs when the binary executes!
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

# Use module functions with namespace prefix
fn main() -> int {
    # SDL functions available directly
    (SDL_Init 0)
    
    # stdlib functions use namespace
    let r: Result<int, string> = Result.Ok { value: 42 }

    # Result helper functions (is_ok/unwrap/etc) are planned once generic
    # functions are supported. For now, use match on Result.
    match r {
        Ok(v) => {
            (println v.value)
        }
        Err(e) => {
            (println e.error)
        }
    }
    
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

## Unsafe Blocks

**Purpose**: Explicitly mark code that performs potentially dangerous operations, requiring conscious safety decisions by programmers.

### What Requires `unsafe`
- **FFI calls**: All `extern` function calls MUST be inside `unsafe` blocks
- **Future**: Unchecked arithmetic, raw pointer operations, inline assembly

### Basic Syntax
```nano
fn call_external_function() -> int {
    let mut result: int = 0
    unsafe {
        set result (some_extern_function)
    }
    return result
}
```

### Multiple and Nested Unsafe Blocks
```nano
fn complex_operation() -> int {
    let mut total: int = 0
    
    /* First unsafe operation */
    unsafe {
        set total (extern_func_1)
    }
    
    /* Safe code in between */
    let x: int = (+ total 10)
    
    /* Second unsafe operation */
    unsafe {
        set total (+ x (extern_func_2))
    }
    
    /* Nested unsafe blocks are allowed */
    unsafe {
        unsafe {
            set total (+ total 1)
        }
    }
    
    return total
}
```

### Why Unsafe Blocks Matter
1. **Explicit Safety Boundaries**: Clear visual markers for code review
2. **Compiler Enforcement**: Can't call `extern` functions without `unsafe`
3. **Audit Trail**: Easy to find all potentially dangerous operations
4. **MISRA/JSF Alignment**: Explicit unsafe operations satisfy coding standards
5. **Gradual Safety**: Safe by default, unsafe by choice

### Example: Wrapping External Libraries
```nano
/* Declare external C function */
extern fn nl_os_getpid() -> int

/* Safe wrapper - encapsulates the unsafe call */
fn get_process_id() -> int {
    let mut pid: int = 0
    unsafe {
        set pid (nl_os_getpid)
    }
    return pid
}

/* Now users can call the safe wrapper */
fn main() -> int {
    let pid: int = (get_process_id)  /* Safe! */
    (println pid)
    return 0
}
```

### Best Practices
- **Minimize unsafe scope**: Keep `unsafe` blocks as small as possible
- **Encapsulate in safe functions**: Wrap extern calls in safe interfaces
- **Document invariants**: Comment why the unsafe operation is actually safe
- **Validate inputs**: Check preconditions before unsafe operations

## Checked Arithmetic Operations (SAFETY)

**NEW**: NanoLang provides checked arithmetic operations that detect overflow, underflow, and division by zero at runtime, returning `Result<int, string>` instead of crashing or wrapping silently.

**MISRA Rule 12.4 Compliance**: These functions satisfy safety-critical requirements for overflow detection.  
**JSF AV Rule 204 Compliance**: Division operations check for zero divisor.

### Available Functions

Import the module:
```nano
import "modules/stdlib/checked_math.nano"
```

Then use:

```nano
/* Safe addition - catches overflow/underflow */
fn example_addition() -> int {
    let result: Result<int, string> = (checked_add 1000000 2000000)
    match result {
        Ok(v) => {
            (println v.value)  /* 3000000 */
        },
        Err(e) => {
            (println e.error)  /* "Integer overflow in addition" */
        }
    }
    return 0
}

/* Safe division - catches division by zero */
fn example_division() -> int {
    let result: Result<int, string> = (checked_div 100 0)
    match result {
        Ok(v) => {
            (println v.value)
        },
        Err(e) => {
            (println e.error)  /* "Division by zero" */
        }
    }
    return 0
}
```

### All Checked Operations

| Function | Returns | Catches |
|----------|---------|---------|
| `checked_add(a, b)` | `Result<int, string>` | Overflow, underflow |
| `checked_sub(a, b)` | `Result<int, string>` | Overflow, underflow |
| `checked_mul(a, b)` | `Result<int, string>` | Overflow, underflow |
| `checked_div(a, b)` | `Result<int, string>` | Division by zero, INT64_MIN / -1 |
| `checked_mod(a, b)` | `Result<int, string>` | Modulo by zero |

### When to Use

- **Safety-critical applications**: Avionics, medical devices, automotive
- **Financial calculations**: Money arithmetic must not overflow silently
- **User input validation**: Prevent malicious overflow attacks
- **Production code**: When correctness matters more than performance

### When NOT to Use

- **Performance-critical inner loops**: Checked ops are slower (~2-3x)
- **Proven-safe calculations**: E.g., loop counters with known bounds
- **Compiler-optimized code**: When overflow is mathematically impossible

### Example: Safe Calculator

See `examples/nl_checked_math_demo.nano` for a complete demonstration.

```nano
import "modules/stdlib/checked_math.nano"

fn safe_calculator(a: int, op: string, b: int) -> Result<int, string> {
    if (str_equals op "+") {
        return (checked_add a b)
    } else { if (str_equals op "-") {
        return (checked_sub a b)
    } else { if (str_equals op "*") {
        return (checked_mul a b)
    } else { if (str_equals op "/") {
        return (checked_div a b)
    } else {
        return Result.Err { error: "Unknown operator" }
    }}}}
}
```

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

## Dynamic Array Operations

**SEMANTICS:** NanoLang arrays use **pure functional semantics** - all operations return new arrays, original arrays are unchanged. This matches the immutable-by-default philosophy.

NanoLang arrays are dynamic and garbage-collected. They can grow and shrink as needed.

### Creating Arrays
```nano
let empty: array<int> = []                   # Empty array
let nums: array<int> = [1, 2, 3, 4, 5]       # Array literal
let strings: array<string> = ["a", "b"]      # String array
let nested: array<array<int>> = [[1], [2]]   # Nested arrays
```

### Adding Elements
```nano
let mut arr: array<int> = []
set arr (array_push arr 42)                  # Append to end
set arr (array_push arr 43)                  # arr is now [42, 43]
```

**Important:** `array_push` returns a new array. You must assign it back to the variable.

### Removing Elements
```nano
let mut arr: array<int> = [10, 20, 30, 40]

# Remove last element
let val: int = (array_pop arr)               # val = 40, arr unchanged
# Note: array_pop doesn't modify the array in place

# Remove at specific index
set arr (array_remove_at arr 1)              # Remove index 1
# arr becomes [10, 30, 40] (removed 20)
```

### Accessing Elements
```nano
let first: int = (at arr 0)                  # Get element (bounds-checked)
(array_set arr 0 100)                        # Set element (bounds-checked)
let len: int = (array_length arr)            # Get length
```

### Complete Example
```nano
fn array_demo() -> int {
    # Create empty array
    let mut numbers: array<int> = []
    
    # Add elements
    set numbers (array_push numbers 10)
    set numbers (array_push numbers 20)
    set numbers (array_push numbers 30)
    # numbers = [10, 20, 30]
    
    # Remove last
    let last: int = (array_pop numbers)
    # last = 30, but numbers still [10, 20, 30]
    
    # Remove at index
    set numbers (array_remove_at numbers 0)
    # numbers = [20, 30]
    
    # Access and create modified version
    let first: int = (at numbers 0)          # first = 20
    set numbers (array_set numbers 0 99)     # numbers = [99, 30] (new array)
    
    return (array_length numbers)            # Returns 2
}

shadow array_demo {
    assert (== (array_demo) 2)
}
```

### Array Operations Summary

**CHOSEN SEMANTICS: Fully Functional (Immutable by Default)**

All array operations return new arrays and leave the original unchanged. This matches NanoLang's immutable-by-default philosophy.

| Operation | Function | Returns | Original Array | Performance |
|-----------|----------|---------|----------------|-------------|
| Append | `array_push(arr, elem)` | new array | unchanged | O(1) amortized |
| Remove last | `array_pop(arr)` | removed element | unchanged | O(1) |
| Remove at index | `array_remove_at(arr, idx)` | new array | unchanged | O(n) |
| Get element | `at(arr, idx)` | element value | unchanged | O(1) |
| Set element | `array_set(arr, idx, val)` | new array | unchanged | O(n) copy |
| Get length | `array_length(arr)` | int | unchanged | O(1) |

**Implementation Detail:** Under the hood, operations use structural sharing and copy-on-write for efficiency. You don't pay for full array copies unless you actually modify elements.

**Best Practice:** If you need to make many modifications, collect them and create a new array. Or consider using a mutable algorithm with explicit copies:

```nano
let mut working_copy: array<int> = arr  # Shallow copy
set working_copy (array_push working_copy 1)
set working_copy (array_push working_copy 2)
# Original arr is still unchanged
```

**Key Principle:** Pure functional semantics → no hidden mutations → easier to reason about → fewer bugs.

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
fn divide(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Result.Err { error: "Division by zero" }
    }
    return Result.Ok { value: (/ a b) }
}

fn main() -> int {
    let result: Result<int, string> = (divide 10 2)

    match result {
        Ok(v) => {
            (println v.value)  # Prints 5
        }
        Err(e) => {
            (println e.error)
        }
    }
    
    return 0
}

shadow divide {
    let r1: Result<int, string> = (divide 10 2)
    match r1 {
        Ok(v) => assert (== v.value 5),
        Err(e) => assert false
    }
    
    let r2: Result<int, string> = (divide 10 0)
    match r2 {
        Ok(v) => assert false,
        Err(e) => assert true
    }
}
```

Note: Helper functions like `is_ok`, `is_err`, `unwrap`, `unwrap_or`, `map`, and
`map_err` are planned once generic functions are supported.

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

## Advanced Features (v0.3+)

### Affine Types for Resource Safety

**Affine types** ensure resources (files, sockets, etc.) are used **at most once**, preventing use-after-free/close bugs at compile time.

```nano
# Declare a resource struct
resource struct FileHandle {
    fd: int
}

# FFI functions that work with resources
extern fn open_file(path: string) -> FileHandle
extern fn close_file(f: FileHandle) -> void
extern fn read_file(f: FileHandle) -> string

fn safe_file_usage() -> int {
    # Create resource
    let f: FileHandle = unsafe { (open_file "data.txt") }
    
    # Use resource (can use multiple times before consuming)
    let data: string = unsafe { (read_file f) }
    
    # Consume resource (transfers ownership, resource is "moved")
    unsafe { (close_file f) }
    
    # ERROR: Cannot use 'f' after it has been consumed!
    # let more: string = unsafe { (read_file f) }  # Compile error!
    
    return 0
}

shadow safe_file_usage {
    assert (== (safe_file_usage) 0)
}
```

**Key Rules**:
1. **Resources must be consumed** before end of scope (or compile error)
2. **Cannot use after consume** (compile-time error)
3. **Cannot consume twice** (compile-time error)
4. Resources transition through states: `UNUSED` → `USED` → `CONSUMED`

**When to Use**:
- File handles
- Network sockets
- Database connections
- GPU resources
- Anything that requires cleanup

---

### Self-Hosted Compiler Architecture

NanoLang is now **self-hosted** - the compiler is written in NanoLang!

**Two Compilers**:
1. **`bin/nanoc`** (symlink to `bin/nanoc_c`) - C reference compiler (Stage 1)
2. **`bin/nanoc_nano`** - Self-hosted NanoLang compiler (Stage 2)

**Note**: NanoLang is a compiled language. The interpreter was removed to eliminate dual implementation burden.

**Compilation Pipeline** (both compilers):
```
Source → Lex → Parse → TypeCheck → Transpile → C Compiler → Binary
```

**Bootstrap Process**:
1. **Stage 1**: C compiler (`nanoc_c`) compiles self-hosted components
2. **Stage 2**: Self-hosted compiler compiles itself
3. **Stage 3**: Stage 2 compiler validates its output matches Stage 1

**Compiler Phase Interfaces** (for self-hosted development):

```nano
# Phase 1: Lexer
fn lex_phase_run(source: string, filename: string) -> LexPhaseOutput {
    # Returns: { tokens, token_count, had_error, diagnostics }
}

# Phase 2: Parser
fn parse_phase_run(lex_output: LexPhaseOutput) -> ParsePhaseOutput {
    # Returns: { parser, had_error, diagnostics }
}

# Phase 3: Type Checker
fn typecheck_phase(parser_state: Parser) -> TypecheckPhaseOutput {
    # Returns: { had_error, diagnostics }
}

# Phase 4: Transpiler
fn transpile_phase(parser_state: Parser, c_file: string) -> TranspilePhaseOutput {
    # Returns: { c_source, had_error, diagnostics, output_path }
}
```

All phase outputs include:
- `had_error: bool`
- `diagnostics: List<CompilerDiagnostic>`

---

### Diagnostic System (Self-Hosted)

The self-hosted compiler uses a structured diagnostic system:

```nano
struct CompilerDiagnostic {
    severity: int,      # DIAG_ERROR, DIAG_WARNING, DIAG_INFO
    phase: int,         # PHASE_LEX, PHASE_PARSE, PHASE_TYPECHECK, PHASE_TRANSPILE
    code: string,       # Error code (e.g., "E001")
    message: string,    # Human-readable message
    location: CompilerSourceLocation
}

struct CompilerSourceLocation {
    file: string,
    line: int,
    column: int
}

# Create diagnostics
fn diag_error(phase: int, code: string, msg: string, loc: CompilerSourceLocation) -> CompilerDiagnostic
fn diag_warning(phase: int, code: string, msg: string, loc: CompilerSourceLocation) -> CompilerDiagnostic
```

**Elm-Style Error Messages** (in development):
- Show source code context
- Highlight problematic code
- Suggest fixes
- Explain *why* something is wrong

---

### Result Type Pattern (Error Handling)

For functions that can fail, use the `Result` union pattern:

```nano
union ResultInt {
    Ok { value: int },
    Err { error: string }
}

fn divide(a: int, b: int) -> ResultInt {
    if (== b 0) {
        return ResultInt.Err { error: "Division by zero" }
    }
    return ResultInt.Ok { value: (/ a b) }
}

shadow divide {
    match (divide 10 2) {
        Ok(v) => { assert (== v.value 5) }
        Err(e) => { assert false }
    }
    match (divide 10 0) {
        Ok(v) => { assert false }
        Err(e) => { assert true }
    }
}
```

---

## LLM Development Guidelines (Updated 2025)

When generating NanoLang code, remember:

- Think "What would the simplest, clearest version look like?"
- Make types explicit
- **Write shadow tests that verify correctness** (MANDATORY)
- Use prefix notation for all operations
- Leverage the type system to catch errors early
- **Use `resource struct` for types that require cleanup**
- **Wrap FFI calls in `unsafe { ... }` blocks**
- Use `Result` unions for fallible operations
- Check for resource leaks in complex functions

---

**For complete language reference, always consult `spec.json` in the root directory.**
