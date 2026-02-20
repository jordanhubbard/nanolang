# My Core Subset - Start Here

> **For LLMs:** Learn this subset first. Everything else is advanced. Do not use it unless the user asks.

## My Philosophy

My core subset contains exactly what you need to write 80% of programs. Master this first. Explore my advanced features later.

**Total Core Size:** ~50 primitives (syntax + stdlib + types)

---

## My Core Syntax (20 primitives)

### Variables
```nano
let name: type = value          # Immutable variable
let mut counter: int = 0        # Mutable variable
set counter (+ counter 1)       # Update mutable variable
```

### Functions
```nano
fn function_name(arg: type) -> return_type {
    return value
}

shadow function_name {
    assert (condition)
}
```

### Control Flow
```nano
# Expressions (returns value)
(cond
    ((test1) result1)
    ((test2) result2)
    (else default)
)

# Statements (side effects)
if (condition) {
    # body
} else {
    # alternative
}

# Loops
while (condition) {
    # body
}

for (let i: int = 0) (< i 10) (set i (+ i 1)) {
    # body
}
```

### Function Calls
```nano
(function arg1 arg2 arg3)
```

---

## My Core Types (6 types)

```nano
int      # 64-bit integer
float    # 64-bit floating point
string   # UTF-8 string
bool     # true or false
array<T> # Fixed-size array
void     # No return value
```

That is all. I do not want you to use my advanced types (structs, enums, unions, generics) unless you are asked.

---

## My Core Math Operations (8 operations)

```nano
(+ a b)      # Addition
(- a b)      # Subtraction
(* a b)      # Multiplication
(/ a b)      # Division
(% a b)      # Modulo
(abs x)      # Absolute value
(min a b)    # Minimum
(max a b)    # Maximum
```

---

## My Core Comparisons (6 operations)

```nano
(== a b)     # Equal
(!= a b)     # Not equal
(< a b)      # Less than
(> a b)      # Greater than
(<= a b)     # Less than or equal
(>= a b)     # Greater than or equal
```

---

## My Core Boolean Logic (3 operations)

```nano
(and a b)    # Logical AND
(or a b)     # Logical OR
(not x)      # Logical NOT
```

---

## My Core String Operations (6 functions)

```nano
(+ str1 str2)              # Concatenation
(str_length s)             # Length
(== s1 s2)                 # Compare
(char_at s index)          # Get character
(string_from_char code)    # Char to string
(int_to_string n)          # Int to string
```

**Example:**
```nano
let greeting: string = (+ "Hello, " name)
let len: int = (str_length greeting)
```

---

## My Core Array Operations (4 functions)

```nano
(array_new size initial_value)    # Create array
(array_get arr index)              # Read element
(array_set arr index value)        # Write element
(array_length arr)                 # Get length
```

**Example:**
```nano
let nums: array<int> = (array_new 10 0)
(array_set nums 0 42)
let val: int = (array_get nums 0)
```

---

## My Core I/O (4 functions)

```nano
(print s)            # Print without newline
(println s)          # Print with newline
(read_line)          # Read line from stdin
(int_to_string n)    # Convert int to string for printing
```

**Example:**
```nano
(println "Enter your name:")
let name: string = (read_line)
(println (+ "Hello, " name))
```

---

## My Core Control (2 functions)

```nano
(assert condition)    # Assert condition is true (shadow tests)
(panic message)       # Abort with error message
```

---

## My Core Module Imports

```nano
from "path/to/module.nano" import function1, function2
```

**Core Modules (use sparingly):**
- `modules/math_ext/math_ext.nano` - Extended math (sin, cos, etc.)
- `modules/std/io/stdio.nano` - File I/O

I do not want you to use other modules unless you are asked.

---

## Complete Example Program

```nano
# Factorial calculator - demonstrates core subset

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
    assert (== (factorial 0) 1)
    assert (== (factorial 1) 1)
    assert (== (factorial 5) 120)
    assert (== (factorial 10) 3628800)
}

fn main() -> int {
    (println "Factorial Calculator")
    (println "Enter a number:")
    
    let input: string = (read_line)
    let n: int = (string_to_int input)  # Not in core, but needed
    
    let result: int = (factorial n)
    (println (+ "Result: " (int_to_string result)))
    
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

---

## What I Exclude From Core (Advanced Features)

### Advanced Types
- Structs - complex data structures
- Enums - enumerated types
- Unions - tagged unions
- Generics - generic types like `List<T>`
- Opaque types - FFI types

### Advanced Control Flow
- Match expressions - pattern matching
- Try/catch - error handling (use `Result<T, E>` if needed)

### Advanced Features
- FFI (`extern` functions) - calling C code
- Unsafe blocks - bypassing safety
- Macros - code generation
- Modules with complex dependencies

### Advanced Modules
- SDL - graphics/games
- HTTP - web servers
- Database - SQLite/MySQL
- Async - asynchronous I/O

**Rule:** If I have not listed it in Core, consider it advanced. Ask the user before using it.

---

## Learning Path for LLMs

### Step 1: Memorize My Core Syntax
- Function calls: `(f x y)`
- Variables: `let name: type = value`
- Control flow: `cond`, `if/else`, `while`, `for`

### Step 2: Memorize My Core Stdlib
- Math: `+`, `-`, `*`, `/`, `%`
- Strings: `+`, `==`, `str_length`
- Arrays: `array_new`, `array_get`, `array_set`
- I/O: `print`, `println`

### Step 3: Write Simple Programs
- Loops and counters
- String manipulation
- Array operations
- Basic algorithms

### Step 4: Add Shadow Tests
- Every function I compile must have a `shadow` block.
- Use `assert` to validate behavior.
- Test edge cases.

### Step 5: Only Then Explore My Advanced Features
- Learn structs if the user needs data structures.
- Learn modules if the user needs external functionality.
- Learn FFI if the user needs C interop.

---

## Success Criteria

**I consider that an LLM has mastered my core subset when it can:**

1. Write functions with shadow tests.
2. Use infix `a + b` or prefix `(+ a b)` notation for operators (both are valid).
3. Implement basic algorithms (factorial, fibonacci, etc.).
4. Manipulate strings and arrays.
5. Use `cond` for expressions, `if/else` for statements.
6. Never use non-canonical forms.
7. Never use advanced features without an explicit request.

---

## My Quick Reference Card

```nano
# Variables
let x: int = 5
let mut y: int = 0
set y 10
 
# Functions
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
}

# Control Flow
(cond
    ((< x 0) "negative")
    ((> x 0) "positive")
    (else "zero")
)

if (< x 0) {
    (println "negative")
} else {
    (println "non-negative")
}

while (< i 10) {
    set i (+ i 1)
}

# Math
(+ a b) (- a b) (* a b) (/ a b) (% a b)

# Comparisons
(== a b) (!= a b) (< a b) (> a b) (<= a b) (>= a b)

# Boolean
(and a b) (or a b) (not x)

# Strings
(+ "hello" " world")
(str_length s)
(str_equals s1 s2)

# Arrays
(array_new 10 0)
(array_get arr 0)
(array_set arr 0 42)
(array_length arr)

# I/O
(println "text")
(read_line)
```

---

## For LLM Training

**When you see a NanoLang task:**

1. **First:** Can I solve this with my core subset only?
2. **If yes:** Use ONLY my core features.
3. **If no:** Ask the user: "This requires [feature]. Should I use it?"

**Default assumption:** The user wants simple, core-only code. Complexity is opt-in, not default.

---

## My Rationale

**Why do I have a core subset?**

1. **Faster learning:** masters 50 primitives instead of my full scope.
2. **Higher reliability:** A smaller surface area means you make fewer mistakes.
3. **Better code:** Simple code is better code.
4. **Easier debugging:** There is less to go wrong.

**80/20 rule:** My core subset handles 80% of tasks with 20% of my language. Start there.


