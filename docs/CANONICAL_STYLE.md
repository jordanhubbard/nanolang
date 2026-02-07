# Canonical NanoLang Style - The One True Way™

> **For LLMs:** This document defines the ONE correct way to write each construct in NanoLang. Always use these forms. Do not deviate.

## Core Principle

**There is exactly ONE canonical way to write each operation.**

When LLMs see multiple equivalent forms, they guess wrong ~50% of the time. This document eliminates guessing.

---

## Function Calls

### ✅ Canonical: Prefix Notation
```nano
(function arg1 arg2 arg3)
(println "Hello")
(str_concat "a" "b")
```

### ❌ Never Use
```
function(arg1, arg2)  # C-style - DOES NOT EXIST
function arg1 arg2    # Haskell-style - DOES NOT EXIST
```

**Rule:** ALL function calls use prefix notation `(f x y)`. No exceptions.

> **Note:** Operators like `+`, `-`, `*`, etc. support both prefix and infix notation. See the [Arithmetic](#arithmetic), [Boolean Logic](#boolean-logic), and [Comparisons](#comparisons) sections below.

---

## Parentheses: Function Calls vs Tuples vs Grouping

NanoLang uses parentheses for three purposes. The parser disambiguates based on what follows the opening `(`:

### The Rule: Commas Make Tuples, Spaces Make Calls

| Pattern | Meaning |
|---------|---------|
| `(a, b, c)` | **Tuple literal** - comma-separated |
| `(fn a b c)` | **Function call** - space-separated |
| `(expr)` | **Grouping** - single expression, no comma |
| `()` | **Empty tuple** |
| `(fn)` | **Zero-arg call** - identifier alone = call |

### Examples

```nano
# Function calls - arguments separated by SPACES
(add 1 2)              # Call add with args 1 and 2
(println "Hello")      # Call println with one arg
(process x y z)        # Call process with three args

# Tuples - elements separated by COMMAS
(1, 2)                 # Tuple with two elements
("a", "b", "c")        # Tuple with three strings
(x, (y, z))            # Nested tuple

# Grouping - single expression
(+ 1 2)                # Groups the prefix op (returns 3)
(3 + 4)                # Groups the infix op (returns 7)

# Mixed: function taking a tuple argument
(fn_expects_tuple (1, 2))   # Call fn_expects_tuple with tuple arg
```

### Parsing `(arg (arg, arg) arg)`

This parses as a **function call** because there are no commas at the top level:

```nano
(process a (x, y) b)
#   │     │   │   └─ third argument: b
#   │     │   └───── second argument: tuple (x, y)
#   │     └───────── first argument: a
#   └─────────────── function name: process
```

The nested `(x, y)` is a tuple because it contains a comma.

### Type Annotations

Tuple types also use comma separation:

```nano
let pair: (int, int) = (1, 2)           # Tuple type and literal
let triple: (string, int, bool) = ("a", 1, true)

fn returns_pair() -> (int, string) {
    return (42, "answer")
}
```

### Common Mistakes

```nano
# ❌ WRONG: This calls fn with args a, b (not a tuple)
(fn a b)    # Function call with TWO arguments

# ✅ RIGHT: To pass a single tuple argument:
(fn (a, b)) # Function call with ONE tuple argument

# ❌ WRONG: Forgetting commas creates a call, not a tuple
let t = (1 2 3)  # ERROR: tries to call "1" as function!

# ✅ RIGHT: Use commas for tuples
let t = (1, 2, 3)  # Creates a tuple
```

---

## Imports and Qualified Calls

### ✅ Canonical: Module Alias + Qualified Call
```nano
import "modules/std/json/json.nano" as json

let value: string = (json.get_string obj "name")
```

### ✅ Canonical: Import Alias for a Specific Function
```nano
from "modules/std/json/json.nano" import parse as json_parse

let data: Json = (json_parse payload)
```

**Rule:** Prefer module aliases and qualified calls to avoid global name collisions.

---

## Conditionals

### For Expressions: Use `cond`

✅ **Canonical:**
```nano
let result: int = (cond
    ((< x 0) -1)
    ((> x 0) 1)
    (else 0)
)
```

❌ **Avoid:**
```nano
let result: int = if (< x 0) { -1 } else { if (> x 0) { 1 } else { 0 } }
```

### For Statements: Use `if/else` (with `else if` chaining)

✅ **Canonical:**
```nano
if (< x 0) {
    (println "negative")
} else if (== x 0) {
    (println "zero")
} else {
    (println "positive")
}
```

**Rule:**
- **Expressions** (returns a value): Use `cond`
- **Statements** (side effects only): Use `if/else` (with `else if` chaining as needed)

---

## String Operations

### ✅ Canonical: String `+`
```nano
let greeting: string = (+ "Hello, " name)
let full_path: string = (+ (+ base "/") filename)
```

### Alternative: `str_concat`
```nano
(str_concat "Hello, " name)  # Equivalent to (+ "Hello, " name)
```

**Note:** `(+ s1 s2)` and `s1 + s2` are syntactic shorthand for `(str_concat s1 s2)`. All three work identically; prefer `+` for consistency with numeric operations.

### ✅ Canonical: String `==`
```nano
if (== name "Alice") { (println "Hello Alice!") }
let same: bool = (== str1 str2)
```

### ❌ Deprecated: `str_equals`
```nano
(str_equals name "Alice")  # OLD WAY - still works but avoid
```

**Rule:** Always use `(== string1 string2)` for comparison.

---

## Arithmetic

NanoLang supports **both prefix and infix** notation for arithmetic operators.

### ✅ Prefix Notation
```nano
(+ a b)
(- a b)
(* a b)
(/ a b)
(% a b)
```

### ✅ Infix Notation
```nano
a + b
a - b
a * b
a / b
a % b
```

### Precedence and Grouping

All infix operators have **equal precedence** and are evaluated **left-to-right** (no PEMDAS). Use parentheses to control evaluation order:

```nano
# Without parens: evaluated left-to-right
let x: int = 2 + 3 * 4    # (2 + 3) * 4 = 20, NOT 2 + 12

# With parens: explicit grouping
let y: int = 2 + (3 * 4)  # 2 + 12 = 14
let z: int = (2 + 3) * 4  # 5 * 4 = 20
```

Unary `-` works without parentheses: `-x`

**Rule:** Both prefix `(+ a b)` and infix `a + b` are valid. Prefix notation avoids precedence ambiguity; infix notation is more readable for simple expressions. Use parentheses to group infix operations when precedence matters.

---

## Boolean Logic

NanoLang supports **both prefix and infix** notation for boolean operators.

### ✅ Prefix Notation
```nano
(and condition1 condition2)
(or condition1 condition2)
(not condition)
```

### ✅ Infix Notation
```nano
condition1 and condition2
condition1 or condition2
not condition
```

### ❌ Never Use
```
condition1 && condition2  # C-style - DOES NOT EXIST
condition1 || condition2  # C-style - DOES NOT EXIST
!condition                # C-style - DOES NOT EXIST
```

Unary `not` works without parentheses: `not flag`

**Rule:** Both prefix `(and a b)` and infix `a and b` are valid. C-style `&&`, `||`, `!` do not exist.

---

## Comparisons

NanoLang supports **both prefix and infix** notation for comparison operators.

### ✅ Prefix Notation
```nano
(== a b)
(!= a b)
(< a b)
(> a b)
(<= a b)
(>= a b)
```

### ✅ Infix Notation
```nano
a == b
a != b
a < b
a > b
a <= b
a >= b
```

**Rule:** Both prefix `(== a b)` and infix `a == b` are valid. Same equal-precedence, left-to-right evaluation applies as with arithmetic operators.

---

## Variables

### ✅ Canonical
```nano
let name: type = value              # Immutable
let mut counter: int = 0            # Mutable
set counter (+ counter 1)           # Mutation
```

### ❌ Never Use
```
var name = value        # JavaScript-style - DOES NOT EXIST
counter = counter + 1   # Assignment with = - DOES NOT EXIST (use set)
counter += 1            # Compound assignment - DOES NOT EXIST
```

**Rule:** 
- Immutable: `let name: type = value`
- Mutable: `let mut name: type = value`
- Update: `set name new_value`

---

## Loops

### ✅ Canonical: `while` loops
```nano
while (< i 10) {
    (println i)
    set i (+ i 1)
}
```

### ✅ Canonical: `for` loops
```nano
for (let i: int = 0) (< i 10) (set i (+ i 1)) {
    (println i)
}
```

### ❌ Never Use
```
forEach(item in list)   # DOES NOT EXIST
for item in list        # Python-style - DOES NOT EXIST
list.forEach(...)       # Method syntax - DOES NOT EXIST
```

**Rule:** Only `while` and `for` exist. No other loop constructs.

---

## Arrays

### ✅ Canonical: Function calls only
```nano
let arr: array<int> = (array_new 10 0)    # Create
let val: int = (array_get arr 0)          # Read
(array_set arr 0 42)                      # Write
let len: int = (array_length arr)         # Length
```

### ❌ Never Use
```
arr[0]          # Subscript syntax - DOES NOT EXIST
arr.get(0)      # Method syntax - DOES NOT EXIST
arr.length      # Property syntax - DOES NOT EXIST
```

**Rule:** All array operations use explicit function calls.

---

## Function Definitions

### ✅ Canonical
```nano
fn function_name(arg1: type1, arg2: type2) -> return_type {
    # body
    return value
}

shadow function_name {
    # tests
    assert (condition)
}
```

### ❌ Never Use
```
def function_name(...):           # Python-style - DOES NOT EXIST
function function_name(...) {...} # JavaScript-style - DOES NOT EXIST
auto function_name(...) -> {...}  # C++20-style - DOES NOT EXIST
```

**Rule:** Only `fn` keyword. Shadow tests are MANDATORY (except for `extern`).

---

## Type Annotations

### ✅ Canonical
```nano
let name: string = "value"
fn add(x: int, y: int) -> int { ... }
let arr: array<int> = (array_new 10 0)
let opt: Result<int, string> = Result.Ok(42)
```

### ❌ Never Use
```
let name = "value"              # Type inference - limited support
String name = "value"           # Java-style - DOES NOT EXIST
auto name = "value"             # C++-style - DOES NOT EXIST
```

**Rule:** Always include explicit type annotations. Clarity over brevity.

---

## Module Imports

### ✅ Canonical
```nano
from "path/to/module.nano" import function1, function2, Type
```

### ❌ Never Use
```
import module                   # Python-style - DOES NOT EXIST
const { f1, f2 } = require(...) # Node.js-style - DOES NOT EXIST
use module::function;           # Rust-style - DOES NOT EXIST
```

**Rule:** Only `from ... import ...` syntax exists.

---

## Comments

### ✅ Canonical
```nano
# Single-line comment
/* Multi-line comment
   spanning multiple lines */
```

### ❌ Never Use
```
// C++ style     # DOES NOT EXIST
-- Haskell style # DOES NOT EXIST
```

**Rule:** `#` for single-line, `/* */` for multi-line.

---

## Summary: LLM Quick Reference

**When generating NanoLang code, ALWAYS:**

1. **Function calls:** `(f x y)` - prefix notation only
2. **Expressions:** `(cond ((test) result) (else default))`
3. **Strings:** `(+ "a" "b")` or `"a" + "b"` - preferred over `str_concat`
4. **Math:** `(+ a b)` or `a + b` - both prefix and infix are valid
5. **Logic:** `(and a b)` or `a and b` - both prefix and infix are valid
6. **Comparisons:** `(== a b)` or `a == b` - both prefix and infix are valid
7. **Variables:** `let name: type = value` and `set name value`
8. **Arrays:** `(array_get arr i)` - function calls only
9. **Types:** Always explicit, never infer
10. **Shadow tests:** MANDATORY for every function (except `extern`)

**Infix operator notes:**
- All infix operators have equal precedence, evaluated left-to-right
- Use parentheses to group: `a * (b + c)`
- Unary `not` and `-` work without parens: `not flag`, `-x`

**Never use:**
- C-style function calls: `arr[i]`, `f(x, y)`
- C-style boolean operators: `&&`, `||`, `!`
- Property access: `obj.prop` (except structs)
- Method calls: `obj.method()`
- Type inference: `let x = 5` (prefer `let x: int = 5`)

---

## Enforcement

**Compiler warnings (future):**
```bash
./bin/nanoc file.nano --strict-canonical
# Warning: Using if/else for expression, prefer cond
```

**Linter (future):**
```bash
./bin/nanolint file.nano
# Error: Non-canonical form detected
```

---

## Why This Matters

**For Humans:** Consistency makes code easier to read and maintain.

**For LLMs:** Eliminates guessing. Every operation has ONE correct form. Model reliability improves dramatically.

**Principle:** When forms are well-defined and documented, LLMs produce correct code reliably.

