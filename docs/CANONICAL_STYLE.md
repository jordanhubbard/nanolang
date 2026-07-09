# My Canonical Style - The One True Way

> **For LLMs:** I define exactly one correct way to write each of my constructs. Use these forms. Do not deviate.

## My Core Principle

**I have exactly one canonical way to write each operation.**

When LLMs see multiple equivalent forms, they guess wrong about half the time. I eliminate guessing.

---

## My Function Calls

### Canonical: Prefix Notation
```nano
(function arg1 arg2 arg3)
(println "Hello")
(str_concat "a" "b")
```

### Never Use
```
function(arg1, arg2)  # C-style - I DO NOT SUPPORT THIS
function arg1 arg2    # Haskell-style - I DO NOT SUPPORT THIS
```

**Rule:** All my function calls use prefix notation `(f x y)`. I make no exceptions.

> **Note:** My operators like `+`, `-`, `*`, and others support both prefix and infix notation. I explain this in the [Arithmetic](#arithmetic), [Boolean Logic](#boolean-logic), and [Comparisons](#comparisons) sections below.

---

## Parentheses: Function Calls vs Tuples vs Grouping

I use parentheses for three purposes. My parser disambiguates based on what follows the opening `(`:

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

### How I Parse `(arg (arg, arg) arg)`

I parse this as a **function call** because I find no commas at the top level:

```nano
(process a (x, y) b)
#   │     │   │   └─ third argument: b
#   │     │   └───── second argument: tuple (x, y)
#   │     └───────── first argument: a
#   └─────────────── function name: process
```

The nested `(x, y)` is a tuple because it contains a comma.

### My Type Annotations

My tuple types also use comma separation:

```nano
let pair: (int, int) = (1, 2)           # Tuple type and literal
let triple: (string, int, bool) = ("a", 1, true)

fn returns_pair() -> (int, string) {
    return (42, "answer")
}
```

### Common Mistakes

```nano
# Never: This calls fn with args a, b (not a tuple)
(fn a b)    # Function call with TWO arguments

# Canonical: To pass a single tuple argument:
(fn (a, b)) # Function call with ONE tuple argument

# Never: Forgetting commas creates a call, not a tuple
let t = (1 2 3)  # ERROR: I try to call "1" as a function

# Canonical: Use commas for tuples
let t = (1, 2, 3)  # Creates a tuple
```

---

## Imports and Qualified Calls

### Canonical: Module Alias + Qualified Call
```nano
import "modules/std/json/json.nano" as json

let value: string = (json.get_string obj "name")
```

### Canonical: Import Alias for a Specific Function
```nano
from "modules/std/json/json.nano" import parse as json_parse

let data: Json = (json_parse payload)
```

**Rule:** I prefer module aliases and qualified calls. This avoids global name collisions.

---

## Conditionals

### For Expressions: Use `cond`

Canonical:
```nano
let result: int = (cond
    ((< x 0) -1)
    ((> x 0) 1)
    (else 0)
)
```

Never:
```nano
let result: int = if (< x 0) { -1 } else { if (> x 0) { 1 } else { 0 } }
```

### For Statements: Use `if/else` (with `else if` chaining)

Canonical:
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
- **Expressions** (returning a value): Use `cond`
- **Statements** (side effects only): Use `if/else` (with `else if` chaining as needed)

---

## String Operations

### Canonical: String `+`
```nano
let greeting: string = (+ "Hello, " name)
let full_path: string = (+ (+ base "/") filename)
```

### Alternative: `str_concat`
```nano
(str_concat "Hello, " name)  # Equivalent to (+ "Hello, " name)
```

**Note:** I treat `(+ s1 s2)` and `s1 + s2` as syntactic shorthand for `(str_concat s1 s2)`. All three work identically. I prefer `+` for consistency with numeric operations.

### Canonical: String `==`
```nano
if (== name "Alice") { (println "Hello Alice!") }
let same: bool = (== str1 str2)
```

### Never: `str_equals`
```nano
(str_equals name "Alice")  # I still support this, but avoid it
```

**Rule:** Always use `(== string1 string2)` when comparing strings.

---

## Arithmetic

I support **both prefix and infix** notation for my arithmetic operators.

### Canonical: Prefix Notation
```nano
(+ a b)
(- a b)
(* a b)
(/ a b)
(% a b)
```

### Canonical: Infix Notation
```nano
a + b
a - b
a * b
a / b
a % b
```

### Precedence and Grouping

All my infix operators have **equal precedence**. I evaluate them **left-to-right**. I do not use PEMDAS. Use parentheses to control evaluation order:

```nano
# Left-to-right evaluation without parentheses
let x: int = 2 + 3 * 4    # (2 + 3) * 4 = 20, NOT 2 + 12

# Explicit grouping with parentheses
let y: int = 2 + (3 * 4)  # 2 + 12 = 14
let z: int = (2 + 3) * 4  # 5 * 4 = 20
```

My unary `-` works without parentheses: `-x`

**Rule:** Both prefix `(+ a b)` and infix `a + b` are valid. My prefix notation avoids precedence ambiguity. My infix notation is often more readable for simple expressions. Use parentheses to group infix operations when precedence matters.

---

## Boolean Logic

I support **both prefix and infix** notation for my boolean operators.

### Canonical: Prefix Notation
```nano
(and condition1 condition2)
(or condition1 condition2)
(not condition)
```

### Canonical: Infix Notation
```nano
condition1 and condition2
condition1 or condition2
not condition
```

### Never Use
```
condition1 && condition2  # C-style - I DO NOT SUPPORT THIS
condition1 || condition2  # C-style - I DO NOT SUPPORT THIS
!condition                # C-style - I DO NOT SUPPORT THIS
```

My unary `not` and `-` work without parentheses: `not flag`, `-x`

**Rule:** Both prefix `(and a b)` and infix `a and b` are valid. I do not have C-style `&&`, `||`, or `!`.

---

## Comparisons

I support **both prefix and infix** notation for my comparison operators.

### Canonical: Prefix Notation
```nano
(== a b)
(!= a b)
(< a b)
(> a b)
(<= a b)
(>= a b)
```

### Canonical: Infix Notation
```nano
a == b
a != b
a < b
a > b
a <= b
a >= b
```

**Rule:** Both prefix `(== a b)` and infix `a == b` are valid. The same equal-precedence, left-to-right evaluation applies as with my arithmetic operators.

---

## Variables

### Canonical
```nano
let name: type = value              # Immutable
let mut counter: int = 0            # Mutable
set counter (+ counter 1)           # Mutation
```

### Never Use
```
var name = value        # JavaScript-style - I DO NOT SUPPORT THIS
counter = counter + 1   # Assignment with = - I DO NOT SUPPORT THIS (use set)
counter += 1            # Compound assignment - I DO NOT SUPPORT THIS
```

**Rule:** 
- Immutable: `let name: type = value`
- Mutable: `let mut name: type = value`
- Update: `set name new_value`

---

## Loops

### Canonical: `while` loops
```nano
while (< i 10) {
    (println i)
    set i (+ i 1)
}
```

### Canonical: `for` loops
```nano
for (let i: int = 0) (< i 10) (set i (+ i 1)) {
    (println i)
}
```

### Never Use
```
forEach(item in list)   # I DO NOT SUPPORT THIS
for item in list        # Python-style - I DO NOT SUPPORT THIS
list.forEach(...)       # Method syntax - I DO NOT SUPPORT THIS
```

**Rule:** I only have `while` and `for`. I provide no other loop constructs.

---

## Arrays

### Canonical: Function calls only
```nano
let arr: array<int> = (array_new 10 0)    # Create
let val: int = (array_get arr 0)          # Read
(array_set arr 0 42)                      # Write
let len: int = (array_length arr)         # Length
```

### Never Use
```
arr[0]          # Subscript syntax - I DO NOT SUPPORT THIS
arr.get(0)      # Method syntax - I DO NOT SUPPORT THIS
arr.length      # Property syntax - I DO NOT SUPPORT THIS
```

**Rule:** All my array operations use explicit function calls.

---

## Function Definitions

### Canonical
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

### Never Use
```
def function_name(...):           # Python-style - I DO NOT SUPPORT THIS
function function_name(...) {...} # JavaScript-style - I DO NOT SUPPORT THIS
auto function_name(...) -> {...}  # C++20-style - I DO NOT SUPPORT THIS
```

**Rule:** Use only the `fn` keyword. My shadow tests are mandatory, except for `extern` functions.

---

## Type Annotations

### Canonical
```nano
let name: string = "value"
fn add(x: int, y: int) -> int { ... }
let arr: array<int> = (array_new 10 0)
let opt: Result<int, string> = Result.Ok(42)
```

### Never Use
```
let name = "value"              # I have limited support for type inference
String name = "value"           # Java-style - I DO NOT SUPPORT THIS
auto name = "value"             # C++-style - I DO NOT SUPPORT THIS
```

**Rule:** Include explicit type annotations. I value clarity over brevity.

---

## Module Imports

### Canonical
```nano
from "path/to/module.nano" import function1, function2, Type
```

### Never Use
```
import module                   # Python-style - I DO NOT SUPPORT THIS
const { f1, f2 } = require(...) # Node.js-style - I DO NOT SUPPORT THIS
use module::function;           # Rust-style - I DO NOT SUPPORT THIS
```

**Rule:** Use only the `from ... import ...` syntax.

---

## Comments

### Canonical
```nano
# Single-line comment
/* Multi-line comment
   spanning multiple lines */
```

### Never Use
```
// C++ style     # I DO NOT SUPPORT THIS
-- Haskell style # I DO NOT SUPPORT THIS
```

**Rule:** Use `#` for single-line comments and `/* */` for multi-line comments.

---

## Summary: LLM Quick Reference

**When you generate my code, always:**

1. **Function calls:** `(f x y)` - use prefix notation only
2. **Expressions:** `(cond ((test) result) (else default))`
3. **Strings:** `(+ "a" "b")` or `"a" + "b"` - I prefer these over `str_concat`
4. **Math:** `(+ a b)` or `a + b` - both prefix and infix are valid
5. **Logic:** `(and a b)` or `a and b` - both prefix and infix are valid
6. **Comparisons:** `(== a b)` or `a == b` - both prefix and infix are valid
7. **Variables:** `let name: type = value` and `set name value`
8. **Arrays:** `(array_get arr i)` - use function calls only
9. **Types:** Keep them explicit, do not infer
10. **Shadow tests:** These are mandatory for every function, unless it is `extern`

**Infix operator notes:**
- All my infix operators have equal precedence and evaluate left-to-right
- Use parentheses to group: `a * (b + c)`
- My unary `not` and `-` work without parentheses: `not flag`, `-x`

**Never use:**
- C-style function calls: `arr[i]`, `f(x, y)`
- C-style boolean operators: `&&`, `||`, `!`
- Property access: `obj.prop` (except for my structs)
- Method calls: `obj.method()`
- Type inference: `let x = 5` (I prefer `let x: int = 5`)

---

## Enforcement

**Compiler warnings (future):**
```bash
./bin/nanoc file.nano --strict-canonical
# Warning: You are using if/else for an expression. I prefer cond.
```

**Linter (future):**
```bash
./bin/nanolint file.nano
# Error: I detected a non-canonical form.
```

---

## Why This Matters

**For Humans:** My consistency makes code easier to read and maintain.

**For LLMs:** This eliminates guessing. Every operation has one correct form. My reliability improves.

**Principle:** When my forms are well-defined and documented, LLMs produce correct code.

