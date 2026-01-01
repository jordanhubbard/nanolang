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
(+ 1 2)
(str_concat "a" "b")
```

### ❌ Never Use
```
function(arg1, arg2)  # C-style - DOES NOT EXIST
function arg1 arg2    # Haskell-style - DOES NOT EXIST
```

**Rule:** ALL function calls use prefix notation `(f x y)`. No exceptions.

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

### For Statements: Use `if/else`

✅ **Canonical:**
```nano
if (< x 0) {
    (println "negative")
} else {
    (println "non-negative")
}
```

**Rule:** 
- **Expressions** (returns a value): Use `cond`
- **Statements** (side effects only): Use `if/else`

---

## String Operations

### ✅ Canonical: String `+`
```nano
let greeting: string = (+ "Hello, " name)
let full_path: string = (+ (+ base "/") filename)
```

### ❌ Deprecated: `str_concat`
```nano
(str_concat "Hello, " name)  # OLD WAY - still works but avoid
```

**Rule:** Always use `(+ string1 string2)` for concatenation.

---

## Arithmetic

### ✅ Canonical: Prefix Operators
```nano
(+ a b)
(- a b)
(* a b)
(/ a b)
(% a b)
```

### ❌ Never Use
```
a + b   # Infix - DOES NOT EXIST
a - b   # Infix - DOES NOT EXIST
```

**Rule:** All arithmetic uses prefix notation. No operator precedence ambiguity.

---

## Boolean Logic

### ✅ Canonical: Prefix Logic
```nano
(and condition1 condition2)
(or condition1 condition2)
(not condition)
```

### ❌ Never Use
```
condition1 && condition2  # C-style - DOES NOT EXIST
condition1 || condition2  # C-style - DOES NOT EXIST
!condition                # C-style - DOES NOT EXIST
```

**Rule:** All boolean logic uses prefix notation.

---

## Comparisons

### ✅ Canonical: Prefix Comparisons
```nano
(== a b)
(!= a b)
(< a b)
(> a b)
(<= a b)
(>= a b)
```

### ❌ Never Use
```
a == b  # Infix - DOES NOT EXIST
a < b   # Infix - DOES NOT EXIST
```

**Rule:** All comparisons use prefix notation.

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
counter = counter + 1   # Assignment syntax - DOES NOT EXIST
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
3. **Strings:** `(+ "a" "b")` - not `str_concat`
4. **Math:** `(+ a b)` - prefix notation only
5. **Logic:** `(and a b)` - prefix notation only
6. **Variables:** `let name: type = value` and `set name value`
7. **Arrays:** `(array_get arr i)` - function calls only
8. **Types:** Always explicit, never infer
9. **Shadow tests:** MANDATORY for every function (except `extern`)

**Never use:**
- C-style syntax: `a + b`, `arr[i]`, `f(x, y)`
- Property access: `obj.prop` (except structs)
- Method calls: `obj.method()`
- Type inference: `let x = 5` (prefer `let x: int = 5`)

---

## Enforcement

**Compiler warnings (future):**
```bash
./bin/nanoc file.nano --strict-canonical
# Warning: Using deprecated str_concat, prefer (+ string1 string2)
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

**Principle:** When there's only one way to do it, LLMs can't get it wrong.

