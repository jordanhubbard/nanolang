# Chapter 22: Canonical Style

**The One True Way™ to write NanoLang code.**

NanoLang has exactly ONE canonical way to write each construct. This eliminates ambiguity and improves LLM code generation.

## 22.1 Core Principle

**ONE syntax per operation. No alternatives, no sugar, no shortcuts.**

When LLMs see multiple equivalent forms, they guess wrong ~50% of the time. NanoLang solves this by having exactly one way to write everything.

## 22.2 Operator Notation: Prefix and Infix

NanoLang supports **both prefix and infix** notation for binary operators:

```nano
# Both are valid and equivalent:
(+ a b)         # Prefix notation
a + b           # Infix notation

(* x 2)         # Prefix
x * 2           # Infix

(== result 42)  # Prefix
result == 42    # Infix
```

**Infix operators:** `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `and`, `or`

**Important:** All infix operators have **equal precedence**, evaluated **left-to-right** (no PEMDAS). Use parentheses to control grouping:

```nano
a * (b + c)    # Parentheses required for addition-first
a + b * c      # Evaluates as (a + b) * c
```

**Unary operators** (`not`, `-`) work without parens: `not flag`, `-x`

**Rule:** Function calls always use prefix notation `(function_name arg1 arg2 ...)`

## 22.3 Expressions vs Statements

### Expressions: Use `cond`

```nano
# ✅ Expressions return values - use cond
let result: int = (cond
    ((< x 0) -1)
    ((> x 0) 1)
    (else 0)
)
```

### Statements: Use `if/else`

```nano
# ✅ Statements have side effects - use if/else
if (< x 0) {
    (println "negative")
} else {
    (println "non-negative")
}
```

**Never mix them up!**

## 22.4 String Concatenation

```nano
# ✅ ALWAYS use + operator
let greeting: string = (+ "Hello, " name)

# ❌ NEVER use str_concat (deprecated)
let bad: string = (str_concat "Hello, " name)
```

## 22.5 Array Access

```nano
# ✅ ALWAYS use array_get
let value: int = (array_get arr 0)

# ❌ NO subscript syntax (doesn't exist)
# let value: int = arr[0]  # This is not valid!
```

## 22.6 Type Annotations

```nano
# ✅ ALWAYS annotate types explicitly
let x: int = 42
let name: string = "Alice"
let items: array<int> = [1, 2, 3]

# ❌ NEVER omit types
# let x = 42  # Type inference is minimal!
```

## 22.7 Function Calls

```nano
# ✅ ALWAYS use prefix notation for function calls
(my_function arg1 arg2)

# ❌ NO parentheses-free calls
# my_function arg1 arg2  # Invalid!
```

**Note:** Infix notation is for **operators only**, not function calls.

## 22.8 Loops

```nano
# ✅ for-in-range
for i in (range 0 10) {
    (println i)
}

# ✅ while loops
while (< i 10) {
    set i (+ i 1)
}

# ❌ NO foreach, do-while, or other loop forms
```

## 22.9 Complete Example

```nano
# ✅ Canonical NanoLang
fn calculate_sum(numbers: array<int>) -> int {
    let len: int = (array_length numbers)
    let mut sum: int = 0
    
    for i in (range 0 len) {
        let value: int = (array_get numbers i)
        set sum (+ sum value)
    }
    
    return sum
}

shadow calculate_sum {
    let nums: array<int> = [1, 2, 3, 4, 5]
    assert (== (calculate_sum nums) 15)
}
```

## Summary

**Canonical Rules:**
- Operators support both prefix `(+ a b)` and infix `a + b` notation
- All infix operators have equal precedence (left-to-right, no PEMDAS)
- Function calls always use prefix notation: `(function_name arg1 arg2)`
- `cond` for expressions, `if/else` (with `else if` chaining) for statements
- `+` for string concatenation
- Explicit type annotations
- `array_get`, never subscripts

**See also:** `docs/CANONICAL_STYLE.md` for complete reference

---

**Previous:** [Chapter 21: Configuration](../part3_modules/21_configuration/index.html)  
**Next:** [Chapter 23: Higher-Level Patterns](23_patterns.md)
