# Chapter 22: Canonical Style

**The One True Way™ to write NanoLang code.**

NanoLang has exactly ONE canonical way to write each construct. This eliminates ambiguity and improves LLM code generation.

## 22.1 Core Principle

**ONE syntax per operation. No alternatives, no sugar, no shortcuts.**

When LLMs see multiple equivalent forms, they guess wrong ~50% of the time. NanoLang solves this by having exactly one way to write everything.

## 22.2 Prefix Notation ONLY

```nano
# ✅ ALWAYS DO THIS
(+ a b)
(* x 2)
(== result 42)

# ❌ NEVER DO THIS
a + b       # No infix operators!
x * 2       # Infix doesn't exist
result == 42  # No infix comparison
```

**Rule:** ALL operations use prefix notation `(operator arg1 arg2 ...)`

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
# ✅ ALWAYS use prefix notation
(my_function arg1 arg2)

# ❌ NO parentheses-free calls
# my_function arg1 arg2  # Invalid!
```

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
- ✅ Prefix notation ONLY
- ✅ `cond` for expressions, `if/else` for statements
- ✅ `+` for string concatenation
- ✅ Explicit type annotations
- ✅ `array_get`, never subscripts
- ✅ ONE way per operation

**See also:** `docs/CANONICAL_STYLE.md` for complete reference

---

**Previous:** [Chapter 21: Configuration](../part3_modules/21_configuration/index.html)  
**Next:** [Chapter 23: LLM Code Generation](23_llm_generation.html)
