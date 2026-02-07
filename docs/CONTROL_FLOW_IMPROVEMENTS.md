# Control Flow Improvements for NanoLang

**Date**: December 31, 2025  
**Status**: Design Proposal  
**Context**: Reducing nested if/else blocks while preserving NanoLang's design principles

---

## Problem Statement

NanoLang currently has deeply nested if/else blocks that reduce readability:

```nano
fn classify_number(n: int) -> int {
    if (< n 0) {
        return -1
    } else {
        if (== n 0) {
            return 0
        } else {
            if (< n 10) {
                return 1
            } else {
                if (< n 100) {
                    return 2
                } else {
                    return 3
                }
            }
        }
    }
}
```

This violates the "clarity over cleverness" principle and makes code harder to maintain.

---

## Current Features (Already Available)

### 1. Match Expressions (For Unions/Enums)

**Status**: ✅ Already implemented

```nano
union Result {
    Ok { value: int },
    Err { code: int }
}

fn handle_result(r: Result) -> int {
    match r {
        Ok(v) => { return v.value }
        Err(e) => { return (- 0 e.code) }
    }
}
```

**Pros**:
- ✅ Pattern matching on union variants
- ✅ Exhaustiveness checking
- ✅ Clean syntax

**Cons**:
- ❌ Only works with unions/enums (not integers, strings, or arbitrary values)
- ❌ Cannot match on ranges or predicates

---

### 2. Guard Clauses (Pattern, Not Syntax)

**Status**: ✅ Works today with early returns

```nano
fn classify_with_guards(n: int) -> int {
    if (< n 0) { return -1 }
    if (== n 0) { return 0 }
    if (< n 10) { return 1 }
    if (< n 100) { return 2 }
    return 3
}
```

**Pros**:
- ✅ No new syntax required
- ✅ Flat structure (no nesting)
- ✅ Easy to read top-to-bottom

**Cons**:
- ⚠️ Requires every branch to return
- ⚠️ Not applicable when you need side effects before returning

---

## Proposed Solutions

### Option A: `cond` Expression (Lisp-Inspired)

**Syntax** (prefix notation, fitting NanoLang style; note that operators also support infix):

```nano
fn classify_cond(n: int) -> int {
    return (cond
        ((< n 0) -1)
        ((== n 0) 0)
        ((< n 10) 1)
        ((< n 100) 2)
        (else 3))
}
```

**Implementation Details**:
- Each clause is `(predicate value)`
- Evaluates predicates top-to-bottom
- Returns the value of the first true predicate
- `else` is required as the final clause
- No need for explicit `return` in each branch

**Pros**:
- ✅ Clean notation (fits NanoLang's style; operators also support infix)
- ✅ Flat structure (no nesting)
- ✅ Familiar to Lisp/Scheme developers
- ✅ Can be used as an expression (returns a value)
- ✅ No complex type inference needed

**Cons**:
- ⚠️ 2× implementation cost (C parser + NanoLang parser)
- ⚠️ New keyword (`cond`, `else`)
- ⚠️ Each clause is limited to a single expression (no statement blocks)

**Enhanced Version with Blocks**:

```nano
fn classify_cond_blocks(n: int) -> int {
    return (cond
        ((< n 0) {
            (println "Negative")
            return -1
        })
        ((== n 0) {
            (println "Zero")
            return 0
        })
        (else {
            (println "Positive")
            return 1
        }))
}
```

---

### Option B: `when` Macro (Clojure-Inspired)

**Syntax**:

```nano
fn classify_when(n: int) -> int {
    (when (< n 0) (return -1))
    (when (== n 0) (return 0))
    (when (< n 10) (return 1))
    (when (< n 100) (return 2))
    return 3
}
```

**Implementation Details**:
- `(when predicate body)` = sugar for `if predicate { body } else {}`
- Only executes body if predicate is true
- No `else` clause (implicit "do nothing")

**Pros**:
- ✅ Simpler than `cond` (just syntactic sugar for `if`)
- ✅ Prefix notation (operators also support infix)
- ✅ Can be used with guard clauses pattern
- ✅ Smaller implementation (transform to existing `if`)

**Cons**:
- ⚠️ Not as powerful as `cond` (no exhaustiveness checking)
- ⚠️ Requires early returns for multi-branch logic
- ⚠️ Still 2× implementation cost

---

### Option C: Enhanced `match` for Scalar Types

**Syntax** (extending existing `match`):

```nano
fn classify_match(n: int) -> int {
    match n {
        (< n 0) => { return -1 }
        (== n 0) => { return 0 }
        (< n 10) => { return 1 }
        (< n 100) => { return 2 }
        _ => { return 3 }
    }
}
```

**Implementation Details**:
- Extend `match` to support predicate expressions
- Evaluate predicates top-to-bottom
- `_` wildcard is the default case

**Pros**:
- ✅ Reuses existing `match` syntax
- ✅ Pattern developers already know
- ✅ Exhaustiveness checking with `_`

**Cons**:
- ⚠️ 2× implementation cost (extend existing match logic)
- ⚠️ Mixing union patterns and predicate patterns may be confusing
- ⚠️ Predicate `n` is implicitly bound (not explicit in pattern)

---

### Option D: `switch` Statement (C-Inspired)

**Syntax**:

```nano
fn classify_switch(n: int) -> int {
    (switch n
        ((< n 0) -1)
        ((== n 0) 0)
        ((< n 10) 1)
        ((< n 100) 2)
        (default 3))
}
```

**Implementation Details**:
- First argument is the value to switch on
- Each case is `(predicate result)`
- `default` is required
- Similar to `cond` but with explicit value binding

**Pros**:
- ✅ Prefix notation (operators also support infix)
- ✅ Explicit value binding (clearer than `cond`)
- ✅ Familiar to C/Java developers

**Cons**:
- ⚠️ Very similar to `cond` (may be redundant)
- ⚠️ 2× implementation cost
- ⚠️ New keyword (`switch`, `default`)

---

## Recommendation: Option A (`cond` Expression)

### Why `cond`?

1. **Best Fit for NanoLang's Style**
   - Clean S-expression syntax for multi-branch conditionals
   - Consistent with NanoLang's philosophy (function calls use prefix; operators support both prefix and infix)

2. **Proven Design**
   - Used in Lisp/Scheme for decades
   - Well-understood semantics
   - Minimal complexity

3. **Flexible Use Cases**
   - Works as expression (returns value)
   - Works as statement (with side effects)
   - Can be nested

4. **Clear Semantics**
   - Top-to-bottom evaluation (short-circuit)
   - Mandatory `else` (exhaustiveness)
   - No hidden control flow

### Implementation Plan

**Phase 1: Parser (C)**
1. Add `TOKEN_COND` and `TOKEN_ELSE` to lexer
2. Add `AST_COND` node type with:
   - Array of `(predicate, body)` pairs
   - Mandatory `else` clause
3. Parse syntax: `(cond (pred1 val1) (pred2 val2) ... (else valN))`

**Phase 2: Type Checker (C)**
1. Check all predicates are boolean expressions
2. Check all branch values have the same type
3. Verify `else` clause is present
4. Return the common type of all branches

**Phase 3: Transpiler (C)**
Transform to nested if/else:
```c
// (cond ((< x 0) -1) ((== x 0) 0) (else 1))
// Transpiles to:
if (x < 0) {
    result = -1;
} else if (x == 0) {
    result = 0;
} else {
    result = 1;
}
```

**Phase 4: Self-Hosted Implementation (NanoLang)**
1. Mirror parser logic in `src_nano/parser.nano`
2. Mirror type checker in `src_nano/typecheck.nano`
3. Mirror transpiler in `src_nano/transpiler.nano`
4. Add shadow tests

**Estimated Effort**:
- C implementation: 3-4 days
- NanoLang implementation: 2-3 days
- Testing & docs: 1-2 days
- **Total: ~2 weeks**

---

## Examples of `cond` in Use

### Example 1: Number Classification

**Before** (Nested if/else):
```nano
fn classify(n: int) -> string {
    if (< n 0) {
        return "negative"
    } else {
        if (== n 0) {
            return "zero"
        } else {
            if (< n 10) {
                return "small"
            } else {
                return "large"
            }
        }
    }
}
```

**After** (With `cond`):
```nano
fn classify(n: int) -> string {
    return (cond
        ((< n 0) "negative")
        ((== n 0) "zero")
        ((< n 10) "small")
        (else "large"))
}
```

### Example 2: Grade Calculator

**Before**:
```nano
fn letter_grade(score: int) -> string {
    if (>= score 90) {
        return "A"
    } else {
        if (>= score 80) {
            return "B"
        } else {
            if (>= score 70) {
                return "C"
            } else {
                if (>= score 60) {
                    return "D"
                } else {
                    return "F"
                }
            }
        }
    }
}
```

**After**:
```nano
fn letter_grade(score: int) -> string {
    return (cond
        ((>= score 90) "A")
        ((>= score 80) "B")
        ((>= score 70) "C")
        ((>= score 60) "D")
        (else "F"))
}
```

### Example 3: With Side Effects

```nano
fn process(x: int) -> int {
    return (cond
        ((< x 0) {
            (println "Negative value")
            return (abs x)
        })
        ((== x 0) {
            (println "Zero value")
            return 1
        })
        (else {
            (println "Positive value")
            return x
        }))
}
```

---

## Alternative: Do Nothing (Use Existing Patterns)

### Pattern 1: Guard Clauses (Recommended Today)

```nano
fn classify(n: int) -> string {
    if (< n 0) { return "negative" }
    if (== n 0) { return "zero" }
    if (< n 10) { return "small" }
    return "large"
}
```

**Pros**:
- ✅ Works today (no implementation needed)
- ✅ Flat structure
- ✅ Clear control flow

**Best for**:
- Functions that return early
- Simple decision trees
- When all branches have the same "shape"

### Pattern 2: Helper Functions

```nano
fn is_negative(n: int) -> bool { return (< n 0) }
fn is_zero(n: int) -> bool { return (== n 0) }
fn is_small(n: int) -> bool { return (< n 10) }

fn classify(n: int) -> string {
    if (is_negative n) { return "negative" }
    if (is_zero n) { return "zero" }
    if (is_small n) { return "small" }
    return "large"
}
```

**Pros**:
- ✅ Named predicates (self-documenting)
- ✅ Reusable logic
- ✅ Easy to test independently

---

## Design Principles Compliance

| Principle | `cond` | Guard Clauses | Helper Functions |
|-----------|--------|---------------|------------------|
| **Prefix/infix notation** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Explicit types** | ✅ Yes | ✅ Yes | ✅ Yes |
| **No implicit behavior** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Simple grammar** | ⚠️ New syntax | ✅ Existing | ✅ Existing |
| **2× implementation** | ⚠️ Required | ✅ Not needed | ✅ Not needed |
| **Clarity** | ✅ High | ✅ High | ✅ Very High |

---

## Conclusion

**Short-term recommendation**: Use **guard clauses** pattern (works today, no implementation cost)

**Long-term recommendation**: Implement **`cond` expression** (best balance of power and simplicity)

**Not recommended**:
- ❌ `when` macro - too similar to existing `if`
- ❌ Enhanced `match` - mixing patterns is confusing
- ❌ `switch` - redundant with `cond`

---

## Next Steps

1. **Gather feedback** from NanoLang users on this proposal
2. **Prototype `cond`** in C parser to validate feasibility
3. **Write design spec** with formal grammar and type rules
4. **Implement Phase 1-4** as outlined above
5. **Update MEMORY.md** with `cond` usage patterns
6. **Refactor existing code** to use `cond` where appropriate

---

## References

- **Lisp `cond`**: http://www.lispworks.com/documentation/HyperSpec/Body/m_cond.htm
- **Scheme `cond`**: https://www.scheme.com/tspl4/control.html
- **Clojure `cond`**: https://clojuredocs.org/clojure.core/cond
- **NanoLang design principles**: See `CONTRIBUTING.md` and `MEMORY.md`

