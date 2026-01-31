# Totality Checker - Implementation Plan

## Goal

Verify all functions are total (always terminate, no runtime crashes possible).

## Problem Statement

Partial functions can cause runtime failures:
```nano
fn head(arr: array<int>) -> int {
    return (at arr 0)  // CRASHES if arr is empty!
}

fn divide(a: int, b: int) -> int {
    return (/ a b)  // CRASHES if b == 0!
}
```

A totality checker would prove these functions can't crash or force defensive coding.

## Proposed Solution

Add static analysis pass that checks:
1. **Pattern Match Exhaustiveness:** All union variants covered
2. **Array Bounds:** Provably in-bounds access
3. **Division by Zero:** Divisor provably non-zero
4. **Recursion Termination:** Structural recursion or explicit proof
5. **Infinite Loops:** While loops have termination proof

## Design Options

### Option A: Conservative (Recommended for v1)

**Approach:** Flag potentially partial functions, require explicit Result<T,E>

```nano
// Before (partial, rejected by checker)
fn head(arr: array<int>) -> int {
    return (at arr 0)  // ERROR: Unproven array access
}

// After (total, passes checker)
fn head(arr: array<int>) -> Result<int, string> {
    if (== (array_length arr) 0) {
        return Result.Error { error: "Empty array" }
    } else {
        return Result.Ok { value: (at arr 0) }
    }
}
```

**Pros:**
- Simple to implement
- Forces defensive programming
- Clear error handling

**Cons:**
- More verbose code
- Requires Result everywhere

### Option B: Dependent Types

**Approach:** Non-empty array type `array<T, n>` where n > 0

```nano
// Type-level proof
fn head(arr: array<int, n>) -> int where (> n 0) {
    return (at arr 0)  // Provably safe!
}
```

**Pros:**
- Elegant, compile-time proof
- No runtime checks needed

**Cons:**
- Major type system extension
- Difficult to implement (~100+ hours)
- Affects all array code

### Option C: Hybrid (Pragmatic)

**Approach:** Checker + opt-in dependent types

```nano
// Option 1: Use Result (conservative)
fn safe_head(arr: array<int>) -> Result<int, string> { ... }

// Option 2: Use assertion (checked)
@requires(not_empty(arr))
fn head(arr: array<int>) -> int {
    return (at arr 0)  // Checked by totality checker
}

// Option 3: Use dependent type (advanced)
fn head_proven(arr: array<int, n>) -> int where (> n 0) {
    return (at arr 0)  // Type-level proof
}
```

## Implementation Strategy

### Phase 1: Pattern Match Exhaustiveness (Core, 12 hours)

**Already have:** Warning for non-exhaustive matches
**Upgrade to:** Error (make mandatory)

```nano
union Status { Ok, Error, Pending }

fn handle(s: Status) -> int {
    match s {
        Ok(v) => return 1,
        Error(e) => return 0
        // MISSING: Pending
    }
}
// ERROR: Non-exhaustive match, function is partial
```

**Implementation:**
1. Enhance existing match exhaustiveness check
2. Make it an error instead of warning
3. Add quick-fix: generate missing cases

### Phase 2: Simple Termination Analysis (20 hours)

**Check:** Structural recursion on lists

```nano
fn sum(arr: array<int>) -> int {
    if (== (array_length arr) 0) {
        return 0
    } else {
        let head: int = (at arr 0)
        let tail: array<int> = (array_slice arr 1 (array_length arr))
        return (+ head (sum tail))  // OK: Recursive call on smaller input
    }
}
```

**Implementation:**
1. Track recursive calls
2. Verify argument size decreases
3. Flag non-decreasing recursion

### Phase 3: Range Analysis (25 hours)

**Check:** Array bounds are proven in-range

```nano
fn get_first_three(arr: array<int>) -> array<int> {
    if (< (array_length arr) 3) {
        return []
    }
    // Checker knows: array_length arr >= 3
    let a: int = (at arr 0)  // Provably safe
    let b: int = (at arr 1)  // Provably safe
    let c: int = (at arr 2)  // Provably safe
    return [a, b, c]
}
```

**Implementation:**
1. Track constraints from if statements
2. Propagate bounds through blocks
3. Verify array accesses against bounds
4. Handle loop invariants

### Phase 4: Division-by-Zero (8 hours)

**Check:** Divisor is provably non-zero

```nano
fn safe_ratio(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Result.Error { error: "Division by zero" }
    }
    // Checker knows: b != 0
    return Result.Ok { value: (/ a b) }  // Provably safe
}
```

**Implementation:**
1. Track zero/non-zero constraints
2. Verify at division points
3. Flag unproven divisions

## Type System Integration

### Totality Annotation

```nano
@total  // Promise to checker
fn safe_function(x: int) -> int {
    return (+ x 1)  // Obviously total
}

shadow safe_function {
    assert (== (safe_function 5) 6)
}
```

### Partiality Warning

```nano
@partial  // Acknowledge partiality
fn dangerous_function(x: int) -> int {
    if (== x 0) {
        # Infinite loop
        let mut i: int = 0
        while true {
            set i (+ i 1)
        }
    }
    return x
}
```

## Integration with Existing Features

### Checked Arithmetic

Use checked arithmetic to prove no overflow:
```nano
fn add_checked(a: int, b: int) -> Result<int, string> {
    return (checked_add a b)  // Total: always returns
}
```

### Unsafe Blocks

Totality checker skips unsafe blocks:
```nano
fn call_ffi() -> int {
    unsafe {
        return (some_c_function)  // Unchecked
    }
}
```

## Estimated Effort

### Minimal (Phase 1 only)
- **12 hours:** Pattern match exhaustiveness
- **Benefit:** Eliminates partial matches
- **Limitation:** Doesn't check array bounds, recursion

### Moderate (Phases 1-2)
- **32 hours:** + Structural recursion
- **Benefit:** Proves recursive functions terminate
- **Limitation:** No array bounds checking

### Full (All Phases)
- **65 hours:** Complete totality checker
- **Benefit:** Provably total functions
- **Limitation:** May flag some safe code (false positives)

## Compatibility

**Breaking Change:** No
**Opt-in:** Yes (via @total annotation or compiler flag)
**Migration:** Existing code unaffected unless --strict-totality

```bash
nanoc --strict-totality file.nano  # Enforce totality
nanoc file.nano                     # Warnings only
```

## References

- Idris: https://www.idris-lang.org/
- Agda: https://wiki.portal.chalmers.se/agda/
- Coq: https://coq.inria.fr/
- Liquid Haskell: https://ucsd-progsys.github.io/liquidhaskell/
- CompCert: http://compcert.inria.fr/

## Status

🟡 **PLANNED** - Comprehensive design complete

**Recommendation:** Start with Phase 1 (pattern exhaustiveness) as it's high-value, low-effort.

**Next Steps:**
1. Implement mandatory exhaustiveness checking
2. Add @total/@partial annotations
3. Implement structural recursion checking
4. Consider range analysis for arrays

Related: nanolang-ygm9

