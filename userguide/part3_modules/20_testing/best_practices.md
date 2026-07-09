# 20.3 Testing Best Practices

**Effective testing strategies for NanoLang programs.**

NanoLang's testing system is intentionally constrained: shadow tests are mandatory, they run at compile time, and there is no way to skip them. This removes a whole class of decisions ("should I test this?") but leaves important choices about *how* to test well. This section covers those choices.

## Shadow Tests vs Property Tests

### Use Shadow Tests For...

**Specific, known correct values.** Shadow tests excel at pinning exact behavior:

```nano
fn celsius_to_fahrenheit(c: float) -> float {
    return (+ (* c 1.8) 32.0)
}

shadow celsius_to_fahrenheit {
    assert (== (celsius_to_fahrenheit 0.0) 32.0)    # water freezes
    assert (== (celsius_to_fahrenheit 100.0) 212.0) # water boils
    assert (== (celsius_to_fahrenheit -40.0) -40.0) # scales cross
}
```

**Documenting behavior at boundaries.** Boundaries are where bugs live. Shadow tests make them explicit:

```nano
fn clamp(v: int, lo: int, hi: int) -> int {
    if (< v lo) { return lo }
    if (> v hi) { return hi }
    return v
}

shadow clamp {
    assert (== (clamp 5  0 10) 5)    # in range: unchanged
    assert (== (clamp -1 0 10) 0)    # just below min: returns min
    assert (== (clamp 11 0 10) 10)   # just above max: returns max
    assert (== (clamp 0  0 10) 0)    # at min: unchanged
    assert (== (clamp 10 0 10) 10)   # at max: unchanged
}
```

**Testing side-effect-free void functions.** When a void function does something observable (like print to stdout), call it in the shadow block to at least verify it doesn't crash:

```nano
fn print_greeting(name: string) -> void {
    (println (+ "Hello, " name))
}

shadow print_greeting {
    (print_greeting "World")
    (print_greeting "")       # edge case: empty string
}
```

**Regression tests.** When you fix a bug, add a shadow test for the exact input that exposed it. That way the bug can never come back silently.

### Use Property Tests For...

**Algebraic or mathematical laws.** These are properties that hold for *all* inputs, not just the ones you happen to think of:

```nano
from "modules/proptest/proptest.nano" import forall_int_pair, int_pair, int_range,
                                             prop_pass, prop_fail, report_passed,
                                             PropertyReport

fn prop_max_commutative(a: int, b: int) -> string {
    let max_ab: int = (cond ((> a b) a) (else b))
    let max_ba: int = (cond ((> b a) b) (else a))
    if (== max_ab max_ba) {
        return (prop_pass)
    } else {
        return (prop_fail "max not commutative")
    }
}

shadow prop_max_commutative {
    let gen: IntPairGenerator = (int_pair (int_range -1000 1000) (int_range -1000 1000))
    let r: PropertyReport = (forall_int_pair "max_commutative" gen prop_max_commutative)
    assert (report_passed r)
}
```

**Invariants that should always hold.** A parser should always return a result of the same length as the input. A serialiser/deserialiser should round-trip.

**Finding edge cases you didn't think of.** If you're not sure what all the edge cases are, let proptest explore the input space:

```nano
fn prop_string_length_nonneg(x: int) -> string {
    let s: string = (int_to_string x)
    if (>= (str_length s) 1) {
        return (prop_pass)
    } else {
        return (prop_fail "empty int_to_string result")
    }
}

shadow prop_string_length_nonneg {
    let report: PropertyReport = (forall_int "int_to_string_nonempty"
                                              (int_range -100000 100000)
                                              prop_string_length_nonneg)
    assert (report_passed report)
}
```

**Both approaches are complementary.** A typical well-tested function has both: shadow tests for specific values and property tests for invariants.

## Test Organization

### One Shadow Block Per Function, Always

Every function needs exactly one shadow block immediately following its definition. Do not put shadow tests far from the function — the colocation is the point:

```nano
# Good: test immediately follows function
fn is_palindrome(s: string) -> bool {
    let len: int = (str_length s)
    if (<= len 1) { return true }
    let mut i: int = 0
    let mut ok: bool = true
    while (and ok (< i (/ len 2))) {
        let left: int = (char_at s i)
        let right: int = (char_at s (- (- len i) 1))
        if (not (== left right)) { set ok false } else { (print "") }
        set i (+ i 1)
    }
    return ok
}

shadow is_palindrome {
    assert (is_palindrome "racecar")
    assert (is_palindrome "a")
    assert (is_palindrome "")
    assert (not (is_palindrome "hello"))
    assert (is_palindrome "abba")
}
```

### Property Tests Belong in Shadow Blocks Too

Run `forall_*` inside a shadow block for the property function:

```nano
fn prop_palindrome_reverse_invariant(x: int) -> string {
    # A reversed palindrome is still a palindrome
    # (This is a contrived property but demonstrates the pattern)
    return (prop_pass)
}

shadow prop_palindrome_reverse_invariant {
    let r: PropertyReport = (forall_int "palindrome_reverse"
                                         (int_range 0 100)
                                         prop_palindrome_reverse_invariant)
    assert (report_passed r)
}
```

### Group Related Functions Together

NanoLang files have no class or namespace scoping, so use comments to group related functions:

```nano
# ============================================================
# String utility functions
# ============================================================

fn str_starts_with(s: string, prefix: string) -> bool {
    let plen: int = (str_length prefix)
    if (> plen (str_length s)) { return false }
    return (== (str_substring s 0 plen) prefix)
}

shadow str_starts_with {
    assert (str_starts_with "hello world" "hello")
    assert (not (str_starts_with "hello" "world"))
    assert (str_starts_with "x" "")
    assert (str_starts_with "" "")
}

fn str_ends_with(s: string, suffix: string) -> bool {
    let slen: int = (str_length s)
    let suflen: int = (str_length suffix)
    if (> suflen slen) { return false }
    return (== (str_substring s (- slen suflen) suflen) suffix)
}

shadow str_ends_with {
    assert (str_ends_with "hello world" "world")
    assert (not (str_ends_with "hello" "world"))
    assert (str_ends_with "x" "")
    assert (str_ends_with "" "")
}
```

## Naming Conventions

### Shadow Blocks

Shadow blocks must be named after the function they test — this is enforced by the compiler. There is no choice here.

### Property Functions

Name property functions with a `prop_` prefix followed by the invariant being described:

```nano
fn prop_abs_nonneg(x: int) -> string { ... }        # abs returns non-negative
fn prop_length_preserving(x: int) -> string { ... } # operation preserves length
fn prop_commutative_add(a: int, b: int) -> string { ... }  # addition commutes
fn prop_sort_idempotent(arr: array<int>) -> string { ... } # sorting twice = sorting once
```

This naming convention makes it immediately clear what invariant each property checks.

## What to Test

### Always Test

- **Every code path** — every `if`/`else` branch should be exercised by at least one assertion
- **Boundary values** — 0, -1, max value, empty string, empty array
- **The happy path** — at least one "normal" correct input
- **Error/edge inputs** — what happens with zero, negative, or empty inputs

### Prioritise Testing

- **Pure functions** — functions with no side effects are the easiest to test and the most valuable
- **Functions with complex conditional logic** — more branches = more opportunities for bugs
- **Functions that encode domain rules** — business logic bugs are expensive

### Accept Shallow Tests For

- **Simple pass-through functions** — if a function just calls another function with the same args, a shallow test is fine
- **Void functions with only side effects** — calling the function to verify it doesn't crash is sufficient
- **`main` functions** — typically `shadow main { assert true }` is acceptable

## Test Isolation

### Shadow Tests Run in Declaration Order

NanoLang shadow tests do not run in a sandbox — they share the program's address space and any mutable global state. If your function uses mutable globals, your shadow test must set them to a known state before each assertion:

```nano
let mut counter: int = 0

fn increment() -> int {
    set counter (+ counter 1)
    return counter
}

shadow increment {
    set counter 0          # reset state before testing
    assert (== (increment) 1)
    assert (== (increment) 2)
    set counter 100
    assert (== (increment) 101)
}
```

### Avoid Testing Two Functions Through One Shadow Block

Each shadow block should test *its* function, not the function plus other functions it calls. If `f` calls `g`, test `g` separately in `g`'s shadow block. Trust that `g` works when testing `f`.

This is already enforced by the language structure (each shadow block is named after a specific function), but it bears repeating as a design principle.

## Handling Side Effects

### Functions That Write Files or Network

Functions that perform I/O should be wrapped so the core logic can be tested without side effects:

```nano
# Hard to test:
fn save_user_data(filename: string, name: string, age: int) -> void {
    let content: string = (+ name (+ "," (int_to_string age)))
    (file_write filename content)
}

# Better: separate serialization from I/O
fn format_user_data(name: string, age: int) -> string {
    return (+ name (+ "," (int_to_string age)))
}

shadow format_user_data {
    assert (== (format_user_data "Alice" 30) "Alice,30")
    assert (== (format_user_data "" 0) ",0")
}

fn save_user_data(filename: string, name: string, age: int) -> void {
    let content: string = (format_user_data name age)
    (file_write filename content)
}

shadow save_user_data {
    # Cannot easily test file I/O in shadow; just ensure it compiles and doesn't crash
    # on a temp path
    (save_user_data "/tmp/test_user.txt" "Test" 1)
}
```

### Functions That Use Time

If a function's behavior depends on the current time, make time an explicit parameter so tests can provide a fixed value:

```nano
# Hard to test predictably:
fn is_business_hours() -> bool { ... }

# Testable:
fn is_business_hours_at(hour: int) -> bool {
    return (and (>= hour 9) (< hour 17))
}

shadow is_business_hours_at {
    assert (is_business_hours_at 9)
    assert (is_business_hours_at 16)
    assert (not (is_business_hours_at 8))
    assert (not (is_business_hours_at 17))
}
```

## Common Patterns

### Testing with a Helper

When multiple shadow blocks need the same setup, extract it into a helper function:

```nano
fn make_test_vector(x: float, y: float) -> Vector2D {
    return (vec_new x y)
}

shadow make_test_vector {
    let v: Vector2D = (make_test_vector 3.0 4.0)
    assert (== v.x 3.0)
    assert (== v.y 4.0)
}

fn vec_length(v: Vector2D) -> float { ... }

shadow vec_length {
    let v: Vector2D = (make_test_vector 3.0 4.0)
    assert (== (vec_length v) 5.0)
}
```

### Testing for "No Crash" on Bad Input

For functions that should handle bad input gracefully rather than crash:

```nano
fn safe_head(arr: array<int>) -> int {
    if (== (array_length arr) 0) { return -1 }
    return (array_get arr 0)
}

shadow safe_head {
    assert (== (safe_head []) -1)         # empty: safe fallback
    assert (== (safe_head [42]) 42)       # single element
    assert (== (safe_head [1, 2, 3]) 1)   # multiple elements
}
```

### Floating-Point Comparisons

Never use `==` for floating-point results that involve computation. Instead, check that the result is within a small epsilon:

```nano
fn approx_equal(a: float, b: float, eps: float) -> bool {
    let diff: float = (- a b)
    let abs_diff: float = (cond ((< diff 0.0) (- 0.0 diff)) (else diff))
    return (< abs_diff eps)
}

shadow approx_equal {
    assert (approx_equal 1.0 1.0 0.001)
    assert (approx_equal 1.0001 1.0 0.001)
    assert (not (approx_equal 1.1 1.0 0.001))
}

fn vec_length_example(v: Vector2D) -> float {
    return (sqrt (+ (* v.x v.x) (* v.y v.y)))
}

shadow vec_length_example {
    let v: Vector2D = (vec_new 3.0 4.0)
    assert (approx_equal (vec_length_example v) 5.0 0.0001)
}
```

## Summary Checklist

Before committing a function, verify:

- [ ] Shadow block exists immediately after the function
- [ ] Shadow block tests the happy path (at least one normal input)
- [ ] Shadow block tests boundary values (0, empty, negative, max)
- [ ] Shadow block exercises both branches of every `if`/`else`
- [ ] Functions with complex domain logic have a `prop_*` property test
- [ ] Mutable global state is reset at the start of shadow blocks that use it
- [ ] Floating-point comparisons use epsilon checks, not `==`
- [ ] Side-effecting logic is separated from pure logic so the pure part can be tested
- [ ] Property function names start with `prop_` and describe the invariant

---

**Previous:** [20.2 coverage](coverage.html)
**Next:** [Chapter 21: Configuration](../21_configuration/index.html)
