# Chapter 20: Testing & Quality

**NanoLang's two-tier testing approach: shadow tests and property-based testing.**

Correctness is a first-class concern in NanoLang. The language bakes testing directly into the syntax rather than leaving it to external frameworks. This chapter covers both testing tiers and the coverage tooling that tells you how thoroughly your code is exercised.

## The Two-Tier Testing Philosophy

### Tier 1: Shadow Tests (Built-in, Mandatory)

Every function in NanoLang must have a `shadow` block. There are no exceptions (except `extern` declarations). Shadow tests run at **compile time** — if a shadow test fails, your program does not compile.

```nano
fn clamp(value: int, lo: int, hi: int) -> int {
    if (< value lo) { return lo }
    if (> value hi) { return hi }
    return value
}

shadow clamp {
    assert (== (clamp 5  0 10) 5)   # in range
    assert (== (clamp -3 0 10) 0)   # below min
    assert (== (clamp 15 0 10) 10)  # above max
    assert (== (clamp 0  0 10) 0)   # at min boundary
    assert (== (clamp 10 0 10) 10)  # at max boundary
}
```

Shadow tests are:
- **Zero overhead** at runtime — they compile away completely
- **Always up to date** — if you change the function and break a test, you find out immediately
- **Self-documenting** — reading a shadow block shows you exactly how the function behaves at boundaries

### Tier 2: Property-Based Testing (proptest module)

Property tests generalise beyond specific examples. Instead of saying "clamp(5, 0, 10) == 5", you say "for any value in range, clamp returns value unchanged — for all integers". The `proptest` module generates hundreds of random inputs, tries to find a counterexample, and shrinks any failure to the smallest possible case.

```nano
from "modules/proptest/proptest.nano" import forall_int, int_range,
                                             prop_pass, prop_fail,
                                             report_passed, report_summary

fn prop_clamp_in_range(value: int) -> string {
    let result: int = (clamp value 0 10)
    if (and (>= result 0) (<= result 10)) {
        return (prop_pass)
    } else {
        return (prop_fail (+ "out of range: " (int_to_string result)))
    }
}

shadow prop_clamp_in_range {
    let report: PropertyReport = (forall_int "clamp_in_range"
                                             (int_range -100 100)
                                             prop_clamp_in_range)
    assert (report_passed report)
}
```

Property tests are:
- **Exhaustive** — they explore the full input space within the specified range
- **Self-shrinking** — when a failure is found, proptest finds the minimal counterexample
- **Repeatable** — the random seed is configurable, so failures can be reproduced exactly

## Choosing Between the Two

| Situation | Use |
|---|---|
| Verifying specific known values | Shadow test |
| Verifying invariants over all inputs | Property test |
| Documenting expected behavior | Shadow test |
| Finding edge cases you haven't thought of | Property test |
| Fast compile-time check | Shadow test |
| Deeper confidence in a core algorithm | Property test |

A good rule of thumb: **start with shadow tests, add property tests for functions with non-trivial domain logic**.

## Shadow Test Syntax

```nano
shadow function_name {
    assert expression
    assert (== (function_name arg1 arg2) expected)
    # multiple asserts are allowed
}
```

The `shadow` keyword is followed by the name of the function being tested (not a function call — just the name). Inside the block, you can call any function and use any let bindings:

```nano
fn abs_value(x: int) -> int {
    if (< x 0) { return (- 0 x) }
    return x
}

shadow abs_value {
    assert (== (abs_value 5) 5)
    assert (== (abs_value -5) 5)
    assert (== (abs_value 0) 0)
    let big: int = (abs_value -1000000)
    assert (== big 1000000)
}
```

## Property Test Quick Start

```nano
from "modules/proptest/proptest.nano" import forall_int, int_range, prop_pass, prop_fail,
                                             report_passed, report_summary, PropertyReport

fn my_abs(x: int) -> int {
    if (< x 0) { return (- 0 x) }
    return x
}

shadow my_abs {
    assert (== (my_abs 5) 5)
    assert (== (my_abs -5) 5)
    assert (== (my_abs 0) 0)
}

fn prop_abs_nonneg(x: int) -> string {
    if (>= (my_abs x) 0) {
        return (prop_pass)
    } else {
        return (prop_fail "abs returned negative")
    }
}

shadow prop_abs_nonneg {
    let report: PropertyReport = (forall_int "abs_nonneg"
                                             (int_range -1000 1000)
                                             prop_abs_nonneg)
    assert (report_passed report)
    (println (report_summary report))
}
```

When you run the shadow test for `prop_abs_nonneg`, proptest generates 100 random integers in [-1000, 1000], calls `prop_abs_nonneg` on each, and asserts all pass.

## Coverage

NanoLang's coverage tooling is provided at the build-system level rather than as a NanoLang module. When you run tests with coverage enabled, the compiler instruments the generated C code so that every executed line is recorded. See [Section 20.2](coverage.html) for how to invoke coverage and read the reports.

---

**Sections:**
- [20.1 proptest — Property-Based Testing](proptest.html)
- [20.2 coverage — Code Coverage](coverage.html)
- [20.3 Testing Best Practices](best_practices.html)

---

**Previous:** [Chapter 19: Terminal UI](../19_terminal_ui/ncurses.html)
**Next:** [20.1 proptest](proptest.html)
