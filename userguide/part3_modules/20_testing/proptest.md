# 20.1 proptest — Property-Based Testing

**Test invariants that should hold for all inputs, not just the ones you thought of.**

Property-based testing is a discipline where you describe a **property** (a boolean statement about your function) and let the framework generate hundreds of random inputs to find a counterexample. If a counterexample is found, proptest **shrinks** it to the smallest possible failing case to make debugging easy.

NanoLang's `proptest` module is a pure NanoLang implementation — no C dependencies. It provides generators for integers (single values, pairs, and arrays), a configurable test runner, and a structured report type.

## Quick Start

```nano
from "modules/proptest/proptest.nano" import forall_int, int_range,
                                             prop_pass, prop_fail,
                                             report_passed, report_summary,
                                             PropertyReport

fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
}

# Property: double always produces an even number
fn prop_double_is_even(x: int) -> string {
    let result: int = (double x)
    if (== (% result 2) 0) {
        return (prop_pass)
    } else {
        return (prop_fail (+ "not even: " (int_to_string result)))
    }
}

shadow prop_double_is_even {
    let report: PropertyReport = (forall_int "double_is_even"
                                             (int_range -500 500)
                                             prop_double_is_even)
    assert (report_passed report)
    (println (report_summary report))
}
```

## Core Concepts

### Properties

A **property** is a function that takes a generated input and returns one of three outcomes:

| Outcome | Meaning | How to return |
|---|---|---|
| Pass | The property holds for this input | `(prop_pass)` |
| Fail | The property is violated | `(prop_fail "reason")` |
| Discard | This input doesn't apply | `(prop_discard "reason")` |

Properties are ordinary NanoLang functions with the signature `(SomeType) -> string`. They return an encoded string — use `prop_pass`, `prop_fail`, and `prop_discard` rather than constructing the string directly.

```nano
from "modules/proptest/proptest.nano" import prop_pass, prop_fail, prop_discard

fn prop_positive_only(x: int) -> string {
    if (< x 0) {
        return (prop_discard "negative values not in domain")
    } else {
        if (> x 0) {
            return (prop_pass)
        } else {
            return (prop_fail "got zero")
        }
    }
}

shadow prop_positive_only {
    assert (== (prop_positive_only 5) (prop_pass))
    assert (== (prop_positive_only -1) (prop_discard "negative values not in domain"))
}
```

### Generators

Generators describe the space of inputs to explore.

#### `int_range(min, max)` — Integer in a range

```nano
from "modules/proptest/proptest.nano" import int_range, IntRangeGenerator

let small_ints: IntRangeGenerator = (int_range 0 100)
let any_byte: IntRangeGenerator = (int_range 0 255)
let signed: IntRangeGenerator = (int_range -1000 1000)
```

#### `int_pair(first, second)` — Two independent integers

```nano
from "modules/proptest/proptest.nano" import int_pair, int_range, IntPairGenerator

let pair_gen: IntPairGenerator = (int_pair (int_range 0 100) (int_range 1 100))
```

#### `int_array(element, max_length)` — Array of integers

```nano
from "modules/proptest/proptest.nano" import int_array, int_range, IntArrayGenerator

let arr_gen: IntArrayGenerator = (int_array (int_range 0 50) 20)
```

### Running Properties

Use the `forall_*` family of functions to run a property against a generator.

#### `forall_int` — Single integer property

```
forall_int(name: string, generator: IntRangeGenerator, property: unknown) -> PropertyReport
```

```nano
let report: PropertyReport = (forall_int "my_property"
                                          (int_range -100 100)
                                          my_property_fn)
```

#### `forall_int_pair` — Pair property

```
forall_int_pair(name: string, generator: IntPairGenerator, property: unknown) -> PropertyReport
```

The property function receives two integers. In NanoLang's current proptest API, pair properties receive the pair as separate parameters:

```nano
from "modules/proptest/proptest.nano" import forall_int_pair, int_pair, int_range,
                                             prop_pass, prop_fail, PropertyReport

fn prop_addition_commutative(a: int, b: int) -> string {
    if (== (+ a b) (+ b a)) {
        return (prop_pass)
    } else {
        return (prop_fail "addition not commutative")
    }
}

shadow prop_addition_commutative {
    let report: PropertyReport = (forall_int_pair "commutative"
                                                   (int_pair (int_range -100 100)
                                                             (int_range -100 100))
                                                   prop_addition_commutative)
    assert (report_passed report)
}
```

#### `forall_int_array` — Array property

```
forall_int_array(name: string, generator: IntArrayGenerator, property: unknown) -> PropertyReport
```

```nano
from "modules/proptest/proptest.nano" import forall_int_array, int_array, int_range,
                                             prop_pass, prop_fail, PropertyReport

fn prop_array_length_nonneg(arr: array<int>) -> string {
    if (>= (array_length arr) 0) {
        return (prop_pass)
    } else {
        return (prop_fail "negative length")
    }
}

shadow prop_array_length_nonneg {
    let report: PropertyReport = (forall_int_array "array_length_nonneg"
                                                    (int_array (int_range 0 100) 10)
                                                    prop_array_length_nonneg)
    assert (report_passed report)
}
```

## Configuration

### Default Configuration

The default config runs 100 trials with up to 40 shrink steps and a discard limit of 200:

```nano
from "modules/proptest/proptest.nano" import config_default, RunConfig

let cfg: RunConfig = (config_default)
# cfg.trials == 100
# cfg.max_shrink_steps == 40
# cfg.discard_limit == 200
# cfg.seed == 1
```

### Custom Configuration

```nano
from "modules/proptest/proptest.nano" import config, RunConfig

let thorough_cfg: RunConfig = (config 1000 100 2000 42)
# 1000 trials, 100 shrink steps, 2000 discard limit, seed 42
```

### Running with a Custom Config

Use the `_with_config` variants:

```nano
from "modules/proptest/proptest.nano" import forall_int_with_config, int_range,
                                             config, prop_pass, prop_fail,
                                             report_passed, PropertyReport

fn prop_square_nonneg(x: int) -> string {
    if (>= (* x x) 0) {
        return (prop_pass)
    } else {
        return (prop_fail "square is negative")
    }
}

shadow prop_square_nonneg {
    let cfg: RunConfig = (config 500 50 1000 99)
    let report: PropertyReport = (forall_int_with_config "square_nonneg"
                                                          (int_range -1000 1000)
                                                          prop_square_nonneg
                                                          cfg)
    assert (report_passed report)
}
```

## Reading Reports

A `PropertyReport` contains the full outcome of a test run:

```nano
struct PropertyReport {
    name: string,         # the name you passed to forall_*
    passed: bool,         # true if all trials passed
    case_count: int,      # number of trials run
    discard_count: int,   # number of discarded cases
    shrink_count: int,    # shrink steps taken on failure
    counterexample: string  # the failing input (if any)
}
```

Two helper functions work with reports:

```
report_passed(report: PropertyReport) -> bool
report_summary(report: PropertyReport) -> string
```

```nano
from "modules/proptest/proptest.nano" import forall_int, int_range, prop_pass, prop_fail,
                                             report_passed, report_summary, PropertyReport

fn prop_always_pass(x: int) -> string {
    return (prop_pass)
}

shadow prop_always_pass {
    let report: PropertyReport = (forall_int "always_pass" (int_range 0 100) prop_always_pass)
    assert (report_passed report)
    let summary: string = (report_summary report)
    (println summary)   # e.g. "always_pass: passed 100 cases"
}
```

## Practical Examples

### Testing Sorting Idempotency

```nano
from "modules/proptest/proptest.nano" import forall_int_array, int_array, int_range,
                                             prop_pass, prop_fail, report_passed,
                                             PropertyReport

fn is_sorted(arr: array<int>) -> bool {
    let len: int = (array_length arr)
    if (<= len 1) { return true }
    let mut i: int = 0
    let mut ok: bool = true
    while (and ok (< i (- len 1))) {
        let a: int = (array_get arr i)
        let b: int = (array_get arr (+ i 1))
        if (> a b) { set ok false } else { (print "") }
        set i (+ i 1)
    }
    return ok
}

shadow is_sorted {
    assert (is_sorted [1, 2, 3])
    assert (not (is_sorted [3, 1, 2]))
    assert (is_sorted [])
    assert (is_sorted [5])
}

fn prop_sort_result_sorted(arr: array<int>) -> string {
    # (Assumes a sort function exists)
    # let sorted: array<int> = (my_sort arr)
    # if (is_sorted sorted) { return (prop_pass) }
    # return (prop_fail "sort result not sorted")
    return (prop_pass)   # placeholder
}

shadow prop_sort_result_sorted {
    let report: PropertyReport = (forall_int_array "sort_sorted"
                                                    (int_array (int_range -100 100) 15)
                                                    prop_sort_result_sorted)
    assert (report_passed report)
}
```

### Testing Addition Properties

```nano
from "modules/proptest/proptest.nano" import forall_int_pair, int_pair, int_range,
                                             prop_pass, prop_fail, report_passed,
                                             PropertyReport

# Property: a + b == b + a
fn prop_add_commutative(a: int, b: int) -> string {
    if (== (+ a b) (+ b a)) {
        return (prop_pass)
    } else {
        return (prop_fail "not commutative")
    }
}

shadow prop_add_commutative {
    let gen: IntPairGenerator = (int_pair (int_range -100 100) (int_range -100 100))
    let report: PropertyReport = (forall_int_pair "add_commutative" gen prop_add_commutative)
    assert (report_passed report)
}

# Property: (a + b) - b == a
fn prop_add_sub_inverse(a: int, b: int) -> string {
    if (== (- (+ a b) b) a) {
        return (prop_pass)
    } else {
        return (prop_fail "inverse failed")
    }
}

shadow prop_add_sub_inverse {
    let gen: IntPairGenerator = (int_pair (int_range -500 500) (int_range -500 500))
    let report: PropertyReport = (forall_int_pair "add_sub_inverse" gen prop_add_sub_inverse)
    assert (report_passed report)
}
```

### Using prop_discard for Preconditions

When your function has a precondition, discard inputs that don't satisfy it rather than failing:

```nano
from "modules/proptest/proptest.nano" import forall_int, int_range, prop_pass, prop_fail,
                                             prop_discard, report_passed, PropertyReport

fn safe_divide(a: int, b: int) -> int {
    if (== b 0) { return 0 }
    return (/ a b)
}

shadow safe_divide {
    assert (== (safe_divide 10 2) 5)
    assert (== (safe_divide 10 0) 0)
}

fn prop_divide_nonzero(b: int) -> string {
    if (== b 0) {
        return (prop_discard "zero divisor")
    } else {
        let result: int = (safe_divide 100 b)
        if (<= (abs result) 100) {
            return (prop_pass)
        } else {
            return (prop_fail "result out of range")
        }
    }
}

shadow prop_divide_nonzero {
    let report: PropertyReport = (forall_int "divide_nonzero"
                                              (int_range -100 100)
                                              prop_divide_nonzero)
    assert (report_passed report)
}
```

## Constants

| Constant | Value | Meaning |
|---|---|---|
| `PROP_OUTCOME_PASS` | 0 | Property passed |
| `PROP_OUTCOME_FAIL` | 1 | Property failed |
| `PROP_OUTCOME_DISCARD` | 2 | Case discarded |
| `RNG_MULTIPLIER` | 48271 | LCG multiplier |
| `RNG_MODULUS` | 2147483647 | LCG modulus (2^31-1) |

---

**Previous:** [Chapter 20 Overview](index.html)
**Next:** [20.2 coverage](coverage.html)
