# Higher-level patterns

Even in a low-level language, patterns like map/reduce/fold can be expressed explicitly.
NanoLang also provides built-in `map`, `filter`, and `reduce` for arrays.

> **For comprehensive coverage** see [Chapter 6: Collections](part1_fundamentals/06_collections.md) and [Chapter 7: Data Structures](part1_fundamentals/07_data_structures.md) for unions and pattern matching.

## Built-in map/filter/reduce

<!--nl-snippet {"name":"ug_patterns_builtin_map_reduce","check":true}-->
```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 3) 6)
}

fn is_positive(x: int) -> bool {
    return (> x 5)
}

shadow is_positive {
    assert (is_positive 6)
    assert (not (is_positive 5))
}

fn sum(a: int, b: int) -> int {
    return (+ a b)
}

shadow sum {
    assert (== (sum 2 3) 5)
}

fn main() -> int {
    let xs: array<int> = (array_new 4 0)
    (array_set xs 0 1)
    (array_set xs 1 2)
    (array_set xs 2 3)
    (array_set xs 3 4)

    let doubled: array<int> = (map xs double)
    # doubled = [2, 4, 6, 8]
    let large: array<int> = (filter doubled is_positive)
    # large = [6, 8] (only elements > 5)
    let total: int = (reduce large 0 sum)

    assert (== (at doubled 0) 2)
    assert (== (at doubled 3) 8)
    assert (== (array_length large) 2)
    assert (== total 14)  # 6 + 8 = 14
    return 0
}

shadow main { assert true }
```

## Unions and match

<!--nl-snippet {"name":"ug_patterns_match_union","check":true}-->
```nano
union SimpleResult {
    Ok { value: int },
    Err { error: string }
}

fn unwrap_or_zero(r: SimpleResult) -> int {
    let value: int = (match r {
        Ok(v) => { return v.value }
        Err(e) => { return 0 }
    })
    return value
}

shadow unwrap_or_zero {
    let ok: SimpleResult = SimpleResult.Ok { value: 7 }
    let err: SimpleResult = SimpleResult.Err { error: "nope" }
    assert (== (unwrap_or_zero ok) 7)
    assert (== (unwrap_or_zero err) 0)
}

fn main() -> int {
    let r: SimpleResult = SimpleResult.Ok { value: 42 }
    assert (== (unwrap_or_zero r) 42)
    return 0
}

shadow main { assert true }
```

## Unsafe and extern calls

Extern functions must be called inside `unsafe { ... }` blocks.

<!--nl-snippet {"name":"ug_patterns_unsafe_extern","check":true}-->
```nano
extern fn get_argc() -> int

fn argc_safe() -> int {
    let mut n: int = 0
    unsafe {
        set n (get_argc)
    }
    return n
}

shadow argc_safe {
    let n: int = (argc_safe)
    assert (> n 0)
}

fn main() -> int {
    assert (> (argc_safe) 0)
    return 0
}

shadow main { assert true }
```

## First-class functions

Functions can be passed as arguments using `fn(...) -> ...` types.

<!--nl-snippet {"name":"ug_patterns_first_class","check":true}-->
```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
}

fn apply_twice(x: int, f: fn(int) -> int) -> int {
    return (f (f x))
}

shadow apply_twice {
    assert (== (apply_twice 3 double) 12)
}

fn main() -> int {
    assert (== (apply_twice 4 double) 16)
    return 0
}

shadow main { assert true }
```

## Fold (reduce)

<!--nl-snippet {"name":"ug_patterns_fold","check":true}-->
```nano
fn fold_sum(xs: array<int>) -> int {
    let mut acc: int = 0
    let mut i: int = 0
    while (< i (array_length xs)) {
        set acc (+ acc (at xs i))
        set i (+ i 1)
    }
    return acc
}

shadow fold_sum {
    let xs: array<int> = (array_new 4 0)
    (array_set xs 0 1)
    (array_set xs 1 2)
    (array_set xs 2 3)
    (array_set xs 3 4)
    assert (== (fold_sum xs) 10)
}

fn main() -> int {
    let xs: array<int> = (array_new 0 0)
    assert (== (fold_sum xs) 0)
    return 0
}

shadow main { assert true }
```

## Logging, tracing, and coverage

The stdlib includes structured logging plus lightweight tracing/coverage hooks.

<!--nl-snippet {"name":"ug_patterns_logging_trace_coverage","check":true}-->
```nano
from "stdlib/log.nano" import log_info
from "stdlib/coverage.nano" import coverage_init, coverage_record, coverage_report
from "stdlib/coverage.nano" import trace_init, trace_record, trace_report

fn add_one(x: int) -> int {
    (trace_record "CALL" "add_one" (int_to_string x))
    (coverage_record "userguide" 1 1)
    return (+ x 1)
}

shadow add_one {
    assert (== (add_one 4) 5)
}

fn main() -> int {
    (coverage_init)
    (trace_init)
    (log_info "demo" "instrumentation demo")
    assert (== (add_one 9) 10)
    (coverage_report)
    (trace_report)
    return 0
}

shadow main { assert true }
```
