# Control flow

NanoLang has expression-style `cond` and statement-style `if/else`, plus `while` and `for` loops.

> **For comprehensive coverage** of all control flow constructs with detailed examples, see [Chapter 5: Control Flow](part1_fundamentals/05_control_flow.md).

## `if/else` (statement)

Use `if/else` for statements with side effects:

<!--nl-snippet {"name":"ug_control_flow_if","check":true}-->
```nano
fn describe(x: int) -> void {
    if (< x 0) {
        (println "negative")
    } else {
        (println "non-negative")
    }
}

shadow describe {
    (describe -5)
    (describe 10)
}

fn main() -> int {
    (describe 42)
    return 0
}

shadow main { assert true }
```

## `cond` (expression)

<!--nl-snippet {"name":"ug_control_flow_cond","check":true}-->
```nano
fn sign(x: int) -> int {
    return (cond
        ((< x 0) -1)
        ((> x 0) 1)
        (else 0)
    )
}

shadow sign {
    assert (== (sign -5) -1)
    assert (== (sign 0) 0)
    assert (== (sign 7) 1)
}

fn main() -> int {
    assert (== (sign 123) 1)
    return 0
}

shadow main { assert true }
```

## Loops

<!--nl-snippet {"name":"ug_control_flow_loops","check":true}-->
```nano
fn sum_1_to_n(n: int) -> int {
    let mut sum: int = 0
    let mut i: int = 1
    while (<= i n) {
        set sum (+ sum i)
        set i (+ i 1)
    }
    return sum
}

shadow sum_1_to_n {
    assert (== (sum_1_to_n 1) 1)
    assert (== (sum_1_to_n 5) 15)
}

fn main() -> int {
    assert (== (sum_1_to_n 10) 55)
    return 0
}

shadow main { assert true }
```

## `for` loop

Use `for` when iterating over a range:

<!--nl-snippet {"name":"ug_control_flow_for","check":true}-->
```nano
fn sum_range(n: int) -> int {
    let mut sum: int = 0
    for i in (range 0 n) {
        set sum (+ sum i)
    }
    return sum
}

shadow sum_range {
    assert (== (sum_range 5) 10)
    assert (== (sum_range 0) 0)
}

fn main() -> int {
    assert (== (sum_range 10) 45)
    return 0
}

shadow main { assert true }
```

---

**See also:** [Chapter 5: Control Flow](part1_fundamentals/05_control_flow.md) for complete documentation including nested conditionals, loop patterns, and best practices.
