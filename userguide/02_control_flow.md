# Control flow

NanoLang has expression-style `cond` and statement-style `if`.

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

