# Higher-level patterns

Even in a low-level language, patterns like map/reduce/fold can be expressed explicitly.
NanoLang also provides built-in `map`, `filter`, and `reduce` for arrays.

## Built-in map/filter/reduce

<!--nl-snippet {"name":"ug_patterns_builtin_map_reduce","check":true}-->
```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 3) 6)
}

fn is_even(x: int) -> bool {
    return (== (% x 2) 0)
}

shadow is_even {
    assert (is_even 4)
    assert (not (is_even 5))
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
    let evens: array<int> = (filter doubled is_even)
    let total: int = (reduce evens 0 sum)

    assert (== (at doubled 0) 2)
    assert (== (at doubled 3) 8)
    assert (== (array_length evens) 2)
    assert (== total 12)
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
