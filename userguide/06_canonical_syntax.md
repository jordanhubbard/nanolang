# Canonical syntax rules

NanoLang is intentionally strict: there is one canonical way to write most constructs.

## Prefix operators

<!--nl-snippet {"name":"ug_canonical_prefix","check":true}-->
```nano
fn distance_1d(a: int, b: int) -> int {
    return (cond
        ((> a b) (- a b))
        (else (- b a))
    )
}

shadow distance_1d {
    assert (== (distance_1d 10 3) 7)
    assert (== (distance_1d 3 10) 7)
    assert (== (distance_1d 5 5) 0)
}

fn main() -> int {
    assert (== (distance_1d -5 5) 10)
    return 0
}

shadow main { assert true }
```
