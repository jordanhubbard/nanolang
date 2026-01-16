# Higher-level patterns

Even in a low-level language, patterns like map/reduce/fold can be expressed explicitly.

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
