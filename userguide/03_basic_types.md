# Basic types

Core types: `int`, `float`, `bool`, `string`, `array<T>`, `void`.

## Strings

String concatenation is done with `+`.

<!--nl-snippet {"name":"ug_types_strings","check":true,"expect_stdout":"hello world\n"}-->
```nano
fn greeting() -> string {
    return (+ "hello" (+ " " "world"))
}

shadow greeting {
    assert (str_equals (greeting) "hello world")
}

fn main() -> int {
    (println (greeting))
    return 0
}

shadow main { assert true }
```

## Arrays

<!--nl-snippet {"name":"ug_types_arrays","check":true}-->
```nano
fn sum_first_three(xs: array<int>) -> int {
    return (+ (at xs 0) (+ (at xs 1) (at xs 2)))
}

shadow sum_first_three {
    let xs: array<int> = (array_new 3 0)
    (array_set xs 0 10)
    (array_set xs 1 20)
    (array_set xs 2 30)
    assert (== (sum_first_three xs) 60)
}

fn main() -> int {
    let xs: array<int> = (array_new 3 0)
    (array_set xs 0 1)
    (array_set xs 1 2)
    (array_set xs 2 3)
    assert (== (sum_first_three xs) 6)
    return 0
}

shadow main { assert true }
```
