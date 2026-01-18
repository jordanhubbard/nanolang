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
    assert (== (greeting) "hello world")
}

fn main() -> int {
    (println (greeting))
    return 0
}

shadow main { assert true }
```

## Booleans

<!--nl-snippet {"name":"ug_types_bools","check":true}-->
```nano
fn both_true(a: bool, b: bool) -> bool {
    return (and a b)
}

shadow both_true {
    assert (both_true true true)
    assert (not (both_true true false))
}

fn main() -> int {
    assert (or (both_true true true) (both_true false true))
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

## Generics

Generic types are written as `Type<T>` and are monomorphized at compile time.

<!--nl-snippet {"name":"ug_types_generics","check":true}-->
```nano
fn sum_list(xs: List<int>) -> int {
    let mut total: int = 0
    let len: int = (list_int_length xs)
    let mut i: int = 0
    while (< i len) {
        let v: int = (list_int_get xs i)
        set total (+ total v)
        set i (+ i 1)
    }
    return total
}

shadow sum_list {
    let xs: List<int> = (list_int_new)
    (list_int_push xs 1)
    (list_int_push xs 2)
    (list_int_push xs 3)
    assert (== (sum_list xs) 6)
}

fn main() -> int {
    let xs: List<int> = (list_int_new)
    (list_int_push xs 10)
    (list_int_push xs 20)
    assert (== (sum_list xs) 30)
    return 0
}

shadow main { assert true }
```

## Tuples

<!--nl-snippet {"name":"ug_types_tuples","check":true}-->
```nano
fn sum_pair() -> int {
    let t: (int, int) = (4, 6)
    return (+ t.0 t.1)
}

shadow sum_pair {
    assert (== (sum_pair) 10)
}

fn main() -> int {
    assert (== (sum_pair) 10)
    return 0
}

shadow main { assert true }
```

## Structs

<!--nl-snippet {"name":"ug_types_structs","check":true}-->
```nano
struct Point2D {
    x: int,
    y: int
}

fn origin_distance(p: Point2D) -> int {
    return (+ p.x p.y)
}

shadow origin_distance {
    let p: Point2D = Point2D { x: 2, y: 3 }
    assert (== (origin_distance p) 5)
}

fn main() -> int {
    let p: Point2D = Point2D { x: 1, y: 1 }
    assert (== (origin_distance p) 2)
    return 0
}

shadow main { assert true }
```

## Enums

<!--nl-snippet {"name":"ug_types_enums","check":true}-->
```nano
enum Status {
    IDLE = 0,
    RUNNING = 1,
    DONE = 2
}

fn is_running(s: Status) -> bool {
    return (== s Status.RUNNING)
}

shadow is_running {
    assert (is_running Status.RUNNING)
    assert (not (is_running Status.IDLE))
}

fn main() -> int {
    assert (is_running Status.RUNNING)
    return 0
}

shadow main { assert true }
```
