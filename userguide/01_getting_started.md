# Getting started

This guide teaches NanoLang in the canonical style (prefix and infix operator notation, explicit types, and shadow tests).

> **For comprehensive coverage** of installation, setup, and your first program, see [Chapter 1: Getting Started](part1_fundamentals/01_getting_started.md).

## Hello world

<!--nl-snippet {"name":"ug_getting_started_hello","check":true,"expect_stdout":"Hello, NanoLang!\n"}-->
```nano
fn hello_message() -> string {
    return "Hello, NanoLang!"
}

shadow hello_message {
    assert (== (hello_message) "Hello, NanoLang!")
}

fn main() -> int {
    (println (hello_message))
    return 0
}

shadow main { assert true }
```

## Basic arithmetic and return values

<!--nl-snippet {"name":"ug_getting_started_arithmetic","check":true}-->
```nano
fn add3(a: int, b: int, c: int) -> int {
    return (+ a (+ b c))
}

shadow add3 {
    assert (== (add3 1 2 3) 6)
}

fn main() -> int {
    assert (== (add3 10 20 30) 60)
    return 0
}

shadow main { assert true }
```
