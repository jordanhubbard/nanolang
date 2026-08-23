# Data and Errors

I represent collections and domain states explicitly. Choose a type that makes invalid states difficult to express.

## Arrays and Tuples

```nano
let values: array<int> = [10, 20, 30]
let first = (at values 0)
(array_set values 1 25)

let pair: (int, string) = (7, "seven")
let number = pair.0
```

Arrays are bounds checked. Some array functions mutate storage and return `void`; others return a new array. Consult [Builtins](../generated/builtins.md) rather than inferring ownership from a familiar name.

## Structs and Enums

```nano
struct Point {
    x: int
    y: int
}

enum Direction {
    North
    South
    East
    West
}

let origin = Point { x: 0, y: 0 }
```

Named types use `UpperCamelCase`. Values and functions use `snake_case`.

## Unions and Match

```nano
union ParseResult {
    Value(int)
    Error(string)
}

fn unwrap_or(result: ParseResult, fallback: int) -> int {
    return (match result {
        Value(value) => value
        Error(_) => fallback
    })
}

shadow unwrap_or {
    assert (== (unwrap_or Value(7) 0) 7)
    assert (== (unwrap_or Error("bad") 3) 3)
}
```

Use `Result<T, E>` or another explicit union for failures callers can handle. Match variants where you can recover or add context. The postfix `?` operator propagates compatible result errors; use an explicit `match` when cleanup or context matters.

## Strings and Binary Strings

`string` is text. `bstring` is length-explicit binary data and can contain zero bytes. Conversion and Unicode behavior are documented per builtin or module. Do not assume byte indices are Unicode character indices.

## Collections

Built-in arrays and hash maps are separate from the C-backed collection modules. The generated [Builtins](../generated/builtins.md) page lists the exact builtin spellings. The generated [Modules](../generated/modules.md) page lists module declarations and native build boundaries.
