# Language

I keep ordinary syntax explicit. I accept a few conveniences, but I do not hide precedence or function signatures.

## Calls and Operators

Calls are parenthesized and prefix:

```nano
(println "ready")
(distance x y)
(math.clamp value low high)
```

Operators accept prefix and infix notation:

```nano
let prefix = (+ 2 (* 3 4))
let infix = 2 + (3 * 4)
```

All infix binary operators have equal precedence and associate left to right. `2 + 3 * 4` means `(2 + 3) * 4`. Add parentheses when grouping matters.

## Bindings

Bindings are immutable unless marked `mut`. Mutation uses `set`:

<!--nl-snippet {"name":"refresh_language_bindings","check":true}-->
```nano
fn count_three() -> int {
    let mut count = 0
    set count (+ count 1)
    set count (+ count 1)
    set count (+ count 1)
    return count
}

shadow count_three {
    assert (== (count_three) 3)
}

fn main() -> int {
    assert (== (count_three) 3)
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

Local annotations are optional when the initializer determines the type:

```nano
let count = 3
let name = "Ada"
```

Annotate empty collections, generic values, foreign handles, resource values, and boundaries where inference would obscure intent. Function parameters and return types remain explicit.

## Scalar Types

The common scalar types are `int`, `u8` or `byte`, `float`, `bool`, `string`, `bstring`, and `void`. Arrays, tuples, named records, enums, unions, function types, generic types, open records, and opaque foreign types extend that set.

## Control Flow

An `if` may omit `else`:

```nano
if needs_redraw {
    (draw scene)
}
```

`cond` selects among expression values and requires `else`:

```nano
let sign = (cond
    ((< value 0) -1)
    ((> value 0) 1)
    (else 0)
)
```

Loops use `while` or `for`:

```nano
while (< index count) {
    set index (+ index 1)
}

for index in (range 0 count) {
    (visit index)
}
```

`break` and `continue` are available inside loops.

## Functions

```nano
fn gcd(a: int, b: int) -> int {
    if (== b 0) {
        return a
    }
    return (gcd b (% a b))
}

shadow gcd {
    assert (== (gcd 48 18) 6)
}
```

I support recursion, first-class functions, closures, generics, preconditions with `requires`, and postconditions with `ensures`. Contracts are checked properties of executions; they are not formal proofs.

## Comments

```nano
# ordinary comment
// accepted line comment
/* block comment */
/// documentation comment
```

Use `#` for ordinary commentary and `///` for documentation consumed by tooling.
