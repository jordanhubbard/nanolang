# Basic types

Core types: `int`, `float`, `bool`, `string`, `array<T>`, `void`.

> **For comprehensive coverage** see [Chapter 2: Syntax & Types](part1_fundamentals/02_syntax_types.md) and [Chapter 7: Data Structures](part1_fundamentals/07_data_structures.md) for structs, enums, and unions.

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

## HashMaps

<!--nl-snippet {"name":"ug_types_hashmap","check":true}-->
```nano
fn count_words(words: array<string>) -> HashMap<string, int> {
    let counts: HashMap<string, int> = (map_new)
    let mut i: int = 0
    while (< i (array_length words)) {
        let w: string = (at words i)
        if (map_has counts w) {
            let cur: int = (map_get counts w)
            (map_put counts w (+ cur 1))
        } else {
            (map_put counts w 1)
        }
        set i (+ i 1)
    }
    return counts
}

shadow count_words {
    let words: array<string> = ["a", "b", "a"]
    let counts: HashMap<string, int> = (count_words words)
    assert (== (map_get counts "a") 2)
    assert (== (map_get counts "b") 1)
    assert (== (map_has counts "c") false)
    # Note: HashMap is automatically GC-managed - no manual free needed
}

fn main() -> int {
    let counts: HashMap<string, int> = (count_words ["x", "x", "y"])
    assert (== (map_get counts "x") 2)
    assert (== (map_get counts "y") 1)
    # Note: HashMap is automatically GC-managed - no manual free needed
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

## Automatic Memory Management

NanoLang features **ARC-style (Automatic Reference Counting)** memory management for all heap-allocated objects, including strings, arrays, and opaque types from external libraries.

### Garbage Collection

All dynamically allocated memory is automatically tracked and freed by the garbage collector. You don't need to manually free memory for:

- **Strings** - Dynamically allocated strings are GC-managed
- **Arrays** - Both fixed and dynamic arrays are GC-managed
- **Opaque types** - External library types (HashMap, regex, JSON) are automatically wrapped with ARC cleanup

### Opaque Types

Opaque types represent handles to external C libraries. Common examples:

- `HashMap<K,V>` - Generic hash maps
- `Regex` - Compiled regular expressions
- `Json` - JSON values from the json module

**Key feature:** The compiler automatically wraps and unwraps these types at call boundaries, ensuring proper cleanup without manual intervention.

### Example: Regex (No Manual Free)

```nano
from "modules/std/regex/regex.nano" import Regex, compile, match

fn check_email(text: string) -> bool {
    let pattern: Regex = (compile "^[a-z]+@[a-z]+\\.[a-z]+$")
    if (== pattern 0) {
        return false
    }
    let result: int = (match pattern text)
    return (== result 1)
    # No regex_free needed - automatically cleaned up by GC
}
```

### Example: JSON (No Manual Free)

```nano
from "modules/std/json/json.nano" import Json, parse, get, as_string

fn extract_field(json_text: string) -> string {
    let root: Json = (parse json_text)
    if (== root 0) {
        return ""
    }
    let value: Json = (get root "name")
    return (as_string value)
    # No json_free needed - automatically cleaned up by GC
}
```

### How It Works

The compiler uses **ARC-style wrapping**:

1. **Wrapping**: When an external function returns an opaque type, the compiler wraps it in a GC-managed object with the appropriate cleanup function
2. **Unwrapping**: When passing an opaque type to an external function, the compiler automatically unwraps it to get the original pointer
3. **Cleanup**: When the GC determines an object is no longer referenced, it calls the cleanup function (e.g., `regex_free`, `json_free`)

This approach is inspired by Objective-C ARC and Swift, providing automatic memory management without manual free calls.

### Benefits

- ✅ **No manual memory management** - No `free()` calls needed
- ✅ **Uniform handling** - All opaque types work the same way
- ✅ **Zero-cost abstraction** - Wrapping/unwrapping at compile time
- ✅ **Library compatibility** - Works with any C library that provides cleanup functions

### Memory Safety

The GC automatically prevents:

- **Memory leaks** - Objects are freed when no longer referenced
- **Double-free** - Each object is freed exactly once
- **Use-after-free** - References are tracked and validated

For more details on memory management and profiling, see [Chapter 8: Profiling](08_profiling.md).
```
