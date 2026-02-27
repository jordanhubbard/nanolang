# Tutorial 2: Language Fundamentals

This tutorial covers nanolang's core features in depth: structs, enums, generics, and functional programming.

## Structs (Product Types)

Structs group related data together:

```nano
struct Point {
    x: float
    y: float
}

fn distance(p1: Point, p2: Point) -> float {
    let dx: float = (- p2.x p1.x)
    let dy: float = (- p2.y p1.y)
    return (sqrt (+ (* dx dx) (* dy dy)))
}

fn main() -> int {
    let origin: Point = Point { x: 0.0, y: 0.0 }
    let point: Point = Point { x: 3.0, y: 4.0 }
    let dist: float = (distance origin point)
    (println (float_to_string dist))  // 5.0
    return 0
}

shadow distance {
    let p1: Point = Point { x: 0.0, y: 0.0 }
    let p2: Point = Point { x: 3.0, y: 4.0 }
    assert (== (distance p1 p2) 5.0)
}
```

### Struct Access

```nano
let p: Point = Point { x: 10.0, y: 20.0 }
let x_coord: float = p.x
let y_coord: float = p.y
```

### Nested Structs

```nano
struct Rectangle {
    top_left: Point
    bottom_right: Point
}

fn area(rect: Rectangle) -> float {
    let width: float = (- rect.bottom_right.x rect.top_left.x)
    let height: float = (- rect.bottom_right.y rect.top_left.y)
    return (* width height)
}
```

## Enums (Sum Types)

Enums represent one of several possible values:

```nano
enum Color {
    Red
    Green
    Blue
}

fn color_name(c: Color) -> string {
    match c {
        Color.Red => return "red"
        Color.Green => return "green"
        Color.Blue => return "blue"
    }
}

shadow color_name {
    assert (== (color_name Color.Red) "red")
    assert (== (color_name Color.Blue) "blue")
}
```

### Enums with Values

```nano
enum Shape {
    Circle(radius: float)
    Rectangle(width: float, height: float)
}

fn area_of_shape(s: Shape) -> float {
    match s {
        Shape.Circle(r) => return (* 3.14159 (* r r))
        Shape.Rectangle(w, h) => return (* w h)
    }
}
```

## Union Types (Tagged Unions)

Unions represent one of multiple possible types:

```nano
union Result<T, E> {
    Ok(value: T)
    Err(error: E)
}

fn divide(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Result.Err "Division by zero"
    } else {
        return Result.Ok (/ a b)
    }
}

fn main() -> int {
    let result: Result<int, string> = (divide 10 2)
    match result {
        Result.Ok(value) => {
            (println (string_concat "Result: " (int_to_string value)))
        }
        Result.Err(error) => {
            (println (string_concat "Error: " error))
        }
    }
    return 0
}

shadow divide {
    let success: Result<int, string> = (divide 10 2)
    match success {
        Result.Ok(v) => assert (== v 5)
        Result.Err(e) => assert false  // Should not error
    }
    
    let failure: Result<int, string> = (divide 10 0)
    match failure {
        Result.Ok(v) => assert false  // Should error
        Result.Err(e) => assert (== e "Division by zero")
    }
}
```

### Optional Values

A common pattern using unions:

```nano
union Option<T> {
    Some(value: T)
    None
}

fn find_index(arr: array<int>, target: int) -> Option<int> {
    let mut i: int = 0
    while (< i (len arr)) {
        if (== (get arr i) target) {
            return Option.Some i
        }
        set i (+ i 1)
    }
    return Option.None
}
```

## Generic Types

Generics allow code reuse with type safety:

```nano
struct Box<T> {
    value: T
}

fn unbox<T>(b: Box<T>) -> T {
    return b.value
}

fn main() -> int {
    let int_box: Box<int> = Box { value: 42 }
    let str_box: Box<string> = Box { value: "hello" }
    
    (println (int_to_string (unbox int_box)))    // 42
    (println (unbox str_box))                    // hello
    return 0
}
```

### Generic Arrays

```nano
fn first<T>(arr: array<T>) -> Option<T> {
    if (== (len arr) 0) {
        return Option.None
    } else {
        return Option.Some (get arr 0)
    }
}

shadow first {
    let nums: array<int> = [1, 2, 3]
    let result: Option<int> = (first nums)
    match result {
        Option.Some(v) => assert (== v 1)
        Option.None => assert false
    }
    
    let empty: array<int> = []
    let result2: Option<int> = (first empty)
    match result2 {
        Option.Some(v) => assert false
        Option.None => assert true
    }
}
```

## Higher-Order Functions

Functions are first-class values in nanolang:

```nano
fn apply_twice(f: fn(int) -> int, x: int) -> int {
    return (f (f x))
}

fn double(x: int) -> int {
    return (* x 2)
}

fn main() -> int {
    let result: int = (apply_twice double 3)
    (println (int_to_string result))  // 12
    return 0
}

shadow apply_twice {
    fn inc(x: int) -> int { return (+ x 1) }
    assert (== (apply_twice inc 5) 7)
}
```

### Map and Reduce

Built-in higher-order functions for arrays:

```nano
fn square(x: int) -> int {
    return (* x x)
}

fn add(acc: int, x: int) -> int {
    return (+ acc x)
}

fn main() -> int {
    let numbers: array<int> = [1, 2, 3, 4, 5]
    
    // Map: apply function to each element
    let squares: array<int> = (map numbers square)
    // squares = [1, 4, 9, 16, 25]
    
    // Reduce: accumulate values
    let sum: int = (reduce numbers 0 add)
    // sum = 15
    
    (println (int_to_string sum))
    return 0
}
```

## Pattern Matching

Pattern matching provides exhaustive handling of variants:

```nano
union Message {
    Text(content: string)
    Number(value: int)
    Quit
}

fn process(msg: Message) -> void {
    match msg {
        Message.Text(s) => {
            (println (string_concat "Text: " s))
        }
        Message.Number(n) => {
            (println (string_concat "Number: " (int_to_string n)))
        }
        Message.Quit => {
            (println "Quitting")
        }
    }
}
```

**Exhaustiveness**: The compiler ensures all cases are handled!

## Control Flow

### If-Else

```nano
fn max(a: int, b: int) -> int {
    if (> a b) {
        return a
    } else {
        return b
    }
}
```

### While Loops

```nano
fn sum_to_n(n: int) -> int {
    let mut sum: int = 0
    let mut i: int = 1
    while (<= i n) {
        let sum = (+ sum i)
        set i (+ i 1)
    }
    return sum
}
```

### For-In Loops (Arrays)

```nano
fn print_all(arr: array<string>) -> void {
    let mut i: int = 0
    while (< i (len arr)) {
        (println (get arr i))
        set i (+ i 1)
    }
}
```

## Dynamic Arrays

Arrays are dynamically sized and immutable by default:

```nano
fn build_array() -> array<int> {
    let mut arr: array<int> = []
    let arr = (push arr 1)
    let arr = (push arr 2)
    let arr = (push arr 3)
    return arr
}

shadow build_array {
    let arr: array<int> = (build_array)
    assert (== (len arr) 3)
    assert (== (get arr 0) 1)
    assert (== (get arr 2) 3)
}
```

### Array Operations

```nano
let arr: array<int> = [1, 2, 3, 4, 5]

// Access
let first: int = (get arr 0)
let last: int = (get arr (- (len arr) 1))

// Modify (returns new array)
let arr2: array<int> = (push arr 6)
let arr3: array<int> = (set_at arr 0 99)

// Iterate
let mut i: int = 0
while (< i (len arr)) {
    let elem: int = (get arr i)
    (println (int_to_string elem))
    set i (+ i 1)
}
```

## Type Aliases

Create meaningful names for complex types:

```nano
type UserId = int
type Username = string
type Coordinate = Point

fn get_user_id(name: Username) -> UserId {
    // Implementation
    return 42
}
```

## Best Practices

### 1. Use Shadow Tests Extensively

```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    } else {
        return (* n (factorial (- n 1)))
    }
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 1) 1)
    assert (== (factorial 5) 120)
    assert (== (factorial 10) 3628800)
}
```

### 2. Prefer Immutability

```nano
// ❌ Mutable when not needed
let mut count: int = 0
set count (+ count 1)

// ✅ Use shadowing for transformation
let count: int = 0
let count = (+ count 1)
```

### 3. Use Union Types for Error Handling

```nano
// ❌ Return magic values
fn find(arr: array<int>, target: int) -> int {
    // Returns -1 if not found
    return (- 1)
}

// ✅ Use Option<T>
fn find(arr: array<int>, target: int) -> Option<int> {
    // Explicit success/failure
    return Option.None
}
```

### 4. Leverage Type Inference

```nano
// Verbose but clear
let x: int = 42
let y: string = "hello"

// Inferred (use for simple cases)
let x = 42  // int inferred
let y = "hello"  // string inferred
```

## Common Idioms

### Builder Pattern

```nano
struct Config {
    debug: bool
    port: int
    host: string
}

fn default_config() -> Config {
    return Config {
        debug: false
        port: 8080
        host: "localhost"
    }
}

fn with_debug(c: Config, enabled: bool) -> Config {
    return Config {
        debug: enabled
        port: c.port
        host: c.host
    }
}
```

### Iterator Pattern (Map/Filter/Reduce)

```nano
fn process_numbers(nums: array<int>) -> int {
    // Filter evens, square them, sum them
    let evens: array<int> = (filter nums is_even)
    let squares: array<int> = (map evens square)
    return (reduce squares 0 add)
}
```

## Next Steps

Continue to [Tutorial 3: Module System](03-modules.md) to learn about:
- Importing modules
- Creating your own modules
- Using the standard library
- Working with external C libraries

Or explore:
- [Examples](../../examples/) - Real-world code
- [Modules Reference](../MODULES.md) - Available libraries
- [FFI Guide](../EXTERN_FFI.md) - C integration

