# Chapter 7: Data Structures

**Create custom data types with structs, enums, and unions.**

This chapter covers how to define and use custom data types in NanoLang: structs for product types, enums for named constants, and unions for sum types.

## 7.1 Structs (Product Types)

Structs group multiple fields together into a single type. They're similar to records or objects in other languages.

### Struct Definitions

```nano
struct Point2D {
    x: int,
    y: int
}
```

**Syntax:**
- `struct` keyword
- Type name (PascalCase)
- Fields with types
- Comma-separated (trailing comma optional)

### Creating Struct Instances

```nano
fn create_point() -> Point2D {
    return Point2D { x: 10, y: 20 }
}

shadow create_point {
    let p: Point2D = (create_point)
    assert (== p.x 10)
    assert (== p.y 20)
}
```

**Syntax:** `TypeName { field1: value1, field2: value2 }`

### Accessing Fields

Use dot notation to access struct fields:

```nano
struct Rectangle {
    width: int,
    height: int
}

fn area(r: Rectangle) -> int {
    return (* r.width r.height)
}

shadow area {
    let rect: Rectangle = Rectangle { width: 5, height: 10 }
    assert (== (area rect) 50)
}
```

### Examples

**Person struct:**

```nano
struct Person {
    name: string,
    age: int
}

fn greet_person(p: Person) -> string {
    return (+ "Hello, " p.name)
}

shadow greet_person {
    let alice: Person = Person { name: "Alice", age: 30 }
    assert (== (greet_person alice) "Hello, Alice")
}
```

**Nested structs:**

```nano
struct Address {
    street: string,
    city: string
}

struct Contact {
    name: string,
    address: Address
}

fn get_city(c: Contact) -> string {
    return c.address.city
}

shadow get_city {
    let addr: Address = Address { 
        street: "123 Main St", 
        city: "Springfield" 
    }
    let contact: Contact = Contact { 
        name: "Bob", 
        address: addr 
    }
    assert (== (get_city contact) "Springfield")
}
```

**Struct with different types:**

```nano
struct Temperature {
    value: float,
    unit: string
}

fn to_celsius(t: Temperature) -> float {
    if (== t.unit "F") {
        return (* (- t.value 32.0) (/ 5.0 9.0))
    }
    return t.value
}

shadow to_celsius {
    let temp_f: Temperature = Temperature { value: 212.0, unit: "F" }
    let celsius: float = (to_celsius temp_f)
    assert (and (> celsius 99.9) (< celsius 100.1))
}
```

## 7.2 Enums (Sum Types)

Enums define a type with a fixed set of named constants. Each constant has an associated integer value.

### Enum Definitions

```nano
enum Status {
    IDLE = 0,
    RUNNING = 1,
    DONE = 2
}
```

**Syntax:**
- `enum` keyword
- Type name (PascalCase)
- Named constants (UPPER_CASE)
- Integer values (explicit)

### Enum Values

Access enum values using dot notation:

```nano
fn get_initial_status() -> Status {
    return Status.IDLE
}

shadow get_initial_status {
    let s: Status = (get_initial_status)
    assert (== s Status.IDLE)
}
```

### Using Enums

Enums are great for representing states or categories:

```nano
enum Color {
    RED = 0,
    GREEN = 1,
    BLUE = 2
}

fn color_name(c: Color) -> string {
    if (== c Color.RED) {
        return "Red"
    }
    if (== c Color.GREEN) {
        return "Green"
    }
    return "Blue"
}

shadow color_name {
    assert (== (color_name Color.RED) "Red")
    assert (== (color_name Color.GREEN) "Green")
    assert (== (color_name Color.BLUE) "Blue")
}
```

### Examples

**HTTP status codes:**

```nano
enum HttpStatus {
    OK = 200,
    NOT_FOUND = 404,
    SERVER_ERROR = 500
}

fn is_error(status: HttpStatus) -> bool {
    return (>= status HttpStatus.NOT_FOUND)
}

shadow is_error {
    assert (not (is_error HttpStatus.OK))
    assert (is_error HttpStatus.NOT_FOUND)
    assert (is_error HttpStatus.SERVER_ERROR)
}
```

**Direction enum:**

```nano
enum Direction {
    NORTH = 0,
    EAST = 1,
    SOUTH = 2,
    WEST = 3
}

fn opposite(d: Direction) -> Direction {
    if (== d Direction.NORTH) { return Direction.SOUTH }
    if (== d Direction.SOUTH) { return Direction.NORTH }
    if (== d Direction.EAST) { return Direction.WEST }
    return Direction.EAST
}

shadow opposite {
    assert (== (opposite Direction.NORTH) Direction.SOUTH)
    assert (== (opposite Direction.EAST) Direction.WEST)
}
```

## 7.3 Unions (Tagged Unions)

Unions represent a value that could be one of several variants. Each variant can have associated data.

### Union Definitions

```nano
union Result {
    Ok { value: int },
    Err { error: string }
}
```

**Syntax:**
- `union` keyword
- Type name (PascalCase)
- Variants (PascalCase)
- Each variant can have fields

### Union Constructors

Create union values using variant constructors:

```nano
fn success() -> Result {
    return Result.Ok { value: 42 }
}

fn failure() -> Result {
    return Result.Err { error: "Failed" }
}

shadow success {
    let r: Result = (success)
    # Use match to check variant
}

shadow failure {
    let r: Result = (failure)
    # Use match to check variant
}
```

### Pattern Matching with Unions

Use `match` to handle different union variants:

```nano
union Option {
    Some { value: int },
    None {}
}

fn unwrap_or(opt: Option, default: int) -> int {
    let result: int = (match opt {
        Some(s) => { return s.value }
        None(n) => { return default }
    })
    return result
}

shadow unwrap_or {
    let some: Option = Option.Some { value: 42 }
    let none: Option = Option.None {}
    assert (== (unwrap_or some 0) 42)
    assert (== (unwrap_or none 0) 0)
}
```

### Examples

**Result type for errors:**

```nano
union ParseResult {
    Success { number: int },
    Error { message: string }
}

fn parse_number(s: string) -> ParseResult {
    if (== s "42") {
        return ParseResult.Success { number: 42 }
    }
    return ParseResult.Error { message: "Invalid number" }
}

fn get_or_default(r: ParseResult) -> int {
    let value: int = (match r {
        Success(s) => { return s.number }
        Error(e) => { return -1 }
    })
    return value
}

shadow get_or_default {
    let ok: ParseResult = (parse_number "42")
    let err: ParseResult = (parse_number "bad")
    assert (== (get_or_default ok) 42)
    assert (== (get_or_default err) -1)
}
```

**Shape union:**

```nano
union Shape {
    Circle { radius: float },
    Rectangle { width: float, height: float }
}

fn area_shape(s: Shape) -> float {
    let result: float = (match s {
        Circle(c) => { return (* 3.14159 (* c.radius c.radius)) }
        Rectangle(r) => { return (* r.width r.height) }
    })
    return result
}

shadow area_shape {
    let circle: Shape = Shape.Circle { radius: 5.0 }
    let rect: Shape = Shape.Rectangle { width: 4.0, height: 6.0 }
    
    let area_c: float = (area_shape circle)
    assert (and (> area_c 78.0) (< area_c 79.0))
    
    assert (== (area_shape rect) 24.0)
}
```

## 7.4 Opaque Types

Opaque types hide their internal structure. They're defined in module interfaces but implemented elsewhere.

### What Are Opaque Types?

Opaque types are useful for:
- Hiding implementation details
- Enforcing encapsulation
- FFI with C libraries

```nano
# Declare opaque type
opaque type Handle

# Use in function signatures
fn create_handle() -> Handle
fn use_handle(h: Handle) -> int
```

### When to Use Opaque Types

**Use opaque types when:**
- Wrapping C libraries (file handles, database connections)
- Implementing abstract data types
- Enforcing invariants

### Examples

**File handle (conceptual):**

```nano
# In a file I/O module:
opaque type FileHandle

extern fn open_file(path: string) -> FileHandle
extern fn read_file(handle: FileHandle) -> string
extern fn close_file(handle: FileHandle) -> void
```

💡 **Pro Tip:** Opaque types are mostly used for FFI. For regular NanoLang code, use structs and unions.

## 7.5 Pattern Matching

Pattern matching deconstructs unions and checks variants.

### Match Expressions

```nano
union Status {
    Ready {},
    Running { progress: int },
    Done { result: int }
}

fn status_message(s: Status) -> string {
    let msg: string = (match s {
        Ready(r) => { return "Ready to start" }
        Running(r) => { return "In progress" }
        Done(d) => { return "Complete" }
    })
    return msg
}

shadow status_message {
    assert (== (status_message Status.Ready {}) "Ready to start")
    assert (== (status_message Status.Running { progress: 50 }) "In progress")
    assert (== (status_message Status.Done { result: 100 }) "Complete")
}
```

### Match Syntax

**Components:**
1. `match` keyword
2. Value to match
3. Branches with variant patterns
4. Arrow `=>` pointing to result

```nano
(match value {
    Variant1(v1) => { /* handle v1 */ }
    Variant2(v2) => { /* handle v2 */ }
})
```

### Exhaustive Matching

Match expressions must handle all variants:

```nano
union Binary {
    Zero {},
    One {}
}

fn binary_to_int(b: Binary) -> int {
    let val: int = (match b {
        Zero(z) => { return 0 }
        One(o) => { return 1 }
        # Must handle all variants!
    })
    return val
}

shadow binary_to_int {
    assert (== (binary_to_int Binary.Zero {}) 0)
    assert (== (binary_to_int Binary.One {}) 1)
}
```

### Examples

**Optional values:**

```nano
union Maybe {
    Just { value: int },
    Nothing {}
}

fn is_nothing(m: Maybe) -> bool {
    let result: bool = (match m {
        Just(j) => { return false }
        Nothing(n) => { return true }
    })
    return result
}

shadow is_nothing {
    assert (not (is_nothing Maybe.Just { value: 42 }))
    assert (is_nothing Maybe.Nothing {})
}
```

**Error handling:**

```nano
union Validation {
    Valid { data: int },
    Invalid { reason: string }
}

fn is_valid(v: Validation) -> bool {
    let result: bool = (match v {
        Valid(val) => { return true }
        Invalid(inv) => { return false }
    })
    return result
}

shadow is_valid {
    let good: Validation = Validation.Valid { data: 100 }
    let bad: Validation = Validation.Invalid { reason: "Too large" }
    assert (is_valid good)
    assert (not (is_valid bad))
}
```

**Accessing matched data:**

```nano
union Container {
    Empty {},
    Full { count: int, item: string }
}

fn describe(c: Container) -> string {
    let desc: string = (match c {
        Empty(e) => { return "Container is empty" }
        Full(f) => { 
            return (+ "Contains " (int_to_string f.count))
        }
    })
    return desc
}

shadow describe {
    assert (== (describe Container.Empty {}) "Container is empty")
    assert (== (describe Container.Full { count: 5, item: "apple" }) "Contains 5")
}
```

### Complete Example: Binary Tree

```nano
union Tree {
    Leaf { value: int },
    Node { value: int, left: Tree, right: Tree }
}

fn sum_tree(t: Tree) -> int {
    let total: int = (match t {
        Leaf(l) => { return l.value }
        Node(n) => {
            let left_sum: int = (sum_tree n.left)
            let right_sum: int = (sum_tree n.right)
            return (+ n.value (+ left_sum right_sum))
        }
    })
    return total
}

shadow sum_tree {
    let leaf1: Tree = Tree.Leaf { value: 1 }
    let leaf2: Tree = Tree.Leaf { value: 2 }
    let leaf3: Tree = Tree.Leaf { value: 3 }
    
    let node1: Tree = Tree.Node { 
        value: 4, 
        left: leaf1, 
        right: leaf2 
    }
    
    let root: Tree = Tree.Node { 
        value: 5, 
        left: node1, 
        right: leaf3 
    }
    
    assert (== (sum_tree leaf1) 1)
    assert (== (sum_tree node1) 7)  # 4 + 1 + 2
    assert (== (sum_tree root) 15)  # 5 + (4+1+2) + 3
}
```

### Summary

In this chapter, you learned:
- ✅ Structs group fields into product types
- ✅ Enums define named integer constants
- ✅ Unions represent sum types with variants
- ✅ Pattern matching handles union variants
- ✅ Match expressions must be exhaustive
- ✅ Opaque types hide implementation details

### Practice Exercises

```nano
# 1. Create a struct for 3D points
struct Point3D {
    x: float,
    y: float,
    z: float
}

fn distance_from_origin(p: Point3D) -> float {
    let x2: float = (* p.x p.x)
    let y2: float = (* p.y p.y)
    let z2: float = (* p.z p.z)
    return (sqrt (+ x2 (+ y2 z2)))
}

shadow distance_from_origin {
    let p: Point3D = Point3D { x: 3.0, y: 4.0, z: 0.0 }
    let dist: float = (distance_from_origin p)
    assert (and (> dist 4.9) (< dist 5.1))
}

# 2. Create an enum for card suits
enum Suit {
    HEARTS = 0,
    DIAMONDS = 1,
    CLUBS = 2,
    SPADES = 3
}

fn is_red(s: Suit) -> bool {
    return (or (== s Suit.HEARTS) (== s Suit.DIAMONDS))
}

shadow is_red {
    assert (is_red Suit.HEARTS)
    assert (is_red Suit.DIAMONDS)
    assert (not (is_red Suit.CLUBS))
    assert (not (is_red Suit.SPADES))
}

# 3. Create a union for calculator operations
union CalcOp {
    Add { a: int, b: int },
    Subtract { a: int, b: int },
    Multiply { a: int, b: int }
}

fn evaluate(op: CalcOp) -> int {
    let result: int = (match op {
        Add(add_op) => { return (+ add_op.a add_op.b) }
        Subtract(sub_op) => { return (- sub_op.a sub_op.b) }
        Multiply(mul_op) => { return (* mul_op.a mul_op.b) }
    })
    return result
}

shadow evaluate {
    assert (== (evaluate CalcOp.Add { a: 5, b: 3 }) 8)
    assert (== (evaluate CalcOp.Subtract { a: 5, b: 3 }) 2)
    assert (== (evaluate CalcOp.Multiply { a: 5, b: 3 }) 15)
}
```

---

**Previous:** [Chapter 6: Collections](06_collections.html)  
**Next:** [Chapter 8: Modules & Imports](08_modules.html)
