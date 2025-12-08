# Structs Design for nanolang

**Status:** Design Phase  
**Priority:** #1 for Self-Hosting  
**Principles:** Safety, Immutability by Default, No Pointers

## Overview

Structs are aggregate types that group related data together. They are essential for representing compiler data structures (tokens, AST nodes, symbols).

## Core Principles (Must Maintain)

✅ **Immutability by Default**
- Struct fields are immutable by default
- Use `mut` keyword for mutable structs
- Individual fields cannot be selectively mutable (too complex)

✅ **Safety**
- No pointers (no null pointer errors)
- No manual memory management
- Structs are values, copied on assignment
- Type-safe field access

✅ **Verification**
- All struct operations verified at compile time
- Shadow tests required for functions using structs
- Type checker ensures field types match

❌ **No Self-Referential Structs**
- No pointers means no linked lists via struct fields
- Use array indices instead (safer)
- This is a feature, not a bug (forces safer designs)

## Syntax Design

### Struct Declaration

```nano
# Define a struct type
struct StructName {
    field1: type1,
    field2: type2,
    field3: type3
}
```

**Rules:**
- `struct` keyword introduces declaration
- Name follows nanolang identifier rules (starts with letter/underscore)
- Fields declared with `name: type` syntax
- Comma-separated fields
- All fields must have types (no inference)
- No methods (keep minimal - use functions)
- No inheritance (keep simple)

**Examples:**

```nano
# Token for lexer
struct Token {
    type: int,
    value: string,
    line: int,
    column: int
}

# Point in 2D space
struct Point {
    x: float,
    y: float
}

# Person record
struct Person {
    name: string,
    age: int,
    active: bool
}
```

---

### Struct Literals (Immutable)

```nano
# Create an immutable struct instance
let instance: StructName = StructName {
    field1: value1,
    field2: value2,
    field3: value3
}
```

**Rules:**
- Struct name followed by `{ }`
- All fields must be initialized
- Order doesn't matter (named fields)
- Type checking ensures field types match
- Result is immutable by default

**Examples:**

```nano
# Immutable token
let tok: Token = Token {
    type: TOKEN_NUMBER,
    value: "42",
    line: 1,
    column: 5
}

# Immutable point
let origin: Point = Point {
    x: 0.0,
    y: 0.0
}

# Can initialize in any order
let person: Person = Person {
    age: 30,
    name: "Alice",
    active: true
}
```

---

### Struct Literals (Mutable)

```nano
# Create a mutable struct instance
let mut instance: StructName = StructName {
    field1: value1,
    field2: value2
}
```

**Rules:**
- Use `mut` keyword to make struct mutable
- Allows field updates via `set`
- All fields are mutable (no per-field mutability)

**Examples:**

```nano
# Mutable point
let mut pos: Point = Point { x: 0.0, y: 0.0 }

# Can modify later
set pos.x 10.0
set pos.y 20.0
```

---

### Field Access (Reading)

```nano
let value: type = struct_instance.field_name
```

**Rules:**
- Dot notation for field access
- Works on both mutable and immutable structs
- Type checker ensures field exists
- Returns value of field

**Examples:**

```nano
let tok: Token = Token {
    type: TOKEN_NUMBER,
    value: "42",
    line: 1,
    column: 5
}

# Read fields
let t: int = tok.type        # 0 (TOKEN_NUMBER)
let v: string = tok.value     # "42"
let l: int = tok.line         # 1
let c: int = tok.column       # 5

# Use in expressions
if (== tok.type TOKEN_NUMBER) {
    print tok.value
}
```

---

### Field Update (Writing)

```nano
set struct_instance.field_name new_value
```

**Rules:**
- Only works on mutable structs
- Compile error if struct is immutable
- Type checker ensures value type matches field type
- Syntax mirrors variable assignment

**Examples:**

```nano
# Immutable - CANNOT modify
let tok: Token = Token { type: 0, value: "42", line: 1, column: 5 }
# set tok.value "100"  # ERROR: tok is immutable

# Mutable - CAN modify
let mut pos: Point = Point { x: 0.0, y: 0.0 }
set pos.x 10.0        # OK
set pos.y 20.0        # OK

# Type checking enforced
# set pos.x "hello"   # ERROR: type mismatch (expected float, got string)
```

---

### Nested Structs

```nano
struct Inner {
    value: int
}

struct Outer {
    inner: Inner,
    name: string
}

# Create nested struct
let obj: Outer = Outer {
    inner: Inner { value: 42 },
    name: "test"
}

# Access nested fields
let v: int = obj.inner.value  # 42
```

**Rules:**
- Structs can contain other structs
- Use dot notation for nested access
- All nesting is by value (no pointers)
- Immutability applies recursively

---

## Type System Integration

### New Type

```c
// In nanolang.h
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_BOOL,
    TYPE_STRING,
    TYPE_VOID,
    TYPE_ARRAY,
    TYPE_STRUCT,    // NEW
    TYPE_UNKNOWN
} Type;

// Extended type info for structs
typedef struct TypeInfo {
    Type base_type;
    struct TypeInfo *element_type;  // For arrays
    char *struct_name;              // For structs (NEW)
} TypeInfo;
```

### Struct Definition Storage

```c
// Store struct definitions in environment
typedef struct {
    char *name;              // Struct name
    char **field_names;      // Array of field names
    Type *field_types;       // Array of field types
    int field_count;         // Number of fields
} StructDef;

// Add to Environment
typedef struct {
    Symbol *symbols;
    int symbol_count;
    Function *functions;
    int function_count;
    StructDef *structs;      // NEW
    int struct_count;        // NEW
} Environment;
```

---

## Examples

### Example 1: Token Structure (Lexer)

```nano
# Define token type
struct Token {
    type: int,
    value: string,
    line: int,
    column: int
}

# Function that creates tokens
fn make_token(t: int, v: string, l: int, c: int) -> Token {
    return Token {
        type: t,
        value: v,
        line: l,
        column: c
    }
}

shadow make_token {
    let tok: Token = (make_token 0 "42" 1 5)
    assert (== tok.type 0)
    assert (str_equals tok.value "42")
    assert (== tok.line 1)
    assert (== tok.column 5)
}

# Function that uses tokens
fn token_to_string(tok: Token) -> string {
    return (str_format "Token({0}, {1}, {2}:{3})" 
                       tok.type tok.value tok.line tok.column)
}

shadow token_to_string {
    let tok: Token = Token { type: 0, value: "42", line: 1, column: 5 }
    let s: string = (token_to_string tok)
    # Would produce: "Token(0, 42, 1:5)"
    assert (str_contains s "42")
}
```

---

### Example 2: Point Mathematics

```nano
struct Point {
    x: float,
    y: float
}

fn distance(p1: Point, p2: Point) -> float {
    let dx: float = (- p2.x p1.x)
    let dy: float = (- p2.y p1.y)
    let dx_squared: float = (* dx dx)
    let dy_squared: float = (* dy dy)
    let sum: float = (+ dx_squared dy_squared)
    return (sqrt sum)
}

shadow distance {
    let origin: Point = Point { x: 0.0, y: 0.0 }
    let p: Point = Point { x: 3.0, y: 4.0 }
    let d: float = (distance origin p)
    assert (== d 5.0)  # 3-4-5 triangle
}

fn translate(p: Point, dx: float, dy: float) -> Point {
    return Point {
        x: (+ p.x dx),
        y: (+ p.y dy)
    }
}

shadow translate {
    let p: Point = Point { x: 1.0, y: 2.0 }
    let p2: Point = (translate p 3.0 4.0)
    assert (== p2.x 4.0)
    assert (== p2.y 6.0)
}
```

---

### Example 3: Mutable Struct Updates

```nano
struct Counter {
    count: int,
    name: string
}

fn increment_counter(c: mut Counter) -> void {
    set c.count (+ c.count 1)
}

# Note: Parameters are immutable, so we return new struct
fn increment_counter_immutable(c: Counter) -> Counter {
    return Counter {
        count: (+ c.count 1),
        name: c.name
    }
}

shadow increment_counter_immutable {
    let c: Counter = Counter { count: 0, name: "test" }
    let c2: Counter = (increment_counter_immutable c)
    assert (== c.count 0)   # Original unchanged
    assert (== c2.count 1)  # New one incremented
}

fn main() -> int {
    let mut counter: Counter = Counter { count: 0, name: "main" }
    
    # Modify in place
    set counter.count 5
    set counter.count (+ counter.count 1)
    
    print counter.count  # Prints 6
    return 0
}

shadow main {
    assert (== (main) 0)
}
```

---

## Implementation Plan

### Phase 1: Lexer (Week 1)

**Add new tokens:**
```c
TOKEN_STRUCT,  // "struct" keyword
// TOKEN_DOT already exists for field access
```

**Tasks:**
- [x] Add "struct" to keywords
- [x] Test tokenization of struct declarations

---

### Phase 2: Parser (Weeks 1-2)

**Add new AST nodes:**
```c
typedef enum {
    // ... existing ...
    AST_STRUCT_DEF,      // struct definition
    AST_STRUCT_LITERAL,  // struct { ... }
    AST_FIELD_ACCESS,    // obj.field
} ASTNodeType;

// Struct definition node
struct {
    char *name;
    char **field_names;
    Type *field_types;
    int field_count;
} struct_def;

// Struct literal node
struct {
    char *struct_name;
    char **field_names;
    ASTNode **field_values;
    int field_count;
} struct_literal;

// Field access node
struct {
    ASTNode *object;     // The struct instance
    char *field_name;    // The field being accessed
} field_access;
```

**Tasks:**
- [ ] Parse struct declarations: `struct Name { ... }`
- [ ] Parse struct literals: `Name { field: value, ... }`
- [ ] Parse field access: `obj.field`
- [ ] Parse nested field access: `obj.inner.value`
- [ ] Handle in variable declarations
- [ ] Handle in expressions
- [ ] Write parser tests

---

### Phase 3: Type Checker (Weeks 2-3)

**Tasks:**
- [ ] Store struct definitions in environment
- [ ] Check struct declarations (no duplicate structs)
- [ ] Check struct literals (all fields present, correct types)
- [ ] Check field access (struct exists, field exists, correct type)
- [ ] Check field updates (struct is mutable, type matches)
- [ ] Handle nested structs
- [ ] Write type checker tests

**Error messages:**
```
Error: Struct 'Token' already defined at line 5
Error: Unknown struct type 'Tokn' (did you mean 'Token'?)
Error: Struct 'Token' requires field 'line', but it's missing
Error: Field 'value' has type string, but got int
Error: Cannot access field 'foo' on struct 'Token' (no such field)
Error: Cannot modify field 'value' - struct 'tok' is immutable
```

---

### Phase 4: Interpreter/Evaluator (Week 3)

**Value representation:**
```c
typedef struct {
    char *struct_name;       // Type name
    char **field_names;      // Field names
    Value *field_values;     // Field values
    int field_count;
} StructValue;

typedef struct {
    ValueType type;
    bool is_return;
    union {
        long long int_val;
        double float_val;
        bool bool_val;
        char *string_val;
        Array *array_val;
        StructValue *struct_val;  // NEW
    } as;
} Value;
```

**Tasks:**
- [ ] Evaluate struct literals (create StructValue)
- [ ] Evaluate field access (extract field value)
- [ ] Evaluate field updates (modify field value)
- [ ] Handle struct copying (value semantics)
- [ ] Memory management (free struct values)
- [ ] Write interpreter tests

---

### Phase 5: C Transpiler (Weeks 4-5)

**C code generation:**

```nano
# nanolang
struct Point {
    x: float,
    y: float
}
```

Transpiles to:

```c
typedef struct {
    double x;
    double y;
} Point;
```

```nano
# nanolang
let p: Point = Point { x: 1.0, y: 2.0 }
```

Transpiles to:

```c
Point p = (Point){ .x = 1.0, .y = 2.0 };
```

```nano
# nanolang
let x: float = p.x
```

Transpiles to:

```c
double x = p.x;
```

**Tasks:**
- [ ] Generate C struct typedefs
- [ ] Generate C struct literals (compound literals)
- [ ] Generate field access
- [ ] Generate field updates
- [ ] Handle nested structs
- [ ] Write transpiler tests
- [ ] Test generated C code compiles

---

### Phase 6: Testing & Documentation (Week 6)

**Test files:**
- [ ] `tests/unit/struct_basic.nano` - Basic struct usage
- [ ] `tests/unit/struct_nested.nano` - Nested structs
- [ ] `tests/unit/struct_mutable.nano` - Mutable structs
- [ ] `tests/unit/struct_immutable.nano` - Immutability enforcement
- [ ] `tests/negative/struct_errors/` - Error cases
- [ ] `examples/17_structs.nano` - Example program

**Documentation:**
- [ ] Update SPECIFICATION.md with struct syntax
- [ ] Update QUICK_REFERENCE.md
- [ ] Update GETTING_STARTED.md with struct example
- [ ] Add to STDLIB.md if needed
- [ ] Update IMPLEMENTATION_STATUS.md

---

## Memory Model

### Value Semantics (No Pointers!)

```nano
let p1: Point = Point { x: 1.0, y: 2.0 }
let p2: Point = p1  # COPY, not reference

set p2.x 10.0  # Only modifies p2
# p1.x is still 1.0
```

**Rules:**
- Structs are always copied on assignment
- No shared references (no aliasing)
- No pointer arithmetic
- Safe by design

**Performance implications:**
- Copying large structs can be slow
- That's okay - safety > performance
- Transpiled C code can be optimized later
- Most compiler data structures are small

---

## Limitations (By Design)

### ❌ No Self-Referential Structs

```nano
# This is NOT possible (and that's good!)
struct Node {
    value: int,
    next: Node  # ERROR: infinite size!
}
```

**Why not:** Without pointers, we can't have linked structures.

**Alternative:** Use array indices:

```nano
struct Node {
    value: int,
    next_index: int  # -1 for null
}

# Store nodes in array
let mut nodes: array<Node> = (array_new 100 default_node)
```

**Benefits:**
- No null pointer errors
- Bounds checking prevents invalid indices
- Forces safer designs
- Cache-friendly (array locality)

---

### ❌ No Methods

```nano
# This is NOT possible
struct Point {
    x: float,
    y: float,
    
    fn distance(self, other: Point) -> float {  # ERROR
        # ...
    }
}
```

**Why not:** Keep language minimal, avoid complexity.

**Alternative:** Use regular functions:

```nano
struct Point {
    x: float,
    y: float
}

fn point_distance(p1: Point, p2: Point) -> float {
    # ...
}
```

---

### ❌ No Constructor Functions

```nano
# This is NOT automatic
struct Point {
    x: float,
    y: float
}

# Must use struct literal
let p: Point = Point { x: 1.0, y: 2.0 }
```

**Why not:** Keep language minimal.

**Alternative:** Write your own constructor function:

```nano
fn make_point(x: float, y: float) -> Point {
    return Point { x: x, y: y }
}

let p: Point = (make_point 1.0 2.0)
```

---

## Error Handling

### Compile-Time Errors

```nano
# Missing field
let p: Point = Point { x: 1.0 }
# ERROR: Struct 'Point' requires field 'y', but it's missing

# Wrong type
let p: Point = Point { x: "hello", y: 2.0 }
# ERROR: Field 'x' has type float, but got string

# Unknown field
let p: Point = Point { x: 1.0, y: 2.0, z: 3.0 }
# ERROR: Struct 'Point' has no field named 'z'

# Unknown struct
let p: Pont = Pont { x: 1.0, y: 2.0 }
# ERROR: Unknown struct type 'Pont' (did you mean 'Point'?)

# Immutability violation
let p: Point = Point { x: 1.0, y: 2.0 }
set p.x 10.0
# ERROR: Cannot modify field 'x' - struct 'p' is immutable (use 'mut')
```

---

## C Transpilation Examples

### Example 1: Simple Struct

**nanolang:**
```nano
struct Point {
    x: float,
    y: float
}

fn main() -> int {
    let p: Point = Point { x: 1.0, y: 2.0 }
    print p.x
    return 0
}
```

**Generated C:**
```c
#include <stdio.h>
#include <stdint.h>

typedef struct {
    double x;
    double y;
} Point;

int main(void) {
    Point p = (Point){ .x = 1.0, .y = 2.0 };
    printf("%f\n", p.x);
    return 0;
}
```

### Example 2: Mutable Struct

**nanolang:**
```nano
struct Counter {
    count: int
}

fn main() -> int {
    let mut c: Counter = Counter { count: 0 }
    set c.count 5
    print c.count
    return 0
}
```

**Generated C:**
```c
typedef struct {
    int64_t count;
} Counter;

int main(void) {
    Counter c = (Counter){ .count = 0 };
    c.count = 5;
    printf("%lld\n", (long long)c.count);
    return 0;
}
```

---

## Timeline

**Total Time:** 6-8 weeks

- **Week 1:** Lexer + Parser start
- **Week 2:** Parser complete + Type checker start
- **Week 3:** Type checker + Interpreter
- **Week 4-5:** C Transpiler
- **Week 6:** Testing + Documentation
- **Weeks 7-8:** Buffer for bugs and polish

---

## Success Criteria

✅ All tests pass:
- [ ] Basic struct creation
- [ ] Field access (read)
- [ ] Field modification (mutable only)
- [ ] Nested structs
- [ ] Type checking
- [ ] Error messages

✅ Can represent compiler data structures:
- [ ] Token struct works
- [ ] Can create and manipulate tokens
- [ ] Ready for lexer implementation

✅ Documentation complete:
- [ ] Specification updated
- [ ] Examples added
- [ ] Tutorial updated

---

## Next Steps After Structs

Once structs are complete, move to:
1. **Enums** - Simpler than structs, builds on same concepts
2. **Lists** - Dynamic collections of structs
3. **File I/O** - Read/write functions
4. **String ops** - Character access and formatting
5. **System execution** - Invoke gcc

Then: **Start writing lexer in nanolang!**

---

**Status:** Ready to implement  
**Priority:** #1  
**Estimated Completion:** 6-8 weeks  
**Dependencies:** None (can start immediately)

