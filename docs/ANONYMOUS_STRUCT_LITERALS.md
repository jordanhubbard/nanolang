# Anonymous Struct Literals

## Overview

NanoLang now supports **anonymous struct literals with type inference**. This allows you to write struct literals without explicitly naming the struct type when the context makes the type clear.

## Syntax

### Named Struct Literals (Existing)
```nano
let point: Point = Point { x: 10, y: 20 }
```

### Anonymous Struct Literals (NEW)
```nano
let point: Point = { x: 10, y: 20 }
```

The struct type (`Point`) is inferred from the variable declaration.

## Supported Contexts

### 1. Let Statements
```nano
struct Point {
    x: int,
    y: int
}

fn example() -> void {
    # Type inferred from declaration
    let p: Point = { x: 5, y: 10 }
    (println (+ "x=" (int_to_string p.x)))
}
```

### 2. Return Statements
```nano
fn create_point(px: int, py: int) -> Point {
    # Type inferred from function return type
    return { x: px, y: py }
}
```

### 3. Nested Struct Literals
```nano
struct Line {
    start: Point,
    end: Point
}

fn create_line() -> Line {
    # Both inner and outer struct literals can be anonymous
    return {
        start: { x: 0, y: 0 },
        end: { x: 100, y: 100 }
    }
}
```

## Type Safety

The compiler verifies that:
- All required fields are present
- Field names match the struct definition
- Field types match the expected types
- No extra fields are provided

```nano
struct Point {
    x: int,
    y: int
}

fn test() -> Point {
    # ✅ Valid
    return { x: 5, y: 10 }
    
    # ❌ Error: Missing field 'y'
    # return { x: 5 }
    
    # ❌ Error: Unknown field 'z'
    # return { x: 5, y: 10, z: 15 }
    
    # ❌ Error: Type mismatch (float vs int)
    # return { x: 5.5, y: 10 }
}
```

## Implementation Details

### Parser
- Added `TOKEN_LBRACE` case to recognize `{ field: value }` syntax
- Heuristic: If `{` is followed by `identifier :`, parse as struct literal
- Stores `struct_name = NULL` initially

### Type Checker
- Infers struct name from context:
  - Let statements: Use declared type (`let x: Type = { ... }`)
  - Return statements: Use function return type
- Fills in `struct_name` field before type checking
- Reports error if type cannot be inferred

### Transpiler
- No changes needed!
- Uses the inferred `struct_name` to generate C code
- Generates identical code as named struct literals

## Benefits

1. **More Concise**: Less repetition when type is obvious
2. **Type-Safe**: Compiler still validates everything
3. **Zero Overhead**: Transpiles to same C code
4. **Familiar**: Similar to struct literals in Go, Rust, etc.

## Limitations

### Future Work
- **Function Call Arguments**: Not yet supported
  ```nano
  fn draw(p: Point) -> void { ... }
  
  # ❌ Currently not supported
  # (draw { x: 5, y: 10 })
  ```

- **Arrays of Structs**: Transpiler doesn't support `array<UserStruct>` yet
  ```nano
  # ❌ Blocked by transpiler limitation
  # let points: array<Point> = [{ x: 1, y: 2 }, { x: 3, y: 4 }]
  ```

### Workarounds
For function calls, use a temporary variable:
```nano
let p: Point = { x: 5, y: 10 }
(draw p)
```

## Examples

See `tests/test_anonymous_struct_literal.nano` for a complete test suite demonstrating:
- Anonymous struct literals in let statements
- Anonymous struct literals in return statements  
- Comparison with named struct literals
- Type inference validation

## Testing

Run the test suite:
```bash
./bin/nanoc tests/test_anonymous_struct_literal.nano -o bin/test
./bin/test
```

Expected output:
```
Testing create_point_named... PASSED
Testing create_point_anonymous... PASSED
Testing create_point_in_let... PASSED
Testing main... ✅ All anonymous struct literal tests passed!
PASSED
All shadow tests passed!
```

## Related Features

- Struct definitions: `docs/STRUCTS.md`
- Type inference: `docs/TYPE_INFERENCE.md`
- Pretty printing: `stdlib/tidy.nano`

