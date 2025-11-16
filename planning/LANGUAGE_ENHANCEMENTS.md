# Nanolang Language Enhancements - Design Document

## Overview

Five critical language features to dramatically improve nanolang's usability for game development and general programming:

1. **Tuple Returns** - Functions can return multiple values
2. **Unary Operators** - Support `-x`, `not x`, etc.
3. **Explicit Type Casting** - `cast<type>(value)` for explicit conversions
4. **Top-Level Constants** - Allow immutable `let` at file scope
5. **Arrays of Structs** - Full GC integration for `array<StructType>`

---

## 1. Tuple Returns

### Syntax

```nano
# Function declaration with tuple return type
fn vec_normalize(x: float, y: float) -> (float, float) {
    let len: float = (sqrt (+ (* x x) (* y y)))
    if (> len 0.01) {
        return ((/ x len), (/ y len))
    } else {
        return (0.0, 0.0)
    }
}

# Destructuring assignment
let (nx: float, ny: float) = (vec_normalize 3.0 4.0)

# Can also assign to existing variables
let mut a: float = 0.0
let mut b: float = 0.0
set (a, b) = (vec_normalize 5.0 12.0)
```

### Implementation Plan

**Parser Changes (`src/parser.c`):**
- Add `ASTNodeType` for `NODE_TUPLE_TYPE` and `NODE_TUPLE_LITERAL`
- Parse `(type1, type2, ...)` in return type position
- Parse `(expr1, expr2, ...)` in return statement
- Parse `let (var1: type1, var2: type2) = expr` for destructuring
- Parse `set (var1, var2) = expr` for tuple assignment

**Type System Changes (`src/types.h`, `src/types.c`):**
- Add `TYPE_TUPLE` to `TypeTag` enum
- Add `TupleType` struct:
  ```c
  typedef struct {
      int element_count;
      Type **element_types;
  } TupleType;
  ```
- Update `Type` union to include `TupleType *tuple_type`

**Type Checker Changes (`src/typechecker.c`):**
- Type check tuple return statements
- Type check tuple destructuring
- Ensure tuple element types match

**Evaluator Changes (`src/eval.c`):**
- Add `VAL_TUPLE` to `ValueType` enum
- Add tuple value representation:
  ```c
  typedef struct {
      int element_count;
      Value *elements;
  } TupleValue;
  ```
- Update `Value` union to include `TupleValue *tuple_val`
- Implement tuple creation and destructuring

---

## 2. Unary Operators

### Syntax

```nano
# Numeric negation
let neg_x: float = (- x)
let neg_y: int = (- 5)

# Logical negation
let is_false: bool = (not true)
let is_valid: bool = (not (> x 10))

# Possible future additions
let abs_x: int = (abs x)      # Already exists as function
let sqrt_x: float = (sqrt x)  # Already exists as function
```

### Implementation Plan

**Parser Changes (`src/parser.c`):**
- Detect unary operators in prefix expressions
- Check argument count: 1 for unary, 2+ for binary
- Parse `(- x)` as unary negation
- Parse `(not x)` as logical negation

**Type Checker Changes (`src/typechecker.c`):**
- Add unary operator type rules:
  - `(- int)` → `int`
  - `(- float)` → `float`
  - `(not bool)` → `bool`

**Evaluator Changes (`src/eval.c`):**
- Implement unary negation for int and float
- Implement logical negation for bool
- Update `eval_expr()` to handle unary operations

---

## 3. Explicit Type Casting

### Syntax

```nano
# Cast integer to float
let x: int = 42
let xf: float = (cast<float> x)

# Cast float to integer (truncation)
let y: float = 3.14
let yi: int = (cast<int> y)

# Cast bool to int (0 or 1)
let b: bool = true
let bi: int = (cast<int> b)

# Alternative: keep existing functions, add cast as syntax sugar
let xf: float = (int_to_float x)  # Existing
let xf: float = (cast<float> x)   # New sugar
```

### Implementation Plan

**Option A: Add `cast<type>` as syntax**
- Parser recognizes `cast<type>` as special builtin
- Type checker validates cast operations
- Evaluator performs conversions

**Option B: Keep existing conversion functions, add shortcuts**
- Add shorter builtin names: `i2f`, `f2i`, `i2s`, etc.
- Keep explicit function calls
- Simpler implementation

**Recommended: Option A for consistency with modern languages**

**Parser Changes (`src/parser.c`):**
- Parse `cast<type>` as special token
- Extract target type from angle brackets
- Create `NODE_CAST` AST node with target type and expression

**Type Checker Changes (`src/typechecker.c`):**
- Define valid cast operations:
  - `int` ↔ `float`
  - `int` ↔ `bool`
  - `int` ↔ `string`
  - `float` ↔ `string`
  - `bool` ↔ `string`
- Reject invalid casts (e.g., `array` → `int`)

**Evaluator Changes (`src/eval.c`):**
- Implement cast operations using existing conversion logic
- Reuse `int_to_float`, `float_to_int`, etc.

---

## 4. Top-Level Immutable Constants

### Syntax

```nano
# Top-level constants (file scope)
let PI: float = 3.14159
let WORLD_WIDTH: int = 800
let WORLD_HEIGHT: int = 600
let GAME_TITLE: string = "My Game"

# Can use in any function
fn calculate_circle_area(radius: float) -> float {
    return (* PI (* radius radius))
}

# ERROR: Cannot mutate top-level constants
let mut COUNTER: int = 0  # ERROR: 'mut' not allowed at top level

# ERROR: Cannot reassign top-level constants
set PI 3.14  # ERROR: Cannot reassign constant
```

### Implementation Plan

**Parser Changes (`src/parser.c`):**
- Allow `let` statements at top level (before functions)
- Reject `let mut` at top level with error message
- Parse top-level constants into separate AST node list

**Type Checker Changes (`src/typechecker.c`):**
- Add top-level constants to global environment
- Mark them as immutable
- Type check constant initializers (must be literals or constant expressions)

**Evaluator Changes (`src/eval.c`):**
- Evaluate top-level constants before functions
- Add to global environment
- Constants are evaluated once at program start

**Constraints:**
- Constants must be initialized with **literal values** or **constant expressions**
- No function calls in constant initializers (to avoid side effects)
- Constants are truly immutable (not just convention)

---

## 5. Arrays of Structs (GC Integration)

### Current State

Arrays of structs are syntactically supported but may have GC issues:

```nano
struct Point { x: float, y: float }

let mut points: array<Point> = []
set points (array_push points (Point { x: 1.0, y: 2.0 }))
```

### Issues to Fix

1. **Type System**: Ensure `array<StructType>` is properly represented
2. **GC Integration**: Struct values in arrays must be tracked
3. **Memory Management**: Arrays must retain/release struct elements

### Implementation Plan

**Already Implemented:**
- `ElementType` enum has `ELEM_STRUCT`
- `DynArray` can store struct pointers
- GC tracks nested objects

**Remaining Work:**
- Test arrays of structs thoroughly
- Ensure GC correctly marks struct fields
- Handle nested structs (struct containing struct)

**Test Cases:**
```nano
struct Vector2D { x: float, y: float }
struct Entity { pos: Vector2D, vel: Vector2D }

fn test_array_of_structs() -> bool {
    let mut entities: array<Entity> = []
    
    let e1: Entity = Entity {
        pos: Vector2D { x: 0.0, y: 0.0 },
        vel: Vector2D { x: 1.0, y: 0.0 }
    }
    
    set entities (array_push entities e1)
    
    let retrieved: Entity = (at entities 0)
    return (== retrieved.pos.x 0.0)
}
```

---

## Implementation Order

1. ✅ **Unary Operators** (simplest, high impact)
2. ✅ **Top-Level Constants** (simple, enables cleaner code)
3. ✅ **Explicit Type Casting** (medium complexity)
4. ✅ **Tuple Returns** (most complex, highest value)
5. ✅ **Arrays of Structs** (verification and testing)

---

## Testing Strategy

For each feature:
1. Write comprehensive tests in `tests/`
2. Test edge cases and error conditions
3. Ensure shadow-tests work with new features
4. Update existing examples to use new features

---

## Documentation Updates

After implementation:
1. Update `docs/SPECIFICATION.md` with new syntax
2. Update `docs/QUICK_REFERENCE.md` with examples
3. Add migration guide for existing code
4. Update `docs/GETTING_STARTED.md` with new features

---

## Success Metrics

✅ **Feature Complete When:**
- Parser handles new syntax without errors
- Type checker validates correctly
- Evaluator executes correctly
- All tests pass
- Documentation updated
- Examples use new features

✅ **Quality Goals:**
- Zero regressions in existing tests
- Consistent with nanolang's prefix notation
- Clear error messages for invalid usage
- Performance: no significant slowdown

---

## Timeline Estimate

- **Unary Operators**: 30 minutes
- **Top-Level Constants**: 45 minutes
- **Explicit Type Casting**: 1 hour
- **Tuple Returns**: 2 hours
- **Arrays of Structs**: 30 minutes (mostly testing)
- **Testing & Documentation**: 1 hour

**Total**: ~6 hours of focused implementation

---

## Future Considerations

After these features are complete, consider:
- **Pattern matching** on tuples
- **Struct field mutation** syntax sugar
- **Optional types** (`?int`, `?string`)
- **Result types** for error handling
- **Generics** (if needed)

But for now, these 5 features will dramatically improve nanolang! 🚀

