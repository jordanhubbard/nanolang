# Type Casting Implementation Plan

## Goal
Implement `cast<type>(value)` syntax for explicit type conversions.

## Syntax
```nano
let x: int = 42
let xf: float = (cast<float> x)  # int → float

let y: float = 3.14
let yi: int = (cast<int> y)  # float → int (truncate)

let b: bool = true
let bi: int = (cast<int> b)  # bool → int (1 or 0)
```

## Implementation Strategy

Since nanolang uses prefix notation and already has conversion functions (`int_to_float`, etc.), the simplest approach is to add shorter builtin functions:

### Option A: Simple Builtin Functions (RECOMMENDED)
Add these builtins:
- `(cast_float x)` - Convert to float
- `(cast_int x)` - Convert to int  
- `(cast_bool x)` - Convert to bool
- `(cast_string x)` - Convert to string

**Pros:**
- Simple to implement
- Consistent with nanolang's design
- No parser changes needed

**Cons:**
- Not as elegant as `cast<type>` syntax
- Different from C-style casts

### Option B: Full `cast<type>` Syntax
Parse `(cast<type> value)` as special syntax.

**Pros:**
- More elegant
- Familiar to C/C++ developers

**Cons:**
- Complex parser changes
- Need to handle `<type>` parsing
- More work for similar result

## Decision: Option A

Let's implement simple builtin functions first. If the user wants the fancier syntax later, we can add it.

## Implementation Steps

1. **Add builtins to `src/eval.c`:**
   - `builtin_cast_int`
   - `builtin_cast_float`
   - `builtin_cast_bool`
   - `builtin_cast_string`

2. **Register in `src/env.c`:**
   - Add to `builtin_functions` array

3. **Add to type checker `src/typechecker.c`:**
   - Add to `builtin_function_names`

4. **Test:**
   - Create `tests/test_casting.nano`
   - Test all valid conversions
   - Test edge cases

## Valid Conversions

| From   | To     | Behavior                          |
|--------|--------|-----------------------------------|
| int    | float  | Direct conversion                 |
| int    | bool   | 0 → false, non-zero → true       |
| int    | string | Number to string                  |
| float  | int    | Truncate (floor)                  |
| float  | bool   | 0.0 → false, non-zero → true     |
| float  | string | Number to string                  |
| bool   | int    | false → 0, true → 1              |
| bool   | string | "false" or "true"                 |
| string | int    | Parse (error if invalid)          |
| string | float  | Parse (error if invalid)          |
| string | bool   | "true"/"1" → true, else false    |

## Examples

```nano
# Numeric conversions
let x: int = 42
let xf: float = (cast_float x)  # 42.0

let y: float = 3.14
let yi: int = (cast_int y)  # 3

# Bool conversions
let flag: bool = true
let flag_i: int = (cast_int flag)  # 1

# String conversions
let n: int = 42
let s: string = (cast_string n)  # "42"
```

## Timeline
- Builtins: 30 minutes
- Testing: 15 minutes
- **Total: 45 minutes**

