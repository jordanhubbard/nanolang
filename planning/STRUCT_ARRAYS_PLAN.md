# Struct Array Implementation Plan

## Status
🔴 **Not yet implemented** - Interpreter supports struct arrays, transpiler does not

## Problem
The transpiler currently only supports arrays of primitive types (int, float, string, bool). Arrays of structs like `array<Tile>` fail to compile.

## Current Errors
```c
/tmp/test.c:754:42: error: passing 'nl_Tile' (aka 'struct nl_Tile') to parameter of incompatible type 'double'
/tmp/test.c:764:16: error: returning 'int64_t' from a function with incompatible result type 'nl_Tile'
```

## What Needs to be Implemented

### 1. Runtime Support (src/runtime/dyn_array.c)
Currently `ELEM_STRUCT` exists but has no functions. Need to add:

```c
// Push struct onto array (makes copy)
DynArray* dyn_array_push_struct(DynArray* arr, void* struct_ptr, size_t struct_size);

// Get struct from array (returns pointer to struct in array)
void* dyn_array_get_struct(DynArray* arr, int64_t index);

// Set struct in array (copies struct)
void dyn_array_set_struct(DynArray* arr, int64_t index, void* struct_ptr, size_t struct_size);

// Pop struct from array (returns copy via out parameter)
void dyn_array_pop_struct(DynArray* arr, void* out_struct, size_t struct_size, bool* success);
```

### 2. Transpiler Type Detection (src/transpiler.c)
Update `generate_expression()` to detect struct array operations:

```c
// In AST_CALL handling for array operations:
if (strcmp(func_name, "array_push") == 0) {
    // Check if array element type is struct
    Type elem_type = get_array_element_type(arr_expr, env);
    if (elem_type == TYPE_STRUCT) {
        const char* struct_name = get_struct_type_name(arr_expr, env);
        // Generate: dyn_array_push_struct(arr, &value, sizeof(nl_StructName))
    } else {
        // Existing int/float handling
    }
}
```

### 3. Generate Struct-Specific Wrappers (src/transpiler.c)
For each struct used in arrays, generate wrapper functions:

```c
// Generated for array<Tile>:
static DynArray* nl_array_push_Tile(DynArray* arr, nl_Tile val) {
    return dyn_array_push_struct(arr, &val, sizeof(nl_Tile));
}

static nl_Tile nl_array_at_Tile(DynArray* arr, int64_t idx) {
    nl_Tile* ptr = (nl_Tile*)dyn_array_get_struct(arr, idx);
    return *ptr;  // Return copy
}

static void nl_array_set_Tile(DynArray* arr, int64_t idx, nl_Tile val) {
    dyn_array_set_struct(arr, idx, &val, sizeof(nl_Tile));
}
```

### 4. Array Literal Support
Handle `let arr: array<Tile> = []` and `let arr: array<Tile> = [tile1, tile2]`:

```c
static DynArray* dynarray_literal_Tile(int count, ...) {
    DynArray* arr = dyn_array_new(ELEM_STRUCT);
    arr->struct_size = sizeof(nl_Tile);
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
        nl_Tile val = va_arg(args, nl_Tile);
        dyn_array_push_struct(arr, &val, sizeof(nl_Tile));
    }
    va_end(args);
    return arr;
}
```

### 5. Function Signatures
Handle functions returning `array<Struct>`:

```c
// nanolang: fn generate_world(seed: int) -> array<Tile>
// C: DynArray* nl_generate_world(int64_t seed)
```

Handle functions taking `array<Struct>` as parameter:

```c
// nanolang: fn get_tile(world: array<Tile>, x: int, y: int) -> Tile
// C: nl_Tile nl_get_tile(DynArray* world, int64_t x, int64_t y)
```

## Testing

### Test Case: terrain_explorer_sdl.nano
Currently interpreter-only. Should compile and run after implementation.

### Expected Behavior
```bash
cd examples
make terrain-explorer-sdl  # Should compile without errors
../bin/terrain_explorer_sdl  # Should run
```

## Implementation Steps

1. **Add runtime functions** (dyn_array.c) - 30 min
   - Implement push/get/set/pop for structs
   - Test with simple struct array

2. **Update transpiler type system** (transpiler.c) - 60 min
   - Add `get_array_element_type()` helper
   - Add `get_struct_type_name()` helper
   - Track which structs are used in arrays

3. **Generate struct-specific wrappers** (transpiler.c) - 60 min
   - Generate `nl_array_push_<StructName>` etc.
   - Update `at()`, `array_push()`, `array_set()` call sites

4. **Array literals** (transpiler.c) - 30 min
   - Handle empty array literals: `[]` with struct type
   - Handle non-empty literals: `[struct1, struct2]`

5. **Test and fix** - 60 min
   - Compile terrain_explorer_sdl.nano
   - Fix any remaining issues
   - Verify zero warnings (per .cursorrules)

**Total estimated time:** 4 hours

## Blockers
None - all infrastructure is in place. Just needs implementation.

## Priority
**HIGH** - One of four remaining interpreter-only examples. Blocking full transpiler coverage.

