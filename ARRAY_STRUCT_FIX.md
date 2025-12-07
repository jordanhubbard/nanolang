# array<struct> Fix - Compiler Phase Communication

## Status: ✅ FIXED (commit 4952d65)

## Problem

`array<struct>` failed to compile with error:
```
error: passing 'nl_Point' to parameter of incompatible type 'int64_t'
```

Transpiler generated `dyn_array_push_int()` instead of `dyn_array_push_struct()`.

## Investigation Process

### Initial Hypothesis ❌
Thought struct_type_name metadata wasn't being set in parser/typechecker.

### Discovery 🔍
Added debug logging throughout compiler:
```
TYPECHECKER: Setting struct_type_name='Point' for array<Point> var 'points' (sym=0x105707d60)
TRANSPILER:  array='points' found, sym->struct_type_name=NULL (sym=0x105707e68)
```

**Different memory addresses!** Typechecker and transpiler were looking at different Symbol objects.

### Root Cause 🎯

**Two problems:**

1. **Typechecker deleted function-local symbols**
   - After checking each function, typechecker restored environment:
     ```c
     env->symbol_count = saved_symbol_count;  // Deleted all function-local vars!
     ```
   - This removed symbols with metadata before transpiler could access them

2. **Transpiler created new symbols without metadata**
   - When transpiling `let` statements, transpiler created fresh symbols:
     ```c
     env_define_var_with_type_info(env, name, type, elem_type, NULL, ...);
     ```
   - Passed `NULL` for TypeInfo, never set `struct_type_name`
   - These new symbols shadowed (non-existent) typechecker symbols

## Solution

### Part 1: Preserve Symbols (typechecker.c)
Don't delete function-local symbols after typechecking:
```c
/* DON'T restore environment - transpiler needs these symbols! */
/* Old code removed:
 *   env->symbol_count = saved_symbol_count;
 */
```

**Why this is safe:**
- C has function-local scope, so no name collisions in generated code
- Transpiler needs type metadata from these symbols
- Only costs extra memory (symbols stay in env), not correctness

### Part 2: Set Metadata (transpiler_iterative_v3_twopass.c)
After creating symbol for `let` statement, set `struct_type_name`:
```c
env_define_var_with_type_info(env, name, type, elem_type, NULL, ...);

/* For array<struct>, set struct_type_name */
if (type == TYPE_ARRAY && elem_type == TYPE_STRUCT && ast_type_name) {
    Symbol *sym = env_get_var(env, name);
    if (sym) {
        sym->struct_type_name = strdup(ast_type_name);
    }
}
```

**Why both fixes needed:**
- Part 1 alone: Typechecker symbols exist, but transpiler creates new ones that shadow them
- Part 2 alone: Transpiler symbols have metadata, but only for params (not local vars)
- Both together: Transpiler symbols have metadata extracted from AST

## Verification

### Test File
```nano
struct Point { x: int, y: int }

fn main() -> int {
    let mut points: array<Point> = []
    let p1: Point = Point { x: 10, y: 20 }
    set points (array_push points p1)
    let p2: Point = (at points 0)
    (println p2.x)
    return 0
}
```

### Results
✅ Compiles without errors  
✅ Generates correct `dyn_array_push_struct(points, &p1, sizeof(nl_Point))`  
✅ Runs and outputs: `10`  
✅ terrain_explorer_sdl.nano now compiles (was interpreter-only)

## Impact

### Now Possible
- `array<Point>`, `array<Particle>`, `array<Tile>` - any array of struct
- Games like terrain_explorer (uses `array<Tile>`)
- Proper asteroids implementation (array<Asteroid>, array<Bullet>)
- Physics engines with `array<RigidBody>`

### Workaround No Longer Needed
The parallel arrays pattern (separate array for each field) was necessary before this fix.
Now can use proper struct arrays.

## Lessons Learned

1. **Environment is shared, but symbols aren't permanent**
   - Same `env` pointer passed to all phases
   - BUT symbols can be deleted between phases
   - Assumption: "same env = symbols persist" was wrong

2. **Transpiler creates its own symbols**
   - Doesn't just read typechecker's symbols
   - Creates fresh symbols for params AND local vars
   - Needs to extract metadata from AST, not rely on typechecker

3. **Debug with memory addresses**
   - Printing `(void*)sym` revealed different objects
   - Without addresses, would have kept searching typechecker code
   - Memory addresses proved the "different symbol" hypothesis

## Estimated Time
- Original estimate for proper fix: 12-20 hours
- Actual time (with focused investigation): ~2 hours
- Key: Right debugging approach (memory addresses) quickly pinpointed issue
