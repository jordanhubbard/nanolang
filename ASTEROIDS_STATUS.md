# Asteroids Implementation Status

## Summary

Asteroids game **cannot currently be implemented** due to a fundamental transpiler limitation: **array<struct> does not compile**.

## The Real Blocker

### What I Initially Thought Was Missing ❌
- SDL drawing primitives (lines, circles)
- Event handling functions
- Missing SDL module capabilities

### What's Actually Missing ✅
**array<struct> transpiler support** - The metadata flow issue discovered during this session.

## Technical Details

### SDL Module Is Fine
The SDL module has everything needed:
- `SDL_RenderDrawLine(renderer, x1, y1, x2, y2)` - ✅ Available
- `SDL_RenderDrawPoint(renderer, x, y)` - ✅ Available  
- Full event polling via `nl_sdl_poll_keypress()` - ✅ Available

Examples that use these successfully:
- `examples/sdl_examples/drawing_primitives.nano`
- `examples/sdl_examples/texture_demo.nano`

### The array<struct> Issue

**What fails:**
```nano
struct Particle {
    x: float,
    y: float,
    vx: float,
    vy: float,
    life: float
}

fn create_explosion() -> array<Particle> {
    let mut particles: array<Particle> = []  # Type annotation present
    let p: Particle = Particle { x: 100.0, y: 100.0, vx: 50.0, vy: 50.0, life: 1.0 }
    set particles (array_push particles p)  # ERROR: "array_push() requires a dynamic array"
    return particles
}
```

**Error message:**
```
Error: array_push() requires a dynamic array (use [] to create one)
```

**Root cause:**
The transpiler's metadata flow from parser → typechecker → transpiler doesn't properly propagate struct type names for `array<StructName>` declarations. See commit 6afb258 for the partial infrastructure that was added.

### Workarounds That DO Work

**1. Parallel Arrays (like particles_sdl.nano):**
```nano
# Instead of array<Particle>, use:
let mut particle_x: array<float> = []
let mut particle_y: array<float> = []
let mut particle_vx: array<float> = []
let mut particle_vy: array<float> = []
let mut particle_life: array<float> = []

# Then access via same index:
set particle_x (array_push particle_x x_val)
set particle_y (array_push particle_y y_val)
# etc.
```

**2. Top-level arrays with push:**
```nano
# At main() scope, not in functions:
let mut asteroids: array<Asteroid> = []
set asteroids (array_push asteroids asteroid1)  # Works at top level!
```

**Why parallel arrays work:**
- Each array is `array<float>` or `array<int>` - primitive types
- No struct metadata needed
- Transpiler generates correct `dyn_array_push_float`, etc.

## What Would Be Needed for Asteroids

### Option 1: Fix array<struct> (proper solution)
Complete the work started in commit 6afb258:
1. Fix metadata flow in parser for `array<StructName>` type annotations
2. Ensure `Symbol.struct_type_name` is set correctly for array element types
3. Verify transpiler receives struct name when generating array operations

Estimated effort: 4-8 hours of debugging the type system flow.

### Option 2: Use Parallel Arrays (workaround)
Rewrite asteroids to use parallel arrays like particles_sdl:
```nano
# Instead of array<Asteroid>:
let mut asteroid_x: array<float> = []
let mut asteroid_y: array<float> = []
let mut asteroid_vx: array<float> = []
let mut asteroid_vy: array<float> = []
let mut asteroid_active: array<bool> = []

# Instead of array<array<Particle>> for explosions:
let mut explosion_ids: array<int> = []  # Which explosion each particle belongs to
let mut explosion_particle_x: array<float> = []
let mut explosion_particle_y: array<float> = []
# ... etc
```

Estimated effort: 2-3 hours to rewrite with parallel arrays.

### Option 3: Rectangle-based Graphics (different game)
Simplify to avoid needing many entities:
- Ship as filled rectangles
- Few large "asteroids" as rectangles
- Simpler collision detection

Would work but loses the essence of asteroids (many small objects).

## Examples That Work Around This

**particles_sdl.nano** - Uses parallel arrays for particles:
- Compiles successfully ✅
- 90K binary
- Uses 5 separate arrays (x, y, vx, vy, life) instead of array<Particle>

**All other working examples** avoid array<struct>:
- checkers_sdl - Uses board as 2D logic, not array of pieces
- boids_sdl - Unknown (need to check)
- falling_sand - Uses 2D grid of ints, not array<Cell>

## Recommendation

**Short term:** Document this as a known limitation and use parallel arrays for games requiring many dynamic entities.

**Medium term:** Fix array<struct> in the transpiler (the metadata flow issue).

**Long term:** Consider if struct-in-array patterns are common enough to warrant priority.

## Related Files

- `src/transpiler_iterative_v3_twopass.c` - Lines 418-473 (array_push handling)
- `src/typechecker.c` - Lines 1566-1570 (array<struct> type name storage)  
- `src/parser.c` - Lines 1467-1475 (array<struct> parsing)
- `tests/test_array_struct_simple.nano` - Test case that fails to compile

## Testing

To verify the issue:
```bash
./bin/nanoc tests/test_array_struct_simple.nano -o /tmp/test
# Fails with: "array_push() requires a dynamic array"
```

To see working alternative:
```bash
./bin/nanoc examples/particles_sdl.nano -o bin/particles_sdl
./bin/particles_sdl  # Works! Uses parallel arrays.
```

---

**Created:** December 7, 2024  
**Status:** Known limitation - asteroids deferred until array<struct> is fixed
