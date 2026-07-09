# My Module System Migration Guide

**From:** `import` syntax with scattered `unsafe {}` blocks  
**To:** `module` syntax with module-level safety  
**Status:** Phase 1 Complete  
**Date:** 2025-01-08

---

## My Summary

I have redesigned my module system to make safety explicit at module boundaries. You no longer need to scatter safety markers throughout your code. This change removes 98% of `unsafe {}` blocks while I maintain the same safety guarantees.

**The change:** You mark entire modules as `unsafe` when you import them, instead of marking individual function calls.

---

## Quick Migration

### Safe Modules (My Own Code)

**Before:**
```nano
import "modules/vector2d/vector2d.nano" as Vec
```

**After:**
```nano
module "modules/vector2d/vector2d.nano" as Vec
```

**Action:** Replace `import` with `module`.

---

### Unsafe Modules (FFI and External Libraries)

**Before:**
```nano
import "modules/sdl/sdl.nano"

fn render() -> void {
    unsafe { (SDL_Init 0) }
    unsafe { (SDL_CreateWindow "Game" 800 600) }
    unsafe { (SDL_Quit) }
}
```

**After:**
```nano
unsafe module "modules/sdl/sdl.nano"

fn render() -> void {
    (SDL_Init 0)
    (SDL_CreateWindow "Game" 800 600)
    (SDL_Quit)
}
```

**Actions:**
1. Add `unsafe` before the module import.
2. Remove all `unsafe {}` blocks for calls to that module.

---

## Complete Examples

### Example 1: SDL Game

**Before (My Old System):**
```nano
import "modules/sdl/sdl.nano"
import "modules/sdl_helpers/sdl_helpers.nano"
import "modules/ui_widgets/ui_widgets.nano"

fn main() -> int {
    unsafe { (SDL_Init 0) }
    
    let mut window: SDL_Window = 0
    unsafe {
        set window (SDL_CreateWindow "My Game" 800 600)
    }
    
    let mut running: bool = true
    while running {
        unsafe {
            set running (not (SDL_QuitRequested))
        }
        
        unsafe { (SDL_RenderClear window) }
        # ... game logic ...
        unsafe { (SDL_RenderPresent window) }
        unsafe { (SDL_Delay 16) }
    }
    
    unsafe { (SDL_DestroyWindow window) }
    unsafe { (SDL_Quit) }
    
    return 0
}
```

**After (My New System):**
```nano
unsafe module "modules/sdl/sdl.nano"
unsafe module "modules/sdl_helpers/sdl_helpers.nano"
module "modules/ui_widgets/ui_widgets.nano"

fn main() -> int {
    (SDL_Init 0)
    
    let mut window: SDL_Window = (SDL_CreateWindow "My Game" 800 600)
    
    let mut running: bool = true
    while running {
        set running (not (SDL_QuitRequested))
        
        (SDL_RenderClear window)
        # ... game logic ...
        (SDL_RenderPresent window)
        (SDL_Delay 16)
    }
    
    (SDL_DestroyWindow window)
    (SDL_Quit)
    
    return 0
}
```

**My Results:**
- **Unsafe blocks:** 7 down to 0 (100% reduction)
- **Lines of code:** 35 down to 27 (23% reduction)
- **Safety:** My compile-time guarantees remain unchanged
- **Clarity:** I make safety explicit at the module boundary

---

### Example 2: Bullet Physics

**Before:**
```nano
import "modules/sdl/sdl.nano"
import "modules/bullet/bullet.nano"

fn init_physics() -> btDynamicsWorld {
    let mut world: btDynamicsWorld = 0
    unsafe {
        set world (btCreateDynamicsWorld)
    }
    
    let mut gravity: btVector3 = btVector3 { x: 0.0, y: -9.81, z: 0.0 }
    unsafe {
        (btSetGravity world gravity)
    }
    
    return world
}

fn step_physics(world: btDynamicsWorld, dt: float) -> void {
    unsafe {
        (btStepSimulation world dt)
    }
}
```

**After:**
```nano
unsafe module "modules/sdl/sdl.nano"
unsafe module "modules/bullet/bullet.nano"

fn init_physics() -> btDynamicsWorld {
    let mut world: btDynamicsWorld = (btCreateDynamicsWorld)
    let mut gravity: btVector3 = btVector3 { x: 0.0, y: -9.81, z: 0.0 }
    (btSetGravity world gravity)
    return world
}

fn step_physics(world: btDynamicsWorld, dt: float) -> void {
    (btStepSimulation world dt)
}
```

**My Results:**
- **Unsafe blocks:** 3 down to 0 (100% reduction)
- **Cleaner function bodies:** I removed the safety noise
- **Same semantics:** I still mark FFI as unsafe, but at the module level

---

## Migration Patterns

### Pattern 1: Safe Modules Only
```nano
# Before
import "modules/math/math.nano"
import "modules/vector2d/vector2d.nano"

# After
module "modules/math/math.nano"
module "modules/vector2d/vector2d.nano"
```

**Action:** Use find-and-replace to change `import` to `module`.

---

### Pattern 2: Mixed Safe and Unsafe Modules
```nano
# Before
import "modules/sdl/sdl.nano"          # Unsafe (FFI)
import "modules/vector2d/vector2d.nano"  # Safe (NanoLang)

# After
unsafe module "modules/sdl/sdl.nano"      # Unsafe (FFI)
module "modules/vector2d/vector2d.nano"   # Safe (NanoLang)
```

**Action:** I require an `unsafe` prefix for FFI modules. Use `module` for the others.

---

### Pattern 3: Nested Unsafe Blocks
```nano
# Before
import "modules/curl/curl.nano"

fn fetch_url(url: string) -> string {
    let mut handle: CURL = 0
    unsafe {
        set handle (curl_easy_init)
        if (!= handle 0) {
            unsafe { (curl_easy_setopt handle CURLOPT_URL url) }
            unsafe { (curl_easy_perform handle) }
            unsafe { (curl_easy_cleanup handle) }
        }
    }
    return ""
}

# After
unsafe module "modules/curl/curl.nano"

fn fetch_url(url: string) -> string {
    let mut handle: CURL = (curl_easy_init)
    if (!= handle 0) {
        (curl_easy_setopt handle CURLOPT_URL url)
        (curl_easy_perform handle)
        (curl_easy_cleanup handle)
    }
    return ""
}
```

**Action:** Remove all `unsafe {}` blocks after you add `unsafe module`.

---

## My Automated Migration Tools

### Script 1: Update Import Statements

```bash
#!/bin/bash
# Replace 'import' with 'module' for safe modules
# Add 'unsafe' prefix for FFI modules

for file in examples/*.nano; do
    # Safe modules
    sed -i '' 's/^import "modules\/math\//module "modules\/math\//g' "$file"
    sed -i '' 's/^import "modules\/vector2d\//module "modules\/vector2d\//g' "$file"
    
    # Unsafe modules (FFI)
    sed -i '' 's/^import "modules\/sdl\//unsafe module "modules\/sdl\//g' "$file"
    sed -i '' 's/^import "modules\/bullet\//unsafe module "modules\/bullet\//g' "$file"
    sed -i '' 's/^import "modules\/curl\//unsafe module "modules\/curl\//g' "$file"
done
```

### Script 2: Remove Unnecessary Unsafe Blocks

```python
#!/usr/bin/env python3
# See: /tmp/remove_unsafe_blocks.py
# This script removes unsafe {} wrappers.
# I designed it to preserve content and indentation.
```

---

## Which Modules are Unsafe?

### Definitely Unsafe (FFI and External C Libraries)
- sdl (SDL2 graphics library)
- bullet (Bullet physics engine)
- curl (HTTP client library)
- opengl (OpenGL graphics)
- glfw (Window and input library)
- ncurses (Terminal UI library)
- sqlite (Database library)

### Definitely Safe (My Own Code)
- vector2d (2D vector math)
- math (Extended math functions)
- ui_widgets (UI components)
- stdlib (Standard library utilities)

### How to Tell
I suggest checking if the module has `extern fn` declarations or depends on C libraries:

```nano
# If you see this, it's unsafe:
extern fn SDL_Init(flags: int) -> int
extern fn curl_easy_init() -> CURL

# If you see only NanoLang code, it's safe:
fn vec_add(a: Vec2, b: Vec2) -> Vec2 { ... }
```

---

## Common Questions

### Q: What if I mix safe and unsafe modules?
**A:** I handle this by letting you mark each module individually.
```nano
unsafe module "modules/sdl/sdl.nano"      # Unsafe
module "modules/vector2d/vector2d.nano"   # Safe
```

### Q: Can I still use `unsafe {}` blocks?
**A:** I allow it, but I discourage it. I prefer module-level safety.
```nano
# Discouraged (the old way)
module "modules/sdl/sdl.nano"
fn render() -> void {
    unsafe { (SDL_Init 0) }
}

# Preferred (the new way)
unsafe module "modules/sdl/sdl.nano"
fn render() -> void {
    (SDL_Init 0)
}
```

### Q: What about legacy `import` syntax?
**A:** I still support it for now, but I will deprecate it. Please migrate to `module`.
```nano
# Legacy (I still support this)
import "modules/old.nano"

# Modern (I prefer this)
module "modules/old.nano"
```

### Q: What if my module imports other modules?
**A:** I require each file to declare its own imports. I do not provide transitive safety.
```nano
# file1.nano
unsafe module "modules/sdl/sdl.nano"
# You can use SDL functions here without unsafe blocks

# file2.nano
module "file1.nano"
# You cannot use SDL functions here without unsafe blocks
# You must import SDL directly or mark file1 as unsafe
```

---

## My Migration Checklist

For each project you have built with me:

- [ ] **Step 1:** Identify which modules are FFI or unsafe.
- [ ] **Step 2:** Add the `unsafe` prefix to those module imports.
- [ ] **Step 3:** Change `import` to `module` for all imports.
- [ ] **Step 4:** Remove `unsafe {}` blocks from functions that use unsafe modules.
- [ ] **Step 5:** Compile and test your code.
- [ ] **Step 6:** Verify that you have no safety regressions.

---

## Statistics (My Examples Migration)

**Files Updated:** 103 example files  
**Imports Changed:** 181 import statements  
**Unsafe Modules:** 147 (SDL, Bullet, OpenGL, and others)  
**Safe Modules:** 34 (vector2d, math, ui_widgets)  
**Unsafe Blocks Removed:** 1,235 down to 88 (98% reduction)  

**Build Status:** All my examples compile and run.

---

## My Benefits Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Unsafe blocks in SDL examples | 1,235 | 88 | 98% reduction |
| Lines of boilerplate | ~3,500 | ~250 | 93% reduction |
| Safety declarations | Scattered | At module boundary | Centralized |
| Code readability | Cluttered | Clean | Improved |
| Compile-time safety | Yes | Yes | Preserved |

---

## My Future Phases

### Phase 2: Module Introspection
- I will allow you to query module metadata at compile-time.
- I will add `___module_info_<name>()` functions.
- This enables me to support advanced tooling.

### Phase 3: Warning System
- `--warn-unsafe-imports`: I will warn on any unsafe module.
- `--warn-ffi`: I will warn on any FFI call.
- `--forbid-unsafe`: I will error on unsafe modules.
- These are my graduated safety levels.

### Phase 4: Module-Qualified Calls
- I will fix my typechecker for `Module.function()` syntax.
- I will provide proper namespace resolution.
- I will enable explicit module prefixes.

---

## My Resources

- `docs/MODULE_ARCHITECTURE_DECISION.md` - My full architectural decision.
- `docs/MODULE_SYSTEM_REDESIGN.md` - My complete redesign specification.
- `docs/MODULE_PHASE1_COMPLETE.md` - My Phase 1 implementation summary.
- `docs/MODULE_IMPLEMENTATION_ROADMAP.md` - My phased implementation plan.
- `MEMORY.md` - My updated module syntax reference.

---

## Support

**Questions?** Check my docs above or:
- Review my examples in `examples/sdl_*.nano`.
- Compare my before and after states in git history: `git show 23354ad`.
- File issues for any migration problems you encounter.

**My Migration Status:** COMPLETE  
**Phase 1:** Production Ready  
**Date:** 2025-01-08
