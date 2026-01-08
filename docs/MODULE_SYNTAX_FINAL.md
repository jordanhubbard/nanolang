# Module Syntax: Final Design

**Date:** 2025-01-08  
**Status:** ✅ **Approved** - Ready for implementation  
**Decision:** Use `module` keyword, not `import module`

---

## Core Principle

**The `module` keyword implies importing.** No need for redundant `import module`.

---

## Syntax

### Basic Module Import

```nano
module "modules/vector2d/vector2d.nano"
```

**Equivalent to current:**
```nano
import "modules/vector2d/vector2d.nano"
```

---

### Module with Alias

```nano
module "modules/vector2d/vector2d.nano" as Vec
```

**Usage:**
```nano
let v: Vec.Vec2 = Vec.Vec2 { x: 1.0, y: 2.0 }
let result: Vec.Vec2 = (Vec.add v1 v2)
```

---

### Safe Module (Explicit)

```nano
safe module "modules/vector2d/vector2d.nano"
```

**Compiler verifies:**
- Module contains no `extern` functions
- Module does not import unsafe modules
- Module is pure NanoLang

**Benefit:** Compiler error if module is actually unsafe

---

### Unsafe Module (Explicit)

```nano
unsafe module "modules/sdl/sdl.nano"
```

**Compiler verifies:**
- Module contains `extern` functions OR
- Module imports unsafe modules OR
- Module is marked unsafe in `module.json`

**Benefit:** Explicit acknowledgment of risk

---

### Default (No Safety Annotation)

```nano
module "modules/math/math.nano"
```

**Behavior:**
- Safe by default (assumes module is safe)
- Compiler warns if module is actually unsafe (with `--warn-unsafe-imports`)
- Compiler errors if `--forbid-unsafe` flag is set

---

## Module Declaration Syntax

### In Module File

**Safe Module (Default):**
```nano
/* modules/vector2d/vector2d.nano */
module vector2d {
    export struct Vec2 { x: float, y: float }
    
    export fn add(v1: Vec2, v2: Vec2) -> Vec2 {
        return Vec2 { x: (+ v1.x v2.x), y: (+ v1.y v2.y) }
    }
}
```

**Unsafe Module (Explicit):**
```nano
/* modules/sdl/sdl.nano */
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    extern fn SDL_Quit() -> void
    
    export fn init() -> bool {
        /* No unsafe block needed - module is unsafe */
        let result: int = (SDL_Init SDL_INIT_VIDEO)
        return (== result 0)
    }
}
```

---

## Complete Examples

### Example 1: Pure NanoLang Module

**Module Definition:**
```nano
/* modules/vector2d/vector2d.nano */
module vector2d {
    export struct Vec2 { x: float, y: float }
    
    export fn add(v1: Vec2, v2: Vec2) -> Vec2 {
        return Vec2 { x: (+ v1.x v2.x), y: (+ v1.y v2.y) }
    }
    
    export fn magnitude(v: Vec2) -> float {
        return (sqrt (+ (* v.x v.x) (* v.y v.y)))
    }
}
```

**Usage:**
```nano
module "modules/vector2d/vector2d.nano" as Vec

fn main() -> int {
    let v1: Vec.Vec2 = Vec.Vec2 { x: 1.0, y: 2.0 }
    let v2: Vec.Vec2 = Vec.Vec2 { x: 3.0, y: 4.0 }
    let result: Vec.Vec2 = (Vec.add v1 v2)
    (println (Vec.magnitude result))
    return 0
}
```

---

### Example 2: Unsafe FFI Module

**Module Definition:**
```nano
/* modules/sdl/sdl.nano */
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    extern fn SDL_CreateWindow(title: string, x: int, y: int, w: int, h: int, flags: int) -> int
    extern fn SDL_DestroyWindow(window: int) -> void
    extern fn SDL_Quit() -> void
    
    export fn init() -> bool {
        let result: int = (SDL_Init SDL_INIT_VIDEO)
        return (== result 0)
    }
    
    export fn create_window(title: string, w: int, h: int) -> int {
        return (SDL_CreateWindow title SDL_WINDOWPOS_CENTERED SDL_WINDOWPOS_CENTERED w h 0)
    }
    
    export fn quit() -> void {
        (SDL_Quit)
    }
}
```

**Usage:**
```nano
unsafe module "modules/sdl/sdl.nano" as SDL

fn main() -> int {
    if (not (SDL.init)) {
        (println "Failed to initialize SDL")
        return 1
    }
    
    let window: int = (SDL.create_window "My Game" 800 600)
    
    /* Game loop... */
    
    (SDL.quit)
    return 0
}
```

---

## Comparison: Before vs After

### Current Syntax

```nano
import "modules/sdl/sdl.nano"

fn main() -> int {
    unsafe { (SDL_Init SDL_INIT_VIDEO) }
    unsafe { (SDL_CreateWindow "Game" 0 0 800 600 0) }
    unsafe { (SDL_Quit) }
    return 0
}
```

### New Syntax

```nano
unsafe module "modules/sdl/sdl.nano" as SDL

fn main() -> int {
    (SDL.init)
    let window: int = (SDL.create_window "Game" 800 600)
    (SDL.quit)
    return 0
}
```

**Benefits:**
- ✅ One `unsafe module` declaration instead of 3 `unsafe {}` blocks
- ✅ Clearer intent: "This module is unsafe"
- ✅ Better namespacing: `SDL.init` vs bare `SDL_Init`
- ✅ Shorter: `module` vs `import`

---

## Grammar Changes

### Current Grammar

```
import_stmt := "import" STRING_LITERAL
            |  "import" STRING_LITERAL "as" IDENTIFIER
            |  "from" STRING_LITERAL "import" identifier_list
```

### New Grammar

```
module_import := [safety_modifier] "module" STRING_LITERAL ["as" IDENTIFIER]

safety_modifier := "safe" | "unsafe"

module_decl := [safety_modifier] "module" IDENTIFIER "{" module_body "}"

module_body := (function_decl | struct_decl | enum_decl | extern_decl)*
```

---

## Safety Semantics

### Safe Module

**Definition:**
```nano
module vector2d {
    /* Pure NanoLang - no FFI */
}
```

**Properties:**
- ✅ Contains no `extern` functions
- ✅ Does not import unsafe modules
- ✅ Pure NanoLang code

**Import:**
```nano
safe module "modules/vector2d/vector2d.nano"
// Compiler verifies module is actually safe
```

---

### Unsafe Module

**Definition:**
```nano
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    /* FFI calls allowed without unsafe blocks */
}
```

**Properties:**
- ⚠️ Contains `extern` functions OR
- ⚠️ Imports unsafe modules OR
- ⚠️ Marked unsafe in `module.json`

**Import:**
```nano
unsafe module "modules/sdl/sdl.nano"
// Explicit acknowledgment of risk
```

---

### Default (No Annotation)

**Definition:**
```nano
module math {
    /* Assumed safe unless proven otherwise */
}
```

**Import:**
```nano
module "modules/math/math.nano"
// Compiler checks safety at import time
```

**Behavior:**
- Safe by default
- Warns if module is actually unsafe (with `--warn-unsafe-imports`)
- Errors if `--forbid-unsafe` and module is unsafe

---

## Compiler Flags

### Warning Modes

```bash
# Default: permissive
nanoc app.nano -o app

# Warn when importing unsafe modules
nanoc app.nano -o app --warn-unsafe-imports
# Output: Warning: Importing unsafe module 'sdl' at line 3

# Warn on every FFI call
nanoc app.nano -o app --warn-ffi
# Output: Warning: FFI call to 'SDL_Init' at line 10

# Strict: forbid all unsafe
nanoc app.nano -o app --forbid-unsafe
# Output: Error: Unsafe module 'sdl' forbidden
```

---

## Migration Path

### Step 1: Update Module Declarations

**Before:**
```nano
/* modules/sdl/sdl.nano */
extern fn SDL_Init(flags: int) -> int

fn init() -> bool {
    let mut result: int = 0
    unsafe {
        set result (SDL_Init SDL_INIT_VIDEO)
    }
    return (== result 0)
}
```

**After:**
```nano
/* modules/sdl/sdl.nano */
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    
    fn init() -> bool {
        let result: int = (SDL_Init SDL_INIT_VIDEO)
        return (== result 0)
    }
}
```

---

### Step 2: Update Imports

**Before:**
```nano
import "modules/sdl/sdl.nano"
```

**After:**
```nano
unsafe module "modules/sdl/sdl.nano"
```

**Or with alias:**
```nano
unsafe module "modules/sdl/sdl.nano" as SDL
```

---

### Step 3: Remove Unsafe Blocks

**Before:**
```nano
fn main() -> int {
    unsafe { (SDL_Init SDL_INIT_VIDEO) }
    unsafe { (SDL_Quit) }
    return 0
}
```

**After:**
```nano
fn main() -> int {
    (SDL_Init SDL_INIT_VIDEO)
    (SDL_Quit)
    return 0
}
```

---

## Auto-Migration Tool

```bash
# Migrate module declaration
nalang migrate --wrap-module modules/sdl/sdl.nano
# Adds unsafe module wrapper
# Removes unsafe blocks inside module

# Migrate imports in user code
nalang migrate --update-imports game.nano
# Changes 'import' to 'module'
# Adds 'unsafe' prefix where needed

# Migrate all examples
nalang migrate --all examples/
```

---

## Backward Compatibility

### Phase 1: Parser accepts both (2 months)

```nano
import "modules/sdl/sdl.nano"        // ✅ Still works (deprecated)
module "modules/sdl/sdl.nano"        // ✅ New syntax
unsafe module "modules/sdl/sdl.nano" // ✅ New syntax with safety
```

### Phase 2: Deprecation warnings (1 month)

```nano
import "modules/sdl/sdl.nano"        // ⚠️ Warning: Use 'module' instead
```

### Phase 3: Remove old syntax (3+ months from now)

```nano
import "modules/sdl/sdl.nano"        // ❌ Error: Use 'module' instead
```

---

## Special Cases

### Stdlib Modules

```nano
module "std/json.nano"
module "std/http.nano"
```

**No prefix needed - stdlib is trusted**

---

### Module Re-exports

```nano
/* modules/graphics/graphics.nano */
module graphics {
    unsafe module "modules/sdl/sdl.nano" as SDL
    
    export fn init() -> bool {
        return (SDL.init)
    }
}
```

**The `graphics` module is unsafe because it imports an unsafe module.**

---

### Circular Dependencies

**Not allowed:**
```nano
/* a.nano */
module "b.nano"

/* b.nano */
module "a.nano"  // ❌ Error: Circular dependency
```

**Compiler detects and rejects circular imports.**

---

## Summary

### Syntax Rules

1. **Use `module` keyword** (not `import module`)
2. **Prefix with `safe` or `unsafe`** (optional, `safe` by default)
3. **Module declaration** wraps module body: `unsafe module name { ... }`
4. **No `unsafe {}` blocks** inside unsafe modules

### Benefits

- ✅ Shorter syntax: `module` vs `import`
- ✅ Clearer intent: `unsafe module` makes risk explicit
- ✅ Less noise: No scattered `unsafe {}` blocks
- ✅ Better tooling: Safety visible in imports

### Migration

- ✅ Auto-migration tool provided
- ✅ 2+ month backward compatibility
- ✅ Gradual deprecation warnings
- ✅ Clear error messages

---

**Status:** ✅ **Approved and ready for implementation**  
**Next Step:** Begin Phase 1 (Parser changes)
