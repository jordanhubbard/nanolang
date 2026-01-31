# Module System Architectural Redesign

**Status:** 🔴 **Proposal** - Requires architectural decision  
**Priority:** P4 - Critical for language maturity  
**Date:** 2025-01-08  
**Motivation:** Current module system feels "grafted on" rather than first-class

---

## Executive Summary

The current NanoLang module system works but feels bolted-on compared to languages like Python, Go, Ruby, and Elixir. This proposal redesigns modules as **first-class language citizens** with:

1. **Module-level safety annotations** (no more `unsafe {}` wrappers everywhere)
2. **Compile-time module introspection** (reflection for modules, not just structs)
3. **Graduated warning system** (`-Warn-unsafe-calls`, `-Warn-ffi`, etc.)
4. **Module namespaces as types** (callable, inspectable, composable)
5. **Clear distinction** between safe NanoLang modules and unsafe FFI modules

---

## Problem Statement

### Current Pain Points

#### 1. **Unsafe Blocks Everywhere**

**Current:**
```nano
import "modules/sdl/sdl.nano"

fn render_frame() -> int {
    unsafe { (SDL_Init SDL_INIT_VIDEO) }
    let window: Window = unsafe { (SDL_CreateWindow "Game" 0 0 800 600 0) }
    unsafe { (SDL_RenderPresent renderer) }
    unsafe { (SDL_Quit) }
    return 0
}
```

**Problems:**
- 4 `unsafe` blocks for routine SDL operations
- Visual noise obscures actual logic
- No way to mark entire module as unsafe
- Safe NanoLang wrappers still need `unsafe`

---

#### 2. **No Module Introspection**

**Current:** Modules are invisible at runtime
```nano
import "modules/vector2d/vector2d.nano"

// ❌ IMPOSSIBLE:
// - List exported functions
// - Check if module is safe/unsafe
// - Get module version
// - Inspect module dependencies
```

**Comparison - Elixir:**
```elixir
> Vector2D.__info__(:functions)
[add: 2, subtract: 2, dot: 2, magnitude: 1]

> Vector2D.__info__(:attributes)
[:vsn, :author, :safe]
```

**Comparison - Python:**
```python
import inspect
import vector2d

inspect.getmembers(vector2d, inspect.isfunction)
# [('add', <function>), ('subtract', <function>), ...]

vector2d.__dict__['__safe__']  # True or False
```

---

#### 3. **No Graduated Safety Controls**

**Current:** All-or-nothing
- Either allow all `unsafe`, or disallow all
- No warnings, only hard errors
- Can't distinguish FFI from safe-but-marked-unsafe code

**Desired:**
```bash
# Compilation modes
nanoc app.nano -o app                       # Default: allow unsafe
nanoc app.nano -o app --forbid-unsafe       # Hard error on any unsafe
nanoc app.nano -o app --warn-unsafe-calls   # Warning on unsafe calls
nanoc app.nano -o app --warn-ffi           # Warning only on FFI, not safe modules
```

---

#### 4. **Module Safety is Invisible**

**Current:** No way to tell if module is safe
```nano
import "modules/vector2d/vector2d.nano"     // Pure NanoLang - safe
import "modules/sdl/sdl.nano"                // FFI to C - unsafe

// Both look identical! User has no idea which is safe.
```

**Desired:** Explicit safety annotations
```nano
import "modules/vector2d/vector2d.nano"     // Compiler knows: safe
import unsafe "modules/sdl/sdl.nano"        // Explicit: unsafe FFI
```

---

## Comparison with Other Languages

### Python

**Module as First-Class Object:**
```python
import math

# Module introspection
dir(math)  # ['sqrt', 'sin', 'cos', ...]
math.__name__  # 'math'
math.__file__  # '/usr/lib/python3/math.py'
math.__dict__  # Full namespace

# Function metadata
import inspect
inspect.signature(math.sqrt)  # (x, /)
```

**What Works:**
- ✅ Modules are objects with properties
- ✅ Full introspection via `dir()`, `inspect`
- ✅ Dynamic attribute access

**What's Missing in NanoLang:**
- ❌ Modules aren't values (can't pass/inspect)
- ❌ No runtime metadata
- ❌ No programmatic discovery

---

### Go

**Package-Level Organization:**
```go
package myapp

import (
    "fmt"           // Standard library
    "unsafe"        // Explicitly unsafe operations
    "github.com/user/repo"  // External package
)

// Package metadata via godoc
// go doc fmt
// go list -json fmt
```

**What Works:**
- ✅ Clear package boundaries
- ✅ Explicit `import "unsafe"` for unsafe operations
- ✅ Compile-time package validation
- ✅ Tool support for package inspection

**What's Missing in NanoLang:**
- ❌ No package-level safety declaration
- ❌ No tooling for module inspection
- ❌ Modules not treated as compilation units

---

### Ruby

**Module Introspection:**
```ruby
require 'vector2d'

# List methods
Vector2D.methods(false)  # [:add, :subtract, :dot]

# Check if method exists
Vector2D.respond_to?(:add)  # true

# Module metadata
Vector2D.ancestors  # [Vector2D, Object, Kernel, BasicObject]
Vector2D.instance_methods  # [...all methods...]

# Safe vs unsafe marking
Vector2D.class_variable_get(:@@unsafe)  # true/false
```

**What Works:**
- ✅ Modules are objects
- ✅ Full runtime introspection
- ✅ Method discovery
- ✅ Custom metadata via class variables

---

### Elixir

**Module as Compile-Time Entity:**
```elixir
defmodule Vector2D do
  @moduledoc "2D vector mathematics"
  @vsn "1.0.0"
  @safe true  # Custom attribute

  def add(v1, v2), do: # ...
end

# Introspection
Vector2D.__info__(:functions)    # [add: 2, subtract: 2]
Vector2D.__info__(:attributes)   # [:vsn, :moduledoc, :safe]
Vector2D.module_info()           # Full module metadata

# Compile-time checks
@unsafe true
def call_c_function(), do: :crypto.hash(:sha256, "data")
```

**What Works:**
- ✅ Module attributes for metadata
- ✅ Compile-time introspection
- ✅ Clear module boundaries
- ✅ Custom safety annotations possible

**What's Missing in NanoLang:**
- ❌ No module attributes
- ❌ No `__info__` equivalent
- ❌ Can't annotate modules with metadata

---

## Proposed Design

### Goal: Modules as First-Class Citizens

**Principles:**
1. **Modules are values** - Can be passed, inspected, queried
2. **Safety is explicit** - Clear distinction between safe and unsafe
3. **Introspection built-in** - Reflection for modules, not just structs
4. **Backward compatible** - Existing code still works
5. **Progressive enhancement** - Warnings, not errors by default

---

### Feature 1: Module-Level Safety Annotations

#### Syntax

**Safe Module (Default):**
```nano
module vector2d {
    export fn add(v1: Vec2, v2: Vec2) -> Vec2 {
        return Vec2 { x: (+ v1.x v2.x), y: (+ v1.y v2.y) }
    }
}
```

**Unsafe Module (Explicit):**
```nano
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    extern fn SDL_Quit() -> void
    
    export fn init() -> bool {
        /* No unsafe block needed - entire module is unsafe */
        let result: int = (SDL_Init SDL_INIT_VIDEO)
        return (== result 0)
    }
}
```

**Benefits:**
- ✅ One annotation instead of dozens of `unsafe {}` blocks
- ✅ Clear visual indicator in module header
- ✅ Compiler tracks safety at module level
- ✅ Safe wrappers don't need `unsafe` blocks

---

#### Import Annotations

**Current (No Safety Info):**
```nano
import "modules/sdl/sdl.nano"
```

**Proposed (Explicit Safety):**
```nano
import safe "modules/vector2d/vector2d.nano"     // Compiler verifies
import unsafe "modules/sdl/sdl.nano"              // Required for unsafe modules

// Shorthand (inferred from module declaration)
import "modules/vector2d/vector2d.nano"  // Safe by default
```

**Compilation Modes:**
```bash
# Default: allow all
nanoc app.nano -o app

# Warn on unsafe imports
nanoc app.nano -o app --warn-unsafe-imports

# Warn on FFI calls (even from unsafe modules)
nanoc app.nano -o app --warn-ffi

# Error on any unsafe
nanoc app.nano -o app --forbid-unsafe
```

---

### Feature 2: Module Introspection

#### Module Metadata Struct

**Auto-Generated by Compiler:**
```nano
struct ModuleInfo {
    name: string,
    version: string,
    is_safe: bool,
    has_ffi: bool,
    exported_functions: array<FunctionInfo>,
    dependencies: array<string>,
    source_file: string
}

struct FunctionInfo {
    name: string,
    param_count: int,
    param_types: array<string>,
    return_type: string,
    is_exported: bool,
    is_extern: bool
}
```

---

#### Introspection API

**At Compile Time:**
```nano
import "modules/sdl/sdl.nano" as SDL

/* Compiler generates __module_info_SDL constant */
fn check_sdl_safety() -> bool {
    let info: ModuleInfo = (__module_info_SDL)
    
    if info.is_safe {
        (println "SDL module is safe")
    } else {
        (println "SDL module uses unsafe operations")
        (println (+ "  Has FFI: " (if info.has_ffi { "yes" } else { "no" })))
    }
    
    return info.is_safe
}
```

**Auto-Generated Functions:**
```nano
/* For module "sdl" */
extern fn __module_info_sdl() -> ModuleInfo
extern fn __module_exported_functions_sdl() -> array<FunctionInfo>
extern fn __module_has_function_sdl(name: string) -> bool
extern fn __module_dependencies_sdl() -> array<string>
```

---

#### Reflection Functions (Same Pattern as Structs)

**Compiler Auto-Generates:**
```c
/* For module "sdl" */
inline ModuleInfo __module_info_sdl(void) {
    return (ModuleInfo){
        .name = "sdl",
        .version = "1.0.0",
        .is_safe = 0,           /* unsafe module */
        .has_ffi = 1,           /* has extern functions */
        .exported_functions = /* ... */,
        .dependencies = /* ["sdl_helpers"] */,
        .source_file = "modules/sdl/sdl.nano"
    };
}

inline int __module_has_function_sdl(const char* name) {
    if (strcmp(name, "init") == 0) return 1;
    if (strcmp(name, "create_window") == 0) return 1;
    /* ... */
    return 0;
}
```

---

### Feature 3: Module Namespaces

#### Qualified Calls (Already Works)

**Current:**
```nano
import "modules/vector2d/vector2d.nano" as Vec

fn test() -> Vec2 {
    return (Vec.add v1 v2)  /* ✅ Works but typechecker treats as field access */
}
```

**Problem:** Parser/typechecker treats `Vec.add` as field access, not module-qualified call.

**Solution:** New parse node `PNODE_MODULE_QUALIFIED_CALL`

---

#### Module as Namespace Type

**Proposed:**
```nano
/* Module can be passed as value */
fn test_math_module(m: Module) -> int {
    if (__module_has_function m "sqrt") {
        (println "Module has sqrt function")
    } else {
        (println "Module does not have sqrt")
    }
    return 0
}

import "modules/vector2d/vector2d.nano" as Vec

fn main() -> int {
    let vec_module: Module = (get_module "vector2d")
    (test_math_module vec_module)
    return 0
}
```

**Type System Support:**
```nano
/* Module type */
type Module = opaque  /* Internally: pointer to ModuleInfo */

/* Built-in functions */
extern fn get_module(name: string) -> Module
extern fn module_info(m: Module) -> ModuleInfo
extern fn module_call(m: Module, func_name: string, args: array<any>) -> any  /* Future */
```

---

### Feature 4: Graduated Warning System

#### Compiler Flags

**Safety Levels:**
```bash
# Level 0: Permissive (default)
nanoc app.nano -o app
# - Allows all unsafe blocks
# - Allows unsafe modules
# - No warnings

# Level 1: Awareness
nanoc app.nano -o app --warn-unsafe-imports
# - Warns when importing unsafe modules
# - Shows: "Warning: Importing unsafe module 'sdl'"

# Level 2: Audit
nanoc app.nano -o app --warn-unsafe-calls
# - Warns on every call to function from unsafe module
# - Shows: "Warning: Calling unsafe function SDL_Init from module 'sdl'"

# Level 3: FFI Only
nanoc app.nano -o app --warn-ffi
# - Warns only on actual FFI calls (extern functions)
# - Ignores safe wrappers in unsafe modules

# Level 4: Strict
nanoc app.nano -o app --forbid-unsafe
# - Hard error on any unsafe module import
# - Hard error on any unsafe block
# - Only allows certified-safe modules
```

---

#### Per-Module Overrides

**In Code:**
```nano
@allow_unsafe
import "modules/sdl/sdl.nano"  /* Won't warn even with --warn-unsafe-imports */

@trusted_wrapper
fn render() -> void {
    /* Calls to SDL functions won't warn because function is marked trusted */
    (SDL_RenderPresent renderer)
}
```

**In module.json:**
```json
{
  "name": "sdl_safe_wrappers",
  "safety": "trusted",
  "wraps_unsafe": ["sdl"],
  "justification": "Provides safe abstractions over SDL FFI"
}
```

---

### Feature 5: Module Metadata in module.json

#### Extended module.json

**Current:**
```json
{
  "name": "sdl",
  "version": "1.0.0",
  "c_sources": [],
  "pkg_config": ["sdl2"]
}
```

**Proposed:**
```json
{
  "name": "sdl",
  "version": "1.0.0",
  "author": "NanoLang Team",
  "license": "MIT",
  "description": "SDL2 windowing and rendering",
  
  "safety": {
    "level": "unsafe",
    "has_ffi": true,
    "ffi_functions": ["SDL_Init", "SDL_Quit", "SDL_CreateWindow"],
    "safe_wrappers": ["init", "quit", "create_window"],
    "audit_date": "2025-01-01",
    "auditor": "security-team"
  },
  
  "exports": {
    "functions": ["init", "quit", "create_window"],
    "types": ["Window", "Renderer", "Surface"]
  },
  
  "dependencies": ["sdl_helpers"],
  "c_sources": [],
  "pkg_config": ["sdl2"]
}
```

---

## Implementation Plan

### Phase 1: Module Safety Annotations (1-2 weeks)

**Goals:**
- Add `unsafe module` syntax
- Eliminate need for `unsafe {}` blocks inside unsafe modules
- Add `import unsafe` syntax

**Changes:**
- Parser: Recognize `unsafe module { ... }`
- Environment: Track module safety level
- Typechecker: Auto-allow FFI calls in unsafe modules
- Transpiler: Generate safety metadata

**Files:**
- `src/parser_iterative.c` - Parse `unsafe module`
- `src/typechecker.c` - Module safety checking
- `src/nanolang.h` - Add `is_unsafe` to module structs

---

### Phase 2: Module Introspection (1-2 weeks)

**Goals:**
- Auto-generate `__module_info_*` functions
- Add `ModuleInfo` and `FunctionInfo` structs to runtime
- Provide query functions

**Changes:**
- Transpiler: Generate module metadata functions (like struct reflection)
- Runtime: Add `ModuleInfo` struct type
- Compiler: Collect exported function info

**Files:**
- `src/transpiler.c` - `generate_module_metadata()`
- `src/runtime/` - Add `module_info.c`
- `src_nano/typecheck.nano` - Module type support

---

### Phase 3: Warning System (1 week)

**Goals:**
- Add compiler flags for safety levels
- Implement per-module warnings
- Add diagnostic output

**Changes:**
- Main: Parse new CLI flags
- Typechecker: Track call sites for warnings
- Diagnostics: Add warning categories

**Files:**
- `src/main.c` - Add CLI flags
- `src/typechecker.c` - Emit warnings
- `src/diagnostics.c` - Warning infrastructure

---

### Phase 4: Module-Qualified Calls (Already Tracked)

**Goals:**
- Fix `Module.function()` treated as field access
- Add proper type resolution

**Changes:**
- Parser: `PNODE_MODULE_QUALIFIED_CALL`
- Typechecker: Resolve function in module namespace

**Issue:** `nanolang-3oda` (already created)

---

### Phase 5: Module as First-Class Value (Future)

**Goals:**
- `Module` opaque type
- Pass modules as values
- Dynamic module queries

**This is advanced** - requires significant runtime support.

---

## Migration Path

### Existing Code (Keep Working)

**Current:**
```nano
import "modules/sdl/sdl.nano"

fn main() -> int {
    unsafe { (SDL_Init SDL_INIT_VIDEO) }
    return 0
}
```

**Still Works!** No breaking changes.

---

### Upgrade to New System

**Step 1: Mark module as unsafe**
```nano
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    /* ... */
}
```

**Step 2: Remove unsafe blocks**
```nano
import "modules/sdl/sdl.nano"

fn main() -> int {
    (SDL_Init SDL_INIT_VIDEO)  /* No unsafe block needed! */
    return 0
}
```

**Step 3: Gradual strictness**
```bash
# Start with warnings
nanoc app.nano --warn-unsafe-imports

# Move to stricter checking
nanoc app.nano --warn-ffi

# Eventually: full safety
nanoc app.nano --forbid-unsafe  # Forces you to use safe wrappers
```

---

## Examples

### Example 1: Pure NanoLang Module (Safe)

**modules/vector2d/vector2d.nano:**
```nano
safe module vector2d {  /* Explicit: this is safe */
    export struct Vec2 {
        x: float,
        y: float
    }
    
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
import safe "modules/vector2d/vector2d.nano" as Vec

fn main() -> int {
    let v1: Vec.Vec2 = Vec.Vec2 { x: 1.0, y: 2.0 }
    let v2: Vec.Vec2 = Vec.Vec2 { x: 3.0, y: 4.0 }
    let result: Vec.Vec2 = (Vec.add v1 v2)
    
    /* Query module safety at compile time */
    let info: ModuleInfo = (__module_info_vector2d)
    assert info.is_safe  /* Compile-time check */
    
    return 0
}
```

---

### Example 2: FFI Module with Safe Wrappers

**modules/sdl/sdl.nano:**
```nano
unsafe module sdl {  /* Entire module is unsafe */
    /* Raw FFI - no unsafe blocks needed */
    extern fn SDL_Init(flags: int) -> int
    extern fn SDL_Quit() -> void
    extern fn SDL_CreateWindow(title: string, x: int, y: int, w: int, h: int, flags: int) -> int
    
    /* Safe wrapper - still in unsafe module, but provides safety */
    export fn init() -> bool {
        let result: int = (SDL_Init SDL_INIT_VIDEO)
        return (== result 0)
    }
    
    export fn quit() -> void {
        (SDL_Quit)
    }
    
    export fn create_window(title: string, w: int, h: int) -> int {
        return (SDL_CreateWindow title SDL_WINDOWPOS_CENTERED SDL_WINDOWPOS_CENTERED w h 0)
    }
}
```

**Usage:**
```nano
import unsafe "modules/sdl/sdl.nano" as SDL

fn main() -> int {
    /* Safe wrappers - no visual noise */
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

**With Warnings:**
```bash
$ nanoc game.nano --warn-unsafe-imports
Warning: Importing unsafe module 'sdl' at line 1
  This module contains FFI calls to external C library

$ nanoc game.nano --warn-ffi
Warning: Function SDL_Init calls FFI at modules/sdl/sdl.nano:8
Warning: Function SDL_Quit calls FFI at modules/sdl/sdl.nano:9
Warning: Function SDL_CreateWindow calls FFI at modules/sdl/sdl.nano:10
```

---

### Example 3: Module Introspection

```nano
import "modules/sdl/sdl.nano" as SDL
import "modules/vector2d/vector2d.nano" as Vec

fn audit_modules() -> void {
    (println "=== Module Safety Audit ===")
    
    /* Check SDL */
    let sdl_info: ModuleInfo = (__module_info_sdl)
    (print "SDL: ")
    if sdl_info.is_safe {
        (println "SAFE")
    } else {
        (println "UNSAFE")
        if sdl_info.has_ffi {
            (println "  - Contains FFI calls")
        } else {
            (print "")
        }
    }
    
    /* Check vector2d */
    let vec_info: ModuleInfo = (__module_info_vector2d)
    (print "vector2d: ")
    if vec_info.is_safe {
        (println "SAFE")
    } else {
        (println "UNSAFE")
    }
    
    /* List exported functions */
    (println "\nSDL Exported Functions:")
    let sdl_funcs: array<FunctionInfo> = (__module_exported_functions_sdl)
    let mut i: int = 0
    while (< i (array_length sdl_funcs)) {
        let func: FunctionInfo = (at sdl_funcs i)
        (print "  - ")
        (print func.name)
        if func.is_extern {
            (println " (FFI)")
        } else {
            (println " (safe wrapper)")
        }
        set i (+ i 1)
    }
}

shadow audit_modules {
    (audit_modules)  /* Run audit in tests */
}
```

**Output:**
```
=== Module Safety Audit ===
SDL: UNSAFE
  - Contains FFI calls
vector2d: SAFE

SDL Exported Functions:
  - init (safe wrapper)
  - quit (safe wrapper)
  - create_window (safe wrapper)
  - SDL_Init (FFI)
  - SDL_Quit (FFI)
  - SDL_CreateWindow (FFI)
```

---

## Benefits

### For Users

1. **Less Visual Noise:** One `unsafe module` instead of dozens of `unsafe {}` blocks
2. **Clear Safety Boundaries:** Instantly see if module is safe or unsafe
3. **Graduated Warnings:** Choose your safety level with compiler flags
4. **Module Discovery:** Introspect modules like you can introspect structs
5. **Better Error Messages:** "Function from unsafe module 'sdl'" vs vague errors

### For Language

1. **Competitive Feature:** Matches Python/Go/Elixir module systems
2. **Safety Story:** Clear path from unsafe FFI to safe wrappers
3. **Tooling Support:** Enable `nalang doc`, `nalang list-modules`, etc.
4. **Ecosystem Growth:** Easy to audit third-party modules
5. **First-Class Modules:** Foundation for advanced metaprogramming

### For Ecosystem

1. **Module Marketplace:** Safety ratings visible upfront
2. **Certification:** Mark modules as audited/trusted
3. **Dependency Analysis:** See full safety impact of dependencies
4. **Documentation Generation:** Auto-generate docs from module metadata
5. **Testing Infrastructure:** Query modules to generate tests

---

## Open Questions

### 1. Module Declarations

**Option A: Module wrapper (explicit)**
```nano
unsafe module sdl {
    /* All functions here */
}
```

**Option B: Module attribute (lightweight)**
```nano
@unsafe
/* Functions follow */
```

**Recommendation:** Option A - clearer scope

---

### 2. Backward Compatibility

**Question:** Should we break existing code?

**Options:**
- A) Keep `unsafe {}` blocks, add module-level as option
- B) Deprecate `unsafe {}`, require migration
- C) Auto-detect: if module has `extern`, it's unsafe

**Recommendation:** Option C with warnings in transition period

---

### 3. Module Type System

**Question:** Should `Module` be a real runtime type?

**Pros:**
- Pass modules as function arguments
- Store modules in arrays
- Dynamic module loading

**Cons:**
- Requires runtime support
- Complicates type system
- May not fit compile-time philosophy

**Recommendation:** Start with compile-time only, add runtime later if needed

---

## Related Issues

- `nanolang-3oda` - Module-qualified calls (Parser fix)
- `nanolang-qlv2` - 100% self-hosting (Typechecker improvements)
- New: Module safety annotations (This proposal)
- New: Module introspection system (This proposal)

---

## References

- Current: `docs/MODULE_SYSTEM.md`
- Current: `docs/EXTERN_FFI.md`
- Current: `MEMORY.md` - Unsafe blocks section
- Related: `docs/REFLECTION_API.md` - Struct introspection
- Inspiration: Python `inspect` module
- Inspiration: Elixir module attributes
- Inspiration: Go `import "unsafe"`
- Inspiration: Ruby module introspection

---

## Decision Required

**This is an architectural decision that needs project owner approval.**

**Questions for Decision:**
1. ✅ Should we proceed with `unsafe module` syntax?
2. ✅ Should we generate module introspection functions?
3. ✅ Should we add warning flags to compiler?
4. ⏳ Should `Module` be a runtime type or compile-time only?
5. ⏳ How aggressive should be backward compatibility break?

---

**Status:** 🔴 **AWAITING ARCHITECTURAL DECISION**  
**Author:** AI Assistant (Claude Sonnet 4.5)  
**Date:** 2025-01-08  
**Estimated Implementation:** 4-6 weeks for full redesign
