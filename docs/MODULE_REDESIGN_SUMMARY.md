# Module System Redesign: Session Summary

**Date:** 2025-01-08  
**Status:** ✅ **Approved and Tracked**  
**Duration:** ~2 hours of architectural work

---

## What We Accomplished

### 1. **Comprehensive Architectural Analysis**

Created 6 detailed documents (4,100+ lines):
- **MODULE_ARCHITECTURE_DECISION.md** - Executive summary with decision matrix
- **MODULE_SYSTEM_REDESIGN.md** - Full technical specification (1,400 lines)
- **MODULE_SYSTEMS_COMPARISON.md** - Python/Go/Ruby/Elixir analysis (900 lines)
- **MODULE_IMPLEMENTATION_ROADMAP.md** - Phase-by-phase plan (1,000 lines)
- **MODULE_BEFORE_AFTER.md** - Visual examples showing 98% reduction in noise
- **MODULE_SYNTAX_FINAL.md** - Approved syntax specification

---

### 2. **Syntax Decision**

**Approved:** Use `module` keyword (not `import module`)

```nano
// Module import
unsafe module "modules/sdl/sdl.nano" as SDL

// Module declaration
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    // No unsafe blocks needed!
}
```

**Key Benefits:**
- ✅ Shorter: `module` vs `import module`
- ✅ Clearer: module keyword implies import
- ✅ Consistent: safety as prefix modifier

---

### 3. **Issues Created (4 phases tracked)**

| Issue | Title | Priority | Time | Status |
|-------|-------|----------|------|--------|
| **nanolang-dc8u** | Phase 1: Module-level safety | P4 Critical | 1-2 weeks | Ready |
| **nanolang-zqke** | Phase 2: Module introspection | P4 Critical | 1-2 weeks | Ready |
| **nanolang-rkc3** | Phase 3: Warning system | P3 High | 1 week | Ready |
| **nanolang-asqo** | Phase 4: Module-qualified calls | P2 Medium | 1 week | Ready |

**Total Timeline:** 4-6 weeks for all phases

---

## Approved Features

### Feature 1: Module-Level Safety (Phase 1)

**Before (45 unsafe blocks):**
```nano
import "modules/sdl/sdl.nano"

fn main() -> int {
    unsafe { (SDL_Init SDL_INIT_VIDEO) }
    unsafe { (SDL_CreateWindow "Game" 0 0 800 600 0) }
    unsafe { (SDL_RenderPresent renderer) }
    unsafe { (SDL_Quit) }
    return 0
}
```

**After (1 annotation):**
```nano
unsafe module "modules/sdl/sdl.nano" as SDL

fn main() -> int {
    (SDL.init)
    let window: int = (SDL.create_window "Game" 800 600)
    (SDL.quit)
    return 0
}
```

**Impact:** 98% reduction in unsafe block noise

---

### Feature 2: Module Introspection (Phase 2)

**Auto-generated functions:**
```c
static inline ModuleInfo ___module_info_sdl(void);
static inline int ___module_has_function_sdl(const char* name);
```

**Usage:**
```nano
extern fn ___module_info_sdl() -> ModuleInfo

let info: ModuleInfo = (___module_info_sdl)
if info.has_ffi {
    (println "SDL contains FFI - use with caution")
}
```

**Same pattern as struct reflection!**

---

### Feature 3: Warning System (Phase 3)

```bash
# Level 1: Warn on unsafe imports
nanoc game.nano --warn-unsafe-imports

# Level 2: Warn on all unsafe calls
nanoc game.nano --warn-unsafe-calls

# Level 3: Warn on FFI only
nanoc game.nano --warn-ffi

# Level 4: Strict mode
nanoc game.nano --forbid-unsafe
```

**Users choose safety level**

---

### Feature 4: Module-Qualified Calls (Phase 4)

**Fix:** `Module.function()` parsed correctly

```nano
module "modules/vector2d/vector2d.nano" as Vec

let result: Vec2 = (Vec.add v1 v2)  // ✅ Now works!
```

**Currently broken:** Treated as field access

---

## Implementation Plan

### Phase 1: Module-Level Safety (1-2 weeks) 🎯 **START HERE**

**Changes:**
1. Parser: Recognize `unsafe module { ... }` and `module "path"`
2. Environment: Track module safety in AST
3. Typechecker: Auto-allow FFI in unsafe modules
4. Transpiler: Generate safety metadata

**Files:**
- `src/parser_iterative.c`
- `src/nanolang.h`
- `src/typechecker.c`
- `src/transpiler_iterative_v3_twopass.c`

**Deliverable:** Eliminate 45 unsafe blocks → 1 annotation

---

### Phase 2: Module Introspection (1-2 weeks)

**Changes:**
1. Transpiler: Generate `___module_info_*()` functions
2. Runtime: Add `ModuleInfo` struct
3. Compiler: Collect exported function metadata

**Files:**
- `src/transpiler_iterative_v3_twopass.c`
- `src/runtime/module_info.h` (new)
- `src/runtime/module_info.c` (new)

**Deliverable:** Query modules like structs

---

### Phase 3: Warning System (1 week)

**Changes:**
1. CLI: Parse `--warn-unsafe-imports`, `--warn-ffi`, `--forbid-unsafe`
2. Typechecker: Emit warnings
3. Diagnostics: Clear messages

**Files:**
- `src/main.c`
- `src/typechecker.c`

**Deliverable:** 4 safety levels

---

### Phase 4: Module-Qualified Calls (1 week)

**Changes:**
1. Parser: New AST node `PNODE_MODULE_QUALIFIED_CALL`
2. Typechecker: Resolve in module namespace
3. Transpiler: Generate correct C calls

**Files:**
- `src/parser_iterative.c`
- `src/typechecker.c`
- `src/transpiler_iterative_v3_twopass.c`

**Deliverable:** `Module.function()` works

---

## Key Metrics

### Before Redesign

| Metric | Value |
|--------|-------|
| Unsafe blocks (SDL examples) | 45 |
| Module introspection | None |
| Safety visibility | Hidden |
| Module.function() | Broken |

### After Redesign

| Metric | Value |
|--------|-------|
| Unsafe blocks (SDL examples) | 1 |
| Module introspection | 5+ functions per module |
| Safety visibility | Explicit at import |
| Module.function() | Working |

**Result:** 98% reduction in unsafe block noise, full introspection, clear safety model

---

## Comparison with Other Languages

| Feature | Python | Go | Elixir | NanoLang (Current) | NanoLang (Proposed) |
|---------|--------|----|----- |-------------------|---------------------|
| **Module introspection** | ✅ `dir()` | ✅ `go doc` | ✅ `__info__` | ❌ | ✅ `___module_info_*()` |
| **Safety annotation** | ❌ | ✅ `import "unsafe"` | ✅ Custom attrs | ❌ | ✅ `unsafe module` |
| **Warning system** | ⚠️ Linters | ✅ `go vet` | ⚠️ Custom | ❌ | ✅ `--warn-ffi` |
| **Qualified calls** | ✅ | ✅ | ✅ | ❌ | ✅ |

**Conclusion:** Proposed system brings NanoLang to parity with mature languages

---

## Migration Strategy

### Backward Compatibility Timeline

**Phase 1a (Weeks 1-2):** Parser accepts both syntaxes
```nano
import "modules/sdl/sdl.nano"        // ✅ Still works
module "modules/sdl/sdl.nano"        // ✅ New syntax
unsafe module "modules/sdl/sdl.nano" // ✅ New syntax
```

**Phase 1b (Weeks 3-4):** Typechecker uses module safety
```nano
import "modules/sdl/sdl.nano"        // ✅ Works with deprecation warning
```

**Phase 1c (Month 2+):** Old syntax deprecated
```nano
import "modules/sdl/sdl.nano"        // ⚠️ Warning: Use 'module' instead
```

**Phase 1d (Month 3+):** Old syntax removed
```nano
import "modules/sdl/sdl.nano"        // ❌ Error: Use 'module' instead
```

**Minimum transition period:** 2 months

---

### Auto-Migration Tool

```bash
# Migrate module declarations
nalang migrate --wrap-module modules/sdl/sdl.nano
# Output: Added unsafe module wrapper, removed 4 unsafe blocks

# Migrate imports
nalang migrate --update-imports game.nano
# Output: Updated 3 imports to new syntax

# Migrate all examples
nalang migrate --all examples/
# Output: Updated 15 files, removed 45 unsafe blocks
```

---

## Next Steps

### Immediate (This Week)

**Option A: Start Phase 1 Implementation**
- Begin parser changes for `module` keyword
- Add safety tracking to Environment
- Estimated: 1-2 weeks

**Option B: Review and Refine**
- Read all documentation
- Provide feedback on syntax/approach
- Adjust plan based on feedback

**Option C: Defer**
- Archive this work for later
- Focus on self-hosting or other priorities

---

### Recommended: Start Phase 1

**Why:**
- Highest impact (eliminates 98% of unsafe noise)
- Foundation for Phases 2-4
- Independent value (ship even if later phases deferred)
- Clear scope (1-2 weeks)

**What to implement:**
1. ✅ Parser: `unsafe module name { ... }`
2. ✅ Parser: `module "path"` with optional `unsafe` prefix
3. ✅ Environment: Track module safety
4. ✅ Typechecker: Auto-allow FFI in unsafe modules
5. ✅ Tests: Backward compatibility + new syntax

**Deliverable:** SDL examples go from 45 `unsafe {}` blocks to 1 `unsafe module` annotation

---

## Files for Review

### Start Here
1. **docs/MODULE_ARCHITECTURE_DECISION.md** - Executive summary (20 min read)
2. **docs/MODULE_SYNTAX_FINAL.md** - Approved syntax (10 min read)

### For Details
3. **docs/MODULE_BEFORE_AFTER.md** - Visual examples (10 min read)
4. **docs/MODULE_SYSTEM_REDESIGN.md** - Full spec (1 hour read)
5. **docs/MODULE_SYSTEMS_COMPARISON.md** - Language comparison (30 min read)
6. **docs/MODULE_IMPLEMENTATION_ROADMAP.md** - Detailed plan (45 min read)

---

## Issue Tracker

All phases tracked in Beads:

```bash
# View all module system issues
bd list --label module-system --json

# View ready-to-work issues
bd ready --json

# Start Phase 1
bd update nanolang-dc8u --status in_progress
```

**Issues:**
- **nanolang-dc8u** - Phase 1: Module-level safety
- **nanolang-zqke** - Phase 2: Module introspection
- **nanolang-rkc3** - Phase 3: Warning system
- **nanolang-asqo** - Phase 4: Module-qualified calls

---

## Decision Made

✅ **User approved Phases 1-4**  
✅ **Syntax finalized: use `module` keyword**  
✅ **Issues created and tracked**  
⏳ **Ready to begin implementation**

---

## Summary in 3 Points

1. **Problem Identified:** Modules feel "grafted on", require scattered unsafe blocks, no introspection
2. **Solution Designed:** Module-level safety, auto-generated metadata, warning system, qualified calls
3. **Next Step:** Begin Phase 1 implementation (1-2 weeks for module-level safety)

---

**Total Architectural Work:** ~2 hours  
**Documents Created:** 6 (4,100+ lines)  
**Issues Tracked:** 4 phases  
**Estimated Implementation:** 4-6 weeks total  
**Ready to Start:** Phase 1 (nanolang-dc8u)

---

**Status:** ✅ **Architecture Complete - Ready for Implementation**  
**Date:** 2025-01-08  
**Next Action:** Begin Phase 1 parser changes (or await further feedback)
