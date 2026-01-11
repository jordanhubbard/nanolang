# Module System: Architectural Decision Required

**Date:** 2025-01-08  
**Status:** 🔴 **Decision Required**  
**Type:** Architectural / Language Design

---

## The Problem

The current module system works but feels **"grafted on"** rather than being a first-class language feature:

### Pain Point #1: Unsafe Blocks Everywhere

**Current Reality:**
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

**What Users Want:**
```nano
unsafe module "modules/sdl/sdl.nano"  // or: unsafe module sdl { ... }

fn render_frame() -> int {
    (SDL_Init SDL_INIT_VIDEO)      // Clean!
    let window: Window = (SDL_CreateWindow "Game" 0 0 800 600 0)
    (SDL_RenderPresent renderer)
    (SDL_Quit)
    return 0
}
```

---

### Pain Point #2: No Module Introspection

**Other Languages:**
```python
# Python
import math
dir(math)  # ['sqrt', 'sin', 'cos', ...]

# Elixir
Math.__info__(:functions)  # [sqrt: 1, sin: 1, ...]

# Go
$ go doc math  # Lists all exports
```

**NanoLang:**
```nano
import "modules/vector2d/vector2d.nano"
// ❌ NO WAY to list exports
// ❌ NO WAY to check safety
// ❌ NO WAY to query metadata
```

---

### Pain Point #3: No Safety Visibility

**Current:**
```nano
import "modules/vector2d/vector2d.nano"  // Safe? Unsafe? Who knows!
import "modules/sdl/sdl.nano"            // Both look identical
```

**Desired:**
```nano
safe module "modules/vector2d/vector2d.nano"    // Compiler verifies
unsafe module "modules/sdl/sdl.nano"             // Explicit warning
```

---

## The Solution: Modules as First-Class Citizens

### Feature 1: Module-Level Safety

**Before:**
- 45 `unsafe {}` blocks in SDL examples
- Visual noise obscures logic
- Hard to audit unsafe code

**After:**
```nano
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    // All FFI calls allowed here - no unsafe blocks needed!
}
```

**Benefit:** One line instead of dozens of blocks

---

### Feature 2: Module Introspection

**Auto-Generated Functions:**
```nano
extern fn ___module_info_sdl() -> ModuleInfo
extern fn ___module_has_function_sdl(name: string) -> bool

struct ModuleInfo {
    name: string,
    is_safe: bool,
    has_ffi: bool,
    exported_functions: array<FunctionInfo>
}
```

**Usage:**
```nano
let info: ModuleInfo = (___module_info_sdl)
if info.is_safe {
    (println "SDL is safe")
} else {
    (println "SDL contains unsafe FFI")
}
```

**Benefit:** Same reflection capabilities we just added for structs!

---

### Feature 3: Graduated Warning System

**Compilation Modes:**
```bash
# Default: allow all
nanoc app.nano -o app

# Warn on unsafe imports
nanoc app.nano -o app --warn-unsafe-imports

# Warn on all FFI calls
nanoc app.nano -o app --warn-ffi

# Strict: error on any unsafe
nanoc app.nano -o app --forbid-unsafe
```

**Benefit:** Users choose their safety level

---

## Comparison with Other Languages

| Feature | Python | Go | Elixir | NanoLang Now | NanoLang Proposed |
|---------|--------|----|----- |--------------|-------------------|
| Module Introspection | ✅ | ✅ | ✅ | ❌ | ✅ |
| Safety Annotation | ❌ | ✅ `import "unsafe"` | ✅ Custom attrs | ❌ | ✅ `unsafe module` |
| Warning System | ⚠️ Linters | ✅ `go vet` | ⚠️ Custom | ❌ | ✅ Compiler flags |
| Module as Value | ✅ | ⚠️ Via reflection | ✅ | ❌ | ⏳ Future |

**Key Insight:** Go's `import "unsafe"` is the gold standard for FFI safety

---

## Implementation Phases

### Phase 1: Module-Level Safety (1-2 weeks)

**Changes:**
- Parser: Recognize `unsafe module { ... }`
- Typechecker: Allow FFI in unsafe modules without `unsafe {}`
- Import: Support `import unsafe`

**Deliverable:** One `unsafe module` annotation instead of dozens of `unsafe {}` blocks

---

### Phase 2: Module Introspection (1-2 weeks)

**Changes:**
- Transpiler: Generate `___module_info_*` functions (like struct reflection)
- Runtime: Add `ModuleInfo` struct
- Compiler: Collect exported function metadata

**Deliverable:** Query modules at compile-time like we query structs

---

### Phase 3: Warning System (1 week)

**Changes:**
- CLI: Add `--warn-unsafe-imports`, `--warn-ffi`, `--forbid-unsafe` flags
- Typechecker: Emit warnings based on mode
- Diagnostics: Clear, actionable messages

**Deliverable:** Users control their safety level

---

### Phase 4: Module-Qualified Calls (1 week)

**Changes:**
- Parser: Fix `Module.function()` treated as field access
- Typechecker: Resolve function in module namespace

**Deliverable:** Clean module call syntax

**Already Tracked:** Issue `nanolang-3oda`

---

**Total Time:** 4-6 weeks for Phases 1-4

---

## Why This Matters

### For Users

1. **Less Visual Noise** - Code reads like Python/Go, not like Rust
2. **Clear Safety Model** - Instantly see what's safe vs unsafe
3. **Better Tooling** - Docs, linters, analyzers can query modules
4. **Graduated Safety** - Start permissive, tighten over time

### For Language

1. **Competitive Feature** - Matches mature languages
2. **Ecosystem Foundation** - Module marketplace, certifications, ratings
3. **Better Diagnostics** - "Function from unsafe module 'sdl'" vs vague errors
4. **Metaprogramming** - Generate code based on module metadata

### For Self-Hosting

1. **Reduces Manual Metadata** - Module introspection replaces manual tables
2. **Better Type Inference** - Know which modules are safe
3. **Easier Auditing** - Query modules programmatically

**Direct Impact on Self-Hosting Issues:**
- `nanolang-3oda` - Module-qualified calls
- `nanolang-qlv2` - Type inference via module metadata

---

## Comparison: Before & After

### Before (Current System)

**Pros:**
- ✅ Works for basic use cases
- ✅ Explicit unsafe blocks for safety

**Cons:**
- ❌ Unsafe blocks scattered everywhere
- ❌ No module introspection
- ❌ No safety visibility
- ❌ Feels bolted-on
- ❌ Hard to audit dependencies

---

### After (Proposed System)

**Pros:**
- ✅ Module-level safety (one annotation)
- ✅ Full introspection (like structs)
- ✅ Graduated warnings (user choice)
- ✅ First-class modules (like Python/Elixir)
- ✅ Easy dependency auditing

**Cons:**
- ⚠️ 4-6 weeks implementation time
- ⚠️ Need migration strategy

---

## Migration Path

### Backward Compatibility

**Strategy:** Gradual transition, no hard breaks

1. **Phase 1a:** Parser recognizes `unsafe module` (no enforcement)
2. **Phase 1b:** Typechecker uses module safety (old syntax still works)
3. **Phase 1c:** Deprecation warnings for scattered `unsafe {}`
4. **Phase 1d:** (Far future) Remove old syntax

**Timeline:** 2+ months transition period

---

### Auto-Migration Tool

```bash
# Convert module with extern functions
$ nalang migrate --add-unsafe-module modules/sdl/sdl.nano

# Before:
import "modules/sdl/sdl.nano"
extern fn SDL_Init(flags: int) -> int
fn init() -> bool {
    unsafe { (SDL_Init SDL_INIT_VIDEO) }
}

# After:
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    fn init() -> bool {
        (SDL_Init SDL_INIT_VIDEO)  // No unsafe block!
    }
}
```

---

## Risk Analysis

### Risk 1: Breaking Existing Code

**Mitigation:**
- Backward compatibility mode (6+ months)
- Deprecation warnings before removal
- Auto-migration tool

**Probability:** Low (designed to be non-breaking)

---

### Risk 2: Self-Hosted Compiler Complexity

**Mitigation:**
- Implement in C compiler first
- Test thoroughly
- Port to NanoLang after proven stable

**Probability:** Medium (managed by phased approach)

---

### Risk 3: Implementation Time

**Mitigation:**
- Independent phases (ship incrementally)
- Each phase adds value
- Can pause after any phase

**Probability:** Medium (realistic estimates)

---

## Decision Matrix

### Option A: Proceed with Full Redesign (Phases 1-4)

**Timeline:** 4-6 weeks  
**Impact:** High - Modernizes module system  
**Risk:** Medium - Requires careful migration  
**Benefit:** First-class modules matching Python/Go/Elixir

**Recommendation:** ✅ **Yes** - Addresses fundamental design issue

---

### Option B: Proceed with Safety Only (Phase 1)

**Timeline:** 1-2 weeks  
**Impact:** Medium - Reduces unsafe block noise  
**Risk:** Low - Minimal changes  
**Benefit:** Cleaner code, easier auditing

**Recommendation:** ✅ **Yes, at minimum** - Quick win

---

### Option C: Defer Entire Redesign

**Timeline:** 0 weeks  
**Impact:** None - Keep status quo  
**Risk:** None - No changes  
**Benefit:** None - Module system still feels grafted-on

**Recommendation:** ❌ **No** - Problem persists

---

## Questions for Decision

1. **Approve module-level safety?** (`unsafe module`)
   - ✅ Recommended: Yes

2. **Approve module introspection?** (Auto-generate metadata)
   - ✅ Recommended: Yes

3. **Approve warning system?** (Compiler flags)
   - ✅ Recommended: Yes

4. **Approve module-qualified calls?** (Parser fix)
   - ✅ Recommended: Yes (already tracked in `nanolang-3oda`)

5. **Approve module as value?** (Runtime type)
   - ⏳ Recommended: Defer to future

6. **Migration timeline?**
   - ⏳ Recommended: 2+ months transition period

---

## Related Documents

1. **`docs/MODULE_SYSTEM_REDESIGN.md`** - Full design specification
2. **`docs/MODULE_SYSTEMS_COMPARISON.md`** - Comparison with Python/Go/Ruby/Elixir
3. **`docs/MODULE_IMPLEMENTATION_ROADMAP.md`** - Detailed phase-by-phase plan
4. **Current:** `docs/MODULE_SYSTEM.md` - Existing documentation
5. **Current:** `MEMORY.md` - Unsafe blocks section

---

## Recommended Actions

### Immediate (Today)

1. ✅ Review this decision document
2. ✅ Approve/reject Phase 1 (module-level safety)
3. ✅ Approve/reject Phase 2 (introspection)

### Short-Term (This Week)

1. ⏳ Create GitHub issues for approved phases
2. ⏳ Begin Phase 1 implementation
3. ⏳ Write migration guide

### Medium-Term (This Month)

1. ⏳ Complete Phase 1 & 2
2. ⏳ Test in self-hosted compiler
3. ⏳ Update all examples

---

## Success Criteria

### Phase 1 Success

- 50%+ reduction in `unsafe {}` block count
- All SDL examples use `unsafe module`
- Zero regressions

### Phase 2 Success

- Self-hosted compiler uses module introspection
- 10+ examples showing module queries
- Docs published

### Overall Success

- Module system feels first-class (like Python/Go)
- Users can audit safety easily
- Ecosystem can build on reflection

---

## Final Recommendation

**Proceed with Phases 1-4 over 4-6 weeks**

**Rationale:**
1. Addresses real pain point (unsafe block noise)
2. Brings NanoLang to parity with Python/Go/Elixir
3. Foundations already laid (struct reflection proves pattern)
4. Phased approach allows incremental delivery
5. Backward compatible with careful migration

**Benefits outweigh costs.** This is the right architectural investment.

---

**Status:** 🔴 **DECISION REQUIRED**  
**Owner:** Project Lead  
**Date:** 2025-01-08  
**Next Step:** Approve/reject Phases 1-2
