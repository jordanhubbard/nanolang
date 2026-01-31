# Phase 2 Status: Module Introspection Implementation

**Status:** 🟡 **Infrastructure Complete, Needs Debugging**  
**Issue:** nanolang-zqke  
**Date:** 2025-01-08  
**Time Invested:** ~2 hours

---

## What Was Implemented ✅

### **1. ModuleInfo Struct** (`src/nanolang.h`)

**Rich metadata tracking:**
```c
typedef struct {
    char *name;                /* Module name (e.g., "sdl", "vector2d") */
    char *path;                /* Module file path */
    bool is_unsafe;            /* Is this module marked unsafe? */
    bool has_ffi;              /* Does this module contain extern functions? */
    char **exported_functions; /* List of exported function names */
    int function_count;
    char **exported_structs;   /* List of exported struct names */
    int struct_count;
} ModuleInfo;
```

**Replaces:** Simple `char** unsafe_modules` with structured metadata.

---

### **2. Environment Tracking** (`src/env.c`)

**New functions:**
```c
void env_register_module(Environment *env, const char *name, const char *path, bool is_unsafe);
ModuleInfo *env_get_module(Environment *env, const char *name);
bool env_is_current_module_unsafe(Environment *env);
void env_mark_module_has_ffi(Environment *env, const char *name);
```

**Functionality:**
- Register modules during import processing
- Query module metadata by name
- Track unsafe context
- Mark modules containing FFI

---

### **3. Typechecker Module Registration** (`src/typechecker.c`)

**Pre-pass processing:**
```c
/* Extract module name from path */
"modules/sdl/sdl.nano" → "sdl"
"modules/vector2d/vector2d.nano" → "vector2d"

/* Register with metadata */
env_register_module(env, module_name, path, is_unsafe);
```

**Tracks:**
- All imported modules (safe + unsafe)
- Module paths
- Unsafe flag from import statement

---

### **4. Transpiler Code Generation** (`src/transpiler.c`)

**Auto-generated functions** (similar to struct reflection):

```c
/* For each module, generate: */

inline bool ___module_is_unsafe_sdl(void) {
    return 1;  /* or 0 */
}

inline bool ___module_has_ffi_sdl(void) {
    return 1;  /* or 0 */
}

inline const char* ___module_name_sdl(void) {
    return "sdl";
}

inline const char* ___module_path_sdl(void) {
    return "modules/sdl/sdl.nano";
}
```

**Generated for:** Every module imported in the program.

---

## Current Status 🟡

### **✅ What Works**
1. Compiler builds successfully
2. Infrastructure in place (ModuleInfo, env functions, etc.)
3. Transpiler generates module metadata functions
4. Module registration during typechecking
5. Module name extraction from paths

### **🟡 What Needs Debugging**
1. **Module metadata functions return incorrect values** (returning "void")
2. Possible issue: Module names not matching between registration and function generation
3. Need to verify `env->module_count` is > 0 during transpilation
4. Need to debug module name normalization

### **❌ What's Not Implemented Yet**
1. `has_ffi` flag tracking (always false currently)
2. `exported_functions` list (NULL currently)
3. `exported_structs` list (NULL currently)
4. Advanced introspection (function signatures, struct fields from modules)

---

## Test Case

**Input:** `/tmp/test_module_introspection.nano`
```nano
unsafe module "modules/sdl/sdl.nano" as SDL
module "modules/vector2d/vector2d.nano" as Vec

extern fn ___module_is_unsafe_sdl() -> bool
extern fn ___module_name_sdl() -> string

fn main() -> int {
    let is_unsafe: bool = false
    unsafe { set is_unsafe (___module_is_unsafe_sdl) }
    
    let name: string = ""
    unsafe { set name (___module_name_sdl) }
    
    (println name)
    return 0
}
```

**Expected Output:**
```
sdl
```

**Actual Output:**
```
void
```

**Issue:** Functions exist but return wrong values (type mismatch or no data).

---

## Debugging Steps (Next Session)

### **1. Verify Module Registration**
Add debug output to typechecker:
```c
fprintf(stderr, "DEBUG: Registered module '%s' (unsafe=%d)\n", module_name, is_unsafe);
```

### **2. Verify Module Count**
In transpiler, before generating:
```c
fprintf(stderr, "DEBUG: Generating metadata for %d modules\n", env->module_count);
```

### **3. Check Module Names**
Ensure extracted names match:
- Registration: `"modules/sdl/sdl.nano"` → `"sdl"`
- Function generation: `___module_is_unsafe_sdl` expects `mod->name == "sdl"`

### **4. Check Generation**
Inspect generated C file:
```bash
$ ./bin/nanoc test.nano -o test --keep-c
$ grep "___module_" /tmp/nanoc_*.c
```

---

## Files Changed

| File | Lines | Changes |
|------|-------|---------|
| `src/nanolang.h` | +23, -2 | Added ModuleInfo struct + prototypes |
| `src/env.c` | +85, -6 | Module registration and query functions |
| `src/typechecker.c` | +41, -6 | Module registration in pre-pass |
| `src/transpiler.c` | +49, -0 | Module metadata generation |

**Total:** ~177 lines changed

---

## Phase 2 Goals (Original)

| Goal | Status |
|------|--------|
| Track module metadata | ✅ Done |
| Generate `___module_info_*()` functions | ✅ Done (needs debug) |
| Query module safety at compile-time | ✅ Infrastructure ready |
| Query module FFI status | 🟡 Infrastructure ready (not tracked yet) |
| List exported functions | ❌ Not implemented |
| List exported structs | ❌ Not implemented |

**Completion:** ~60% (infrastructure), ~30% (functional)

---

## Phase 2 Completion Estimate

### **To Complete Core Features:**
1. Debug module metadata values (1-2 hours)
2. Implement FFI tracking (1 hour)
3. Test with multiple modules (1 hour)
4. Documentation (30 mins)

**Total:** ~4 hours additional work

### **For Full Phase 2 (with export lists):**
1. Core features (4 hours)
2. Track exported functions (2 hours)
3. Track exported structs (1 hour)
4. Advanced introspection API (2 hours)

**Total:** ~9 hours additional work

---

## Next Steps (Prioritized)

### **Option A: Debug & Ship Core (4 hours)**
1. Add debug output to trace module registration
2. Verify `env->module_count` during transpilation
3. Fix module name extraction/matching
4. Test with SDL/vector2d examples
5. Document working API

**Result:** Working module introspection for safety queries

---

### **Option B: Continue to Phase 3 (Warning System)**
- Defer Phase 2 debugging to later
- Implement `--warn-unsafe-imports`, `--warn-ffi` flags
- Simpler implementation (compiler flags)
- Phase 2 infrastructure will be available when needed

**Result:** User-facing safety controls ready

---

### **Option C: Fix Phase 4 (Module-Qualified Calls)**
- Fix `Module.function()` typechecker bug
- Critical for ergonomic module usage
- Requires typechecker work

**Result:** `SDL.init()` syntax works correctly

---

## Recommendation

**Proceed with Option B (Phase 3)** for the following reasons:
1. Phase 2 infrastructure is complete and can be debugged later
2. Phase 3 provides immediate user value (safety warnings)
3. Phase 3 is simpler and faster to implement (~1 week)
4. Phase 2 debugging can be done incrementally as needed

**Or, if user prefers:**
- **Option A:** Complete Phase 2 fully (4-9 hours)
- **Option C:** Fix module-qualified calls first (critical usability)

---

## Commits

1. `db8252f` - feat: Phase 2 infrastructure for module introspection

**Changes:** 177 lines, 4 files  
**Status:** Compiler builds, infrastructure complete, needs debugging

---

## Summary

**Phase 2 is 60% complete:**
- ✅ Infrastructure: ModuleInfo, environment tracking, transpiler generation
- 🟡 Functionality: Functions generate but return incorrect values
- ❌ Advanced: Export lists, FFI tracking not implemented

**Estimated completion:** 4-9 additional hours depending on scope.

**Recommendation:** Move to Phase 3 (warning system) and debug Phase 2 incrementally.

---

**Status:** 🟡 **Paused at infrastructure complete**  
**Next:** User decision (Option A, B, or C)  
**Date:** 2025-01-08
