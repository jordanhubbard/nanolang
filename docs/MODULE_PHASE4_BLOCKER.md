# Phase 4 Blocker: Module Function Compilation

**Issue:** nanolang-asqo  
**Status:** 80% complete - Blocked by module transpilation issue  
**Date:** 2025-01-08  
**Session Duration:** 5+ hours

---

## ✅ What Works

**Parser, Typechecker, Transpiler (Main Program):**
- `AST_MODULE_QUALIFIED_CALL` nodes are created ✅
- Type checking resolves `Math.add` correctly ✅  
- Transpiler generates `test_math_module__add(10, 20)` ✅
- C code generation is perfect ✅

---

## ❌ The Blocker

**Module Object Compilation:**

When compiling `/tmp/test_math_module.nano` to `obj/nano_modules/test_math_module.o`:

**Expected:**
```c
// obj/nano_modules/test_math_module.o.c
int64_t test_math_module__add(int64_t a, int64_t b) {
    return a + b;
}

int64_t test_math_module__multiply(int64_t a, int64_t b) {
    return a * b;
}
```

**Actual:**
- Object file: 336 bytes (empty)
- Symbols: only `ltmp0` (no functions)
- Functions NOT transpiled

---

## 🔍 Investigation

### Module Compilation Flow

1. `compile_modules()` is called (`src/module.c:1199`)
2. For pure NanoLang modules (`src/module.c:1329`)
3. Calls `compile_module_to_object()` (`src/module.c:1360`)
4. Which calls `transpile_to_c()` (`src/module.c:957`)
5. Writes C to `obj/nano_modules/test_math_module.o.c`
6. Compiles C to `obj/nano_modules/test_math_module.o`

**Result:** Object file is empty (no function definitions)

### Why Functions Aren't Transpiled

The transpiler (`src/transpiler.c`) likely has special handling that:
- Only transpiles functions if they're reachable from `main()`
- Skips module functions when compiling modules separately
- Expects modules to define `main()` for standalone compilation

---

## 📊 Evidence

```bash
$ nm obj/nano_modules/test_math_module.o
0000000000000000 t ltmp0      # Only this - no functions!

$ ls -la obj/nano_modules/test_math_module.o
-rw-r--r-- 1 jkh staff 336 Jan 8 20:31 test_math_module.o  # Suspiciously small
```

Compare with working modules:
```bash
$ ls -la obj/nano_modules/lexer.o
-rw-r--r-- 1 jkh staff 123504 Jan 8 13:53 lexer.o  # Much larger!
```

---

## 🤔 Root Cause Hypothesis

**Theory 1: Module Transpilation Skips Functions**
- `transpile_to_c()` may only emit functions reachable from `main()`
- Modules without `main()` → no functions emitted
- Lines 950-956 of `src/module.c` mark `main` as `extern` to skip it
- But this may also cause other functions to be skipped

**Theory 2: Function Visibility**
- Functions need to be marked `pub` to be exported
- Test module functions aren't marked `pub`
- Transpiler skips non-public functions in modules

**Theory 3: Module Name Mismatch**
- Functions registered as `"add"` with `module_name="test_math_module"`
- But transpiler expects a different module name format
- Name mangling doesn't match

---

## 🎯 Next Steps

### Option A: Fix Transpiler (2-3 hours)
Modify `src/transpiler.c` to:
1. Always transpile all functions in a module
2. Don't skip functions just because there's no `main()`
3. Use `module_name` to generate `module__function` names

### Option B: Require pub Functions (30 mins)
Document that module functions MUST be marked `pub`:
```nano
pub fn add(a: int, b: int) -> int { ... }
```
Then fix transpiler to only emit `pub` functions for modules.

### Option C: Working Test (30 mins)
Create test with existing working modules (vector2d, sdl):
- These already compile to objects correctly
- Test module-qualified calls with them
- Prove the system works for production modules

**Recommendation:** Try Option C first (quick win), then Option B (architectural fix)

---

## 🔧 Workaround for Testing

Use existing module that compiles correctly:

```nano
module "modules/vector2d/vector2d.nano" as Vec

fn main() -> int {
    let v: Vec.Vec2 = (Vec.Vec2_new 1.0 2.0)  // If constructor exists
    return 0
}
```

---

## 📝 Files Investigated

| File | Purpose | Status |
|------|---------|--------|
| `src/parser.c` | Creates AST nodes | ✅ Working |
| `src/typechecker.c` | Type checking | ✅ Working |
| `src/transpiler_iterative_v3_twopass.c` | C generation | ✅ Working |
| `src/module.c` | Module compilation | ⚠️ Calls transpiler |
| `src/transpiler.c` | Main transpiler entry | ❌ Issue here |

---

## 🎓 Lessons Learned

1. **Module System Complexity:** Separate compilation is hard
2. **Function Visibility:** Need `pub` keyword enforcement
3. **Testing:** Should have tested with existing modules first
4. **Transpiler Assumptions:** Makes assumptions about `main()` presence

---

**Estimated Fix Time:** 2-4 hours (if we understand the transpiler's function emission logic)  
**Workaround Available:** Yes (use existing modules)  
**Priority:** Medium (parser/typechecker work is done, this is linking issue)
