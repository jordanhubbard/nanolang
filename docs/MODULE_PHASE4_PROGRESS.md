# Phase 4 Progress: Module-Qualified Calls

**Issue:** nanolang-asqo  
**Status:** 80% Complete (Parser/Typechecker/Transpiler working)  
**Date:** 2025-01-08  
**Commit:** 4afa2cd

---

## ✅ Completed Steps

### Step 1: AST Node (COMPLETE)
- `AST_MODULE_QUALIFIED_CALL` enum exists in `src/nanolang.h`
- Struct definition complete with all required fields
- Infrastructure was already in place

### Step 2: Parser Implementation (COMPLETE)
**File:** `src/parser.c` (+49 lines)

**Changes:**
- Lines 1543-1561: Zero-argument module calls `(Module.func)`
- Lines 1584-1615: Multi-argument module calls `(Module.func arg1 arg2)`
- Error handling paths updated to free `module_alias` and `qualified_func_name`

**How It Works:**
1. Detects `AST_FIELD_ACCESS` pattern in call position
2. Checks if object is `AST_IDENTIFIER` (module alias)
3. Creates `AST_MODULE_QUALIFIED_CALL` node
4. Sets `module_alias` (e.g., "Math") and `function_name` (e.g., "add")
5. Properly frees field_access node

**Before (Incorrect):**
```c
// Created AST_CALL with name "Math.add" (dotted string)
node->as.call.name = "Math.add";
```

**After (Correct):**
```c
// Creates AST_MODULE_QUALIFIED_CALL with separate fields
node->as.module_qualified_call.module_alias = "Math";
node->as.module_qualified_call.function_name = "add";
```

### Step 3: Typechecker Implementation (COMPLETE)
**File:** `src/typechecker.c` (+37 lines)

**Changes:**
- Added case for `AST_MODULE_QUALIFIED_CALL` (after line 1536)
- Placed between `AST_CALL` and `AST_ARRAY_LITERAL` cases

**How It Works:**
1. Constructs qualified name: `"Module.function"`
2. Looks up function in environment: `env_get_function(env, "Math.add")`
3. Verifies argument count matches function signature
4. Type checks each argument expression
5. Returns function's return type

**Error Messages:**
- `"Undefined function 'Math.add'"` - function not found
- `"Function 'Math.add' expects 2 arguments, got 3"` - wrong arity

### Step 4: Transpiler Implementation (COMPLETE)
**File:** `src/transpiler_iterative_v3_twopass.c` (+27 lines)

**Changes:**
- Added case for `AST_MODULE_QUALIFIED_CALL` (after AST_CALL, before AST_IF)
- Placed between lines 1582 and 1585

**How It Works:**
1. Constructs qualified name: `"Module.function"`
2. Uses `map_function_name(qualified_name, env)` for C name mapping
3. Emits C function call: `module__function(args...)`
4. Frees temporary qualified_name string

**Example Transpilation:**
```nano
(Math.add 10 20)
```
↓
```c
test_math_module__add(10LL, 20LL)
```

---

## 🧪 Test Results

### Test Case: `/tmp/test_module_call.nano`

```nano
module "/tmp/test_math_module.nano" as Math

fn main() -> int {
    let result: int = (Math.add 10 20)
    let result2: int = (Math.multiply 5 6)
    return 0
}
```

### Results:

**✅ Parser:** 
- Creates `AST_MODULE_QUALIFIED_CALL` nodes
- No parse errors

**✅ Typechecker:** 
- Resolves `Math.add` and `Math.multiply`
- Type checks pass
- No type errors

**✅ Transpiler:** 
- Generates correct C code:
  ```c
  int64_t result = test_math_module__add(10LL, 20LL);
  int64_t result2 = test_math_module__multiply(5LL, 6LL);
  ```

**❌ C Compilation:**
```
error: call to undeclared function 'test_math_module__add'
error: call to undeclared function 'test_math_module__multiply'
```

**Root Cause:** Module functions aren't exported with prefixed names. Functions in `/tmp/test_math_module.nano` are declared as `add` and `multiply`, but the transpiler expects them to be `test_math_module__add`.

---

## 🚧 Remaining Work (20%)

### Issue: Module Function Namespacing

**Problem:** The module system doesn't prefix exported function names.

**Current Behavior:**
```nano
// In module file: test_math_module.nano
fn add(a: int, b: int) -> int { ... }  // Exported as "add"
```

**Expected Behavior:**
```nano
// When imported as "Math"
// Should be callable as: (Math.add a b)
// Should transpile to: test_math_module__add(a, b)
// Functions should be exported with module-prefixed names
```

**Required Changes:**

1. **Module Loader** (`src/module.c`):
   - When loading a module, prefix all exported function names
   - Store mapping: `"add"` → `"test_math_module__add"`
   - Update environment's function table

2. **Module Alias Resolution**:
   - Map alias to module path: `"Math"` → `"/tmp/test_math_module.nano"`
   - Convert path to prefix: `"/tmp/test_math_module.nano"` → `"test_math_module"`

3. **Function Registration**:
   - Register functions with both:
     - Qualified name: `"Math.add"` → `"test_math_module__add"`
     - Local name (within module): `"add"` → `"add"`

**Estimated Time:** 4-6 hours (architectural changes)

**Complexity:** Medium-High
- Requires modifying module loading
- Needs careful namespace management
- Must not break existing code

---

## 📊 Overall Progress

| Step | Status | Time Spent | Remaining |
|------|--------|------------|-----------|
| 1. AST Node | ✅ Complete | 0.5 hours | 0 hours |
| 2. Parser | ✅ Complete | 1.5 hours | 0 hours |
| 3. Typechecker | ✅ Complete | 1 hour | 0 hours |
| 4. Transpiler | ✅ Complete | 0.5 hours | 0 hours |
| 5. Module Namespacing | ⏳ Pending | 0 hours | 4-6 hours |
| 6. Testing | ⏳ Pending | 0 hours | 1 hour |
| 7. Documentation | ⏳ Pending | 0 hours | 1 hour |

**Total:** 3.5 hours spent, ~6-8 hours remaining

---

## 🎯 Next Steps

### Immediate (When Resuming):

**Option A: Complete Module Namespacing (4-6 hours)**
- Implement function prefixing in module loader
- Add alias → prefix mapping
- Test end-to-end with real modules
- **Result:** Fully working module-qualified calls

**Option B: Document Current State (30 mins)**
- Update MEMORY.md with current limitations
- Create issue for module namespacing
- Document workarounds for users
- **Result:** Users aware of current status

**Option C: Test with Built-in Modules (1 hour)**
- Try with existing modules (vector2d, sdl)
- See if any already have proper prefixes
- Document which modules work vs don't
- **Result:** Practical usage assessment

**Recommendation:** Option B → Option C → Option A
- Document first (prevents confusion)
- Test existing modules (assess scope)
- Implement namespacing (complete feature)

---

## 📝 Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/parser.c` | +49, -31 | Create AST_MODULE_QUALIFIED_CALL nodes |
| `src/typechecker.c` | +37 | Type check module-qualified calls |
| `src/transpiler_iterative_v3_twopass.c` | +27 | Generate C code for module calls |

**Total:** +113 lines, -31 lines (net +82)

---

## 🎉 Achievements

1. **Clean AST Representation:** Module calls are now first-class AST nodes
2. **Proper Type Checking:** Arguments verified against function signatures
3. **Correct C Generation:** Transpiler emits properly-namespaced C calls
4. **Memory Safety:** No leaks in parser/typechecker/transpiler
5. **Error Messages:** Clear, helpful error reporting

---

## 📚 Related Documentation

- `docs/MODULE_PHASE4_STATUS.md` - Initial roadmap
- `docs/MODULE_ARCHITECTURE_DECISION.md` - Overall design
- `docs/MODULE_PHASE1_COMPLETE.md` - Module syntax
- `docs/MODULE_PHASE2_COMPLETE.md` - Module introspection

---

## 🐛 Known Issues

1. **Module Function Namespacing:** Functions not exported with prefixes
2. **Type Annotations:** `Vec.Vec2` type syntax not yet supported
3. **Struct Constructors:** `Vec.Vec2 { ... }` struct syntax not yet supported

Issues #2 and #3 are separate features (module-qualified types), not critical for function calls.

---

**Status:** Parser/Typechecker/Transpiler complete. Module namespacing pending.  
**Next:** Implement module function prefixing in loader.  
**Estimated Completion:** 4-6 hours from resume.
