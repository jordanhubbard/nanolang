# Phase 4 Status: Module-Qualified Calls

**Issue:** nanolang-asqo  
**Status:** Step 1 Complete (AST infrastructure exists)  
**Date:** 2025-01-08  
**Progress:** 10% (infrastructure only)

---

## Problem Statement

Module-qualified calls like `(Vec.add v1 v2)` currently fail because the parser treats them as field access.

### Current Errors

```nano
module "modules/vector2d/vector2d.nano" as Vec

fn test() -> int {
    let v1: Vec.Vec2 = Vec.Vec2 { x: 1.0, y: 2.0 }  // ❌ Parse error
    let v3: Vec.Vec2 = (Vec.add v1 v2)              // ❌ "Undefined function"
    return 0
}
```

**Errors:**
- Type annotations: `Vec.Vec2` not recognized
- Struct literals: `Vec.Vec2 { ... }` not recognized  
- Function calls: `(Vec.add ...)` treated as undefined
- All parsed as field access or rejected

---

## Implementation Status

### ✅ Step 1: AST Node (COMPLETE)

**File:** `src/nanolang.h`

**Enum Value (Line 158):**
```c
AST_MODULE_QUALIFIED_CALL,  /* Module-qualified call: (Module.function args...) */
```

**Struct Definition (Lines 235-241):**
```c
struct {
    char *module_alias;       /* "Vec" from "as Vec" */
    char *function_name;      /* "add" */
    ASTNode **args;
    int arg_count;
    char *return_struct_type_name;
} module_qualified_call;
```

**Status:** ✅ Already exists in header, added previously but never implemented

---

### ⏳ Step 2: Parser Implementation (NOT STARTED)

**Files to Modify:**
- `src/parser.c` (main)
- `src/parser_iterative.c` (if exists)

**Requirements:**

1. **Detect Pattern:** `IDENTIFIER DOT IDENTIFIER LPAREN`
   - First identifier → check if it's a module alias
   - If yes → parse as `AST_MODULE_QUALIFIED_CALL`
   - If no → parse as field access (existing behavior)

2. **Parse Arguments:**
   - Use existing argument parsing logic
   - Set `module_alias` and `function_name`
   - Build args array

3. **Distinguish from Field Access:**
   ```nano
   (Vec.add v1 v2)     // → AST_MODULE_QUALIFIED_CALL
   obj.field           // → AST_FIELD_ACCESS (existing)
   ```

**Implementation Complexity:** Medium  
**Estimated Time:** 1-2 days  
**Current Status:** 0 matches for `AST_MODULE_QUALIFIED_CALL` in `src/*.c`

---

### ⏳ Step 3: Typechecker Implementation (NOT STARTED)

**File:** `src/typechecker.c`

**Requirements:**

1. **Add Case for `AST_MODULE_QUALIFIED_CALL`:**
   ```c
   case AST_MODULE_QUALIFIED_CALL: {
       // Look up module namespace
       // Find function in module's exported symbols
       // Type check arguments
       // Return function's return type
   }
   ```

2. **Module Namespace Lookup:**
   - Get module alias from node
   - Find module in environment
   - Look up function in module's namespace
   - Verify function exists

3. **Argument Typechecking:**
   - Check each argument against function signature
   - Report type mismatches with clear errors
   - Handle variadic functions (if needed)

4. **Return Type:**
   - Return function's declared return type
   - Handle struct return types correctly
   - Propagate type information to parent nodes

**Implementation Complexity:** Medium-High  
**Estimated Time:** 1-2 days  
**Current Status:** Not implemented

---

### ⏳ Step 4: Transpiler Implementation (NOT STARTED)

**Files to Modify:**
- `src/transpiler.c` (main)
- `src/transpiler_iterative_v3_twopass.c` (two-pass transpiler)

**Requirements:**

1. **Add Case for `AST_MODULE_QUALIFIED_CALL`:**
   ```c
   case AST_MODULE_QUALIFIED_CALL: {
       // Generate C function call
       // Use module's C prefix (e.g., nl_vector2d_add)
       // Transpile arguments
       // Handle return value
   }
   ```

2. **C Function Name Generation:**
   - Module "vector2d" + function "add" → `nl_vector2d_add`
   - Handle module path → C prefix mapping
   - Respect C naming conventions

3. **Argument Transpilation:**
   - Transpile each argument expression
   - Generate argument list in C
   - Handle type conversions if needed

**Implementation Complexity:** Low-Medium  
**Estimated Time:** 1 day  
**Current Status:** Not implemented

---

### ⏳ Step 5: Testing (NOT STARTED)

**Test Cases Required:**

1. **Basic Module-Qualified Call:**
   ```nano
   module "math.nano" as Math
   let result: int = (Math.add 1 2)
   ```

2. **Multiple Arguments:**
   ```nano
   module "vector2d/vector2d.nano" as Vec
   let v3: Vec.Vec2 = (Vec.add v1 v2)
   ```

3. **Return Type Verification:**
   ```nano
   let x: int = (Module.func)        // int return
   let s: string = (Module.func2)    // string return
   ```

4. **Error Cases:**
   ```nano
   (UnknownModule.func)              // Error: module not found
   (Module.unknownFunc)              // Error: function not found
   (Module.func wrong_type_arg)      // Error: type mismatch
   ```

5. **Field Access Still Works:**
   ```nano
   let v: Vec2 = Vec2 { x: 1.0, y: 2.0 }
   let x: float = v.x                // Should still work!
   ```

**Estimated Time:** 1 day  
**Current Status:** Not started

---

## Related Issues

### **Module-Qualified Types**

**Problem:** Type annotations like `Vec.Vec2` also fail

```nano
let v: Vec.Vec2 = ...  // ❌ Parser error
```

**Solution:** Similar pattern detection in type parsing  
**Status:** Separate work, but related

### **Module-Qualified Struct Literals**

**Problem:** Struct constructors fail

```nano
let v: Vec.Vec2 = Vec.Vec2 { x: 1.0, y: 2.0 }  // ❌ Parser error
```

**Solution:** Detect pattern in struct literal parsing  
**Status:** Can be bundled with function calls

---

## Implementation Roadmap

### **Week 1: Parser + Typechecker**
- **Day 1-2:** Parser implementation
  - Pattern detection (Module.function)
  - AST node creation
  - Unit tests

- **Day 3-4:** Typechecker implementation
  - Module namespace lookup
  - Function resolution
  - Argument type checking
  - Return type inference

- **Day 5:** Integration testing
  - Test parser → typechecker flow
  - Fix any issues

### **Week 2: Transpiler + Polish**
- **Day 1:** Transpiler implementation
  - C function name generation
  - Argument transpilation
  - Test generated C code

- **Day 2:** Comprehensive testing
  - All test cases
  - Edge cases
  - Error messages

- **Day 3:** Documentation
  - Update MEMORY.md
  - Add examples
  - Migration guide (if needed)

- **Day 4-5:** Review and polish
  - Code review
  - Performance testing
  - Final integration

---

## Current Blockers

### **None!**

All prerequisites are complete:
- ✅ Module system (Phase 1)
- ✅ Module introspection (Phase 2)
- ✅ AST infrastructure (Step 1)

Ready to proceed with parser implementation.

---

## Next Steps

**Immediate (When Resuming):**

1. **Study Parser Code:**
   - How does current call parsing work?
   - Where is field access handled?
   - How to detect module aliases?

2. **Design Pattern Detection:**
   - Lookahead logic for `ID DOT ID LPAREN`
   - Module alias vs struct name distinction
   - Edge case handling

3. **Implement Parser:**
   - Add `parse_module_qualified_call()` function
   - Integrate into main expression parsing
   - Test with examples

**Recommended Start Time:** After rest/break (fresh mind for parser work)

---

## Success Criteria

**Phase 4 Complete When:**
- ✅ `(Module.function args...)` works correctly
- ✅ Parser distinguishes module calls from field access
- ✅ Typechecker resolves functions in module namespaces
- ✅ Transpiler generates correct C code
- ✅ All test cases pass
- ✅ Examples updated
- ✅ Documentation complete

**Estimated Total Time:** 1 week (5-7 days of focused work)

---

## Files Changed (Estimated)

| File | Lines Changed | Complexity |
|------|---------------|------------|
| `src/parser.c` | +100-150 | Medium |
| `src/typechecker.c` | +80-120 | Medium-High |
| `src/transpiler.c` | +50-80 | Low-Medium |
| `examples/*.nano` | +50-100 | Low |
| `docs/*.md` | +200-300 | Low |

**Total Estimated:** ~600-800 lines changed

---

## Risk Assessment

**Low Risk:**
- AST infrastructure complete
- Pattern is well-understood
- Similar to existing field access logic

**Medium Risk:**
- Module namespace lookup complexity
- Type inference edge cases
- C name generation collisions

**Mitigation:**
- Thorough testing at each step
- Incremental implementation
- Review existing field access code

---

## References

**Design Documents:**
- `docs/MODULE_ARCHITECTURE_DECISION.md` - Overall architecture
- `docs/MODULE_PHASE1_COMPLETE.md` - Module syntax
- `docs/MODULE_PHASE2_COMPLETE.md` - Module introspection

**Related Code:**
- `src/parser.c` - Call parsing, field access
- `src/typechecker.c` - Type inference, namespace lookup
- `src/transpiler.c` - C code generation

**Test Files:**
- `/tmp/test_module_qualified.nano` - Current failing test
- `examples/vector2d_demo.nano` - Vector2D usage

---

**Status:** Ready to implement when resuming work  
**Next:** Parser implementation (Step 2)  
**Estimated Completion:** 1 week from start
