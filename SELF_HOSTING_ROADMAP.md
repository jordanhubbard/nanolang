# Self-Hosting Roadmap - Path to 100%

**Current Status:** v0.2.0 - Compiler 100% functional, Self-hosting components at ~60%  
**Goal:** 100% self-hosted compiler written in nanolang  
**Timeline:** Incremental approach, fix blockers first

## Current Self-Hosted Codebase

| Component | Lines | Compiles? | Works? | Completeness |
|-----------|-------|-----------|--------|--------------|
| parser_mvp.nano | 2,767 | ❌ | ⏳ | 85% |
| typechecker_minimal.nano | 797 | ❌ | ⏳ | 60% |
| transpiler_minimal.nano | 1,081 | ❌ | ⏳ | 65% |
| **Total** | **4,645** | ❌ | ⏳ | **~70%** |

## Blocking Issues (Must Fix First)

### 1. Parser Compilation Errors
**File:** `src_nano/parser_mvp.nano`  
**Errors:**
- Line 1061, 2203: 'else if' syntax not supported (must use nested if-else)
- Parser expects `if { } else { }` but code uses `else if { }`

**Fix:** Refactor all `else if` chains to nested `if-else` blocks

### 2. Typechecker Compilation Errors  
**File:** `src_nano/typechecker_minimal.nano`  
**Errors:**
- Multiple "Cannot determine struct type for field access" errors
- C compilation error: "type name requires a specifier or qualifier"
- Issues with struct field access in generated C code

**Fix:** 
- Add type annotations for struct fields
- Fix struct member access patterns
- Ensure proper type declarations

### 3. Transpiler Compilation Errors
**File:** `src_nano/transpiler_minimal.nano`  
**Errors:**
- `extern struct parser_get_function(...)` - invalid syntax
- Should be: `extern Parser* parser_get_function(...)`
- Struct return types in extern declarations malformed

**Fix:**
- Fix extern function declarations for struct returns
- Use proper opaque pointer types or full struct definitions

## Phase 1: Make Components Compile ✅

**Goal:** All three components compile with C compiler  
**Priority:** CRITICAL

### Tasks:
1. ✅ Fix parser `else if` → nested `if-else` (Est: 30 min)
2. ✅ Fix typechecker struct field access (Est: 45 min)
3. ✅ Fix transpiler extern declarations (Est: 30 min)
4. ✅ Test compilation: `make stage2` succeeds

**Success Criteria:**
- `bin/nanoc src_nano/parser_mvp.nano -o bin/parser_mvp` compiles
- `bin/nanoc src_nano/typechecker_minimal.nano -o bin/typechecker_minimal` compiles
- `bin/nanoc src_nano/transpiler_minimal.nano -o bin/transpiler_minimal` compiles
- All three binaries execute without segfault

## Phase 2: Make Components Functional 🔄

**Goal:** Each component produces correct output  
**Priority:** HIGH

### 2.1 Parser Functionality
- ✅ Parse all test cases in `tests/selfhost/`
- ✅ Verify AST construction matches C reference
- ✅ Test with simple programs (arithmetic, functions, control flow)

### 2.2 Typechecker Functionality
- ⏳ Type check all test cases without errors
- ⏳ Catch type mismatches correctly
- ⏳ Handle all type categories (int, bool, string, struct, function)
- ⏳ Symbol table works correctly

### 2.3 Transpiler Functionality  
- ⏳ Generate valid C code for all test cases
- ⏳ Handle all expression types
- ⏳ Handle all statement types
- ⏳ Generated code compiles with gcc
- ⏳ Generated binaries run correctly

**Success Criteria:**
- 8/8 tests in `tests/selfhost/` pass with self-hosted components
- Self-hosted compiler output matches C compiler output

## Phase 3: Full Integration 🎯

**Goal:** Self-hosted compiler compiles itself  
**Priority:** ULTIMATE

### 3.1 Bootstrap Test
```bash
# Stage 1: C compiler
make stage1

# Stage 2: Compile self-hosted components with C compiler  
bin/nanoc src_nano/compiler.nano -o bin/nanoc_stage2

# Stage 3: Compile self-hosted components with stage2
bin/nanoc_stage2 src_nano/compiler.nano -o bin/nanoc_stage3

# Stage 4: Verify stage2 and stage3 produce identical output (bit-for-bit)
diff bin/nanoc_stage2 bin/nanoc_stage3
```

### 3.2 Self-Hosting Verification
**The ultimate test:** Compiler compiles itself, then the output compiles itself again, and the two outputs are identical (fixed point).

**Success Criteria:**
- `bin/nanoc_stage2` successfully compiles all of `src_nano/`
- `bin/nanoc_stage3` (compiled by stage2) produces identical binaries to stage2
- **100% SELF-HOSTED ACHIEVED** 🎉

## Missing Features (Phase 4 - After Self-Hosting)

These aren't needed for self-hosting but would complete the language:

### Language Features:
- ⏳ Arrays/Lists (currently using external runtime)
- ⏳ Tuples (partially implemented, need full support in self-hosted)
- ⏳ Pattern matching (unions)
- ⏳ Generics (beyond List<T>)
- ⏳ Module system (imports)
- ⏳ Closures / first-class functions

### Compiler Features:
- ⏳ Optimization passes
- ⏳ Better error messages
- ⏳ Warning system
- ⏳ Code formatting
- ⏳ Language server protocol (LSP)

## Current File Organization

```
src/              # C reference compiler (4,500 lines)
src_nano/         # Self-hosted compiler in nanolang
  ├── parser_mvp.nano           # 2,767 lines
  ├── typechecker_minimal.nano  # 797 lines
  ├── transpiler_minimal.nano   # 1,081 lines
  ├── compiler.nano             # Integration
  └── [other components]

tests/selfhost/   # Test suite for self-hosted compiler (8 tests)
```

## Bootstrap Chicken-and-Egg Problem Discovered

**Root Cause:** The self-hosted components use advanced features that aren't yet implemented:
- Generic `List<ASTNumber>`, `List<ASTIdentifier>`, etc. (only `List<int>`, `List<string>`, `List<Token>` exist)
- Complex struct field access on function parameters
- Large deeply-nested functions that expose parser/compiler limitations

**What This Means:**
We can't compile the self-hosted components with the current compiler because they use features needed TO BUILD a compiler that supports those features. Classic bootstrap problem!

**Two Paths Forward:**

### Path A: Enhance C Compiler First (Recommended)
1. Add generic list support for arbitrary struct types to C compiler
2. Fix struct field access type inference in C compiler
3. Add parser improvements (better nesting, error recovery)
4. THEN compile self-hosted components with enhanced C compiler

### Path B: Simplify Self-Hosted Components (Harder)
1. Rewrite Parser/Typechecker/Transpiler to use arrays instead of `List<T>`
2. Reduce nesting depth in parser functions
3. Add explicit type annotations everywhere
4. May sacrifice code quality/readability

**Decision:** Path A is better - enhance the C compiler to support the features we need, THEN bootstrap.

## Next Steps (Revised)

**Phase 0: Enhance C Compiler** (NEW - REQUIRED)
1. Implement generic list codegen for any struct type
   - Currently: Manual list_int.c, list_string.c files
   - Needed: Auto-generate list_TYPE.c for any TYPE at compile time
2. Fix struct field access type inference
   - Allow `param.field` where `param: StructType`
3. Parser improvements for deeply nested code
4. Test enhancements with existing test suite

**Phase 1: Compile Self-Hosted Components**
1. Fix parser brace matching issues
2. Fix extern declarations
3. Verify all 3 components compile

**Phase 2-4:** (unchanged)

**Next Session:**
1. Test parser on all 8 selfhost tests
2. Complete typechecker type validation
3. Complete transpiler code generation
4. Achieve Phase 2 completion

**Future Sessions:**
1. Full bootstrap testing (Phase 3)
2. Self-compilation verification
3. Performance optimization
4. **DECLARE 100% SELF-HOSTING** 🎉

## Metrics Tracking

| Metric | Current | Target | Progress |
|--------|---------|--------|----------|
| Components compile | 0/3 | 3/3 | 0% |
| Selfhost tests pass | 0/8 | 8/8 | 0% |
| Self-compilation works | ❌ | ✅ | 0% |
| Bootstrap fixed point | ❌ | ✅ | 0% |
| **Overall Self-Hosting** | **~70%** | **100%** | **70%** |

---

**Last Updated:** 2025-11-30  
**Version:** 0.2.0  
**Status:** Phase 1 In Progress - Fixing Compilation Blockers
