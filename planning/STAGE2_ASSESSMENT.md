# Stage 2 Assessment: Full Self-Hosting Challenges

**Date:** November 13, 2025  
**Status:** In Progress - Challenges Identified

---

## Current Progress

### ✅ Completed
1. **Stage 0**: C compiler fully functional
2. **Lexer in Nanolang**: `src_nano/lexer_main.nano`
   - 577 lines, fully functional
   - Compiles successfully
   - All shadow tests pass
   - Uses `list_token` for token storage
3. **AST Types (Minimal)**: `src_nano/ast_types.nano`
   - Basic struct definitions
   - Compiles successfully
   - Demonstrates struct usage

---

## Critical Challenges Identified

### 1. Union Types Not Supported

**Problem:**
- C `ASTNode` uses a union to represent different node types
- Nanolang doesn't have union support

**C Code Example:**
```c
struct ASTNode {
    ASTNodeType type;
    union {
        long long number;
        struct { char *name; ASTNode **args; int arg_count; } call;
        struct { ASTNode *condition; ASTNode *then_branch; ASTNode *else_branch; } if_stmt;
        // ... 20+ more variants
    } as;
};
```

**Solutions:**
- **Option A:** Add union support to nanolang (3-4 weeks)
- **Option B:** Use separate struct types for each node (complex, many types)
- **Option C:** Use tagged union pattern with multiple fields (wastes memory)
- **Option D:** Use generic approach with lists and type checking at runtime

### 2. Pointer Types and Dynamic Allocation

**Problem:**
- AST requires pointer types (`ASTNode**`, `char**`)
- Dynamic allocation of AST nodes
- Recursive data structures

**Current Nanolang Support:**
- Lists (`list_int`, `list_string`, `list_token`) for arrays
- Structs can contain other structs by value
- No explicit pointer types
- No manual memory management

**Implications:**
- Can't represent `ASTNode**` (array of AST pointers)
- Can't create tree structures easily
- Can't implement recursive descent parser naturally

### 3. Heterogeneous Collections

**Problem:**
- Need arrays of mixed types (e.g., different AST node types in a block)
- Need arrays of structs with different internal structures

**Current Nanolang Support:**
- `list_int`, `list_string`, `list_token` - all homogeneous
- Arrays have fixed element types
- No generic list or `void*` equivalent

### 4. String Building and Manipulation

**Problem:**
- Transpiler needs efficient string building (StringBuilder in C)
- Need string formatting, concatenation, large buffers

**Current Nanolang Support:**
- `str_concat` for basic concatenation
- No efficient string builder
- String operations allocate new strings (inefficient)

**Required:**
- StringBuilder implementation
- String formatting (like sprintf)
- Efficient multi-append operations

### 5. File I/O

**Problem:**
- Compiler needs to read source files
- Write generated C code
- Execute system commands (gcc)

**Current Nanolang Support:**
- None - no file I/O functions
- No `extern` functions for `fopen`, `fread`, `fwrite`
- No `system()` call

**Required:**
- Add extern declarations for file operations
- Add safe wrappers for file I/O

### 6. Complex Type System

**Problem:**
- Type checker needs to represent complex types:
  - `array<int>` - typed arrays
  - `struct Point` - named struct types  
  - `list_int`, `list_string` - list types
  - Function signatures with parameter types

**Current Representation:**
- Type enum (TYPE_INT, TYPE_ARRAY, etc.)
- Additional fields for struct names, element types
- Complex to represent in nanolang without pointers

---

## Estimated Implementation Timeline

### Phase 1: Language Extensions (8-12 weeks)
1. **Union Types** (3-4 weeks)
   - Design union syntax
   - Lexer/parser updates
   - Type checker support
   - Transpiler codegen
   - Testing

2. **Generic Lists** (2-3 weeks)
   - `list<T>` generic type
   - Type parameter support
   - Runtime implementation
   - Testing

3. **File I/O** (1-2 weeks)
   - Extern declarations for file ops
   - Safe wrappers
   - Testing

4. **String Builder** (1 week)
   - Efficient string building type
   - append, format operations
   - Testing

5. **System Execution** (1 week)
   - Extern for `system()`
   - Safe command building
   - Testing

### Phase 2: Compiler Components (16-20 weeks)
1. **Complete AST Definitions** (2-3 weeks)
   - All 25+ node types
   - Helper functions
   - Testing

2. **Parser** (6-8 weeks)
   - Recursive descent implementation
   - All expression types
   - All statement types
   - Error handling
   - Testing

3. **Type Checker** (5-6 weeks)
   - Symbol table management
   - Type inference
   - Type checking all expressions
   - Error messages
   - Testing

4. **Transpiler** (6-7 weeks)
   - C code generation for all node types
   - String building
   - Indentation and formatting
   - Testing

5. **Main Driver** (1-2 weeks)
   - Command-line parsing
   - Orchestration
   - Error handling
   - Testing

### Phase 3: Integration & Validation (4-6 weeks)
1. **Component Integration** (2-3 weeks)
2. **Testing Suite** (1-2 weeks)
3. **Bug Fixes** (1-2 weeks)
4. **Performance Optimization** (1-2 weeks)

**Total Estimated Time: 28-38 weeks (7-9 months)**

---

## Simplified Alternative: Stage 1.5 (Hybrid)

Instead of full self-hosting, implement a hybrid approach:

### Stage 1.5 Components:
1. ✅ **Lexer**: Nanolang (`lexer_main.nano`)
2. **Parser**: Nanolang (simplified subset)
3. **Type Checker**: C (keep existing)
4. **Transpiler**: C (keep existing)
5. **Main**: C with nanolang lexer bridge

**Benefits:**
- Proves lexer works in production
- Demonstrates self-hosting is feasible
- Much faster (2-4 weeks)
- Can iterate on design

**Implementation:**
1. Create C bridge function to convert `list_token` → `Token*`
2. Compile nanolang lexer with Stage 0
3. Link compiled lexer with C parser/typechecker/transpiler
4. Test with all examples

**Timeline: 2-4 weeks**

---

## Recommendation

Given the challenges identified, I recommend:

### Short-Term (1-2 months):
1. **Complete Stage 1.5** (hybrid compiler)
   - Nanolang lexer + C rest
   - Validates self-hosting approach
   - Production-ready lexer

2. **Add Minimal Language Features**
   - File I/O support
   - Basic union types
   - Generic `list<T>` for common types

### Medium-Term (3-6 months):
3. **Implement Parser in Nanolang**
   - Work around union limitations
   - Test extensively
   - Stage 1.75: Nanolang lexer+parser, C typechecker+transpiler

4. **Document Limitations**
   - What needs unions
   - What needs pointers
   - Design decisions

### Long-Term (6-12 months):
5. **Complete Type Checker in Nanolang**
6. **Complete Transpiler in Nanolang**
7. **Full Stage 2 Integration**
8. **Stage 3: Bootstrap** (compile Stage 2 with Stage 2)

---

## Current Files Status

### Working:
- `src_nano/lexer_main.nano` (577 lines) ✅
- `src_nano/ast_types.nano` (minimal) ✅
- `src_nano/token_types.nano` ✅

### In Progress:
- `planning/STAGE2_ASSESSMENT.md` (this file) 🚧

### TODO:
- `src_nano/env_types.nano`
- `src_nano/parser.nano`
- `src_nano/typechecker.nano`
- `src_nano/transpiler.nano`
- `src_nano/main.nano`
- `src_nano/compiler.nano` (integration)

---

## Decision Points

1. **Should we pursue full Stage 2 now?**
   - ✅ Pros: Complete self-hosting
   - ❌ Cons: 7-9 months, requires language extensions

2. **Should we do Stage 1.5 first?**
   - ✅ Pros: Quick validation, production lexer
   - ✅ Pros: Identifies real issues
   - ❌ Cons: Still need Stage 2 later

3. **Should we extend the language first?**
   - ✅ Pros: Makes self-hosting easier
   - ❌ Cons: Moves goalposts, delays self-hosting

**Recommended Path:** Stage 1.5 → Language Extensions → Full Stage 2

---

**Last Updated:** 2025-11-13  
**Next Steps:** Decide on approach (1.5 vs full Stage 2)

