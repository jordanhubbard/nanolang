# Full Self-Hosting Plan

## Current Status: Pseudo Self-Hosting

What we have now (Stage 1 & 2):
- ✅ Compiler wrapper written in nanolang (237 lines)
- ❌ Uses C FFI for actual compilation (calls lexer.c, parser.c, etc.)
- ❌ NOT truly self-hosting - just a wrapper

## Goal: TRUE Self-Hosting

Requirements:
1. **Entire compiler in pure nanolang** (no C FFI for core compilation)
2. **Entire interpreter in pure nanolang**
3. **Stage 3 verification**: Stage 2 compiles itself → Stage 3, identical output

## Scope of Work

### C Code to Rewrite (13,361 lines)

```
Component              C Lines    Status         nanolang Lines
------------------------------------------------------------------
lexer.c                327        EXISTS         447 (lexer_complete.nano)
parser.c               2,581      EXISTS         321 (parser_complete.nano)
typechecker.c          3,360      EXISTS         468 (typechecker_minimal.nano)
transpiler.c           3,063      EXISTS         510 (transpiler_minimal.nano)
eval.c (interpreter)   3,155      MISSING        ~3,000 needed
env.c                  875        MISSING        ~800 needed
module.c               ?          MISSING        ~500 needed
------------------------------------------------------------------
TOTAL                  13,361                    ~6,046 (partial)
```

### Existing nanolang Components

Already in src_nano/:
- ✅ lexer_complete.nano (447 lines) - Full lexer implementation
- ✅ parser_complete.nano (321 lines) - Parser with AST
- ✅ typechecker_minimal.nano (468 lines) - Type checking logic
- ✅ transpiler_minimal.nano (510 lines) - C code generation
- ❌ No eval.nano (interpreter)
- ❌ No env.nano (environment/symbol table)
- ❌ No module.nano (module system)

**Total existing: ~1,746 lines**
**Still needed: ~4,300 lines**

## Implementation Plan

### Phase 1: Assess & Test Existing Components (Current)

1. Test if lexer_complete.nano compiles and works
2. Test if parser_complete.nano compiles and works
3. Test if typechecker_minimal.nano compiles and works
4. Test if transpiler_minimal.nano compiles and works
5. Identify gaps and missing functionality

### Phase 2: Complete Missing Components

1. **env.nano** (~800 lines)
   - Symbol table management
   - Scope handling
   - Function/variable lookup
   - Type environment

2. **eval.nano** (~3,000 lines)
   - Expression evaluation
   - Statement execution
   - Function calls
   - Control flow
   - Memory management

3. **module.nano** (~500 lines)
   - Import/export handling
   - Module resolution
   - Dependency management

### Phase 3: Integration

1. Create **compiler_pure.nano** that integrates:
   - lexer_complete.nano
   - parser_complete.nano
   - typechecker_minimal.nano (or complete)
   - transpiler_minimal.nano (or complete)
   - env.nano
   - module.nano

2. Create **interpreter_pure.nano** that integrates:
   - lexer_complete.nano
   - parser_complete.nano
   - typechecker_minimal.nano
   - eval.nano
   - env.nano
   - module.nano

### Phase 4: Build Pure Stage 2

1. Use Stage 1 (C-backed) to compile compiler_pure.nano → Stage 2 (pure)
2. Test Stage 2 can compile simple programs
3. Test Stage 2 can compile complex programs
4. Test Stage 2 can compile ALL examples

### Phase 5: Stage 3 Verification

1. Use Stage 2 (pure) to compile compiler_pure.nano → Stage 3
2. Compare Stage 2 output vs Stage 3 output
3. **Verify bit-identical** (or functionally identical)
4. If identical: **TRUE SELF-HOSTING ACHIEVED!**

### Phase 6: Interpreter Self-Hosting

1. Use Stage 2 to build interpreter_pure.nano
2. Test interpreter can run programs
3. Test interpreter can run the compiler!
4. Full language bootstrap complete

## Technical Challenges

### 1. Data Structures

C uses:
- Complex structs (Token, ASTNode, etc.)
- Linked lists
- Dynamic arrays
- Hash tables

nanolang needs:
- Struct definitions for all AST nodes
- List<Token>, List<ASTNode>
- String manipulation utilities
- Memory management

### 2. Recursive Data Structures

AST nodes contain pointers to other AST nodes. nanolang needs:
- Proper struct typing
- Potentially opaque types or int handles
- Reference management

### 3. Error Handling

C uses:
- Return codes
- Global error state
- fprintf to stderr

nanolang needs:
- Consistent error propagation
- Error messages
- Graceful failure

### 4. String Processing

Heavy string manipulation for:
- Lexing
- Code generation
- Error messages

nanolang needs:
- Efficient string operations
- String builder pattern
- Character access

### 5. File I/O

Compiler needs:
- Read source files
- Write generated C code
- Import resolution

Already have file_io.nano, but may need expansion.

## Success Criteria

✅ **Stage 2 (pure nanolang) compiles itself**
✅ **Stage 3 output identical to Stage 2**
✅ **All examples compile and run with Stage 2**
✅ **All tests pass with Stage 2**
✅ **Interpreter written in nanolang works**
✅ **No C FFI for core compilation logic**

## Timeline Estimate

- Phase 1 (Assessment): 1-2 hours
- Phase 2 (Missing components): 10-20 hours
- Phase 3 (Integration): 5-10 hours
- Phase 4 (Build & Test): 3-5 hours
- Phase 5 (Stage 3 verification): 2-3 hours
- Phase 6 (Interpreter): 5-10 hours

**Total: 26-50 hours of development**

## Current Next Steps

1. ✅ Understand the scope (this document)
2. ⏳ Test lexer_complete.nano - does it work?
3. ⏳ Test parser_complete.nano - does it work?
4. ⏳ Test typechecker_minimal.nano - does it work?
5. ⏳ Test transpiler_minimal.nano - does it work?
6. ⏳ Identify what's missing or broken
7. ⏳ Start implementing env.nano
8. ⏳ Start implementing eval.nano

## Notes

- This is a MAJOR undertaking - essentially rewriting the entire language in itself
- May need to add language features to nanolang to support this (e.g., better struct support)
- Performance will be slower than C, but that's expected
- The goal is **correctness first, performance later**
- This proves the language is **Turing complete** and **expressive enough** to implement itself
