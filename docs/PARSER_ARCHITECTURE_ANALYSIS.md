# Parser Architecture Analysis

**Date**: November 23, 2025  
**Status**: Analysis Document

## Current Architecture

### Shared Parser Implementation

Both the interpreter (`./bin/nano`) and compiler (`./bin/nanoc`) use the **same parser**:

- **Lexer**: `tokenize()` in `src/lexer.c` (~327 lines)
- **Parser**: `parse_program()` in `src/parser.c` (~2,568 lines)
- **Type Checker**: `type_check()` in `src/typechecker.c` (~3,304 lines)

Both call the same functions in the same order:
1. `tokenize(source, &token_count)` - Lexing
2. `parse_program(tokens, token_count)` - Parsing
3. `process_imports(program, env, modules, input_file)` - Module loading
4. `type_check(program, env)` - Type checking

### Code Size

Total compiler/interpreter codebase:
- `parser.c`: 2,568 lines
- `lexer.c`: 327 lines
- `typechecker.c`: 3,304 lines
- `transpiler.c`: 3,042 lines (compiler only)
- `eval.c`: 3,151 lines (interpreter only)
- **Total**: ~12,392 lines of C code

## Why Output Differs

The parity script shows differences, but these are **not parsing differences**:

### 1. Warning Output Differences

**Interpreter** shows warnings during type checking:
```
Warning: Function names 'or_op' and 'not_op' are very similar...
```

**Compiler** also shows warnings, but they may be:
- Filtered by the parity script (`grep -v` filters)
- Mixed with shadow test output
- Sent to stderr vs stdout differently

**Root Cause**: Both use `type_check()` with `warnings_enabled = true`, so warnings should appear in both. The difference is in **output capture**, not parsing.

### 2. Shadow Test Output

**Compiler** runs shadow tests during compilation:
```
Running shadow tests...
Testing and_op... PASSED
Testing or_op... PASSED
```

**Interpreter** doesn't run shadow tests (they're compile-time only).

**Root Cause**: Shadow tests are part of the compilation process, not parsing.

### 3. Error Message Differences

Some examples show different error messages, but this is typically due to:
- Different error reporting contexts
- Different error recovery strategies
- Different validation timing

**Root Cause**: Error handling differences, not parsing differences.

## Do We Need a Unified Parser?

### Current State: ✅ Already Unified

The parser **is already unified** - both interpreter and compiler use the exact same:
- Lexer (`tokenize`)
- Parser (`parse_program`)
- Type checker (`type_check`)

### The Real Question: Self-Hosting

The question isn't about unifying the parser - it's about **self-hosting**: rewriting the compiler in nanolang itself.

## Self-Hosting Path

### Current Architecture (C Implementation)

```
┌─────────────────┐
│  Source File    │
│  (.nano)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Lexer (C)      │  ← tokenize()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Parser (C)     │  ← parse_program()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Type Checker(C) │  ← type_check()
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│Eval (C)│ │Transpile │
│        │ │   (C)    │
└────────┘ └──────────┘
```

### Self-Hosted Architecture (nanolang Implementation)

```
┌─────────────────┐
│  Source File    │
│  (.nano)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Lexer (nanolang)│  ← Written in nanolang
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Parser (nanolang)│  ← Written in nanolang
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Type Checker     │  ← Written in nanolang
│  (nanolang)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│Eval    │ │Transpile │
│(nanolang)│ │(nanolang)│
└────────┘ └──────────┘
```

## Benefits of Self-Hosting

### 1. Single Source of Truth

Currently, the parser logic exists only in C. If we want to:
- Add new language features
- Fix parsing bugs
- Improve error messages
- Refactor the parser

We must modify C code. With self-hosting:
- Parser logic is in nanolang
- Easier to understand and modify
- Language evolves with the compiler

### 2. Guaranteed Consistency

With a self-hosted compiler:
- Interpreter and compiler use the **same nanolang code**
- No possibility of divergence
- Single parser implementation
- Single type checker implementation

### 3. Language Evolution

Self-hosting enables:
- Compiler written in the language it compiles
- Language features tested on the compiler itself
- Dogfooding: compiler is the first complex program
- Confidence that language is powerful enough

### 4. Reduced Maintenance Burden

Currently:
- ~12,392 lines of C code to maintain
- C-specific bugs and memory management
- Platform-specific issues

With self-hosting:
- Compiler written in nanolang (safer, simpler)
- Memory management handled by GC
- Platform portability through C runtime

## Current Status: Not Self-Hosted

The compiler is **not yet self-hosted**. All components are in C:
- ✅ Lexer: C (`src/lexer.c`)
- ✅ Parser: C (`src/parser.c`)
- ✅ Type Checker: C (`src/typechecker.c`)
- ✅ Transpiler: C (`src/transpiler.c`)
- ✅ Interpreter: C (`src/eval.c`)

## Path to Self-Hosting

### Phase 1: Prerequisites (Current)

We need these features to write a compiler in nanolang:

1. **Structs** ✅ (Implemented)
   - Represent tokens, AST nodes, symbols
   - Field access and initialization

2. **Enums** ✅ (Implemented)
   - Represent token types, AST node types
   - Type-safe constants

3. **Generic Lists** ✅ (Implemented)
   - `List<Token>`, `List<ASTNode>`, etc.
   - Dynamic data structures

4. **File I/O** ✅ (Implemented)
   - Read source files
   - Write C output

5. **Advanced String Operations** ✅ (Implemented)
   - Character access, parsing, formatting

6. **Module System** ✅ (Implemented)
   - Split compiler across files

### Phase 2: Rewrite Components (Future)

Once prerequisites are met, rewrite in nanolang:

1. **Lexer** (nanolang)
   - Read source file
   - Tokenize into `List<Token>`
   - Return token stream

2. **Parser** (nanolang)
   - Take `List<Token>`
   - Build `ASTNode` tree
   - Return `ASTNode*`

3. **Type Checker** (nanolang)
   - Take `ASTNode*`
   - Build symbol table
   - Validate types
   - Return validated AST

4. **Transpiler** (nanolang)
   - Take validated AST
   - Generate C code
   - Write to file

5. **Interpreter** (nanolang)
   - Take validated AST
   - Execute directly
   - Return results

### Phase 3: Bootstrap (Future)

1. Write compiler in nanolang
2. Compile compiler with C compiler (bootstrap)
3. Use nanolang compiler to compile itself
4. Verify output matches C compiler
5. Switch to nanolang compiler as primary

## Recommendations

### Short Term: Fix Output Differences

The current "parsing differences" are actually **output handling differences**:

1. **Unify Warning Output**
   - Ensure warnings appear in both interpreter and compiler
   - Standardize warning format
   - Make warnings optional (flag)

2. **Unify Error Messages**
   - Ensure same error messages in both
   - Same error recovery behavior
   - Same validation timing

3. **Update Parity Script**
   - Better handle shadow test output
   - Filter warnings consistently
   - Compare semantic equivalence, not exact output

### Long Term: Self-Hosting

1. **Complete Prerequisites**
   - All features needed for compiler construction
   - Test with compiler-like programs

2. **Start with Lexer**
   - Simplest component
   - Good test of string/file I/O
   - Can coexist with C lexer

3. **Gradual Migration**
   - Rewrite one component at a time
   - Keep C version as fallback
   - Verify parity at each step

4. **Bootstrap Process**
   - Document bootstrap procedure
   - Test bootstrap on multiple platforms
   - Ensure reproducibility

## Conclusion

### Current State

- ✅ **Parser is already unified** - both use same C code
- ⚠️ **Output differences exist** - but not parsing differences
- ❌ **Not self-hosted** - all code is in C

### Answer to Question

**"Do we need a unified parser written in nanolang for both?"**

**Short answer**: No, the parser is already unified (same C code). But **yes**, we should eventually rewrite it in nanolang for self-hosting.

**Why self-hosting matters**:
1. Single source of truth (no C/nanolang divergence)
2. Language evolution (compiler tests language features)
3. Reduced maintenance (nanolang is safer than C)
4. Guaranteed consistency (interpreter = compiler = same code)

**Current priority**: Fix output differences (warnings, error messages) to ensure true parity, then work toward self-hosting as a long-term goal.
