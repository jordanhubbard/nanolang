# Self-Hosting Next Steps

**Date:** January 2025  
**Status:** Components Complete, Integration Needed

## Current State Assessment

### ✅ Completed Components

1. **Lexer** (`src_nano/lexer_main.nano` - 616 lines)
   - ✅ Complete tokenization
   - ✅ Returns `list_token` type
   - ✅ All shadow tests passing

2. **Parser** (`src_nano/parser_mvp.nano` - 2,336 lines)
   - ✅ Complete recursive descent parser
   - ✅ Parses expressions, statements, definitions
   - ✅ Uses `List<LexToken>` type
   - ✅ All shadow tests passing

3. **Type Checker** (`src_nano/typechecker_minimal.nano` - 468 lines)
   - ✅ Basic type system infrastructure
   - ✅ Symbol table structure
   - ✅ Type equality checking
   - ✅ All shadow tests passing

4. **Transpiler** (`src_nano/transpiler_minimal.nano` - 510 lines)
   - ✅ C code generation logic complete
   - ✅ Expression/statement/function generation
   - ✅ All shadow tests passing
   - ⚠️ Known issue: `array<string>` type problems

5. **File I/O** (`src_nano/file_io.nano`)
   - ✅ Extern declarations for read_file/write_file
   - ✅ File operations available

### ⚠️ Integration Issues

1. **Type Mismatches**
   - Lexer returns `list_token` (runtime type)
   - Parser expects `List<LexToken>` (generic type)
   - Need adapter functions to convert between types

2. **Missing Top-Level Functions**
   - Parser has `parse_definition`, `parse_statement`, `parse_expression`
   - But no `parse_program()` that parses a complete file
   - Need wrapper function that orchestrates parsing

3. **Component Interfaces**
   - Type checker doesn't have a `typecheck(ast)` function
   - Transpiler doesn't have a `transpile(ast)` function
   - Need wrapper functions that use the internal APIs

## Next Steps (Priority Order)

### Step 1: Create Type Adapters (1-2 days)

**Goal:** Bridge type mismatches between components

**Tasks:**
1. Create `convert_list_token_to_lex_tokens()` function
   - Converts `list_token` → `List<LexToken>`
   - Handles token count calculation
   - Maps token types correctly

2. Create `convert_parser_to_ast()` function
   - Extracts AST from Parser state
   - Creates unified AST representation
   - Handles all node types

**Files to create:**
- `src_nano/type_adapters.nano`

### Step 2: Create Parser Wrapper (1 day)

**Goal:** Add `parse_program()` function

**Tasks:**
1. Create `parse_program(tokens: List<LexToken>) -> Parser` function
   - Uses `parser_new()` to create parser state
   - Calls `parse_definition()` in loop
   - Handles errors and returns complete parser state

2. Add to `parser_mvp.nano` or create `parser_wrapper.nano`

**Function signature:**
```nano
fn parse_program(tokens: List<LexToken>, token_count: int) -> Parser {
    let p: Parser = (parser_new tokens token_count)
    
    /* Parse all definitions */
    while (not (parser_is_at_end p)) {
        if (parser_has_error p) {
            break
        } else {
            (print "")
        }
        set p (parse_definition p)
    }
    
    return p
}
```

### Step 3: Create Type Checker Wrapper (1-2 days)

**Goal:** Add `typecheck()` function

**Tasks:**
1. Create `typecheck(parser: Parser) -> TypedParser` function
   - Creates type environment
   - Walks AST nodes from parser
   - Validates types
   - Returns typed AST or error

2. Add to `typechecker_minimal.nano` or create wrapper

**Function signature:**
```nano
fn typecheck(parser: Parser) -> TypedParser {
    let env: TypeEnvironment = (env_new)
    /* Walk AST and type check */
    /* Return typed parser state */
}
```

### Step 4: Create Transpiler Wrapper (1 day)

**Goal:** Add `transpile()` function

**Tasks:**
1. Create `transpile(typed_parser: TypedParser) -> string` function
   - Extracts functions from parser
   - Generates C code for each function
   - Combines into complete C program
   - Returns C code string

2. Fix `array<string>` type issues if possible
3. Add to `transpiler_minimal.nano` or create wrapper

**Function signature:**
```nano
fn transpile(typed_parser: TypedParser) -> string {
    /* Extract functions from parser */
    /* Generate C code */
    /* Return complete C program */
}
```

### Step 5: Complete Integration Pipeline (1-2 days)

**Goal:** Connect all components end-to-end

**Tasks:**
1. Update `compiler_main.nano` or create `compiler_integration.nano`
2. Wire up: lexer → parser → typechecker → transpiler
3. Add proper error handling
4. Add progress reporting

**Pipeline:**
```nano
fn compile_file(input_path: string, output_path: string) -> int {
    /* 1. Read source */
    let source: string = (read_file input_path)
    
    /* 2. Tokenize */
    let tokens: list_token = (tokenize source)
    let lex_tokens: List<LexToken> = (convert_list_token_to_lex_tokens tokens)
    
    /* 3. Parse */
    let parser: Parser = (parse_program lex_tokens (list_token_length tokens))
    
    /* 4. Type check */
    let typed_parser: TypedParser = (typecheck parser)
    
    /* 5. Transpile */
    let c_code: string = (transpile typed_parser)
    
    /* 6. Write output */
    return (write_file output_path c_code)
}
```

### Step 6: End-to-End Testing (1 day)

**Goal:** Test complete compilation pipeline

**Tasks:**
1. Create simple test program (`test_hello.nano`)
2. Compile with self-hosted compiler
3. Verify generated C code
4. Compile C code with gcc
5. Run executable
6. Verify output

**Test Program:**
```nano
fn main() -> int {
    (println "Hello, World!")
    return 0
}
```

## Implementation Strategy

### Option A: Incremental Integration (Recommended)
1. Start with type adapters
2. Add parser wrapper
3. Add type checker wrapper
4. Add transpiler wrapper
5. Connect everything
6. Test end-to-end

**Pros:** Testable at each step, clear progress  
**Cons:** May need multiple iterations

### Option B: Complete Integration First
1. Create all wrappers at once
2. Connect everything
3. Fix issues as they arise

**Pros:** Faster initial integration  
**Cons:** Harder to debug, more complex

## Estimated Timeline

- **Type Adapters:** 1-2 days
- **Parser Wrapper:** 1 day
- **Type Checker Wrapper:** 1-2 days
- **Transpiler Wrapper:** 1 day
- **Integration:** 1-2 days
- **Testing:** 1 day

**Total:** 6-9 days

## Success Criteria

✅ Can compile simple hello world program  
✅ Generated C code compiles with gcc  
✅ Executable runs and produces correct output  
✅ All components integrated without type errors  
✅ Error handling works correctly  

## Known Blockers

1. **Type System Issues**
   - `array<string>` support in C compiler
   - May need workarounds or fixes

2. **Missing Function Signatures**
   - Need to determine exact interfaces
   - May need to modify components slightly

3. **Error Handling**
   - Need consistent error propagation
   - Need helpful error messages

## Files to Create/Modify

**New Files:**
- `src_nano/type_adapters.nano` - Type conversion functions
- `src_nano/parser_wrapper.nano` - parse_program wrapper (or add to parser_mvp.nano)
- `src_nano/typechecker_wrapper.nano` - typecheck wrapper (or add to typechecker_minimal.nano)
- `src_nano/transpiler_wrapper.nano` - transpile wrapper (or add to transpiler_minimal.nano)
- `src_nano/compiler_integration.nano` - Complete integration (or update compiler_main.nano)

**Test Files:**
- `examples/self_host_test_hello.nano` - Simple test program

## Next Action

**Start with Step 1: Create Type Adapters**

This is the foundation for everything else. Once we can convert between `list_token` and `List<LexToken>`, we can connect the lexer and parser.

---

**Status:** Ready to begin implementation  
**Priority:** High - Blocking self-hosting completion  
**Owner:** Self-hosting team
