# NanoLang True Self-Hosting Architecture

## Current Status: Almost There!

We have **all the components** needed for true self-hosting (5,700 lines of NanoLang):

```
src_nano/
├── lexer_main.nano          (610 lines)  ✅ Complete lexer
├── parser_mvp.nano          (2,772 lines) ✅ Complete parser  
├── typechecker_minimal.nano (796 lines)   ✅ Complete type checker
├── transpiler_minimal.nano  (1,069 lines) ✅ Complete transpiler
├── compiler.nano            (248 lines)   ✅ CLI framework
├── ast_shared.nano          (199 lines)   ✅ Shared AST
└── file_io.nano             (27 lines)    ✅ File I/O
```

## The Gap

**What we have:**
- `nanoc_selfhost.nano` (140 lines) - Wrapper that calls `bin/nanoc`
- **Not true self-hosting** - it's a shell script in NanoLang clothing

**What we need:**
- `nanoc_full.nano` - Integrated compiler using NanoLang components
- **True self-hosting** - no delegation to C!

## Architecture: nanoc_full.nano

```nano
/* ============================================================
 * nanoc_full.nano - TRUE Self-Hosted NanoLang Compiler
 * ============================================================
 * Uses NanoLang implementations of all compiler phases.
 * NO delegation to C compiler internals!
 */

/* Step 1: File I/O */
import "file_io.nano"           // read_file(), write_file()

/* Step 2: Lexical Analysis */  
import "lexer_main.nano"        // tokenize(source) -> List<Token>

/* Step 3: Parsing */
import "parser_mvp.nano"        // parse_program(tokens) -> Parser (AST)

/* Step 4: Type Checking */
import "typechecker_minimal.nano"  // typecheck_parser(parser) -> int

/* Step 5: Code Generation */
import "transpiler_minimal.nano"   // transpile_parser(parser) -> string (C code)

/* Step 6: C Compilation (OK to use system compiler) */
// system("gcc generated.c -o output")  // This is FINE!

fn compile_nanolang(input: string, output: string) -> int {
    // 1. Read source
    let source: string = (read_file input)
    
    // 2. Tokenize (NanoLang!)
    let tokens: List<Token> = (tokenize source)
    
    // 3. Parse (NanoLang!)
    let token_count: int = (List_Token_length tokens)
    let parser: Parser = (parse_program tokens token_count)
    
    // 4. Type check (NanoLang!)
    let result: int = (typecheck_parser parser)
    if (!= result 0) {
        return result
    }
    
    // 5. Generate C code (NanoLang!)
    let c_code: string = (transpile_parser parser)
    
    // 6. Write C file
    let c_file: string = (str_concat output ".c")
    (write_file c_file c_code)
    
    // 7. Compile C to binary (using system C compiler is OK!)
    let cmd: string = (str_concat "gcc " c_file)
    let cmd2: string = (str_concat cmd " -o ")
    let cmd3: string = (str_concat cmd2 output)
    return (system cmd3)
}
```

## Why This Is True Self-Hosting

✅ **Lexing in NanoLang** - `lexer_main.nano` does the work  
✅ **Parsing in NanoLang** - `parser_mvp.nano` builds AST  
✅ **Type checking in NanoLang** - `typechecker_minimal.nano` validates  
✅ **Code gen in NanoLang** - `transpiler_minimal.nano` produces C  
✅ **System C compiler** - Just compiles the generated C (THIS IS FINE!)

**Using gcc/clang to compile the generated C is NOT cheating!**
- GCC does this (uses system assembler)
- Rust did this (OCaml → Rust → C → binary)
- The KEY: Compiler logic is in NanoLang ✅

## Current Blocker: Import System

**Problem:**
```nano
import "parser_mvp.nano"        // Struct Token conflicts!
import "typechecker_minimal.nano"  // Struct Token conflicts!
```

Both define `struct Token` and other structs. Need:
1. **Import aliases** (e.g., `import "parser.nano" as Parser`)
2. **Namespacing** (e.g., `Parser.Token`, `TypeChecker.Type`)

**Workaround for NOW:**
We can **merge the components** into one file temporarily:

```
nanoc_integrated.nano (all-in-one, ~5,700 lines)
├── Shared structs (deduplicated)
├── Lexer functions
├── Parser functions  
├── Type checker functions
├── Transpiler functions
└── Main CLI
```

## GCC-Style 3-Stage Bootstrap

Once we have `nanoc_full.nano`:

```bash
# Stage 0: Bootstrap (C compiler builds tools)
gcc src/*.c -o build/bootstrap/nanoc_c

# Stage 1: Self-hosted compiler (C → NanoLang)
build/bootstrap/nanoc_c src_nano/nanoc_full.nano -o build/stage1/nanoc

# Stage 2: Reproducibility check (NanoLang → NanoLang)
build/stage1/nanoc src_nano/nanoc_full.nano -o build/stage2/nanoc

# Stage 3: Verify identical output
cmp build/stage1/nanoc build/stage2/nanoc  # Should be identical!

# FINAL: Replace the C compiler!
cp build/stage2/nanoc bin/nanoc
# bin/nanoc is now SELF-HOSTED! 🎉
```

After this:
- `bin/nanoc` is compiled from NanoLang source
- `bin/nanoc` was compiled by a NanoLang compiler  
- C compiler archived to `build/bootstrap/nanoc_c`
- **TRUE SELF-HOSTING ACHIEVED!**

## Implementation Plan

### Phase 1: Merge Components (IMMEDIATE)
1. Create `nanoc_integrated.nano` (merge all src_nano/ components)
2. Deduplicate struct definitions
3. Test it compiles with C compiler
4. Verify it can compile simple programs

### Phase 2: Bootstrap Test
1. Stage 0: C compiler → nanoc_integrated
2. Stage 1: nanoc_integrated → nanoc_integrated (self-compile!)
3. Verify output works

### Phase 3: Replace bin/nanoc
1. Update Makefile for proper bootstrap
2. Replace `bin/nanoc` with self-hosted version
3. Archive C compiler to `build/bootstrap/`
4. **CELEBRATE! 🎉**

### Phase 4: Cleanup (FUTURE)
1. Fix import system (aliases/namespacing)
2. Split back into modular components
3. Incremental improvements

## Success Criteria

**True self-hosting means:**

✅ `bin/nanoc` is a binary compiled from `src_nano/nanoc_full.nano`  
✅ `bin/nanoc` was compiled by a NanoLang compiler (not C)  
✅ `bin/nanoc` uses NanoLang code for lexing, parsing, type checking, transpiling  
✅ The only C in the pipeline is the system C compiler (gcc/clang)  
✅ We can delete the C source and still compile NanoLang programs  

**NOT required:**
❌ Replacing gcc/clang (that would be a C compiler, not a NanoLang compiler)  
❌ Writing assembly directly (unnecessary complexity)  
❌ Avoiding all C (we transpile to C - that's our design!)

## Analogy

Think of it like TypeScript:
- **TypeScript compiler**: Written in TypeScript ✅
- **Produces**: JavaScript (another language) ✅  
- **Runs on**: Node.js/V8 (C++) ✅
- **Is it self-hosted?**: YES! ✅

NanoLang:
- **NanoLang compiler**: Written in NanoLang ✅
- **Produces**: C code (another language) ✅
- **Uses**: gcc/clang (C/C++) ✅  
- **Is it self-hosted?**: YES! ✅

---

**Next Steps:** Create `nanoc_integrated.nano` and do the bootstrap!
