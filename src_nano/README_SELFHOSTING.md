# Self-Hosting Status

## Current Reality

The nanolang compiler has **individual self-hosted components** but **cannot yet fully compile itself** due to:

### 1. Module Integration Issues
- **Duplicate struct definitions**: parser_mvp.nano, typechecker_minimal.nano, and transpiler_minimal.nano all define overlapping structs (ASTNumber, ASTFunction, Parser, etc.)
- **Multiple main() functions**: Each component has its own test main() that conflicts when imported together
- **Type mismatches**: lexer returns `List<Token>` but parser expects `List<LexToken>`

### 2. Nanolang Language Limitations  
- **Scoping bugs**: Variables declared before if/else blocks are not visible in nested else branches
- **No if-without-else**: Parser requires every `if` to have an `else` block
- **Struct field access in nested blocks**: Typechecker incorrectly reports "unused variable" for struct fields accessed in nested else blocks

##What Works

✅ **Individual components** compile and run:
- `lexer_main.nano` - Tokenization (617 lines)
- `parser_mvp.nano` - Parsing (2,768 lines)  
- `typechecker_minimal.nano` - Type checking (797 lines)
- `transpiler_minimal.nano` - C code generation (1,081 lines)

✅ **Minimal C helpers** (acceptable for bootstrapping):
- `file_io.c` - read_file(), write_file() (59 lines)
- `cli_args.c` - get_argc(), get_argv() (34 lines)

## What Doesn't Work

❌ **Removed fake self-hosting** (used C compiler via FFI):
- ~~`stage1_compiler.nano`~~ - Just called C functions
- ~~`compiler_extern.c`~~ - Wrapped C lexer/parser/typechecker (269 lines)
- ~~`compiler_extern.nano`~~ - Extern declarations for the above

## Path to True Self-Hosting

1. **Fix module system**: Allow importing components without duplicate struct definitions
2. **Fix scoping**: Variables should be visible in all if/else branches  
3. **Add conditional imports**: Import only what's needed, not test code
4. **Unified AST types**: Share struct definitions across modules
5. **Integration layer**: Create compiler_main.nano that properly wires components together

## Current Build Process

The Makefile builds self-hosted components individually (Stage 2), but they cannot be combined into a working compiler yet. The C reference compiler (`bin/nanoc`) is still required.
