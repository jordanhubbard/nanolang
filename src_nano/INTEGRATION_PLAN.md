# NanoLang Integrated Compiler - Implementation Plan

## Problem: Struct Duplication

**Duplicated across 3-4 files:**
- `Parser` (4x) - parser_mvp, typechecker, transpiler, ast_shared
- `ASTNumber` (4x)
- `ASTIdentifier` (4x)  
- `ASTBinaryOp` (4x)
- `ASTLet` (4x)
- `ASTReturn` (4x)
- `ASTFunction` (4x)
- And many more...

**Why this happens:**
Each component was developed independently and redefined the structs it needed.

## Solution: Three Approaches

### Option 1: Merge All (FASTEST - DO THIS FIRST!)

Create `src_nano/nanoc_integrated.nano` as a single file:

```
nanoc_integrated.nano (~6,000 lines)
├─ Shared Structs (canonical definitions, once)
│  ├─ Token, Parser, AST nodes
│  └─ Type, TypeEnvironment, CodeGenState
├─ Lexer Functions (from lexer_main.nano)
├─ Parser Functions (from parser_mvp.nano)
├─ Type Checker Functions (from typechecker_minimal.nano)
├─ Transpiler Functions (from transpiler_minimal.nano)
├─ CLI/Main (from compiler.nano)
└─ Shadow Tests
```

**Process:**
1. Copy struct definitions from `ast_shared.nano` (canonical)
2. Add unique structs (`Type`, `TypeEnvironment`, `CodeGenState`)
3. Copy ALL functions from each component
4. Rename conflicting function names if needed
5. Wire together in `main()`

**Result:** One self-contained compiler!

### Option 2: Fix Imports (BETTER - DO LATER)

Once NanoLang supports import aliases:

```nano
import "ast_shared.nano"
import "lexer_main.nano" as Lexer
import "parser_mvp.nano" as Parser  
import "typechecker_minimal.nano" as TC
import "transpiler_minimal.nano" as Trans

fn compile(input: string) -> int {
    let tokens = Lexer.tokenize(source)
    let parser = Parser.parse_program(tokens)
    let result = TC.typecheck_parser(parser)
    let c_code = Trans.transpile_parser(parser)
    ...
}
```

### Option 3: Modular System (FUTURE)

Proper module system with:
- Separate compilation units
- Namespacing
- Symbol resolution
- Link-time integration

## Implementation: Option 1 (NOW)

### Step 1: Create Canonical Structs Section

```nano
/* ==================================================================
 * SHARED TYPE DEFINITIONS (Canonical - used by all phases)
 * ================================================================== */

/* Token from lexer */
struct Token {
    type: int,
    value: string,
    line: int,
    column: int
}

/* Parser state and AST storage */
struct Parser {
    tokens: List<Token>,
    position: int,
    token_count: int,
    /* AST node storage */
    numbers: array<ASTNumber>,
    identifiers: array<ASTIdentifier>,
    /* ... all other AST arrays ... */
    function_count: int,
    has_error: bool
}

/* AST Node types */
struct ASTNumber { value: string, line: int, column: int }
struct ASTIdentifier { name: string, line: int, column: int }
struct ASTBinaryOp { op: int, left_id: int, right_id: int, ... }
/* ... all other AST structs ... */

/* Type system structs */
struct Type { kind: int, name: string }
struct TypeEnvironment { symbols: array<Symbol>, count: int }
struct Symbol { name: string, sym_type: Type, is_mut: bool, is_fn: bool }

/* Code generation state */
struct CodeGenState { temp_counter: int, indent_level: int }
```

### Step 2: Copy Lexer Functions

```nano
/* ==================================================================
 * PHASE 1: LEXICAL ANALYSIS (from lexer_main.nano)
 * ================================================================== */

enum TokenType {
    NUMBER = 0,
    IDENTIFIER = 3,
    STRING = 2,
    /* ... */
}

fn is_identifier_start(c: int) -> bool { ... }
fn is_identifier_char(c: int) -> bool { ... }
fn check_keyword_group1(s: string) -> int { ... }
/* ... all lexer functions ... */
fn tokenize(source: string) -> List<Token> { ... }
```

### Step 3: Copy Parser Functions

```nano
/* ==================================================================
 * PHASE 2: PARSING (from parser_mvp.nano)
 * ================================================================== */

fn parser_new(tokens: List<Token>, count: int) -> Parser { ... }
fn parse_primary(p: Parser) -> Parser { ... }
fn parse_expression(p: Parser) -> Parser { ... }
/* ... all parser functions ... */
fn parse_program(tokens: List<Token>, count: int) -> Parser { ... }
```

### Step 4: Copy Type Checker Functions

```nano
/* ==================================================================
 * PHASE 3: TYPE CHECKING (from typechecker_minimal.nano)
 * ================================================================== */

fn type_int() -> Type { ... }
fn types_equal(t1: Type, t2: Type) -> bool { ... }
fn check_expr_node(parser: Parser, node_id: int, ...) -> Type { ... }
/* ... all typechecker functions ... */
fn typecheck_parser(parser: Parser) -> int { ... }
```

### Step 5: Copy Transpiler Functions

```nano
/* ==================================================================
 * PHASE 4: CODE GENERATION (from transpiler_minimal.nano)
 * ================================================================== */

fn codegen_new() -> CodeGenState { ... }
fn gen_indent(level: int) -> string { ... }
fn type_to_c(nano_type: string) -> string { ... }
fn generate_expression(parser: Parser, node_id: int, ...) -> string { ... }
/* ... all transpiler functions ... */
fn transpile_parser(parser: Parser) -> string { ... }
```

### Step 6: Wire Together in Main

```nano
/* ==================================================================
 * MAIN COMPILATION PIPELINE
 * ================================================================== */

fn compile_file(input: string, output: string) -> int {
    /* 1. Read source */
    let source: string = (read_file input)
    
    /* 2. Lexical analysis */
    let tokens: List<Token> = (tokenize source)
    let token_count: int = (List_Token_length tokens)
    
    /* 3. Parse */
    let parser: Parser = (parse_program tokens token_count)
    
    /* 4. Type check */
    let tc_result: int = (typecheck_parser parser)
    if (!= tc_result 0) {
        (println "Type checking failed!")
        return 1
    }
    
    /* 5. Generate C code */
    let c_code: string = (transpile_parser parser)
    
    /* 6. Write C file */
    let c_file: string = (str_concat output ".c")
    (write_file c_file c_code)
    
    /* 7. Compile C to binary */
    let gcc_cmd: string = (str_concat "gcc " c_file)
    let gcc_cmd2: string = (str_concat gcc_cmd " -o ")
    let gcc_cmd3: string = (str_concat gcc_cmd2 output)
    let gcc_result: int = (system gcc_cmd3)
    
    return gcc_result
}

fn main() -> int {
    /* Parse CLI args */
    let args: CompilerArgs = (parse_args)
    
    if args.show_help {
        (show_usage)
        return 0
    }
    
    /* Compile! */
    return (compile_file args.input_file args.output_file)
}
```

## Build Process

```bash
# Stage 0: C compiler builds integrated compiler
bin/nanoc src_nano/nanoc_integrated.nano -o build/stage1/nanoc

# Stage 1: Integrated compiler compiles itself!
build/stage1/nanoc src_nano/nanoc_integrated.nano -o build/stage2/nanoc

# Stage 2: Verify reproducibility
cmp build/stage1/nanoc build/stage2/nanoc

# Success! Replace bin/nanoc
cp build/stage2/nanoc bin/nanoc
mv bin/nanoc bin/nanoc  # Old C version archived
```

## Testing Strategy

### 1. Compile a Simple Program

```nano
// test_simple.nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

fn main() -> int {
    return (add 2 3)
}

shadow add {
    assert (== (add 2 3) 5)
}

shadow main {
    assert (== (main) 5)
}
```

Test:
```bash
build/stage1/nanoc test_simple.nano -o test_simple
./test_simple
echo $?  # Should be 5
```

### 2. Self-Compile Test

```bash
# Can the NanoLang compiler compile itself?
build/stage1/nanoc src_nano/nanoc_integrated.nano -o build/test/nanoc_test
# Success = TRUE SELF-HOSTING!
```

## Success Criteria

✅ `nanoc_integrated.nano` compiles with C compiler  
✅ `nanoc_integrated` can compile simple NanoLang programs  
✅ `nanoc_integrated` can compile itself (self-compile)  
✅ Stage 1 and Stage 2 outputs are identical (reproducible)  
✅ `bin/nanoc` is replaced with self-hosted version  

## Timeline

- **Day 1:** Create `nanoc_integrated.nano` (merge components)
- **Day 2:** Test Stage 0 (C → NanoLang compiler)  
- **Day 3:** Test Stage 1 (NanoLang → NanoLang compiler)
- **Day 4:** Verify bootstrap, replace bin/nanoc
- **Day 5:** CELEBRATE! 🎉

---

**Next:** Start creating `nanoc_integrated.nano`!
