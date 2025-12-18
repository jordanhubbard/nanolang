# Self-Hosting Feature Gap Analysis

## Current vs Required Features

### ✅ What We Have (Sufficient for Simple Programs)

```
┌─────────────────────────────────────────┐
│         CURRENT NANOLANG v0.1           │
├─────────────────────────────────────────┤
│                                         │
│  Types:                                 │
│  • int, float, bool, string, void      │
│  • array<T> (fixed-size, bounds-safe)  │
│                                         │
│  Control Flow:                          │
│  • if/else statements                   │
│  • while loops                          │
│  • for loops (with range)               │
│                                         │
│  Data:                                  │
│  • Variables (let, mut)                 │
│  • Fixed-size arrays [1, 2, 3]         │
│  • Functions with params               │
│                                         │
│  Standard Library (24 functions):       │
│  • I/O: print, println, assert         │
│  • Math: abs, min, max, sqrt, pow, ... │
│  • String: length, concat, substring   │
│  • Array: at, length, new, set         │
│  • OS: getcwd, getenv, range           │
│                                         │
└─────────────────────────────────────────┘
```

**Good for:**
- Mathematical computations
- Simple algorithms (factorial, fibonacci, primes)
- Array processing
- String manipulation (limited)

**Not sufficient for:**
- Compilers (no compound types, no file I/O)
- Complex data structures (no structs)
- Growing collections (arrays are fixed-size)
- File operations (can't read/write files)

---

### ❌ What We Need (For Compiler Construction)

```
┌─────────────────────────────────────────┐
│    FEATURES NEEDED FOR SELF-HOSTING     │
├─────────────────────────────────────────┤
│                                         │
│  1. STRUCTS ⭐ (Priority #1)           │
│     struct Token {                      │
│         type: int,                      │
│         value: string,                  │
│         line: int                       │
│     }                                   │
│                                         │
│  2. ENUMS                               │
│     enum TokenType {                    │
│         TOKEN_NUMBER,                   │
│         TOKEN_STRING,                   │
│         TOKEN_LPAREN                    │
│     }                                   │
│                                         │
│  3. DYNAMIC LISTS                       │
│     let mut tokens: list<Token>         │
│     (list_push tokens tok)              │
│     (list_get tokens 0)                 │
│                                         │
│  4. FILE I/O                            │
│     let source: string =                │
│         (file_read "prog.nano")         │
│     (file_write "out.c" code)           │
│                                         │
│  5. STRING OPS                          │
│     (str_char_at s 0)                   │
│     (str_format "{0} = {1}" x y)        │
│     (str_split "a,b" ",")               │
│                                         │
│  6. SYSTEM EXECUTION                    │
│     (system "gcc -o prog prog.c")       │
│                                         │
└─────────────────────────────────────────┘
```

---

## Compiler Data Structure Requirements

### Lexer Needs

```
Token = {
    type: TokenType (enum),
    value: string,
    line: int,
    column: int
}

Tokens = list<Token>  (dynamic collection)

Need to:
- Read source file (file I/O)
- Parse character by character (str_char_at)
- Build growing token list (dynamic lists)
- Represent token types (enums)
```

### Parser Needs

```
ASTNode (needs tagged unions or variant types):
- NumberNode { value: int }
- StringNode { value: string }
- BinaryOpNode { op: string, left: ASTNode, right: ASTNode }
- CallNode { name: string, args: list<ASTNode> }
- FunctionNode { name: string, params: list<Param>, body: ASTNode }

Need to:
- Store variable-length AST node lists (dynamic lists)
- Represent different node types (enums/structs)
- Build tree structures (recursive structs)
```

### Type Checker Needs

```
Symbol = {
    name: string,
    type: Type,
    is_mut: bool,
    line: int
}

SymbolTable = dict<string, Symbol>  (or list for linear search)

Need to:
- Store symbols (structs)
- Look up symbols by name (ideally hash table, but list works)
- Track function signatures (structs)
```

### Transpiler Needs

```
Need to:
- Build C code strings (string concatenation, formatting)
- Write output file (file I/O)
- Invoke gcc (system execution)
- Format code nicely (str_format)
```

---

## Feature Comparison Table

| Feature | Current | Needed | Priority | Complexity | Time Est. |
|---------|---------|--------|----------|------------|-----------|
| **Structs** | ❌ | ✅ | P1 | Medium | 6-8 weeks |
| **Enums** | ❌ | ✅ | P1 | Medium | 4-6 weeks |
| **Lists** | ❌ | ✅ | P1 | Medium | 4-6 weeks |
| **File I/O** | ❌ | ✅ | P1 | Low | 2-3 weeks |
| **String ops** | ⚠️ Basic | ✅ Advanced | P1 | Low | 2-3 weeks |
| **System exec** | ❌ | ✅ | P1 | Low | 1-2 weeks |
| **Hash tables** | ❌ | ⚠️ Nice | P2 | High | 6-8 weeks |
| **Result types** | ❌ | ⚠️ Nice | P2 | Medium | 4-6 weeks |
| **Modules** | ❌ | ⚠️ Nice | P2 | High | 8-10 weeks |
| **Generics** | ❌ | ⚠️ Optional | P3 | Very High | 12+ weeks |

**Legend:**
- ✅ Essential
- ⚠️ Nice to have
- ❌ Not needed

**Total time for P1 features:** ~20-28 weeks (5-7 months)

---

## Why These 6 Features?

### 1. Structs - Foundation of Everything

**Without structs:**
```nano
# Can't represent a token properly
let tok_type: int = TOKEN_NUMBER
let tok_value: string = "42"
let tok_line: int = 1
let tok_column: int = 5

# Ugly and error-prone!
```

**With structs:**
```nano
struct Token {
    type: int,
    value: string,
    line: int,
    column: int
}

let tok: Token = Token {
    type: TOKEN_NUMBER,
    value: "42",
    line: 1,
    column: 5
}

# Clean, type-safe, maintainable
```

---

### 2. Enums - Type-Safe Constants

**Without enums:**
```nano
let TOKEN_NUMBER: int = 0
let TOKEN_STRING: int = 1
let TOKEN_LPAREN: int = 2

# Easy to make mistakes with magic numbers
if (== tok.type 2) {  # What is 2?
    # ...
}
```

**With enums:**
```nano
enum TokenType {
    TOKEN_NUMBER = 0,
    TOKEN_STRING = 1,
    TOKEN_LPAREN = 2
}

if (== tok.type TOKEN_LPAREN) {  # Clear!
    # ...
}
```

---

### 3. Dynamic Lists - Growing Collections

**Without lists:**
```nano
# Pre-allocate massive array? How big?
let mut tokens: array<Token> = (array_new 10000 default_token)
let mut token_count: int = 0

# Manually track count, risk overflow
(array_set tokens token_count new_token)
set token_count (+ token_count 1)
```

**With lists:**
```nano
let mut tokens: list<Token> = (list_new)

# Automatically grows as needed
(list_push tokens new_token)

let count: int = (list_length tokens)
```

---

### 4. File I/O - Read Source, Write Output

**Without file I/O:**
```nano
# Can't read source files!
# Can't write generated C code!
# Compiler is useless!
```

**With file I/O:**
```nano
fn compile(source_path: string, output_path: string) -> int {
    let source: string = (file_read source_path)
    let tokens: list<Token> = (tokenize source)
    let ast: ASTNode = (parse tokens)
    let c_code: string = (transpile ast)
    (file_write output_path c_code)
    return 0
}
```

---

### 5. Advanced String Operations - Character Access

**Without char access:**
```nano
# Can't implement lexer!
# How do you parse "(+ 1 2)" character by character?
# Current string functions don't give access to individual chars
```

**With char access:**
```nano
fn tokenize(source: string) -> list<Token> {
    let len: int = (str_length source)
    let mut pos: int = 0
    
    while (< pos len) {
        let c: string = (str_char_at source pos)
        
        if (str_equals c "(") {
            # Handle left paren
        } else {
            if (str_equals c ")") {
                # Handle right paren
            } else {
                # Handle other characters
            }
        }
        
        set pos (+ pos 1)
    }
}
```

---

### 6. System Execution - Invoke GCC

**Without system execution:**
```nano
# Generated C code... now what?
# Can't invoke gcc to compile it
# User has to manually run gcc
```

**With system execution:**
```nano
fn compile_to_binary(c_file: string, output: string) -> int {
    # Transpile nanolang → C
    let c_code: string = (transpile ast)
    (file_write c_file c_code)
    
    # Compile C → binary
    let cmd: string = (str_format "gcc -o {0} {1}" output c_file)
    let exit_code: int = (system cmd)
    
    return exit_code
}
```

---

## Development Strategy

### Phase 1: Core Features (Months 1-6)

**Month 1-2: Structs**
- Design syntax
- Implement in C compiler (lexer, parser, type checker, transpiler)
- Write shadow tests
- Test with examples

**Month 3: Enums**
- Design syntax (start with C-style)
- Implement in C compiler
- Write tests

**Month 4: Dynamic Lists**
- Design API
- Implement generic list<T> or specialized versions
- Write tests

**Month 5: File I/O + String Ops**
- Add file_read, file_write, file_exists
- Add str_char_at, str_char_code, str_format
- Add str_split, str_to_int, str_to_float
- Write tests

**Month 6: System Execution**
- Add system() function
- Write tests
- Security considerations

### Phase 2: Write Compiler (Months 7-9)

**Month 7: Lexer in nanolang**
- Tokenize function
- Token struct
- Shadow tests

**Month 8: Parser in nanolang**
- Parse function
- AST structs
- Shadow tests

**Month 9: Type Checker + Transpiler in nanolang**
- Type checking
- C code generation
- Shadow tests

### Phase 3: Bootstrap (Months 10-12)

**Month 10: First Bootstrap**
- Compile nanolang compiler with C compiler
- Test extensively

**Month 11: Second Bootstrap**
- Compile nanolang compiler with itself
- Compare outputs

**Month 12: Optimization + Release**
- Performance tuning
- Bug fixes
- Documentation
- Release v1.0

---

## Risk Analysis

### Low Risk
- File I/O - Standard C functions
- String ops - Standard C functions
- System execution - Standard C function

### Medium Risk
- Structs - Parser complexity, memory layout
- Enums - Type system changes
- Lists - Memory management, generics

### Mitigation
- Incremental development
- Extensive testing at each step
- Keep C compiler as reference
- Shadow tests for everything

---

## Success Metrics

### Technical Success
- ✅ Nanolang compiler compiles itself
- ✅ Bootstrap process repeatable
- ✅ Output identical across bootstrap levels
- ✅ All tests pass (shadow tests + integration)
- ✅ Performance within 2-3x of C compiler

### Design Success
- ✅ Language stays minimal (< 20 keywords)
- ✅ Language stays unambiguous
- ✅ LLMs can still understand easily
- ✅ Shadow tests still mandatory

### Community Success
- ✅ Documentation complete
- ✅ Examples work
- ✅ Community can contribute
- ✅ Self-hosting is reproducible

---

## Conclusion

**Bottom line:** 6 features bridge the gap from "toy language" to "self-hosting compiler."

**Time investment:** ~6-12 months

**Payoff:**
- Validates nanolang design
- Proves language is practical
- Demonstrates LLM-friendliness
- Achieves independence from C
- Earns community respect

**Next step:** Start with structs (most impactful feature).

---

See [SELF_HOSTING_REQUIREMENTS.md](SELF_HOSTING_REQUIREMENTS.md) for detailed design proposals.

