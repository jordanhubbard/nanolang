# Self-Hosting Implementation Checklist

Track progress toward nanolang self-hosting (compiler written in nanolang).

## Essential Features (P1) - Must Have

### 1. Structs ⭐ Priority #1

**Status:** ❌ Not Started

**Syntax Design:**
```nano
struct Token {
    type: int,
    value: string,
    line: int,
    column: int
}

let tok: Token = Token { type: 0, value: "42", line: 1, column: 5 }
let line: int = tok.line
```

**Implementation Tasks:**
- [ ] Design complete syntax (declarations, literals, field access)
- [ ] Add TOKEN_STRUCT to lexer
- [ ] Parse struct declarations
- [ ] Parse struct literals
- [ ] Parse field access (dot notation)
- [ ] Type checker: struct types
- [ ] Type checker: field type checking
- [ ] Transpiler: generate C structs
- [ ] Transpiler: struct initialization
- [ ] Environment: track struct definitions
- [ ] Write shadow tests
- [ ] Update documentation

**Estimated Time:** 6-8 weeks

**Blocker for:** Parser, type checker (AST representation)

---

### 2. Enums (Simple)

**Status:** ❌ Not Started

**Syntax Design:**
```nano
enum TokenType {
    TOKEN_NUMBER = 0,
    TOKEN_STRING = 1,
    TOKEN_LPAREN = 2
}

let t: int = TOKEN_NUMBER
```

**Implementation Tasks:**
- [ ] Design syntax (C-style enums, integer values)
- [ ] Add TOKEN_ENUM to lexer
- [ ] Parse enum declarations
- [ ] Type checker: enum values as constants
- [ ] Transpiler: generate C enums
- [ ] Environment: track enum definitions
- [ ] Write shadow tests
- [ ] Update documentation

**Estimated Time:** 4-6 weeks

**Blocker for:** Better token type representation

---

### 3. Dynamic Lists

**Status:** ❌ Not Started

**Syntax Design:**
```nano
let mut tokens: list<Token> = (list_new)
(list_push tokens tok)
let first: Token = (list_get tokens 0)
let count: int = (list_length tokens)
```

**Implementation Tasks:**
- [ ] Design list API (new, push, get, set, length, clear)
- [ ] Decide: Generic list<T> or specialized versions?
- [ ] Add list type to type system
- [ ] Implement list operations in C runtime
- [ ] Add to transpiler (generate correct C types)
- [ ] Write shadow tests for each operation
- [ ] Test with growing lists (stress test)
- [ ] Update documentation

**Estimated Time:** 4-6 weeks

**Blocker for:** Token storage, AST node storage

---

### 4. File I/O

**Status:** ❌ Not Started

**Syntax Design:**
```nano
let source: string = (file_read "program.nano")
(file_write "output.c" c_code)
let exists: bool = (file_exists "input.nano")
```

**Implementation Tasks:**
- [ ] Add file_read(path: string) -> string
- [ ] Add file_write(path: string, content: string) -> void
- [ ] Add file_exists(path: string) -> bool
- [ ] Implement in C runtime (fopen, fread, fwrite, fclose)
- [ ] Error handling (return empty string on failure)
- [ ] Add to stdlib documentation
- [ ] Write shadow tests
- [ ] Test with real files

**Estimated Time:** 2-3 weeks

**Blocker for:** Reading source files, writing output

---

### 5. Advanced String Operations

**Status:** ⚠️ Partial (have basic string ops)

**Syntax Design:**
```nano
let c: string = (str_char_at "Hello" 0)        # "H"
let code: int = (str_char_code "A")             # 65
let s: string = (str_from_code 65)              # "A"
let parts: array<string> = (str_split "a,b" ",")  # ["a", "b"]
let num: int = (str_to_int "42")                # 42
let f: float = (str_to_float "3.14")            # 3.14
let formatted: string = (str_format "x={0}" "5")  # "x=5"
```

**Implementation Tasks:**
- [ ] Add str_char_at(s: string, index: int) -> string
- [ ] Add str_char_code(s: string) -> int
- [ ] Add str_from_code(code: int) -> string
- [ ] Add str_split(s: string, delim: string) -> array<string>
- [ ] Add str_to_int(s: string) -> int (0 on failure)
- [ ] Add str_to_float(s: string) -> float (0.0 on failure)
- [ ] Add str_format(template: string, arg: string) -> string (basic)
- [ ] Implement in C runtime
- [ ] Add to stdlib documentation
- [ ] Write shadow tests
- [ ] Test with lexer use cases

**Estimated Time:** 2-3 weeks

**Blocker for:** Lexer (character-by-character parsing)

---

### 6. System Execution

**Status:** ❌ Not Started

**Syntax Design:**
```nano
let exit_code: int = (system "gcc -o prog prog.c")
```

**Implementation Tasks:**
- [ ] Add system(cmd: string) -> int
- [ ] Implement using C system() function
- [ ] Consider security implications (command injection)
- [ ] Add to stdlib documentation
- [ ] Write shadow tests (careful with side effects)
- [ ] Test with gcc invocation

**Estimated Time:** 1-2 weeks

**Blocker for:** Invoking C compiler

---

## Quality-of-Life Features (P2) - Nice to Have

### 7. Hash Tables

**Status:** ❌ Not Started

**Benefit:** O(1) symbol lookup instead of O(n)

**Estimated Time:** 6-8 weeks

**Decision:** Start without, optimize later if needed

---

### 8. Result Types (Error Handling)

**Status:** ❌ Not Started

**Benefit:** Graceful error propagation

**Estimated Time:** 4-6 weeks (requires enums + pattern matching)

**Decision:** Use return codes initially

---

### 9. Module System

**Status:** ❌ Not Started

**Benefit:** Split compiler across multiple files

**Estimated Time:** 8-10 weeks

**Decision:** Single file initially, refactor later

---

### 10. Pattern Matching

**Status:** ❌ Not Started

**Benefit:** Better enum handling

**Estimated Time:** 6-8 weeks

**Decision:** Use if/else initially

---

## Compiler Components (After P1 Features Complete)

### Lexer in nanolang

**Status:** ❌ Not Started

**Dependencies:** Structs, Enums, Lists, String ops

**Implementation Tasks:**
- [ ] Define Token struct
- [ ] Define TokenType enum
- [ ] Implement tokenize(source: string) -> list<Token>
- [ ] Handle whitespace
- [ ] Handle comments (#)
- [ ] Handle literals (numbers, strings, bools)
- [ ] Handle keywords
- [ ] Handle operators and delimiters
- [ ] Write comprehensive shadow tests
- [ ] Compare output with C lexer

**Estimated Time:** 3-4 weeks

**Lines of Code:** ~400-500 (similar to lexer.c)

---

### Parser in nanolang

**Status:** ❌ Not Started

**Dependencies:** Lexer, Structs (for AST nodes), Lists

**Implementation Tasks:**
- [ ] Define ASTNode structs (multiple types)
- [ ] Implement parse_program(tokens: list<Token>) -> ASTNode
- [ ] Implement parse_function
- [ ] Implement parse_statement
- [ ] Implement parse_expression
- [ ] Handle both prefix and infix notation
- [ ] Error recovery
- [ ] Write comprehensive shadow tests
- [ ] Compare output with C parser

**Estimated Time:** 4-5 weeks

**Lines of Code:** ~600-700 (similar to parser.c)

---

### Type Checker in nanolang

**Status:** ❌ Not Started

**Dependencies:** Parser, Structs (Symbol table)

**Implementation Tasks:**
- [ ] Define Symbol struct
- [ ] Implement symbol table (list-based initially)
- [ ] Implement type_check(program: ASTNode) -> bool
- [ ] Check function signatures
- [ ] Check expression types
- [ ] Check return paths
- [ ] Write comprehensive shadow tests
- [ ] Compare output with C type checker

**Estimated Time:** 3-4 weeks

**Lines of Code:** ~500-600 (similar to typechecker.c)

---

### Transpiler in nanolang

**Status:** ❌ Not Started

**Dependencies:** Type checker, String operations (code generation), File I/O

**Implementation Tasks:**
- [ ] Implement transpile(program: ASTNode) -> string
- [ ] Generate C function definitions
- [ ] Generate C statements
- [ ] Generate C expressions
- [ ] Handle built-in functions
- [ ] Format C code (basic)
- [ ] Write comprehensive shadow tests
- [ ] Compare output with C transpiler

**Estimated Time:** 3-4 weeks

**Lines of Code:** ~400-500 (similar to transpiler.c)

---

### Main Compiler Driver in nanolang

**Status:** ❌ Not Started

**Dependencies:** All components above, System execution

**Implementation Tasks:**
- [ ] Implement main() function
- [ ] Parse command-line arguments
- [ ] Read source file
- [ ] Call lexer
- [ ] Call parser
- [ ] Call type checker
- [ ] Run shadow tests
- [ ] Call transpiler
- [ ] Write C file
- [ ] Invoke gcc
- [ ] Handle errors
- [ ] Write shadow tests
- [ ] Compare with C compiler

**Estimated Time:** 2-3 weeks

**Lines of Code:** ~200-300 (similar to main.c)

---

## Bootstrap Process

### Bootstrap Level 0 (C Compiler)

**Status:** ✅ Complete

- C compiler (current) compiles nanolang programs
- C compiler supports all P1 features
- Ready to compile nanolang compiler

---

### Bootstrap Level 1

**Status:** ❌ Not Started

**Process:**
1. Write nanolang compiler in nanolang (nanolanc.nano)
2. Compile with C compiler: `./nanoc nanolanc.nano -o nanolanc_v1`
3. Test: `./nanolanc_v1 examples/hello.nano -o hello_test`
4. Verify: Compare with C compiler output

**Success Criteria:**
- [ ] nanolanc_v1 successfully compiles hello.nano
- [ ] Output identical to C compiler
- [ ] All shadow tests pass

---

### Bootstrap Level 2

**Status:** ❌ Not Started

**Process:**
1. Compile nanolang compiler with itself: `./nanolanc_v1 nanolanc.nano -o nanolanc_v2`
2. Test: `./nanolanc_v2 examples/hello.nano -o hello_test`
3. Verify: Compare with v1 output

**Success Criteria:**
- [ ] nanolanc_v2 successfully compiles hello.nano
- [ ] Output identical to v1
- [ ] Compiler can compile itself

---

### Bootstrap Level 3+ (Fixed Point)

**Status:** ❌ Not Started

**Process:**
1. Continue bootstrapping: v2 → v3, v3 → v4, etc.
2. Check for fixed point (vN and vN+1 are identical)

**Success Criteria:**
- [ ] Fixed point reached (vN == vN+1)
- [ ] All tests pass at every level
- [ ] Bootstrapping is repeatable

---

## Timeline Summary

### Phase 1: P1 Features (Months 1-6)

| Month | Feature | Status |
|-------|---------|--------|
| 1-2 | Structs | ❌ |
| 3 | Enums | ❌ |
| 4 | Lists | ❌ |
| 5 | File I/O + String Ops | ❌ |
| 6 | System Execution | ❌ |

### Phase 2: Compiler Rewrite (Months 7-9)

| Month | Component | Status |
|-------|-----------|--------|
| 7 | Lexer | ❌ |
| 8 | Parser | ❌ |
| 9 | Type Checker + Transpiler | ❌ |

### Phase 3: Bootstrap (Months 10-12)

| Month | Activity | Status |
|-------|----------|--------|
| 10 | Bootstrap Level 1 | ❌ |
| 11 | Bootstrap Level 2+ | ❌ |
| 12 | Testing + Optimization | ❌ |

---

## Progress Tracking

**Overall Progress:** 100% (6/6 P1 features complete) ✅

**P1 Features:**
- [x] ✅ Structs (100% - Complete November 2025)
- [x] ✅ Enums (100% - Complete November 2025)
- [x] ✅ Lists (100% - list_int and list_string implemented)
- [x] ✅ File I/O (100% - Complete via stdlib)
- [x] ✅ String ops (100% - 13+ advanced functions implemented)
- [x] ✅ System execution (100% - Complete via stdlib)

**Compiler Components:**
- [ ] Lexer (0%)
- [ ] Parser (0%)
- [ ] Type Checker (0%)
- [ ] Transpiler (0%)
- [ ] Main Driver (0%)

**Bootstrap:**
- [ ] Level 0 (C compiler) ✅
- [ ] Level 1 (0%)
- [ ] Level 2 (0%)
- [ ] Fixed Point (0%)

---

## Next Immediate Actions

1. **Design struct syntax** - Complete specification
2. **Implement structs in C compiler** - Add to lexer, parser, type checker, transpiler
3. **Write struct tests** - Shadow tests for all struct operations
4. **Update documentation** - Add structs to spec, getting started guide

**Start Date:** TBD  
**Target Completion:** TBD  
**Current Owner:** TBD

---

## Success Metrics

### Technical Metrics
- ✅ All P1 features implemented and tested
- ✅ Nanolang compiler written in nanolang
- ✅ Bootstrap process successful (Level 2+)
- ✅ Fixed point reached
- ✅ All shadow tests pass
- ✅ Performance within 2-3x of C compiler

### Code Metrics
- ✅ ~5,000 lines of nanolang code (compiler)
- ✅ 100% shadow test coverage
- ✅ Zero known bugs
- ✅ Documentation complete

### Design Metrics
- ✅ Language stays minimal (< 25 keywords)
- ✅ Language stays unambiguous
- ✅ Both prefix and infix notation supported
- ✅ Shadow tests still mandatory
- ✅ LLM-friendly (measured by LLM code gen success)

---

**Last Updated:** 2025-11-12  
**Status:** Planning Phase  
**Next Review:** After structs implementation

See [SELF_HOSTING_REQUIREMENTS.md](SELF_HOSTING_REQUIREMENTS.md) for detailed requirements and [planning/SELF_HOST_STATUS.md](../planning/SELF_HOST_STATUS.md) for a quick status overview.

