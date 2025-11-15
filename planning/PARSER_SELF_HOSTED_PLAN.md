# Self-Hosted Parser Implementation Plan

**Date:** November 15, 2025  
**Goal:** Complete nanolang parser written in nanolang  
**Current Status:** Foundation complete (213 lines, 10%)

---

## 📊 Current State

**What's Done:**
- ✅ AST node type definitions (9 types)
- ✅ Parser state structure
- ✅ Basic node creation functions
- ✅ Shadow tests pass

**What's Missing:**
- ❌ Token stream management
- ❌ Expression parsing (literals, binary ops, calls)
- ❌ Statement parsing (let, set, if, while, for, return)
- ❌ Function definition parsing
- ❌ Type annotation parsing
- ❌ Block parsing
- ❌ Program parsing
- ❌ Error handling

---

## 🎯 Implementation Strategy

### Phase 1: Token Management (2-3h)
**Goal:** Handle token stream, peek, advance, expect

```nano
fn parser_current(p: Parser) -> LexToken
fn parser_peek(p: Parser, offset: int) -> LexToken
fn parser_advance(p: Parser) -> int
fn parser_expect(p: Parser, expected: int) -> bool
fn parser_is_at_end(p: Parser) -> bool
fn parser_match(p: Parser, token_type: int) -> bool
```

**Deliverable:** Can navigate token stream safely

### Phase 2: Expression Parsing (10-15h)
**Goal:** Parse all expression types

**2A: Primary Expressions (3h)**
- Numbers: `42`, `3.14`
- Strings: `"hello"`
- Bools: `true`, `false`
- Identifiers: `x`, `my_var`

**2B: Prefix Operations (4h)**
- Arithmetic: `(+ 2 3)`, `(* x y)`
- Comparison: `(== a b)`, `(> x 5)`
- Logical: `(and p q)`, `(not flag)`

**2C: Complex Expressions (5h)**
- Function calls: `(func arg1 arg2)`
- Array literals: `[1, 2, 3]`
- Struct literals: `Point{x: 10, y: 20}`
- Field access: `point.x`
- Array access: `(at arr i)`

**2D: Pattern Matching (3h)**
- Match expressions: `match value { ... }`

### Phase 3: Statement Parsing (10-15h)
**Goal:** Parse all statement types

**3A: Variable Declarations (3h)**
- Let statements: `let x: int = 42`
- Mutable: `let mut y: int = 0`

**3B: Assignment (2h)**
- Set statements: `set x 10`

**3C: Control Flow (6h)**
- If/else: `if cond { } else { }`
- While: `while cond { }`
- For: `for i in range { }`
- Return: `return value`

**3D: Blocks (2h)**
- Statement blocks: `{ stmt1 stmt2 }`

### Phase 4: Definition Parsing (15-20h)
**Goal:** Parse top-level definitions

**4A: Function Definitions (8h)**
- Signature: `fn name(params) -> type`
- Parameters: `param: type`
- Body: function body block
- Extern functions: `extern fn name(...)`

**4B: Type Definitions (6h)**
- Struct: `struct Name { fields }`
- Enum: `enum Name { variants }`
- Union: `union Name { variants }`

**4C: Shadow Tests (3h)**
- Shadow blocks: `shadow func { asserts }`

**4D: Type Annotations (3h)**
- Simple types: `int`, `string`, `bool`
- Array types: `array<int>`
- Generic types: `List<Point>`
- Function types: `fn(int) -> int`

### Phase 5: Program Parsing (3-5h)
**Goal:** Top-level orchestration

- Parse sequence of definitions
- Build AST_PROGRAM node
- Return full program AST

### Phase 6: Error Handling (5-8h)
**Goal:** Helpful error messages

- Syntax errors with line/column
- Unexpected token errors
- Missing tokens
- Type annotation errors

### Phase 7: Integration & Testing (10-15h)
**Goal:** Comprehensive test suite

- Unit tests for each parse function
- Integration tests for complete programs
- Regression tests
- Performance tests

---

## 📝 Detailed Implementation: Phase 1 (Token Management)

### Step 1.1: Parser State Extension

```nano
struct Parser {
    tokens: List_LexToken,
    current: int,
    has_error: bool,
    error_message: string
}

fn parser_new(tokens: List_LexToken) -> Parser {
    let p: Parser = Parser{
        tokens: tokens,
        current: 0,
        has_error: false,
        error_message: ""
    }
    return p
}
```

### Step 1.2: Token Navigation

```nano
fn parser_current(p: Parser) -> LexToken {
    if (>= p.current (List_LexToken_length p.tokens)) {
        /* Return EOF token */
        return (create_eof_token)
    } else {
        return (List_LexToken_get p.tokens p.current)
    }
}

fn parser_advance(p: Parser) -> int {
    if (< p.current (List_LexToken_length p.tokens)) {
        set p.current (+ p.current 1)
    } else {
        return 0
    }
    return 1
}

fn parser_is_at_end(p: Parser) -> bool {
    let tok: LexToken = (parser_current p)
    return (== tok.type EOF)  /* Assuming EOF token type */
}

fn parser_match(p: Parser, expected: int) -> bool {
    let tok: LexToken = (parser_current p)
    return (== tok.type expected)
}

fn parser_expect(p: Parser, expected: int) -> bool {
    if (parser_match p expected) {
        (parser_advance p)
        return true
    } else {
        /* Set error */
        set p.has_error true
        set p.error_message "Unexpected token"
        return false
    }
}
```

---

## 🎯 Success Criteria

**Phase 1 Complete When:**
- ✅ Can navigate tokens forward
- ✅ Can check current token type
- ✅ Can match and expect tokens
- ✅ Can detect end of input
- ✅ All shadow tests pass

**Parser Complete When:**
- ✅ Can parse all expression types
- ✅ Can parse all statement types
- ✅ Can parse all definition types
- ✅ Can parse complete programs
- ✅ Generates correct AST
- ✅ Has good error messages
- ✅ 100% shadow test coverage
- ✅ Passes integration tests

---

## ⏱️ Time Estimates

| Phase | Description | Time |
|-------|-------------|------|
| 1 | Token Management | 2-3h |
| 2 | Expression Parsing | 10-15h |
| 3 | Statement Parsing | 10-15h |
| 4 | Definition Parsing | 15-20h |
| 5 | Program Parsing | 3-5h |
| 6 | Error Handling | 5-8h |
| 7 | Integration & Testing | 10-15h |
| **TOTAL** | **Complete Parser** | **55-81h** |

**Optimistic:** 55 hours  
**Realistic:** 65-70 hours  
**Pessimistic:** 81 hours

---

## 📂 File Structure

```
src_nano/
├── lexer_complete.nano          ✅ Done (447 lines)
├── parser_foundation.nano       🔄 Current (213 lines)
├── parser_tokens.nano           ⏳ New - Token management
├── parser_expressions.nano      ⏳ New - Expression parsing
├── parser_statements.nano       ⏳ New - Statement parsing
├── parser_definitions.nano      ⏳ New - Definition parsing
├── parser_complete.nano         ⏳ New - Full parser integration
└── compiler_stage2.nano         ⏳ Future - Full compiler
```

**Alternative: Single File Approach**
- Keep everything in `parser_complete.nano` (~1,500 lines)
- Easier to manage initially
- Split later if needed

---

## 🚀 Getting Started

**Immediate Next Steps:**
1. Extend Parser struct with token list
2. Implement token navigation functions
3. Write shadow tests for token management
4. Start on primary expression parsing

**First Milestone:** Parse and print a simple expression
```nano
/* Input tokens for: (+ 2 3) */
/* Output: ASTBinaryOp{op: PLUS, left: 2, right: 3} */
```

---

**Status:** Ready to implement Phase 1! 🎯  
**Estimated First Session:** 2-3 hours for token management  
**Next Review:** After Phase 1 complete

