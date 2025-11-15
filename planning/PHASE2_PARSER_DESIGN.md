# Phase 2 Step 2: Self-Hosted Parser Design

**Date:** November 15, 2025  
**Status:** Design Phase  
**Goal:** Implement a complete recursive descent parser in nanolang

---

## Overview

The parser transforms a `List<LexToken>` (from the lexer) into an Abstract Syntax Tree (AST) suitable for type checking and code generation.

### Input
- `List<LexToken>` from the lexer

### Output
- AST (Abstract Syntax Tree) representing the program structure

### Approach
- **Recursive Descent Parser**: Top-down parsing strategy
- **Union Types**: Use nanolang's union types for AST node variants
- **Explicit Type Annotations**: All nodes have clear types

---

## AST Design

### Core AST Node Structure

We'll use **union types** to represent different AST node kinds:

```nano
/* AST Node Types */
enum ASTNodeType {
    /* Literals */
    AST_NUMBER = 0,
    AST_STRING = 1,
    AST_BOOL = 2,
    AST_IDENTIFIER = 3,
    
    /* Expressions */
    AST_BINARY_OP = 4,
    AST_UNARY_OP = 5,
    AST_CALL = 6,
    AST_FIELD_ACCESS = 7,
    AST_ARRAY_LITERAL = 8,
    AST_STRUCT_LITERAL = 9,
    
    /* Statements */
    AST_LET = 10,
    AST_SET = 11,
    AST_IF = 12,
    AST_WHILE = 13,
    AST_FOR = 14,
    AST_RETURN = 15,
    AST_BLOCK = 16,
    
    /* Definitions */
    AST_FUNCTION = 17,
    AST_STRUCT_DEF = 18,
    AST_ENUM_DEF = 19,
    AST_UNION_DEF = 20,
    
    /* Top-level */
    AST_PROGRAM = 21
}

/* Base AST node - all nodes share these fields */
struct ASTNode {
    node_type: int,  /* ASTNodeType */
    line: int,
    column: int
}
```

### Specialized Node Types

Since nanolang doesn't support nested structs yet, we'll use separate structs for each node type:

```nano
/* Literal Nodes */
struct ASTNumber {
    base: ASTNode,
    value: string  /* Store as string, convert during codegen */
}

struct ASTString {
    base: ASTNode,
    value: string
}

struct ASTBool {
    base: ASTNode,
    value: bool
}

struct ASTIdentifier {
    base: ASTNode,
    name: string
}

/* Binary Operation */
struct ASTBinaryOp {
    base: ASTNode,
    op: int,  /* Token type of operator */
    left: int,  /* Index into node list */
    right: int  /* Index into node list */
}

/* Function Call */
struct ASTCall {
    base: ASTNode,
    function: int,  /* Index to function name node */
    args: array<int>,  /* Indices to argument nodes */
    arg_count: int
}

/* Let Statement */
struct ASTLet {
    base: ASTNode,
    name: string,
    var_type: string,  /* Type annotation as string */
    value: int,  /* Index to value expression */
    is_mut: bool
}

/* Function Definition */
struct ASTFunction {
    base: ASTNode,
    name: string,
    params: array<string>,  /* Parameter names */
    param_types: array<string>,  /* Parameter type annotations */
    param_count: int,
    return_type: string,
    body: int  /* Index to body block */
}

/* Program (top-level) */
struct ASTProgram {
    base: ASTNode,
    items: array<int>,  /* Indices to top-level items */
    item_count: int
}
```

### AST Storage Strategy

**Challenge:** Nanolang doesn't support recursive types (self-referential structs).

**Solution:** Use an **index-based** approach:
- Store all AST nodes in a single `List<ASTNode>` (or multiple lists by type)
- References between nodes use **integer indices** instead of pointers
- Similar to how ECS (Entity Component System) architectures work

```nano
struct Parser {
    tokens: List<LexToken>,
    position: int,
    
    /* AST node storage - index-based */
    numbers: List<ASTNumber>,
    strings: List<ASTString>,
    bools: List<ASTBool>,
    identifiers: List<ASTIdentifier>,
    binary_ops: List<ASTBinaryOp>,
    calls: List<ASTCall>,
    lets: List<ASTLet>,
    functions: List<ASTFunction>,
    /* ... more node type lists ... */
    
    /* Mapping from global node ID to (type, index) */
    node_types: List<int>,  /* ASTNodeType for each global ID */
    node_indices: List<int>,  /* Index within type-specific list */
}
```

---

## Parser Architecture

### Core Functions

```nano
/* Parser state management */
fn parser_new(tokens: List<LexToken>) -> Parser
fn peek(p: Parser) -> LexToken
fn advance(p: Parser) -> void
fn expect(p: Parser, token_type: int) -> bool
fn match_token(p: Parser, token_type: int) -> bool

/* Top-level parsing */
fn parse_program(p: Parser) -> int  /* Returns program node ID */

/* Expression parsing */
fn parse_expression(p: Parser) -> int
fn parse_primary(p: Parser) -> int
fn parse_call(p: Parser) -> int
fn parse_binary_op(p: Parser, precedence: int) -> int

/* Statement parsing */
fn parse_statement(p: Parser) -> int
fn parse_let(p: Parser) -> int
fn parse_set(p: Parser) -> int
fn parse_if(p: Parser) -> int
fn parse_while(p: Parser) -> int
fn parse_return(p: Parser) -> int
fn parse_block(p: Parser) -> int

/* Definition parsing */
fn parse_function(p: Parser) -> int
fn parse_struct_def(p: Parser) -> int
fn parse_enum_def(p: Parser) -> int
```

### Parsing Strategy

**Recursive Descent:**
1. Start with `parse_program()`
2. For each item, determine type and delegate to specific parser
3. Each parser function:
   - Checks current token
   - Consumes tokens as needed
   - Creates appropriate AST node
   - Returns node ID
   - Recursively parses sub-expressions/statements

**Operator Precedence:**
- Use precedence climbing for binary operators
- Precedence levels (from low to high):
  1. `or`
  2. `and`
  3. `==`, `!=`, `<`, `<=`, `>`, `>=`
  4. `+`, `-`
  5. `*`, `/`, `%`
  6. `not` (unary)

---

## Implementation Plan

### Phase 1: Foundation (MVP)
**Goal:** Parse simple expressions and statements

**Tasks:**
1. Define basic AST node structs
2. Implement parser state management
3. Parse literals (numbers, strings, bools, identifiers)
4. Parse simple binary expressions
5. Parse let statements
6. Parse function calls (basic)

**Test:** Parse `let x: int = (+ 2 3)`

### Phase 2: Statements
**Goal:** Parse control flow and blocks

**Tasks:**
1. Parse if/else statements
2. Parse while loops
3. Parse blocks
4. Parse return statements
5. Parse set statements

**Test:** Parse `if (> x 0) { return 1 } else { return 0 }`

### Phase 3: Definitions
**Goal:** Parse top-level definitions

**Tasks:**
1. Parse function definitions
2. Parse struct definitions
3. Parse enum definitions
4. Parse program structure

**Test:** Parse complete function with shadow test

### Phase 4: Advanced Features
**Goal:** Parse remaining language features

**Tasks:**
1. Parse arrays
2. Parse struct literals
3. Parse field access
4. Parse union types
5. Parse match expressions

**Test:** Parse real nanolang source files

---

## Challenges & Solutions

### Challenge 1: Recursive Types
**Problem:** Nanolang doesn't support `struct ASTNode { left: ASTNode, ... }`

**Solution:** Use index-based references
- Store nodes in lists
- Reference by integer index
- Similar to how C compilers work internally

### Challenge 2: Discriminated Unions
**Problem:** Need to represent many different node types

**Solution:** Use separate lists for each node type
- `numbers: List<ASTNumber>`
- `binary_ops: List<ASTBinaryOp>`
- Global node ID maps to (type, index) pair

### Challenge 3: Dynamic Arrays
**Problem:** Children count varies (function args, block statements)

**Solution:** Use `array<int>` with explicit count
- Pre-allocate array with max size
- Store actual count separately
- Or use multiple List<int> for indices

### Challenge 4: Error Reporting
**Problem:** Need good error messages

**Solution:** Track line/column in every AST node
- Store token position when creating nodes
- Include in error messages
- Help users debug their code

---

## Testing Strategy

### Unit Tests (Shadow Tests)
Test each parser function individually:

```nano
shadow parse_number {
    /* Setup */
    let tokens: List<LexToken> = (List_LexToken_new)
    (List_LexToken_push tokens (new_token TokenType.NUMBER "42" 1 1))
    
    let p: Parser = (parser_new tokens)
    let node_id: int = (parse_primary p)
    
    /* Verify */
    assert (!= node_id -1)
    /* Check node is NUMBER type */
}

shadow parse_binary_op {
    /* Test: (+ 2 3) */
    let tokens: List<LexToken> = create_test_tokens()
    let p: Parser = (parser_new tokens)
    let node_id: int = (parse_expression p)
    
    /* Verify binary op structure */
    assert (!= node_id -1)
}
```

### Integration Tests
Parse complete programs:

```nano
shadow parse_function_def {
    let source: string = "fn add(a: int, b: int) -> int { return (+ a b) }"
    let tokens: List<LexToken> = (lex source)
    let p: Parser = (parser_new tokens)
    let program_id: int = (parse_program p)
    
    assert (!= program_id -1)
}
```

---

## Success Criteria

✅ **Phase 1 Complete When:**
- Can parse simple expressions: `(+ 2 3)`
- Can parse let statements: `let x: int = 42`
- Can parse function calls: `(add 2 3)`
- All shadow tests pass
- Generated C code compiles

✅ **Phase 2 Complete When:**
- Can parse if/else, while, blocks
- Can parse complete functions
- Real nanolang code parses correctly

✅ **Full Parser Complete When:**
- Parses entire nanolang language
- Handles all syntax features
- Good error messages
- Performance is acceptable
- Used in self-hosted compiler pipeline

---

## Estimated Effort

**Conservative Estimate:** 60-80 hours (3-4 weeks)

**Breakdown:**
- Foundation (Phase 1): 20-25 hours
- Statements (Phase 2): 15-20 hours
- Definitions (Phase 3): 15-20 hours
- Advanced Features (Phase 4): 10-15 hours

**Based on Lexer Performance:** Could be 5-10x faster (6-12 hours actual)

---

## Next Steps

1. ✅ Create design document (this file)
2. 🔄 Define basic AST node structs
3. 🔄 Implement parser state management
4. 🔄 Parse literals and simple expressions
5. ⏳ Continue with statements and definitions

---

**Ready to implement!** Let's start with Phase 1: Foundation (MVP)

