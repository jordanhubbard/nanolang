# Union Types Audit - src_nano Files

## Overview
Audited all `src_nano/*.nano` files to identify opportunities for using union types. Union types enable more expressive, type-safe code by representing values that can be one of several variants.

## Files Audited
- ✅ ast_types.nano
- ✅ token_struct.nano
- ✅ token_types.nano
- ✅ lexer_v2.nano
- ✅ compiler_stage2.nano
- ✅ lexer_*.nano (various)

---

## 🎯 Major Opportunity: AST Nodes

### Current Approach (ast_types.nano)
**Problem**: Each AST node type is a separate struct with redundant fields.

```nano
/* Current - Separate structs */
struct ASTNodeNumber {
    node_type: int,     /* ASTNodeType enum */
    line: int,
    column: int,
    value: int
}

struct ASTNodeIdentifier {
    node_type: int,     /* ASTNodeType enum */
    line: int,
    column: int,
    name: string
}

/* Need separate functions for each type */
fn create_number_node(value: int, line: int, column: int) -> ASTNodeNumber { ... }
fn create_identifier_node(name: string, line: int, column: int) -> ASTNodeIdentifier { ... }
```

**Issues**:
1. ❌ Cannot have a single `ASTNode` type
2. ❌ Cannot store mixed AST nodes in arrays/lists
3. ❌ Redundant `node_type` field that duplicates union tag
4. ❌ Type system doesn't enforce variant correctness
5. ❌ Difficult to pattern match on node types

### Recommended: Union Type

```nano
/* Better - Tagged union */
union ASTNode {
    Number { value: int },
    Float { value: float },
    String { value: string },
    Identifier { name: string },
    PrefixOp { op: int, args: array<ASTNode> },
    Call { name: string, args: array<ASTNode> },
    Let { name: string, type: int, value: ASTNode },
    /* ... more variants ... */
}

/* Common fields (line, column) can be tracked separately or added to each variant */

/* Construction is clean */
fn create_number_node(value: int) -> ASTNode {
    return ASTNode.Number { value: value }
}

/* Pattern matching is natural */
fn eval_node(node: ASTNode) -> int {
    return match node {
        Number(n) => n.value,
        Identifier(id) => (lookup_variable id.name),
        PrefixOp(op) => (eval_prefix_op op.op op.args),
        _ => 0
    }
}
```

**Benefits**:
- ✅ Single unified `ASTNode` type
- ✅ Type-safe variant construction
- ✅ Pattern matching with exhaustiveness checking
- ✅ Can store in `List<ASTNode>` (once generics implemented)
- ✅ Compiler enforces correct access
- ✅ Matches C implementation structure

---

## 🎯 Opportunity: Parse Results

### Current Approach
**Problem**: Functions return success/failure using magic values or separate error tracking.

```nano
/* Current - Error-prone */
fn parse_token(source: string, pos: int) -> int {
    /* Returns -1 for error, or token type */
    if (error_condition) {
        return -1  /* Magic number! */
    } else {}
    return token_type
}

/* Caller has to check */
let result: int = (parse_token source 0)
if (== result -1) {
    /* Handle error - but what error? */
} else {}
```

### Recommended: Result Union

```nano
/* Better - Type-safe results */
union ParseResult {
    Success { token: Token, next_pos: int },
    Error { message: string, line: int, column: int }
}

fn parse_token(source: string, pos: int) -> ParseResult {
    if (error_condition) {
        return ParseResult.Error {
            message: "Unterminated string",
            line: current_line,
            column: current_col
        }
    } else {}
    
    return ParseResult.Success {
        token: parsed_token,
        next_pos: (+ pos token_length)
    }
}

/* Type-safe handling */
let result: ParseResult = (parse_token source 0)
return match result {
    Success(s) => (process_token s.token),
    Error(e) => (report_error e.message e.line e.column)
}
```

**Benefits**:
- ✅ No magic error values
- ✅ Compiler enforces error handling
- ✅ Error messages included in result
- ✅ Pattern matching makes handling clear
- ✅ Cannot accidentally ignore errors

---

## 🎯 Opportunity: Type Representations

### Current Approach
**Problem**: Types are represented as integers, losing semantic information.

```nano
/* Current - Just an int */
enum Type {
    TYPE_INT = 0,
    TYPE_ARRAY = 5,
    TYPE_STRUCT = 6
}

/* For arrays, need separate field for element type */
struct TypeInfo {
    base_type: int,      /* Type enum */
    element_type: int,   /* For arrays */
    struct_name: string  /* For structs */
}
```

### Recommended: Type Union

```nano
/* Better - Structured types */
union Type {
    Int {},
    Float {},
    Bool {},
    String {},
    Void {},
    Array { element: Type },
    Struct { name: string },
    Enum { name: string },
    Function { params: array<Type>, return_type: Type }
}

/* Type checking is clearer */
fn check_type_match(expected: Type, actual: Type) -> bool {
    return match expected {
        Int(_) => match actual { Int(_) => true, _ => false },
        Array(a) => match actual {
            Array(b) => (check_type_match a.element b.element),
            _ => false
        },
        _ => false
    }
}
```

**Benefits**:
- ✅ Types are first-class values
- ✅ Recursive type structure (array of array)
- ✅ Type equality checking is natural
- ✅ Compiler catches type errors
- ✅ Matches type theory better

---

## 🎯 Opportunity: Token Values

### Current Approach
**Good enough**: Token struct uses string for all values.

```nano
struct Token {
    type: int,      /* TokenType */
    value: string,  /* Always a string */
    line: int,
    column: int
}
```

### Possible Enhancement: Typed Values

```nano
union TokenValue {
    None {},
    Number { value: int },
    Float { value: float },
    String { value: string },
    Identifier { name: string }
}

struct Token {
    type: int,       /* TokenType */
    value: TokenValue,
    line: int,
    column: int
}
```

**Assessment**: ⚠️ **Low priority** - current approach works fine for lexer. The string representation is actually convenient for error messages and debugging.

---

## 🎯 Opportunity: Lexer State

### Current Approach
**Problem**: Lexer position tracking uses multiple variables.

```nano
/* Current - Multiple variables */
fn lex(source: string) -> array<Token> {
    let mut i: int = 0
    let mut line: int = 1
    let mut column: int = 1
    let mut line_start: int = 0
    /* ... */
}
```

### Possible Enhancement: State Struct/Union

```nano
struct LexerState {
    source: string,
    position: int,
    line: int,
    column: int,
    line_start: int
}

fn advance_char(state: LexerState) -> LexerState {
    let c: int = (char_at state.source state.position)
    if (== c 10) {  /* newline */
        return LexerState {
            source: state.source,
            position: (+ state.position 1),
            line: (+ state.line 1),
            column: 1,
            line_start: (+ state.position 1)
        }
    } else {
        return LexerState {
            source: state.source,
            position: (+ state.position 1),
            line: state.line,
            column: (+ state.column 1),
            line_start: state.line_start
        }
    }
}
```

**Assessment**: ⚠️ **Medium priority** - cleaner but not critical. The functional approach with immutable state would be nice but requires more overhead.

---

## Priority Recommendations

### 🔴 High Priority
1. **ASTNode Union** - Critical for parser/compiler work
   - Enables type-safe AST manipulation
   - Matches C implementation structure
   - Required for serious compiler development

### 🟡 Medium Priority  
2. **ParseResult Union** - Improves error handling
   - Makes error cases explicit
   - Better error messages
   - Type-safe result handling

3. **Type Union** - Better type system representation
   - Cleaner type checking code
   - Recursive types work naturally
   - Better matches theory

### 🟢 Low Priority
4. **TokenValue Union** - Nice to have
   - Current string approach works fine
   - Would add complexity without major benefit

5. **LexerState Struct** - Refactoring opportunity
   - Not urgent, current approach works
   - Would enable functional style

---

## Implementation Plan

### Phase 1: ASTNode Union (Next Step)
```nano
/* Define comprehensive AST union */
union ASTNode {
    /* Literals */
    Number { value: int },
    Float { value: float },
    String { value: string },
    Bool { value: bool },
    
    /* Variables */
    Identifier { name: string },
    
    /* Expressions */
    PrefixOp { op: int, operands: array<ASTNode> },
    Call { name: string, args: array<ASTNode> },
    ArrayLiteral { elements: array<ASTNode> },
    FieldAccess { object: ASTNode, field: string },
    
    /* Statements */
    Let { name: string, var_type: int, value: ASTNode, is_mut: bool },
    Set { name: string, value: ASTNode },
    If { condition: ASTNode, then_branch: ASTNode, else_branch: ASTNode },
    While { condition: ASTNode, body: ASTNode },
    Return { value: ASTNode },
    Block { statements: array<ASTNode> },
    
    /* Definitions */
    Function { name: string, params: array<Param>, body: ASTNode },
    StructDef { name: string, fields: array<Field> },
    EnumDef { name: string, variants: array<string> }
}

/* Update ast_types.nano to use this */
/* Rewrite parser to construct union variants */
```

### Phase 2: Result Types
```nano
union ParseResult<T> {  /* Once generics work */
    Ok { value: T },
    Err { message: string, line: int, column: int }
}

/* Use throughout parser for error handling */
```

### Phase 3: Type Union
```nano
union Type {
    Int {},
    Float {},
    /* ... */
    Array { element: Type },
    Struct { name: string }
}

/* Update type checker to use this */
```

---

## Code Examples

### Example: Pattern Matching AST

```nano
fn ast_to_string(node: ASTNode) -> string {
    return match node {
        Number(n) => (int_to_string n.value),
        String(s) => s.value,
        Identifier(id) => id.name,
        PrefixOp(op) => (str_concat "(" (op_name op.op)),
        Call(c) => (str_concat c.name "()"),
        _ => "<unknown>"
    }
}
```

### Example: Type-Safe AST Construction

```nano
fn parse_number(token: Token) -> ASTNode {
    let value: int = (string_to_int token.value)
    return ASTNode.Number { value: value }
}

fn parse_call(name: string, args: array<ASTNode>) -> ASTNode {
    return ASTNode.Call { name: name, args: args }
}
```

### Example: Error Handling

```nano
fn parse_expression(tokens: array<Token>, pos: int) -> ParseResult {
    if (>= pos (array_length tokens)) {
        return ParseResult.Error {
            message: "Unexpected end of input",
            line: 0,
            column: 0
        }
    } else {}
    
    let tok: Token = (at tokens pos)
    /* ... parse logic ... */
    
    return ParseResult.Success {
        value: parsed_ast,
        next_pos: (+ pos tokens_consumed)
    }
}
```

---

## Conclusion

**Key Findings**:
1. ✅ `ast_types.nano` is the **perfect use case** for union types
2. ✅ Result/Error handling would greatly benefit from unions
3. ✅ Type representations are a good candidate
4. ⚠️ Token values work fine as-is (string)
5. ⚠️ State management is a refactoring opportunity, not critical

**Recommendation**: 
Start with **ASTNode union** in `ast_types.nano`. This is the highest-impact change and will immediately improve the self-hosted compiler implementation quality.

**Blocker**: Need generic `List<ASTNode>` to store AST children effectively. This ties directly into implementing generics (Option A).

**Next Steps**:
1. ✅ Complete this audit (DONE)
2. 🚧 Implement generics (Option A)
3. 🚧 Update `ast_types.nano` to use union types
4. 🚧 Build parser using union-based AST

