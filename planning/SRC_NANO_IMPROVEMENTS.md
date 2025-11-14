# src_nano Improvements with New Features

## Overview

Now that we have:
- ✅ Generic syntax (`List<Token>`)
- ✅ Fixed enum variant access (`TokenType.FN`)
- ✅ Union type support (ready for use)

We can significantly improve the self-hosted compiler code in `src_nano/`.

---

## Current Blockers Resolved

### 1. ✅ Enum Variant Access Fixed

**Before** (workaround with magic numbers):
```nano
fn classify_keyword(word: string) -> int {
    if (strings_equal word "fn") { return 19 } else {}   /* Magic number! */
    if (strings_equal word "let") { return 20 } else {}
    return 4
}
```

**After** (clean enum access):
```nano
fn classify_keyword(word: string) -> int {
    if (strings_equal word "fn") { return TokenType.FN } else {}  /* Clear! */
    if (strings_equal word "let") { return TokenType.LET } else {}
    return TokenType.IDENTIFIER
}
```

### 2. ✅ Generic List Syntax Ready

**Before** (not possible):
```nano
/* Cannot create a list of tokens! */
```

**After** (once runtime supports it):
```nano
fn lex(source: string) -> List<Token> {
    let tokens: List<Token> = (list_token_new)
    /* ... */
    return tokens
}
```

### 3. ✅ Union Types Available

**Current** (separate structs):
```nano
struct ASTNodeNumber {
    node_type: int,
    line: int,
    value: int
}

struct ASTNodeIdentifier {
    node_type: int,
    line: int,
    name: string
}
```

**Future** (union type):
```nano
union ASTNode {
    Number { line: int, value: int },
    Identifier { line: int, name: string },
    /* ... */
}
```

---

## Improvement Opportunities

### File: `src_nano/lexer_v2.nano`

#### Improvement 1: Use Enum Variants

**Current**:
```nano
shadow classify_keyword {
    assert (== (classify_keyword "fn") 19)  /* Magic number */
    assert (== (classify_keyword "let") 20)
}
```

**Improved**:
```nano
shadow classify_keyword {
    assert (== (classify_keyword "fn") TokenType.FN)   /* Clear! */
    assert (== (classify_keyword "let") TokenType.LET)
}
```

**Impact**: All 50+ magic numbers replaced with clear enum names!

#### Improvement 2: Return Type Annotation

**Current**:
```nano
fn classify_keyword(word: string) -> int {
    /* Returns TokenType value */
}
```

**Improved**:
```nano
fn classify_keyword(word: string) -> TokenType {
    if (strings_equal word "fn") { return TokenType.FN } else {}
    return TokenType.IDENTIFIER
}
```

**Impact**: Function signature is self-documenting!

#### Improvement 3: Generic Return Type (Future)

**Current**:
```nano
fn lex(source: string) -> array<Token> {
    /* But array<Token> doesn't work yet */
}
```

**Future**:
```nano
fn lex(source: string) -> List<Token> {
    let tokens: List<Token> = (list_token_new)
    /* Add tokens as we lex */
    (list_token_push tokens new_token)
    return tokens
}
```

**Impact**: Clean, type-safe tokenization!

---

### File: `src_nano/ast_types.nano`

#### Improvement 1: Use Union Types

**Current**:
```nano
/* Each AST node is a separate struct */
struct ASTNodeNumber { node_type: int, line: int, column: int, value: int }
struct ASTNodeString { node_type: int, line: int, column: int, value: string }
struct ASTNodeIdentifier { node_type: int, line: int, column: int, name: string }

/* Cannot have a unified AST node type! */
```

**Improved**:
```nano
/* Single unified AST node type */
union ASTNode {
    Number { line: int, column: int, value: int },
    Float { line: int, column: int, value: float },
    String { line: int, column: int, value: string },
    Identifier { line: int, column: int, name: string },
    PrefixOp { line: int, column: int, op: int, operands: List<ASTNode> },
    Call { line: int, column: int, name: string, args: List<ASTNode> },
    /* ... */
}

/* Pattern matching is natural! */
fn eval_node(node: ASTNode) -> int {
    return match node {
        Number(n) => n.value,
        Identifier(id) => (lookup_variable id.name),
        PrefixOp(op) => (eval_op op.op op.operands),
        _ => 0
    }
}
```

**Impact**: Massive code quality improvement!

---

## Implementation Plan

### Phase 1: Low-Hanging Fruit (1 hour)

**Task**: Update lexer_v2.nano to use enum variants

1. Replace all magic numbers with `TokenType.VARIANT`
2. Update shadow tests to use clear enum names
3. Change return type annotations where applicable

**Files**:
- `src_nano/lexer_v2.nano`

**Expected Result**:
- 50+ magic numbers eliminated
- Code is self-documenting
- Tests are clearer

### Phase 2: Generic List Integration (1-2 hours)

**Task**: Use `List<Token>` for tokenization

**Blocker**: Need runtime support for returning lists

**Current Runtime**:
```c
/* list_token is just an int handle */
int list_token_new();
void list_token_push(int list, Token tok);
```

**Needed**:
```c
/* Proper list type that can be returned */
typedef struct { /* ... */ } List_Token;
List_Token* list_token_new();
void list_token_push(List_Token* list, Token tok);
```

**Alternative**: Keep current approach (return array) for now

### Phase 3: Union Types for AST (2-3 hours)

**Task**: Refactor `ast_types.nano` to use union types

**Dependency**: Need `List<ASTNode>` for children

**Steps**:
1. Define `union ASTNode` with all variants
2. Update helper functions to construct variants
3. Add pattern matching examples
4. Update tests

**Expected Result**:
- Single unified AST type
- Pattern matching for evaluation
- Type-safe tree manipulation

---

## Quick Wins - Implementable Now

### 1. Update Enum Usage in Lexer

**File**: `src_nano/lexer_v2.nano`

**Change**:
```nano
/* OLD */
shadow classify_keyword {
    assert (== (classify_keyword "fn") 19)
    assert (== (classify_keyword "let") 20)
    assert (== (classify_keyword "if") 23)
}

/* NEW */
shadow classify_keyword {
    assert (== (classify_keyword "fn") TokenType.FN)
    assert (== (classify_keyword "let") TokenType.LET)
    assert (== (classify_keyword "if") TokenType.IF)
}
```

**Impact**: Immediate code clarity improvement!

### 2. Use Generic Syntax for Type Annotations

**File**: `src_nano/lexer_v2.nano`

**Change**:
```nano
/* OLD */
let tokens: int = (list_token_new)  /* Type is unclear */

/* NEW */
let tokens: List<Token> = (list_token_new)  /* Type is clear! */
```

**Note**: Still returns `int` from function, but local variable type is documented

### 3. Document Union Opportunities

**File**: All `src_nano/*.nano` files

**Action**: Add comments showing future union improvements

```nano
/* TODO: Once generics support arbitrary types, refactor to:
 * union ASTNode {
 *     Number { value: int },
 *     String { value: string },
 *     ...
 * }
 */
struct ASTNodeNumber { /* current approach */ }
```

---

## Timeline

### Immediate (Today - 1 hour)
- [ ] Update `lexer_v2.nano` enum usage
- [ ] Test with enum variants
- [ ] Document other improvement opportunities

### Short-term (Next Session - 2-3 hours)
- [ ] Full lexer refactor with generics
- [ ] Add union type examples
- [ ] Documentation updates

### Medium-term (Future - 4-6 hours)
- [ ] Refactor AST types to unions
- [ ] Implement parser with union AST
- [ ] Complete self-hosted compiler

---

## Example: Before & After

### lexer_v2.nano - classify_keyword function

**Before**:
```nano
fn classify_keyword(word: string) -> int {
    if (strings_equal word "extern") { return 18 } else {}
    if (strings_equal word "fn") { return 19 } else {}
    if (strings_equal word "let") { return 20 } else {}
    if (strings_equal word "mut") { return 21 } else {}
    if (strings_equal word "set") { return 22 } else {}
    if (strings_equal word "if") { return 23 } else {}
    if (strings_equal word "else") { return 24 } else {}
    if (strings_equal word "while") { return 25 } else {}
    if (strings_equal word "for") { return 26 } else {}
    if (strings_equal word "in") { return 27 } else {}
    if (strings_equal word "return") { return 28 } else {}
    /* ... 40+ more lines of magic numbers ... */
    return 4  /* IDENTIFIER */
}
```

**After**:
```nano
fn classify_keyword(word: string) -> TokenType {
    if (strings_equal word "extern") { return TokenType.EXTERN } else {}
    if (strings_equal word "fn") { return TokenType.FN } else {}
    if (strings_equal word "let") { return TokenType.LET } else {}
    if (strings_equal word "mut") { return TokenType.MUT } else {}
    if (strings_equal word "set") { return TokenType.SET } else {}
    if (strings_equal word "if") { return TokenType.IF } else {}
    if (strings_equal word "else") { return TokenType.ELSE } else {}
    if (strings_equal word "while") { return TokenType.WHILE } else {}
    if (strings_equal word "for") { return TokenType.FOR } else {}
    if (strings_equal word "in") { return TokenType.IN } else {}
    if (strings_equal word "return") { return TokenType.RETURN } else {}
    /* ... 40+ more lines of CLEAR NAMES ... */
    return TokenType.IDENTIFIER
}
```

**Improvement**: Every number is now self-explanatory!

---

## Success Metrics

### Phase 1 Complete When:
- [ ] All magic numbers replaced with enum variants
- [ ] Code compiles and passes tests
- [ ] Documentation updated
- [ ] Code is more readable

### Phase 2 Complete When:
- [ ] `List<Token>` used throughout lexer
- [ ] Generic syntax in all type annotations
- [ ] Tests use clear type names
- [ ] No int-as-list workarounds

### Phase 3 Complete When:
- [ ] AST uses union types
- [ ] Pattern matching works
- [ ] Parser uses unified AST
- [ ] Self-hosted compiler milestone reached

---

## Conclusion

**Status**: Ready to improve!

**Blockers Resolved**: All major blockers (enum access, generics) are fixed

**Next Step**: Update `lexer_v2.nano` with enum variants (1 hour effort)

**Long-term Vision**: Full self-hosted compiler with clean, type-safe code

---

*End of src_nano improvements plan*

