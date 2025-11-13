# Union Types Implementation Plan

**Feature Branch:** `feature/union-types`  
**Timeline:** 15-20 hours  
**Status:** 🚧 In Progress

---

## Goal

Implement discriminated unions with pattern matching for AST and variant type representation.

---

## Syntax Design

### Union Definition
```nano
union ASTNode {
    Number { value: int },
    String { value: string },
    BinOp { left: ASTNode, op: string, right: ASTNode },
    Identifier { name: string }
}
```

### Union Construction
```nano
let num: ASTNode = ASTNode.Number { value: 42 }
let id: ASTNode = ASTNode.Identifier { name: "x" }
let binop: ASTNode = ASTNode.BinOp { 
    left: num, 
    op: "+", 
    right: id 
}
```

### Pattern Matching
```nano
fn eval(node: ASTNode) -> int {
    match node {
        Number(n) => return n.value,
        String(s) => return 0,
        BinOp(op) => {
            let left_val: int = (eval op.left)
            let right_val: int = (eval op.right)
            return (+ left_val right_val)
        },
        Identifier(id) => return 0
    }
}
```

---

## Implementation Phases

### Phase 1: Lexer (2 hours) ✓ Starting Here

**New Tokens:**
- `TOKEN_UNION` - "union" keyword
- `TOKEN_MATCH` - "match" keyword  
- `TOKEN_ARROW` - "=>" for match arms

**Files to Modify:**
- `src/lexer.c` - Add new token types and keywords

**Tasks:**
- [x] Add tokens to `TokenType` enum
- [ ] Add keywords to keyword table
- [ ] Add "=>" two-character operator
- [ ] Test lexer with union syntax

---

### Phase 2: Parser (5-6 hours)

**New AST Node Types:**
```c
typedef struct UnionDef {
    char *name;
    int variant_count;
    char **variant_names;
    // Each variant has fields like a struct
    StructDef *variant_structs;
} UnionDef;

typedef struct MatchExpr {
    ASTNode *expr;  // Expression to match on
    int arm_count;
    char **pattern_names;  // Variant names
    char **binding_names;  // Variable names for patterns
    ASTNode **arm_bodies;  // Code for each arm
} MatchExpr;
```

**Grammar:**
```
union_def := "union" IDENTIFIER "{" variant_list "}"
variant_list := variant ("," variant)*
variant := IDENTIFIER "{" field_list "}"

match_expr := "match" expr "{" match_arm_list "}"
match_arm_list := match_arm ("," match_arm)*
match_arm := IDENTIFIER "(" IDENTIFIER ")" "=>" statement
```

**Files to Modify:**
- `src/parser.c` - Add union and match parsing
- `src/nanolang.h` - Add new AST node types

**Tasks:**
- [ ] Add `AST_UNION_DEF` node type
- [ ] Add `AST_MATCH` node type  
- [ ] Add `AST_UNION_CONSTRUCT` node type
- [ ] Implement `parse_union_def()`
- [ ] Implement `parse_match_expr()`
- [ ] Implement `parse_union_construction()`
- [ ] Test parser with examples

---

### Phase 3: Type Checker (5-6 hours)

**Type System Changes:**
```c
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_STRING,
    TYPE_BOOL,
    TYPE_ARRAY,
    TYPE_STRUCT,
    TYPE_ENUM,
    TYPE_UNION,  // NEW
    TYPE_VOID
} Type;
```

**Environment Changes:**
```c
// Store union definitions
typedef struct {
    char *name;
    int variant_count;
    char **variant_names;
    Type *variant_types;
} UnionTypeInfo;

// Add to Environment struct
UnionTypeInfo *unions;
int union_count;
```

**Type Checking Rules:**
1. Union construction must use valid variant name
2. Variant fields must match defined types
3. Match expression must cover all variants (exhaustiveness)
4. All match arms must return same type
5. Pattern binding gets correct variant type

**Files to Modify:**
- `src/typechecker.c` - Add union type checking
- `src/env.c` - Add union storage to environment

**Tasks:**
- [ ] Add `TYPE_UNION` to type system
- [ ] Implement `check_union_def()`
- [ ] Implement `check_match_expr()` with exhaustiveness
- [ ] Implement `check_union_construction()`
- [ ] Add union types to environment
- [ ] Test type checker with examples

---

### Phase 4: Transpiler (4-5 hours)

**C Code Generation Strategy:**

#### Tagged Union Representation
```c
/* Union: ASTNode */
typedef enum {
    ASTNODE_TAG_NUMBER,
    ASTNODE_TAG_STRING,
    ASTNODE_TAG_BINOP,
    ASTNODE_TAG_IDENTIFIER
} ASTNode_Tag;

typedef struct ASTNode {
    ASTNode_Tag tag;
    union {
        struct { int64_t value; } number;
        struct { const char* value; } string;
        struct { 
            struct ASTNode* left; 
            const char* op; 
            struct ASTNode* right; 
        } binop;
        struct { const char* name; } identifier;
    } data;
} ASTNode;
```

#### Construction
```nano
let num: ASTNode = ASTNode.Number { value: 42 }
```
↓
```c
ASTNode num = {
    .tag = ASTNODE_TAG_NUMBER,
    .data.number = { .value = 42LL }
};
```

#### Pattern Matching
```nano
match node {
    Number(n) => return n.value,
    String(s) => return 0
}
```
↓
```c
switch (node.tag) {
    case ASTNODE_TAG_NUMBER: {
        struct { int64_t value; } n = node.data.number;
        return n.value;
    }
    case ASTNODE_TAG_STRING: {
        struct { const char* value; } s = node.data.string;
        return 0LL;
    }
}
```

**Files to Modify:**
- `src/transpiler.c` - Add union code generation

**Tasks:**
- [ ] Generate tag enum for each union
- [ ] Generate tagged union struct
- [ ] Generate construction code
- [ ] Generate match as switch statement
- [ ] Handle nested unions
- [ ] Test transpiler output

---

### Phase 5: Testing (2-3 hours)

**Test Cases:**

#### Test 1: Simple Union
```nano
union Color {
    Red {},
    Green {},
    Blue { intensity: int }
}

fn color_value(c: Color) -> int {
    match c {
        Red(r) => return 1,
        Green(g) => return 2,
        Blue(b) => return b.intensity
    }
}

shadow color_value {
    let red: Color = Color.Red {}
    assert (== (color_value red) 1)
    
    let blue: Color = Color.Blue { intensity: 5 }
    assert (== (color_value blue) 5)
}
```

#### Test 2: Recursive Union (AST-like)
```nano
union Expr {
    Num { val: int },
    Add { left: Expr, right: Expr }
}

fn eval_expr(e: Expr) -> int {
    match e {
        Num(n) => return n.val,
        Add(a) => return (+ (eval_expr a.left) (eval_expr a.right))
    }
}

shadow eval_expr {
    let two: Expr = Expr.Num { val: 2 }
    let three: Expr = Expr.Num { val: 3 }
    let sum: Expr = Expr.Add { left: two, right: three }
    assert (== (eval_expr sum) 5)
}
```

#### Test 3: Union with Strings
```nano
union Token {
    Number { value: int },
    Identifier { name: string },
    Operator { op: string }
}

fn token_to_string(t: Token) -> string {
    match t {
        Number(n) => return "num",
        Identifier(i) => return i.name,
        Operator(o) => return o.op
    }
}
```

#### Test 4: Exhaustiveness Check (should fail)
```nano
union Status {
    Ok {},
    Error { msg: string },
    Pending {}
}

fn check(s: Status) -> int {
    match s {
        Ok(o) => return 1,
        Error(e) => return 0
        # Missing Pending - should be type error!
    }
}
```

**Test Files:**
- [ ] `tests/unit/test_union_simple.nano`
- [ ] `tests/unit/test_union_recursive.nano`
- [ ] `tests/unit/test_union_strings.nano`
- [ ] `tests/negative/test_union_exhaustiveness.nano`
- [ ] `examples/union_ast.nano` - Full AST example

---

## Implementation Checklist

### Lexer
- [ ] Add TOKEN_UNION, TOKEN_MATCH, TOKEN_ARROW
- [ ] Update keyword table
- [ ] Test lexer tokens

### Parser  
- [ ] Define UnionDef AST node
- [ ] Define MatchExpr AST node
- [ ] Implement parse_union_def()
- [ ] Implement parse_match_expr()
- [ ] Implement parse_union_construction()
- [ ] Test parser output

### Type Checker
- [ ] Add TYPE_UNION
- [ ] Implement union type storage
- [ ] Implement check_union_def()
- [ ] Implement check_match_expr() with exhaustiveness
- [ ] Implement check_union_construction()
- [ ] Test type checking

### Transpiler
- [ ] Generate tag enum
- [ ] Generate tagged union struct
- [ ] Generate union construction
- [ ] Generate match as switch
- [ ] Test C code generation

### Testing
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Write negative tests
- [ ] Write examples
- [ ] Validate all tests pass

### Documentation
- [ ] Update SPECIFICATION.md
- [ ] Update QUICK_REFERENCE.md
- [ ] Add UNION_TYPES_GUIDE.md
- [ ] Update examples/

---

## Progress Tracking

**Current Phase:** Phase 1 - Lexer  
**Hours Spent:** 0  
**Estimated Remaining:** 15-20 hours

---

## Notes

- Union types are heap-allocated in C (pointers)
- Recursive unions need forward declarations
- Match arms must cover all variants (exhaustiveness checking)
- Pattern binding introduces new variable in scope
- Unions can be nested (union inside union)

---

**Status:** Ready to begin implementation! 🚀

