# Tuple Implementation Plan

**Status**: In Progress - Type System Complete ✅, Parser Pending

---

## Completed: Phase 1 - Type System (✅)

### Type Definitions Added

1. **Type Enum**: Added `TYPE_TUPLE` to `Type` enum
2. **Value Type**: Added `VAL_TUPLE` to `ValueType` enum  
3. **AST Node Types**: Added `AST_TUPLE_LITERAL` and `AST_TUPLE_INDEX`

### Data Structures

```c
/* Tuple value structure */
typedef struct {
    Value *elements;          /* Array of values */
    int element_count;        /* Number of elements */
} TupleValue;

/* Extended TypeInfo for tuples */
typedef struct TypeInfo {
    // ... existing fields ...
    
    /* For tuple types: (int, string, bool) */
    Type *tuple_types;               /* Array of tuple element types */
    char **tuple_type_names;         /* For struct/enum/union tuple elements */
    int tuple_element_count;         /* Number of tuple elements */
} TypeInfo;

/* AST nodes */
struct {
    ASTNode **elements;    /* Array of element expressions */
    int element_count;     /* Number of elements */
    Type *element_types;   /* Types (filled by type checker) */
} tuple_literal;

struct {
    ASTNode *tuple;        /* The tuple expression */
    int index;             /* The index (0, 1, 2, ...) */
} tuple_index;
```

### Helper Functions

```c
Value create_tuple(Value *elements, int element_count);
void free_tuple(TupleValue *tuple);
```

**Files Modified**:
- `src/nanolang.h` - Type system and AST structures
- `src/env.c` - Helper functions for creating tuples

---

## TODO: Phase 2 - Parser

### Syntax to Support

```nano
/* Tuple types in function signature */
fn divide_with_remainder(a: int, b: int) -> (int, int)

/* Tuple literal creation */
let result: (int, int) = (10, 3)

/* Tuple literal in return */
return (quotient, remainder)

/* Tuple index access */
let q: int = result.0
let r: int = result.1

/* Tuple destructuring (future) */
let (q, r): (int, int) = (divide_with_remainder 10 3)
```

### Parser Changes Needed

1. **Type Parsing**: `parse_type()` needs to handle `(Type, Type, ...)`
2. **Expression Parsing**: `parse_primary()` needs to disambiguate:
   - `(expr)` - parenthesized expression
   - `(expr, expr, ...)` - tuple literal (2+ elements)
3. **Field Access**: `parse_primary()` needs to handle `tuple.0`, `tuple.1`, etc.
4. **Function Signatures**: Already handles complex return types

### Implementation Strategy

```c
/* In parse_type() */
if (match(p, TOKEN_LPAREN)) {
    /* Could be tuple type: (int, string, bool) */
    Type *types = malloc(...);
    int count = 0;
    
    /* Parse first type */
    types[count++] = parse_type(p);
    
    /* If comma, it's a tuple */
    if (match(p, TOKEN_COMMA)) {
        /* Parse remaining types */
        do {
            types[count++] = parse_type(p);
        } while (match(p, TOKEN_COMMA));
        
        expect(p, TOKEN_RPAREN);
        /* Return TYPE_TUPLE with type info */
    } else {
        /* Just parenthesized type */
        expect(p, TOKEN_RPAREN);
        return types[0];
    }
}

/* In parse_primary() */
if (match(p, TOKEN_LPAREN)) {
    /* Could be:
     * 1. Parenthesized expression: (expr)
     * 2. Tuple literal: (expr, expr, ...)
     */
    ASTNode **elements = malloc(...);
    int count = 0;
    
    /* Parse first expression */
    elements[count++] = parse_expression(p);
    
    /* Check for comma */
    if (match(p, TOKEN_COMMA)) {
        /* It's a tuple literal */
        do {
            elements[count++] = parse_expression(p);
        } while (match(p, TOKEN_COMMA));
        
        expect(p, TOKEN_RPAREN);
        
        /* Create AST_TUPLE_LITERAL node */
        ASTNode *node = malloc(sizeof(ASTNode));
        node->type = AST_TUPLE_LITERAL;
        node->as.tuple_literal.elements = elements;
        node->as.tuple_literal.element_count = count;
        node->as.tuple_literal.element_types = NULL;  /* Filled by type checker */
        return node;
    } else {
        /* Just a parenthesized expression */
        expect(p, TOKEN_RPAREN);
        return elements[0];
    }
}

/* In parse_primary() - handle tuple.0, tuple.1 */
/* After parsing base expression */
while (match(p, TOKEN_DOT)) {
    Token *tok = current_token(p);
    
    /* Check if it's a number (tuple index) */
    if (tok->type == TOKEN_NUMBER) {
        int index = (int)tok->value;  /* Convert to int */
        advance(p);
        
        /* Create AST_TUPLE_INDEX node */
        ASTNode *index_node = malloc(sizeof(ASTNode));
        index_node->type = AST_TUPLE_INDEX;
        index_node->as.tuple_index.tuple = expr;
        index_node->as.tuple_index.index = index;
        expr = index_node;
    } else {
        /* Regular field access: struct.field_name */
        /* ... existing logic ... */
    }
}
```

---

## TODO: Phase 3 - Type Checker

### Type Checking Rules

1. **Tuple Literal Type Inference**:
   ```nano
   let t: (int, string) = (42, "hello")  /* Types must match */
   ```
   - Check element count matches
   - Check each element type matches

2. **Tuple Index Access**:
   ```nano
   let x: int = t.0  /* Must verify index is in bounds */
   let y: string = t.1
   ```
   - Verify base expression is `TYPE_TUPLE`
   - Verify index < element_count
   - Return the appropriate element type

3. **Function Return Types**:
   ```nano
   fn divide(a: int, b: int) -> (int, int) {
       return (10, 3)  /* Must match return type */
   }
   ```
   - Verify return expression is tuple
   - Verify tuple types match function signature

### Implementation

```c
/* In check_expression() */
case AST_TUPLE_LITERAL: {
    /* Type check each element */
    for (int i = 0; i < expr->as.tuple_literal.element_count; i++) {
        Type elem_type = check_expression(expr->as.tuple_literal.elements[i], env);
        /* Store types */
        if (!expr->as.tuple_literal.element_types) {
            expr->as.tuple_literal.element_types = malloc(sizeof(Type) * expr->as.tuple_literal.element_count);
        }
        expr->as.tuple_literal.element_types[i] = elem_type;
    }
    return TYPE_TUPLE;  /* With TypeInfo containing element types */
}

case AST_TUPLE_INDEX: {
    Type tuple_type = check_expression(expr->as.tuple_index.tuple, env);
    
    if (tuple_type != TYPE_TUPLE) {
        fprintf(stderr, "Error: Tuple index access on non-tuple type\n");
        return TYPE_UNKNOWN;
    }
    
    /* Verify index is in bounds */
    int index = expr->as.tuple_index.index;
    /* Get TypeInfo from symbol table or expression */
    /* Return element_types[index] */
}
```

---

## TODO: Phase 4 - Transpiler

### C Code Generation Strategy

Tuples will be transpiled to anonymous structs in C:

```nano
/* nanolang */
fn divide(a: int, b: int) -> (int, int) {
    return ((/ a b), (% a b))
}

let result: (int, int) = (divide 10 3)
let q: int = result.0
```

```c
/* Generated C */
typedef struct {
    int64_t _0;
    int64_t _1;
} Tuple_int_int;

Tuple_int_int divide(int64_t a, int64_t b) {
    Tuple_int_int _return_val;
    _return_val._0 = a / b;
    _return_val._1 = a % b;
    return _return_val;
}

Tuple_int_int result = divide(10, 3);
int64_t q = result._0;
```

### Implementation

```c
/* Generate typedef for tuple type */
void transpile_tuple_typedef(StringBuilder *sb, Type *types, int count) {
    sb_append(sb, "typedef struct {\n");
    for (int i = 0; i < count; i++) {
        sb_append(sb, "    ");
        sb_append(sb, type_to_c(types[i]));
        sb_appendf(sb, " _%d;\n", i);
    }
    sb_append(sb, "} ");
    /* Generate type name: Tuple_int_int or Tuple_int_string_bool */
    sb_append(sb, generate_tuple_type_name(types, count));
    sb_append(sb, ";\n\n");
}

/* Transpile tuple literal */
case AST_TUPLE_LITERAL: {
    /* Generate compound literal */
    const char *type_name = generate_tuple_type_name(
        expr->as.tuple_literal.element_types,
        expr->as.tuple_literal.element_count
    );
    
    sb_appendf(sb, "(%s){", type_name);
    for (int i = 0; i < expr->as.tuple_literal.element_count; i++) {
        if (i > 0) sb_append(sb, ", ");
        sb_appendf(sb, "._" % d " = ", i);
        transpile_expression(sb, expr->as.tuple_literal.elements[i], env);
    }
    sb_append(sb, "}");
    break;
}

/* Transpile tuple index */
case AST_TUPLE_INDEX: {
    transpile_expression(sb, expr->as.tuple_index.tuple, env);
    sb_appendf(sb, "._%d", expr->as.tuple_index.index);
    break;
}
```

---

## TODO: Phase 5 - Interpreter

### Evaluation Strategy

```c
/* Evaluate tuple literal */
case AST_TUPLE_LITERAL: {
    Value *elements = malloc(sizeof(Value) * expr->as.tuple_literal.element_count);
    
    for (int i = 0; i < expr->as.tuple_literal.element_count; i++) {
        elements[i] = eval_expression(expr->as.tuple_literal.elements[i], env);
    }
    
    return create_tuple(elements, expr->as.tuple_literal.element_count);
}

/* Evaluate tuple index */
case AST_TUPLE_INDEX: {
    Value tuple = eval_expression(expr->as.tuple_index.tuple, env);
    
    if (tuple.type != VAL_TUPLE) {
        fprintf(stderr, "Error: Tuple index access on non-tuple\n");
        return create_void();
    }
    
    int index = expr->as.tuple_index.index;
    if (index < 0 || index >= tuple.as.tuple_val->element_count) {
        fprintf(stderr, "Error: Tuple index %d out of bounds\n", index);
        return create_void();
    }
    
    return tuple.as.tuple_val->elements[index];
}

/* Print tuple */
case VAL_TUPLE: {
    TupleValue *tv = val.as.tuple_val;
    printf("(");
    for (int i = 0; i < tv->element_count; i++) {
        if (i > 0) printf(", ");
        print_value(tv->elements[i]);
    }
    printf(")");
    break;
}
```

---

## TODO: Phase 6 - Testing

### Test Cases

```nano
/* Test 1: Basic tuple return */
fn get_pair() -> (int, int) {
    return (10, 20)
}

shadow get_pair {
    let result: (int, int) = (get_pair)
    assert (== result.0 10)
    assert (== result.1 20)
}

/* Test 2: Tuple with mixed types */
fn get_info() -> (string, int, bool) {
    return ("Alice", 30, true)
}

shadow get_info {
    let info: (string, int, bool) = (get_info)
    assert (== info.0 "Alice")
    assert (== info.1 30)
    assert (== info.2 true)
}

/* Test 3: Division with remainder */
fn divide_with_remainder(a: int, b: int) -> (int, int) {
    let quotient: int = (/ a b)
    let remainder: int = (% a b)
    return (quotient, remainder)
}

shadow divide_with_remainder {
    let result: (int, int) = (divide_with_remainder 10 3)
    assert (== result.0 3)
    assert (== result.1 1)
}

/* Test 4: Tuple as parameter (future) */
fn add_pair(p: (int, int)) -> int {
    return (+ p.0 p.1)
}

shadow add_pair {
    assert (== (add_pair (10, 20)) 30)
}
```

---

## Summary

**Current Status**: Type system complete, ready for parser implementation

**Next Steps**:
1. Implement tuple type parsing in `parse_type()`
2. Implement tuple literal parsing in `parse_primary()`
3. Implement tuple index access in field access logic
4. Add type checking for tuples
5. Generate C code for tuples
6. Add interpreter support
7. Create comprehensive tests

**Files to Modify**:
- ✅ `src/nanolang.h` - Type system (DONE)
- ✅ `src/env.c` - Helper functions (DONE)
- ⏳ `src/parser.c` - Parse tuple syntax
- ⏳ `src/typechecker.c` - Type check tuples
- ⏳ `src/transpiler.c` - Generate C code
- ⏳ `src/eval.c` - Interpret tuples
- ⏳ `tests/` - Add comprehensive tests

