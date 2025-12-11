# Implementation Guide - Remaining Parser Features

## Status: Token Helpers Added ✅, Architecture Complete ✅

The parser has **all infrastructure ready**. This guide provides exact code to add for each remaining feature.

## Quick Reference: What's Left

| Feature | Location | Effort | Priority |
|---------|----------|--------|----------|
| FOR loops | parse_statement | 1 hour | HIGH |
| Array literals | parse_primary | 1 hour | HIGH |
| import/opaque/shadow | parse_definition | 1 hour | HIGH |
| Field access | parse_expression postfix | 2.5 hours | HIGH |
| Float literals | parse_primary | 15 min | MEDIUM |
| Struct literals | parse_primary | 2 hours | MEDIUM |
| Match expressions | parse_primary | 4 hours | LOW |
| Union construction | postfix or primary | 2 hours | LOW |
| Tuple literals | parse_primary | 2 hours | LOW |

---

## Feature 1: FOR Loops (1 hour)

### Location: parse_statement function (~line 2446)

Add after the `while` check:

```nano
if (== tok.token_type (token_for)) {
    let p1: Parser = (parser_advance p)  /* consume 'for' */
    if (parser_is_at_end p1) {
        return (parser_with_error p1 true)
    } else {
        let tok2: LexToken = (parser_current p1)
        if (== tok2.token_type (token_identifier)) {
            let var_name: string = tok2.value
            let p2: Parser = (parser_advance p1)  /* consume identifier */
            
            /* Expect 'in' keyword */
            let tok3: LexToken = (parser_current p2)
            if (== tok3.token_type (token_in)) {
                let p3: Parser = (parser_advance p2)  /* consume 'in' */
                
                /* Parse iterable expression */
                let p4: Parser = (parse_expression p3)
                if (parser_has_error p4) {
                    return p4
                } else {
                    let iterable_id: int = p4.last_expr_node_id
                    let iterable_type: int = p4.last_expr_node_type
                    
                    /* Parse body block */
                    let p5: Parser = (parse_block p4)
                    if (parser_has_error p5) {
                        return p5
                    } else {
                        let body_id: int = p5.last_expr_node_id
                        return (parser_store_for p5 var_name iterable_id iterable_type body_id tok.line tok.column)
                    }
                }
            } else {
                return (parser_with_error p2 true)
            }
        } else {
            return (parser_with_error p1 true)
        }
    }
} else {
```

**Also need parser_store_for function** (~40 lines, copy pattern from parser_store_while)

---

## Feature 2: Array Literals (1 hour)

### Location: parse_primary function (~line 1440)

Add after checking for numbers/strings/etc:

```nano
if (== tok.token_type (token_lbracket)) {
    let p1: Parser = (parser_advance p)  /* consume '[' */
    let mut element_count: int = 0
    let mut pcur: Parser = p1
    
    /* Parse array elements */
    while (not (parser_is_at_end pcur)) {
        let tcur: LexToken = (parser_current pcur)
        if (== tcur.token_type (token_rbracket)) {
            /* End of array */
            let p2: Parser = (parser_advance pcur)
            return (parser_store_array_literal p2 element_count tok.line tok.column)
        } else {
            /* Parse element expression */
            let pelem: Parser = (parse_expression pcur)
            if (parser_has_error pelem) {
                return pelem
            } else {
                set element_count (+ element_count 1)
                set pcur pelem
                
                /* Check for comma */
                let tnext: LexToken = (parser_current pcur)
                if (== tnext.token_type (token_comma)) {
                    set pcur (parser_advance pcur)
                } else {
                    /* No comma, should be end bracket */
                    if (== tnext.token_type (token_rbracket)) {
                        let p2: Parser = (parser_advance pcur)
                        return (parser_store_array_literal p2 element_count tok.line tok.column)
                    } else {
                        return (parser_with_error pcur true)
                    }
                }
            }
        }
    }
    return (parser_with_error pcur true)
} else {
```

**Also need parser_store_array_literal function** (~40 lines)

---

## Feature 3: import/opaque/shadow (1 hour)

### Location: parse_definition function (~line 3220)

Add after the union check:

```nano
if (== tok.token_type (token_import)) {
    /* Parse: import "path" as name */
    let p1: Parser = (parser_advance p)  /* consume 'import' */
    let tok2: LexToken = (parser_current p1)
    if (== tok2.token_type (token_string)) {
        let module_path: string = tok2.value
        let p2: Parser = (parser_advance p1)
        let tok3: LexToken = (parser_current p2)
        if (== tok3.token_type (token_as)) {
            let p3: Parser = (parser_advance p2)
            let tok4: LexToken = (parser_current p3)
            if (== tok4.token_type (token_identifier)) {
                let module_name: string = tok4.value
                let p4: Parser = (parser_advance p3)
                return (parser_store_import p4 module_path module_name tok.line tok.column)
            } else {
                return (parser_with_error p3 true)
            }
        } else {
            return (parser_with_error p2 true)
        }
    } else {
        return (parser_with_error p1 true)
    }
} else {
    if (== tok.token_type (token_opaque)) {
        /* Parse: opaque type TypeName */
        let p1: Parser = (parser_advance p)  /* consume 'opaque' */
        /* Expect 'type' keyword - this is actually TOKEN_TYPE_INT starting at different values */
        let p2: Parser = (parser_advance p1)  /* Skip type keyword check for MVP */
        let tok3: LexToken = (parser_current p2)
        if (== tok3.token_type (token_identifier)) {
            let type_name: string = tok3.value
            let p3: Parser = (parser_advance p2)
            return (parser_store_opaque_type p3 type_name tok.line tok.column)
        } else {
            return (parser_with_error p2 true)
        }
    } else {
        if (== tok.token_type (token_shadow)) {
            /* Parse: shadow target_name { body } */
            let p1: Parser = (parser_advance p)  /* consume 'shadow' */
            let tok2: LexToken = (parser_current p1)
            if (== tok2.token_type (token_identifier)) {
                let target_name: string = tok2.value
                let p2: Parser = (parser_advance p1)
                let p3: Parser = (parse_block p2)
                if (parser_has_error p3) {
                    return p3
                } else {
                    let body_id: int = p3.last_expr_node_id
                    return (parser_store_shadow p3 target_name body_id tok.line tok.column)
                }
            } else {
                return (parser_with_error p1 true)
            }
        } else {
```

**Also need 3 store functions:**
- parser_store_import (~40 lines)
- parser_store_opaque_type (~40 lines)  
- parser_store_shadow (~40 lines)

---

## Feature 4: Field Access (2.5 hours) - COMPLEX

### Location: parse_expression function (~line 1597)

This requires modifying the parse_expression flow to add postfix operator handling after parse_primary.

Current flow:
```
parse_expression -> parse_expression_recursive -> parse_primary
```

Need to add after parse_primary returns:
```
parse_expression -> parse_expression_recursive -> parse_primary -> check for dot -> parse_field_access
```

### Implementation:

After calling parse_primary in parse_expression_recursive, add:

```nano
/* After parse_primary call (around line 1570) */
let mut p_postfix: Parser = p_after_primary

/* Handle postfix operators (field access, tuple index) */
while (not (parser_is_at_end p_postfix)) {
    let tok_post: LexToken = (parser_current p_postfix)
    if (== tok_post.token_type (token_dot)) {
        let p1: Parser = (parser_advance p_postfix)  /* consume '.' */
        let tok2: LexToken = (parser_current p1)
        
        /* Check if it's a tuple index (number) or field access (identifier) */
        if (== tok2.token_type (token_number)) {
            /* Tuple index: obj.0, obj.1 */
            let index_str: string = tok2.value
            let p2: Parser = (parser_advance p1)
            let tuple_id: int = p_postfix.last_expr_node_id
            let tuple_type: int = p_postfix.last_expr_node_type
            /* Need to convert string to int - for MVP, store as 0 */
            set p_postfix (parser_store_tuple_index p2 tuple_id tuple_type 0 tok_post.line tok_post.column)
        } else {
            if (== tok2.token_type (token_identifier)) {
                /* Field access: obj.field */
                let field_name: string = tok2.value
                let p2: Parser = (parser_advance p1)
                let object_id: int = p_postfix.last_expr_node_id
                let object_type: int = p_postfix.last_expr_node_type
                set p_postfix (parser_store_field_access p2 object_id object_type field_name tok_post.line tok_post.column)
            } else {
                break
            }
        }
    } else {
        break
    }
}

return p_postfix
```

**Also need 2 store functions:**
- parser_store_field_access (~40 lines)
- parser_store_tuple_index (~40 lines)

---

## Feature 5: Float Literals (15 min) - EASY

### Location: parse_primary, number handling (~line 1465)

Currently all numbers go to parser_store_number. Add check:

```nano
if (== tok.token_type (token_number)) {
    /* Check if it contains a dot (simple float detection) */
    let value: string = tok.value
    /* For MVP: Just use parser_store_number for both */
    /* To properly distinguish: need a string contains check */
    let p1: Parser = (parser_store_number p tok.value tok.line tok.column)
    return (parser_advance p1)
} else {
```

**Note:** Proper float support needs:
1. Check if value contains "." character
2. Call parser_store_float instead of parser_store_number
3. Add parser_store_float function (~40 lines, copy from parser_store_number)

---

## Implementation Order

1. **Add all parser_store functions first** (2 hours)
   - parser_store_for
   - parser_store_array_literal
   - parser_store_import
   - parser_store_opaque_type
   - parser_store_shadow
   - parser_store_field_access
   - parser_store_tuple_index
   - parser_store_float

2. **Add parsing logic** (3-4 hours)
   - FOR loops in parse_statement
   - Array literals in parse_primary
   - import/opaque/shadow in parse_definition
   - Field access postfix in parse_expression
   - Float handling in parse_primary

3. **Test** (1 hour)
   - Compile parser
   - Add shadow tests
   - Test with examples

---

## Testing

After implementation, test with:

```bash
# Compile parser
./bin/nanoc src_nano/parser_mvp.nano

# Test FOR loops
cat > test_for.nano << 'EOF'
fn test() -> int {
    for i in range {
        (print i)
    }
    return 0
}
EOF
./bin/nanoc test_for.nano

# Test arrays
cat > test_arrays.nano << 'EOF'
let nums: array<int> = [1, 2, 3, 4, 5]
EOF
./bin/nanoc test_arrays.nano

# Test field access
cat > test_fields.nano << 'EOF'
struct Point { x: int, y: int }
fn test(p: Point) -> int {
    return p.x
}
EOF
./bin/nanoc test_fields.nano
```

---

## Summary

**Total Effort:** ~7 hours

**Files to Modify:**
- `src_nano/parser_mvp.nano` (add ~500 lines)

**Benefits:**
- FOR loops enable iteration
- Array literals enable data structures
- import/opaque/shadow enable modules and testing
- Field access enables OOP patterns
- Float literals complete numeric types

**Result:** Parser goes from ~60% to ~85% feature complete

---

**Last Updated:** 2025-12-10
**Status:** Ready for implementation
**Branch:** feat/full-parser
