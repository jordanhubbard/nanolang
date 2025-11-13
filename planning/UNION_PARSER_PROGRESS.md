# Union Parser Implementation Progress

**Date:** November 13, 2025  
**Status:** Phase 2 - Parser (40% complete)

---

## ✅ Completed

### 1. Union Definition Parsing
**Function:** `parse_union_def()`

**Syntax Supported:**
```nano
union Color {
    Red {},
    Green {},
    Blue { intensity: int }
}
```

**Implementation:**
- Parses `union` keyword
- Parses union name (e.g., "Color")
- Parses multiple variants
- Each variant can have struct-like fields
- Supports optional commas between variants and fields
- Creates `AST_UNION_DEF` node with all variant information

**Data Structures:**
- `variant_names`: Array of variant name strings
- `variant_field_counts`: Array of field counts per variant
- `variant_field_names`: 2D array of field names per variant
- `variant_field_types`: 2D array of field types per variant

---

## 🚧 In Progress

### 2. Union Construction Parsing
**Function:** `parse_union_construct()` (NOT YET IMPLEMENTED)

**Syntax to Support:**
```nano
let red: Color = Color.Red {}
let blue: Color = Color.Blue { intensity: 5 }
```

**Requirements:**
- Parse `UnionName.VariantName { field: value, ... }` syntax
- Create `AST_UNION_CONSTRUCT` node
- Store union name, variant name, field names, field values

**Location:**  
Should be added to `parse_expression()` when encountering identifier followed by `.`

---

### 3. Match Expression Parsing
**Function:** `parse_match_expr()` (NOT YET IMPLEMENTED)

**Syntax to Support:**
```nano
match color {
    Red(r) => return 1,
    Green(g) => return 2,
    Blue(b) => return b.intensity
}
```

**Requirements:**
- Parse `match expr { pattern => body, ... }` syntax
- Parse pattern bindings: `VariantName(binding_var)`
- Parse `=>` arrow token
- Parse arm bodies (expressions or statements)
- Create `AST_MATCH` node

**Location:**
Should be added to `parse_expression()` or `parse_statement()` when encountering `match` keyword

---

## 📋 Next Steps

1. **Implement `parse_union_construct()`**
   - Modify `parse_expression()` to detect `Identifier.Identifier {` pattern
   - Parse field assignments inside braces
   - Create AST_UNION_CONSTRUCT node

2. **Implement `parse_match_expr()`**
   - Add to `parse_expression()` or `parse_statement()`
   - Parse match arms with pattern bindings
   - Handle `=>` arrows
   - Create AST_MATCH node

3. **Test Parser Output**
   - Write test programs with union syntax
   - Verify AST structure is correct
   - Check error handling for invalid syntax

4. **Move to Type Checker (Phase 3)**
   - Once all three parsing functions are complete
   - Implement type checking for unions
   - Add exhaustiveness checking for match

---

## Implementation Notes

### Parser Integration Points

**Union Construction:**
```c
// In parse_expression() after parsing an identifier:
if (match(p, TOKEN_DOT)) {
    // Could be struct field access OR union construction
    advance(p);  // consume '.'
    if (!match(p, TOKEN_IDENTIFIER)) {
        error("Expected field or variant name after '.'");
    }
    char *field_or_variant = strdup(current_token(p)->value);
    advance(p);
    
    if (match(p, TOKEN_LBRACE)) {
        // This is union construction: UnionName.Variant { ... }
        return parse_union_construct_rest(p, identifier, field_or_variant);
    } else {
        // This is field access: obj.field
        return create_field_access(identifier, field_or_variant);
    }
}
```

**Match Expression:**
```c
// In parse_expression() or parse_statement():
if (match(p, TOKEN_MATCH)) {
    return parse_match_expr(p);
}
```

### AST Node Structures (Already Defined)

```c
// union_construct
struct {
    char *union_name;          // "Color"
    char *variant_name;        // "Blue"
    int field_count;           // 1
    char **field_names;        // ["intensity"]
    ASTNode **field_values;    // [5]
} union_construct;

// match_expr
struct {
    ASTNode *expr;             // Expression to match on
    int arm_count;             // Number of match arms
    char **pattern_variants;   // Variant names ["Red", "Green", "Blue"]
    char **pattern_bindings;   // Binding variables ["r", "g", "b"]
    ASTNode **arm_bodies;      // Code for each arm
} match_expr;
```

---

## Current Blockers

- **None** - Parser definition is complete for unions
- **Ready** - Can proceed with construction and match parsing

---

## Time Estimate

- ✅ **parse_union_def**: 2 hours (COMPLETE)
- ⏱️ **parse_union_construct**: 1-1.5 hours (estimated)
- ⏱️ **parse_match_expr**: 2-2.5 hours (estimated)
- ⏱️ **Testing & debugging**: 0.5 hours

**Total remaining for parser phase: ~4-5 hours**

---

**Next Session:** Implement union construction and match parsing, then proceed to type checker.

