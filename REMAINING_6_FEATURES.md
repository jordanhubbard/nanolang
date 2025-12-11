# Remaining 6 Features - Implementation Plan

## Priority Order

### High Priority (4-5 hours)
1. **Field Access** (2.5 hours) - Critical for OOP
2. **Float Literals** (30 min) - Language completeness

### Medium Priority (4 hours)  
3. **Struct Literals** (2 hours) - Common pattern
4. **Match Expressions** (4 hours) - Advanced control flow

### Low Priority (4 hours)
5. **Union Construction** (2 hours) - Sum types
6. **Tuple Literals** (2 hours) - Product types

**Total: ~10-12 hours**

---

## Feature 1: Field Access (2.5 hours)

**Status:** parser_store_field_access exists ✅

**Strategy:** Add postfix operator loop after parse_primary in parse_expression

**Implementation:**
- Modify parse_expression_recursive to check for dot after primary
- Loop while current token is dot
- Parse field name (identifier) or tuple index (number)
- Call parser_store_field_access or parser_store_tuple_index

**Complexity:** Medium - requires understanding expression parsing flow

---

## Feature 2: Float Literals (30 min)

**Status:** parser_store_float exists ✅

**Strategy:** Add string contains check in parse_primary number handling

**Implementation:**
- In parse_primary where token_number is handled
- Check if value string contains "."
- If yes: call parser_store_float
- If no: call parser_store_number

**Complexity:** Low - simple string check

---

## Feature 3: Struct Literals (2 hours)

**Status:** parser_store_struct_literal exists ✅

**Strategy:** Add struct literal parsing to parse_primary

**Implementation:**
- In parse_primary, check if identifier is followed by lbrace
- Parse field: value pairs separated by commas
- Store each field/value pair
- Call parser_store_struct_literal with field count

**Complexity:** Medium - requires field parsing loop

---

## Feature 4: Match Expressions (4 hours)

**Status:** parser_store_match and parser_store_match_arm exist ✅

**Strategy:** Add match parsing to parse_primary

**Implementation:**
- Check for token_match
- Parse matched expression
- Parse arms: pattern => body separated by commas
- Each arm needs pattern parsing
- Call parser_store_match with arm count

**Complexity:** High - pattern matching is complex

---

## Feature 5: Union Construction (2 hours)

**Status:** parser_store_union_construct exists ✅

**Strategy:** Add to postfix operators after field access

**Implementation:**
- When we see dot after identifier, check next token
- If identifier followed by lbrace: union variant with fields
- Parse variant name and field values
- Call parser_store_union_construct

**Complexity:** Medium - similar to struct literals

---

## Feature 6: Tuple Literals (2 hours)

**Status:** parser_store_tuple_literal exists ✅

**Strategy:** Distinguish from parenthesized expressions

**Implementation:**
- In parse_primary lparen handling
- Parse first expression
- If comma follows: tuple literal
- If rparen follows: parenthesized expression
- Continue parsing tuple elements if comma found

**Complexity:** Medium - disambiguation needed

---

## Implementation Order

1. Float literals (30 min) - Easiest win
2. Field access (2.5 hours) - Most valuable
3. Struct literals (2 hours) - Common use case
4. Union construction (2 hours) - Goes with field access
5. Tuple literals (2 hours) - Expression disambiguation
6. Match expressions (4 hours) - Most complex

**Total: 13 hours conservative estimate**

