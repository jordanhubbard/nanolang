# Parser Refactoring Plan

**Goal**: Split 6,743-line `parser.nano` into maintainable, testable modules.

## Current Structure Analysis
- **147 functions** in a single file
- Heavy interdependencies (Parser state threading)
- Difficult to debug and test

## Proposed Module Structure

### 1. `parser_core.nano` (~800 lines)
**Core parser state and navigation**
- Parser struct initialization (`parser_init_ast_lists`, `parser_new`)
- Position management (`parser_advance`, `parser_is_at_end`, `parser_peek`)
- State management (`parser_with_position`, `parser_with_error`, `parser_allocate_id`)
- Token matching (`parser_match`, `parser_expect`)

### 2. `parser_tokens.nano` (~200 lines)
**Token type helpers** (currently 64 `token_*` functions)
- Convert to enum-style or lookup table
- Reduce from 64 functions to data-driven approach
- `get_token_type(name: string) -> int`

### 3. `parser_types.nano` (~400 lines)
**Type parsing utilities**
- `parse_type_string`
- `is_type_start_token_type`
- `parse_qualified_name`
- `parse_call_name`

### 4. `parser_expressions.nano` (~1,800 lines)
**Expression parsing**
- `parse_primary`
- `parse_expression_recursive`
- `parse_expression`
- `is_binary_op`
- `parse_cond_expression`, `parse_cond_clauses`
- `parse_union_construct`, `parse_struct_literal`, `parse_match`

### 5. `parser_statements.nano` (~1,200 lines)
**Statement parsing**
- `parse_statement`
- `parse_let_statement`, `parse_if_statement`, `parse_while_statement`
- `parse_for_statement`, `parse_return_statement`, `parse_assert_statement`
- `parse_block`, `parse_unsafe_block`

### 6. `parser_definitions.nano` (~1,400 lines)
**Top-level definition parsing**
- `parse_definition`
- `parse_function_definition`, `parse_extern_function_definition`
- `parse_struct_definition`, `parse_enum_definition`, `parse_union_definition`
- `parse_import`, `parse_from_import`, `parse_opaque_type`, `parse_shadow`

### 7. `parser_storage.nano` (~900 lines)
**AST node storage helpers** (23 `parser_store_*` functions)
- `parser_store_number`, `parser_store_string`, `parser_store_identifier`
- `parser_store_binary_op`, `parser_store_call`, `parser_store_call_arg`
- `parser_store_let`, `parser_store_if`, `parser_store_while`
- etc.

## Refactoring Strategy

### Phase 1: Extract Clean Modules (Low Risk)
1. ✅ `parser_tokens.nano` - Pure functions, no dependencies
2. ✅ `parser_core.nano` - Foundation layer
3. ✅ `parser_types.nano` - Type utilities

### Phase 2: Extract Core Logic (Medium Risk)
4. ✅ `parser_storage.nano` - AST builders
5. ✅ `parser_expressions.nano` - Expression parsing

### Phase 3: Extract Top-Level (Higher Risk)
6. ✅ `parser_statements.nano` - Statement parsing
7. ✅ `parser_definitions.nano` - Top-level parsing

### Phase 4: Integration & Testing
8. Update imports in `nanoc_v06.nano`
9. Add shadow tests for each module
10. Test self-compilation

## Testing Strategy
- Add shadow tests for each extracted module
- Test incrementally after each module extraction
- Ensure `bin/nanoc` still compiles after each step
- Final test: `nanoc_v06` compiles itself

## Expected Benefits
- **Maintainability**: ~1,000 lines per module vs 6,743
- **Testability**: Isolated shadow tests per module
- **Debuggability**: Easier to trace bugs in smaller files
- **Self-Hosting**: Fixed bugs → 100% self-compilation

