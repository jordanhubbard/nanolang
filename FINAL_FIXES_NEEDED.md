# Final Fixes Needed for Full Parser

## Current Status

✅ Created branch `feat/full-parser`  
✅ Added all 31 node types to enum  
✅ Added all 16 new AST structs  
✅ Updated Parser struct with 30 lists  
✅ Updated parser_init_ast_lists and parser_new  
✅ Refactored 11/14 parser_store functions  
⚠️ **Need to manually fix 3 functions + 2 enum variants**

## Issue

The Python refactoring script has a bug when re-processing already-processed functions. The backup file `parser_mvp_before_regen.nano` has 11 correctly refactored functions, but 3 still reference the deleted `parser_increment_count` helper.

## Manual Fixes Required

### 1. Fix parser_store_number (line ~1006)

**Current (broken):**
```nano
fn parser_store_number(p: Parser, value: string, line: int, column: int) -> Parser {
    let node: ASTNumber = ASTNumber {
        node_type: ParseNodeType.PNODE_NUMBER,
        line: line,
        column: column,
        value: value
    }
    let node_id: int = p.numbers_count
    (list_ASTNumber_push p.numbers node)
    return (parser_increment_count p ParseNodeType.PNODE_NUMBER node_id 0)
}
```

**Replace with (see parser_store_call as template):**
```nano
fn parser_store_number(p: Parser, value: string, line: int, column: int) -> Parser {
    let node: ASTNumber = ASTNumber {
        node_type: ParseNodeType.PNODE_NUMBER,
        line: line,
        column: column,
        value: value
    }
    let node_id: int = p.numbers_count
    (list_ASTNumber_push p.numbers node)
    
    return Parser {
        tokens: p.tokens,
        position: p.position,
        token_count: p.token_count,
        has_error: p.has_error,
        numbers: p.numbers,
        floats: p.floats,
        strings: p.strings,
        bools: p.bools,
        identifiers: p.identifiers,
        binary_ops: p.binary_ops,
        calls: p.calls,
        array_literals: p.array_literals,
        lets: p.lets,
        sets: p.sets,
        ifs: p.ifs,
        whiles: p.whiles,
        fors: p.fors,
        returns: p.returns,
        blocks: p.blocks,
        prints: p.prints,
        asserts: p.asserts,
        functions: p.functions,
        shadows: p.shadows,
        structs: p.structs,
        struct_literals: p.struct_literals,
        field_accesses: p.field_accesses,
        enums: p.enums,
        unions: p.unions,
        union_constructs: p.union_constructs,
        matches: p.matches,
        imports: p.imports,
        opaque_types: p.opaque_types,
        tuple_literals: p.tuple_literals,
        tuple_indices: p.tuple_indices,
        numbers_count: (+ p.numbers_count 1),    /* INCREMENT THIS ONE */
        floats_count: p.floats_count,
        strings_count: p.strings_count,
        bools_count: p.bools_count,
        identifiers_count: p.identifiers_count,
        binary_ops_count: p.binary_ops_count,
        calls_count: p.calls_count,
        array_literals_count: p.array_literals_count,
        lets_count: p.lets_count,
        sets_count: p.sets_count,
        ifs_count: p.ifs_count,
        whiles_count: p.whiles_count,
        fors_count: p.fors_count,
        returns_count: p.returns_count,
        blocks_count: p.blocks_count,
        prints_count: p.prints_count,
        asserts_count: p.asserts_count,
        functions_count: p.functions_count,
        shadows_count: p.shadows_count,
        structs_count: p.structs_count,
        struct_literals_count: p.struct_literals_count,
        field_accesses_count: p.field_accesses_count,
        enums_count: p.enums_count,
        unions_count: p.unions_count,
        union_constructs_count: p.union_constructs_count,
        matches_count: p.matches_count,
        imports_count: p.imports_count,
        opaque_types_count: p.opaque_types_count,
        tuple_literals_count: p.tuple_literals_count,
        tuple_indices_count: p.tuple_indices_count,
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: 0
    }
}
```

### 2. Fix parser_store_identifier (line ~1024)

Same pattern as above, but:
- Increment `identifiers_count` instead
- Use `last_expr_node_type: 1`

### 3. Fix parser_store_binary_op (line ~1046)

Same pattern, but:
- Increment `binary_ops_count` instead  
- Use `last_expr_node_type: 2`

### 4. Fix parser_store_struct (line ~2303)

**Current:**
```nano
node_type: ParseNodeType.PNODE_STRUCT,
```

**Change to:**
```nano
node_type: ParseNodeType.PNODE_STRUCT_DEF,
```

### 5. Fix parser_store_union (line ~2479)

**Current:**
```nano
node_type: ParseNodeType.PNODE_UNION,
```

**Change to:**
```nano
node_type: ParseNodeType.PNODE_UNION_DEF,
```

## Testing

After fixes, compile:
```bash
cd /Users/jkh/Src/nanolang
./bin/nanoc src_nano/parser_mvp.nano
```

Should compile cleanly (warnings about missing shadow tests are OK).

## Files Reference

- **Working file**: `src_nano/parser_mvp.nano` (needs 5 fixes above)
- **Backup (MVP original)**: `src_nano/parser_mvp_backup.nano`
- **Backup (before script bug)**: `src_nano/parser_mvp_before_regen.nano`
- **Template for expansions**: See `parser_store_call` around line 1057

## Quick Fix Script

Alternatively, you can use sed:

```bash
cd /Users/jkh/Src/nanolang

# Fix enum variants
sed -i '' 's/ParseNodeType\.PNODE_STRUCT,/ParseNodeType.PNODE_STRUCT_DEF,/g' src_nano/parser_mvp.nano
sed -i '' 's/ParseNodeType\.PNODE_UNION,/ParseNodeType.PNODE_UNION_DEF,/g' src_nano/parser_mvp.nano

# For the three functions, manual editing is safer
```

## Estimated Time

- Manual fixes: 15-20 minutes of careful copy-paste
- Testing: 5 minutes
- **Total: ~25 minutes to completion**
