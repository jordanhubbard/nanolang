#!/usr/bin/env python3
"""Generate parser_store functions for new features"""

# Template for all parser fields
PARSER_FIELDS = """        tokens: p.tokens,
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
        numbers_count: p.numbers_count,
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
        fors_count: FORS_COUNT,
        returns_count: p.returns_count,
        blocks_count: p.blocks_count,
        prints_count: p.prints_count,
        asserts_count: p.asserts_count,
        functions_count: p.functions_count,
        shadows_count: SHADOWS_COUNT,
        structs_count: p.structs_count,
        struct_literals_count: p.struct_literals_count,
        field_accesses_count: FIELD_ACCESSES_COUNT,
        enums_count: p.enums_count,
        unions_count: p.unions_count,
        union_constructs_count: p.union_constructs_count,
        matches_count: p.matches_count,
        imports_count: IMPORTS_COUNT,
        opaque_types_count: OPAQUE_TYPES_COUNT,
        tuple_literals_count: p.tuple_literals_count,
        tuple_indices_count: TUPLE_INDICES_COUNT"""

# Generate parser_store_for
def gen_for():
    return '''/* Helper: Store for loop node */
fn parser_store_for(p: Parser, var_name: string, iterable_id: int, iterable_type: int, body_id: int, line: int, column: int) -> Parser {
    let node: ASTFor = ASTFor {
        node_type: ParseNodeType.PNODE_FOR,
        line: line,
        column: column,
        var_name: var_name,
        iterable: iterable_id,
        iterable_type: iterable_type,
        body: body_id
    }
    let node_id: int = p.fors_count
    (list_ASTFor_push p.fors node)
    
    return Parser {
PARSER_FIELDS,
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: -1
    }
}
'''.replace('PARSER_FIELDS', PARSER_FIELDS.replace('FORS_COUNT', '(+ p.fors_count 1)').replace('SHADOWS_COUNT', 'p.shadows_count').replace('FIELD_ACCESSES_COUNT', 'p.field_accesses_count').replace('IMPORTS_COUNT', 'p.imports_count').replace('OPAQUE_TYPES_COUNT', 'p.opaque_types_count').replace('TUPLE_INDICES_COUNT', 'p.tuple_indices_count'))

# Generate parser_store_array_literal
def gen_array_literal():
    return '''/* Helper: Store array literal node */
fn parser_store_array_literal(p: Parser, element_count: int, line: int, column: int) -> Parser {
    let node: ASTArrayLiteral = ASTArrayLiteral {
        node_type: ParseNodeType.PNODE_ARRAY_LITERAL,
        line: line,
        column: column,
        element_count: element_count
    }
    let node_id: int = p.array_literals_count
    (list_ASTArrayLiteral_push p.array_literals node)
    
    return Parser {
PARSER_FIELDS,
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: -1
    }
}
'''.replace('PARSER_FIELDS', PARSER_FIELDS.replace('FORS_COUNT', 'p.fors_count').replace('SHADOWS_COUNT', 'p.shadows_count').replace('FIELD_ACCESSES_COUNT', 'p.field_accesses_count').replace('IMPORTS_COUNT', 'p.imports_count').replace('OPAQUE_TYPES_COUNT', 'p.opaque_types_count').replace('TUPLE_INDICES_COUNT', 'p.tuple_indices_count'))

# Generate parser_store_import
def gen_import():
    return '''/* Helper: Store import node */
fn parser_store_import(p: Parser, module_path: string, module_name: string, line: int, column: int) -> Parser {
    let node: ASTImport = ASTImport {
        node_type: ParseNodeType.PNODE_IMPORT,
        line: line,
        column: column,
        module_path: module_path,
        module_name: module_name
    }
    let node_id: int = p.imports_count
    (list_ASTImport_push p.imports node)
    
    return Parser {
PARSER_FIELDS,
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: -1
    }
}
'''.replace('PARSER_FIELDS', PARSER_FIELDS.replace('FORS_COUNT', 'p.fors_count').replace('SHADOWS_COUNT', 'p.shadows_count').replace('FIELD_ACCESSES_COUNT', 'p.field_accesses_count').replace('IMPORTS_COUNT', '(+ p.imports_count 1)').replace('OPAQUE_TYPES_COUNT', 'p.opaque_types_count').replace('TUPLE_INDICES_COUNT', 'p.tuple_indices_count'))

# Generate parser_store_opaque_type
def gen_opaque_type():
    return '''/* Helper: Store opaque type node */
fn parser_store_opaque_type(p: Parser, type_name: string, line: int, column: int) -> Parser {
    let node: ASTOpaqueType = ASTOpaqueType {
        node_type: ParseNodeType.PNODE_OPAQUE_TYPE,
        line: line,
        column: column,
        type_name: type_name
    }
    let node_id: int = p.opaque_types_count
    (list_ASTOpaqueType_push p.opaque_types node)
    
    return Parser {
PARSER_FIELDS,
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: -1
    }
}
'''.replace('PARSER_FIELDS', PARSER_FIELDS.replace('FORS_COUNT', 'p.fors_count').replace('SHADOWS_COUNT', 'p.shadows_count').replace('FIELD_ACCESSES_COUNT', 'p.field_accesses_count').replace('IMPORTS_COUNT', 'p.imports_count').replace('OPAQUE_TYPES_COUNT', '(+ p.opaque_types_count 1)').replace('TUPLE_INDICES_COUNT', 'p.tuple_indices_count'))

# Generate parser_store_shadow
def gen_shadow():
    return '''/* Helper: Store shadow test node */
fn parser_store_shadow(p: Parser, target_name: string, body_id: int, line: int, column: int) -> Parser {
    let node: ASTShadow = ASTShadow {
        node_type: ParseNodeType.PNODE_SHADOW,
        line: line,
        column: column,
        target_name: target_name,
        body: body_id
    }
    let node_id: int = p.shadows_count
    (list_ASTShadow_push p.shadows node)
    
    return Parser {
PARSER_FIELDS,
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: -1
    }
}
'''.replace('PARSER_FIELDS', PARSER_FIELDS.replace('FORS_COUNT', 'p.fors_count').replace('SHADOWS_COUNT', '(+ p.shadows_count 1)').replace('FIELD_ACCESSES_COUNT', 'p.field_accesses_count').replace('IMPORTS_COUNT', 'p.imports_count').replace('OPAQUE_TYPES_COUNT', 'p.opaque_types_count').replace('TUPLE_INDICES_COUNT', 'p.tuple_indices_count'))

# Generate parser_store_field_access
def gen_field_access():
    return '''/* Helper: Store field access node */
fn parser_store_field_access(p: Parser, object_id: int, object_type: int, field_name: string, line: int, column: int) -> Parser {
    let node: ASTFieldAccess = ASTFieldAccess {
        node_type: ParseNodeType.PNODE_FIELD_ACCESS,
        line: line,
        column: column,
        object: object_id,
        object_type: object_type,
        field_name: field_name
    }
    let node_id: int = p.field_accesses_count
    (list_ASTFieldAccess_push p.field_accesses node)
    
    return Parser {
PARSER_FIELDS,
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: -1
    }
}
'''.replace('PARSER_FIELDS', PARSER_FIELDS.replace('FORS_COUNT', 'p.fors_count').replace('SHADOWS_COUNT', 'p.shadows_count').replace('FIELD_ACCESSES_COUNT', '(+ p.field_accesses_count 1)').replace('IMPORTS_COUNT', 'p.imports_count').replace('OPAQUE_TYPES_COUNT', 'p.opaque_types_count').replace('TUPLE_INDICES_COUNT', 'p.tuple_indices_count'))

# Generate parser_store_float
def gen_float():
    return '''/* Helper: Store float node */
fn parser_store_float(p: Parser, value: string, line: int, column: int) -> Parser {
    let node: ASTFloat = ASTFloat {
        node_type: ParseNodeType.PNODE_FLOAT,
        line: line,
        column: column,
        value: value
    }
    let node_id: int = p.floats_count
    (list_ASTFloat_push p.floats node)
    
    return Parser {
PARSER_FIELDS,
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: -1
    }
}
'''.replace('PARSER_FIELDS', PARSER_FIELDS.replace('FORS_COUNT', 'p.fors_count').replace('SHADOWS_COUNT', 'p.shadows_count').replace('FIELD_ACCESSES_COUNT', 'p.field_accesses_count').replace('IMPORTS_COUNT', 'p.imports_count').replace('OPAQUE_TYPES_COUNT', 'p.opaque_types_count').replace('TUPLE_INDICES_COUNT', 'p.tuple_indices_count'))

# Generate all functions
functions = [
    gen_for(),
    gen_array_literal(),
    gen_import(),
    gen_opaque_type(),
    gen_shadow(),
    gen_field_access(),
    gen_float()
]

print('\n'.join(functions))
print(f"\n/* Generated {len(functions)} parser_store functions */")
