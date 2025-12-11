#!/usr/bin/env python3
"""
Complete fix: Regenerate the 3 parser_store functions AND fix the 3 parser_with functions.
"""

# Template list and count fields from parser_store_call
LIST_FIELDS = """        numbers: p.numbers,
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
        tuple_indices: p.tuple_indices,"""

COUNT_FIELDS_TEMPLATE = """        numbers_count: p.numbers_count,
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
        tuple_indices_count: p.tuple_indices_count,"""

def generate_parser_store_number():
    counts = COUNT_FIELDS_TEMPLATE.replace("numbers_count: p.numbers_count", "numbers_count: (+ p.numbers_count 1)")
    return f"""fn parser_store_number(p: Parser, value: string, line: int, column: int) -> Parser {{
    let node: ASTNumber = ASTNumber {{
        node_type: ParseNodeType.PNODE_NUMBER,
        line: line,
        column: column,
        value: value
    }}
    let node_id: int = p.numbers_count
    (list_ASTNumber_push p.numbers node)
    
    return Parser {{
        tokens: p.tokens,
        position: p.position,
        token_count: p.token_count,
        has_error: p.has_error,
{LIST_FIELDS}
{counts}
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: 0
    }}
}}"""

def generate_parser_store_identifier():
    counts = COUNT_FIELDS_TEMPLATE.replace("identifiers_count: p.identifiers_count", "identifiers_count: (+ p.identifiers_count 1)")
    return f"""fn parser_store_identifier(p: Parser, name: string, line: int, column: int) -> Parser {{
    let node: ASTIdentifier = ASTIdentifier {{
        node_type: ParseNodeType.PNODE_IDENTIFIER,
        line: line,
        column: column,
        name: name
    }}
    let node_id: int = p.identifiers_count
    (list_ASTIdentifier_push p.identifiers node)
    
    return Parser {{
        tokens: p.tokens,
        position: p.position,
        token_count: p.token_count,
        has_error: p.has_error,
{LIST_FIELDS}
{counts}
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: 1
    }}
}}"""

def generate_parser_store_binary_op():
    counts = COUNT_FIELDS_TEMPLATE.replace("binary_ops_count: p.binary_ops_count", "binary_ops_count: (+ p.binary_ops_count 1)")
    return f"""fn parser_store_binary_op(p: Parser, op: int, left_id: int, right_id: int, left_type: int, right_type: int, line: int, column: int) -> Parser {{
    let node: ASTBinaryOp = ASTBinaryOp {{
        node_type: ParseNodeType.PNODE_BINARY_OP,
        line: line,
        column: column,
        op: op,
        left: left_id,
        right: right_id,
        left_type: left_type,
        right_type: right_type
    }}
    let node_id: int = p.binary_ops_count
    (list_ASTBinaryOp_push p.binary_ops node)
    
    return Parser {{
        tokens: p.tokens,
        position: p.position,
        token_count: p.token_count,
        has_error: p.has_error,
{LIST_FIELDS}
{counts}
        next_node_id: (+ p.next_node_id 1),
        last_expr_node_id: node_id,
        last_expr_node_type: 2
    }}
}}"""

def generate_parser_with_position():
    return f"""fn parser_with_position(p: Parser, new_position: int) -> Parser {{
    return Parser {{
        tokens: p.tokens,
        position: new_position,
        token_count: p.token_count,
        has_error: p.has_error,
{LIST_FIELDS}
{COUNT_FIELDS_TEMPLATE}
        next_node_id: p.next_node_id,
        last_expr_node_id: p.last_expr_node_id,
        last_expr_node_type: p.last_expr_node_type
    }}
}}"""

def generate_parser_with_error():
    return f"""fn parser_with_error(p: Parser, error: bool) -> Parser {{
    return Parser {{
        tokens: p.tokens,
        position: p.position,
        token_count: p.token_count,
        has_error: error,
{LIST_FIELDS}
{COUNT_FIELDS_TEMPLATE}
        next_node_id: p.next_node_id,
        last_expr_node_id: p.last_expr_node_id,
        last_expr_node_type: p.last_expr_node_type
    }}
}}"""

def generate_parser_with_calls_count():
    counts = COUNT_FIELDS_TEMPLATE.replace("calls_count: p.calls_count", "calls_count: calls_count")
    return f"""fn parser_with_calls_count(p: Parser, calls_count: int) -> Parser {{
    return Parser {{
        tokens: p.tokens,
        position: p.position,
        token_count: p.token_count,
        has_error: p.has_error,
{LIST_FIELDS}
{counts}
        next_node_id: p.next_node_id,
        last_expr_node_id: p.last_expr_node_id,
        last_expr_node_type: p.last_expr_node_type
    }}
}}"""

import re

with open('src_nano/parser_mvp.nano', 'r') as f:
    content = f.read()

# Replace the three parser_store functions
content = re.sub(
    r'fn parser_store_number\(.*?\n\}',
    generate_parser_store_number(),
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'fn parser_store_identifier\(.*?\n\}',
    generate_parser_store_identifier(),
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'fn parser_store_binary_op\(.*?\n\}',
    generate_parser_store_binary_op(),
    content,
    flags=re.DOTALL
)

# Replace the three parser_with functions
content = re.sub(
    r'fn parser_with_position\(.*?\n\}',
    generate_parser_with_position(),
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'fn parser_with_error\(.*?\n\}',
    generate_parser_with_error(),
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'fn parser_with_calls_count\(.*?\n\}',
    generate_parser_with_calls_count(),
    content,
    flags=re.DOTALL
)

# Fix enum variants
content = content.replace('ParseNodeType.PNODE_STRUCT,', 'ParseNodeType.PNODE_STRUCT_DEF,')
content = content.replace('ParseNodeType.PNODE_UNION,', 'ParseNodeType.PNODE_UNION_DEF,')

with open('src_nano/parser_mvp.nano', 'w') as f:
    f.write(content)

print("✅ Fixed all 6 functions + 2 enum variants")
