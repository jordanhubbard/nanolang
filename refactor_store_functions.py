#!/usr/bin/env python3
"""
Quick script to generate refactored parser_store_* functions.
Each function should follow this pattern:
- Create node struct
- Get node_id from count
- Push to list
- Return parser_increment_count(...)
"""

functions = [
    ("parser_store_call", "ASTCall", "PNODE_CALL", "calls", 3),  # 3 = call expr type
    ("parser_store_let", "ASTLet", "PNODE_LET", "lets", -1),  # -1 = not an expression
    ("parser_store_set", "ASTSet", "PNODE_SET", "sets", -1),
    ("parser_store_if", "ASTIf", "PNODE_IF", "ifs", -1),
    ("parser_store_while", "ASTWhile", "PNODE_WHILE", "whiles", -1),
    ("parser_store_return", "ASTReturn", "PNODE_RETURN", "returns", -1),
    ("parser_store_block", "ASTBlock", "PNODE_BLOCK", "blocks", -1),
    ("parser_store_function", "ASTFunction", "PNODE_FUNCTION", "functions", -1),
    ("parser_store_struct", "ASTStruct", "PNODE_STRUCT_DEF", "structs", -1),
    ("parser_store_enum", "ASTEnum", "PNODE_ENUM", "enums", -1),
    ("parser_store_union", "ASTUnion", "PNODE_UNION_DEF", "unions", -1),
]

print("The pattern for each refactored function:")
print("1. Keep function signature")
print("2. Keep node creation")
print("3. Get node_id from count: let node_id: int = p.{list_name}_count")
print("4. Push to list: (list_{Type}_push p.{list_name} node)")
print("5. Return: return (parser_increment_count p ParseNodeType.{node_type} node_id {expr_type})")
print()

for func_name, ast_type, pnode_type, list_name, expr_type in functions:
    print(f"\n/* {func_name} refactoring: */")
    print(f"    let node_id: int = p.{list_name}_count")
    print(f"    (list_{ast_type}_push p.{list_name} node)")
    print(f"    return (parser_increment_count p ParseNodeType.{pnode_type} node_id {expr_type})")
