#!/usr/bin/env python3
"""
Script to refactor parser_store_* functions to include all 30 list fields.
Reads parser_mvp.nano, finds each parser_store_* function, and regenerates
with all fields included.
"""

import re
import sys

# All list field names in order
LIST_FIELDS = [
    "numbers", "floats", "strings", "bools", "identifiers", "binary_ops", 
    "calls", "array_literals", "lets", "sets", "ifs", "whiles", "fors",
    "returns", "blocks", "prints", "asserts", "functions", "shadows",
    "structs", "struct_literals", "field_accesses", "enums", "unions",
    "union_constructs", "matches", "imports", "opaque_types", 
    "tuple_literals", "tuple_indices"
]

# All count field names in order (same as LIST_FIELDS + "_count")
COUNT_FIELDS = [f"{field}_count" for field in LIST_FIELDS]

# Map of function name -> (node_type, list_name, expr_type)
# expr_type: 0=number, 1=identifier, 2=binary_op, 3=call, -1=statement
STORE_FUNCTIONS = {
    "parser_store_number": ("ParseNodeType.PNODE_NUMBER", "numbers", 0),
    "parser_store_identifier": ("ParseNodeType.PNODE_IDENTIFIER", "identifiers", 1),
    "parser_store_binary_op": ("ParseNodeType.PNODE_BINARY_OP", "binary_ops", 2),
    "parser_store_call": ("ParseNodeType.PNODE_CALL", "calls", 3),
    "parser_store_let": ("ParseNodeType.PNODE_LET", "lets", -1),
    "parser_store_set": ("ParseNodeType.PNODE_SET", "sets", -1),
    "parser_store_if": ("ParseNodeType.PNODE_IF", "ifs", -1),
    "parser_store_while": ("ParseNodeType.PNODE_WHILE", "whiles", -1),
    "parser_store_return": ("ParseNodeType.PNODE_RETURN", "returns", -1),
    "parser_store_block": ("ParseNodeType.PNODE_BLOCK", "blocks", -1),
    "parser_store_function": ("ParseNodeType.PNODE_FUNCTION", "functions", -1),
    "parser_store_struct": ("ParseNodeType.PNODE_STRUCT_DEF", "structs", -1),
    "parser_store_enum": ("ParseNodeType.PNODE_ENUM", "enums", -1),
    "parser_store_union": ("ParseNodeType.PNODE_UNION_DEF", "unions", -1),
}

def find_function_body(lines, start_idx):
    """Find the complete function body from start_idx to matching }"""
    depth = 0
    func_lines = []
    in_func = False
    
    for i in range(start_idx, len(lines)):
        line = lines[i]
        func_lines.append(line)
        
        # Count braces
        for char in line:
            if char == '{':
                depth += 1
                in_func = True
            elif char == '}':
                depth -= 1
                if in_func and depth == 0:
                    return func_lines, i
    
    return func_lines, len(lines) - 1

def extract_node_creation(func_body):
    """Extract the node creation part (let node: Type = Type { ... })"""
    node_lines = []
    in_node = False
    depth = 0
    
    for line in func_body:
        if 'let node:' in line or in_node:
            node_lines.append(line)
            in_node = True
            
            for char in line:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return node_lines
    
    return node_lines

def generate_refactored_function(func_name, signature, node_creation_lines, node_type, list_name, expr_type):
    """Generate a refactored function with all fields"""
    
    # Map AST type names to list functions
    ast_type_map = {
        "numbers": "ASTNumber",
        "floats": "ASTFloat",
        "strings": "ASTString",
        "bools": "ASTBool",
        "identifiers": "ASTIdentifier",
        "binary_ops": "ASTBinaryOp",
        "calls": "ASTCall",
        "array_literals": "ASTArrayLiteral",
        "lets": "ASTLet",
        "sets": "ASTSet",
        "ifs": "ASTIf",
        "whiles": "ASTWhile",
        "fors": "ASTFor",
        "returns": "ASTReturn",
        "blocks": "ASTBlock",
        "prints": "ASTPrint",
        "asserts": "ASTAssert",
        "functions": "ASTFunction",
        "shadows": "ASTShadow",
        "structs": "ASTStruct",
        "struct_literals": "ASTStructLiteral",
        "field_accesses": "ASTFieldAccess",
        "enums": "ASTEnum",
        "unions": "ASTUnion",
        "union_constructs": "ASTUnionConstruct",
        "matches": "ASTMatch",
        "imports": "ASTImport",
        "opaque_types": "ASTOpaqueType",
        "tuple_literals": "ASTTupleLiteral",
        "tuple_indices": "ASTTupleIndex",
    }
    
    ast_type = ast_type_map[list_name]
    
    result = []
    result.append(f"\n/* Helper: Store {list_name[:-1] if list_name.endswith('s') else list_name} node */")
    result.append(signature)
    
    # Add node creation
    for line in node_creation_lines:
        result.append(line.rstrip())
    
    # Add simplified logic
    result.append(f"    let node_id: int = p.{list_name}_count")
    result.append(f"    (list_{ast_type}_push p.{list_name} node)")
    result.append(f"    ")
    result.append(f"    return Parser {{")
    result.append(f"        tokens: p.tokens,")
    result.append(f"        position: p.position,")
    result.append(f"        token_count: p.token_count,")
    result.append(f"        has_error: p.has_error,")
    
    # Add all list fields
    for field in LIST_FIELDS:
        result.append(f"        {field}: p.{field},")
    
    # Add all count fields with increment logic
    for field in COUNT_FIELDS:
        base_field = field.replace("_count", "")
        if base_field == list_name:
            result.append(f"        {field}: (+ p.{field} 1),")
        else:
            result.append(f"        {field}: p.{field},")
    
    result.append(f"        next_node_id: (+ p.next_node_id 1),")
    result.append(f"        last_expr_node_id: node_id,")
    result.append(f"        last_expr_node_type: {expr_type}")
    result.append(f"    }}")
    result.append(f"}}")
    
    return "\n".join(result)

def main():
    input_file = "src_nano/parser_mvp.nano"
    output_file = "src_nano/parser_mvp_refactored.nano"
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Find and replace each parser_store function
    output_lines = []
    i = 0
    refactored_count = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a parser_store function we need to refactor
        matched_func = None
        for func_name in STORE_FUNCTIONS.keys():
            if f"fn {func_name}(" in line:
                matched_func = func_name
                break
        
        if matched_func:
            print(f"Refactoring {matched_func}...")
            
            # Extract function signature
            signature = line.rstrip()
            
            # Find function body
            func_body, end_idx = find_function_body(lines, i + 1)
            
            # Extract node creation
            node_creation = extract_node_creation(func_body)
            
            # Generate refactored version
            node_type, list_name, expr_type = STORE_FUNCTIONS[matched_func]
            refactored = generate_refactored_function(
                matched_func, signature, node_creation, 
                node_type, list_name, expr_type
            )
            
            output_lines.append(refactored + "\n")
            
            # Skip the shadow test that follows
            i = end_idx + 1
            while i < len(lines) and not lines[i].strip().startswith("shadow"):
                i += 1
            if i < len(lines) and lines[i].strip().startswith("shadow"):
                # Skip shadow block
                while i < len(lines):
                    output_lines.append(lines[i])
                    if '}' in lines[i]:
                        i += 1
                        break
                    i += 1
            
            refactored_count += 1
        else:
            output_lines.append(line)
            i += 1
    
    print(f"\nWriting {output_file}...")
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    
    print(f"✅ Refactored {refactored_count} functions")
    print(f"Output written to: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Review the changes: diff src_nano/parser_mvp.nano src_nano/parser_mvp_refactored.nano | less")
    print(f"2. If looks good: mv src_nano/parser_mvp_refactored.nano src_nano/parser_mvp.nano")
    print(f"3. Test compilation: ./bin/nanoc src_nano/parser_mvp.nano")

if __name__ == "__main__":
    main()
