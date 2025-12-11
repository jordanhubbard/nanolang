#!/usr/bin/env python3
"""Precise integration for 100% completion"""

with open('src_nano/parser_mvp.nano', 'r') as f:
    lines = f.readlines()

print("FINAL INTEGRATION - Precise line-by-line approach\n")

changes = []

# ============================================================================
# 1. STRUCT LITERAL INTEGRATION (around line 1626-1628)
# ============================================================================
print("1. Integrating struct literal detection...")

# Find the exact lines
for i, line in enumerate(lines):
    if 'if (== tok.token_type (token_identifier)) {' in line and i > 1620 and i < 1640:
        # Check if next lines match the simple identifier pattern
        if i+1 < len(lines) and 'let p1: Parser = (parser_store_identifier p tok.value tok.line tok.column)' in lines[i+1]:
            if i+2 < len(lines) and 'return (parser_advance p1)' in lines[i+2]:
                # Replace these 3 lines with struct literal check
                indent = ' ' * 24  # 24 spaces of indentation
                new_code = [
                    line,  # Keep the if line
                    f'{indent}    let name: string = tok.value\n',
                    f'{indent}    let p1: Parser = (parser_advance p)  /* consume identifier */\n',
                    f'{indent}    \n',
                    f'{indent}    /* Check for struct literal: Identifier{{...}} */\n',
                    f'{indent}    if (parser_is_at_end p1) {{\n',
                    f'{indent}        return (parser_store_identifier p tok.value tok.line tok.column)\n',
                    f'{indent}    }} else {{\n',
                    f'{indent}        let tok2: LexToken = (parser_current p1)\n',
                    f'{indent}        if (== tok2.token_type (token_lbrace)) {{\n',
                    f'{indent}            let p2: Parser = (parser_advance p1)  /* consume {{ */\n',
                    f'{indent}            return (parse_struct_literal p2 name tok.line tok.column)\n',
                    f'{indent}        }} else {{\n',
                    f'{indent}            /* Just an identifier */\n',
                    f'{indent}            return (parser_store_identifier p1 name tok.line tok.column)\n',
                    f'{indent}        }}\n',
                    f'{indent}}}\n',
                ]
                
                # Replace lines
                lines[i:i+3] = new_code
                changes.append(f"✅ Struct literals integrated at line {i+1}")
                print(f"   ✅ Replaced lines {i+1}-{i+3} with struct literal check")
                break

# ============================================================================
# 2. MATCH EXPRESSION INTEGRATION (add before number check ~line 1610)
# ============================================================================
print("\n2. Integrating match expression...")

# Find number check and add match before it
for i, line in enumerate(lines):
    if '/* Number literal */' in line and i > 1600 and i < 1650:
        # Insert match check before this
        indent = ' ' * 12
        match_code = [
            f'{indent}/* Match expression */\n',
            f'{indent}if (== tok.token_type (token_match)) {{\n',
            f'{indent}    return (parse_match p)\n',
            f'{indent}}} else {{\n',
            f'{indent}    ',
        ]
        
        # Insert before the number check comment
        lines[i:i] = match_code
        
        # Now we need to close the else at the end of all checks
        # Find the end of the parse_primary function
        # Look for the final closing braces
        for j in range(i+100, min(i+200, len(lines))):
            if lines[j].strip() == '}' and lines[j-1].strip().startswith('return (parser_with_error'):
                # Add closing brace for our else
                lines[j] = f'{indent}}}\n' + lines[j]
                changes.append(f"✅ Match expression integrated at line {i+1}")
                print(f"   ✅ Added match check before number literal at line {i+1}")
                break
        break

# ============================================================================
# 3. FIELD ACCESS POSTFIX (in parse_expression_recursive, complex!)
# ============================================================================
print("\n3. Integrating field access postfix operators...")

# Find parse_expression_recursive and the right spot
for i, line in enumerate(lines):
    if 'fn parse_expression_recursive(p: Parser, left_parsed: bool) -> Parser {' in line:
        # Find where we handle the primary expression
        # Look for "let p_primary: Parser = (parse_primary p)"
        for j in range(i, min(i+100, len(lines))):
            if 'let p_primary: Parser = (parse_primary p)' in lines[j]:
                # Find the next "if (parser_is_at_end p_primary)" block
                for k in range(j, min(j+50, len(lines))):
                    if 'if (parser_is_at_end p_primary)' in lines[k]:
                        # Find the else block
                        for m in range(k, min(k+20, len(lines))):
                            if '} else {' in lines[m]:
                                # Insert postfix handling right after this else
                                indent = ' ' * 12
                                postfix_code = [
                                    f'{indent}    /* Handle postfix operators: field access */\n',
                                    f'{indent}    let mut p_with_postfix: Parser = p_primary\n',
                                    f'{indent}    let mut keep_parsing: bool = true\n',
                                    f'{indent}    \n',
                                    f'{indent}    while (and keep_parsing (not (parser_is_at_end p_with_postfix))) {{\n',
                                    f'{indent}        let tok_check: LexToken = (parser_current p_with_postfix)\n',
                                    f'{indent}        \n',
                                    f'{indent}        if (== tok_check.token_type (token_dot)) {{\n',
                                    f'{indent}            let p_dot: Parser = (parser_advance p_with_postfix)\n',
                                    f'{indent}            \n',
                                    f'{indent}            if (parser_is_at_end p_dot) {{\n',
                                    f'{indent}                set keep_parsing false\n',
                                    f'{indent}            }} else {{\n',
                                    f'{indent}                let tok_after_dot: LexToken = (parser_current p_dot)\n',
                                    f'{indent}                \n',
                                    f'{indent}                if (== tok_after_dot.token_type (token_identifier)) {{\n',
                                    f'{indent}                    /* Field access */\n',
                                    f'{indent}                    let field_name: string = tok_after_dot.value\n',
                                    f'{indent}                    let p_field: Parser = (parser_advance p_dot)\n',
                                    f'{indent}                    let obj_id: int = p_with_postfix.last_expr_node_id\n',
                                    f'{indent}                    let obj_type: int = p_with_postfix.last_expr_node_type\n',
                                    f'{indent}                    set p_with_postfix (parser_store_field_access p_field obj_id obj_type field_name tok_check.line tok_check.column)\n',
                                    f'{indent}                }} else {{\n',
                                    f'{indent}                    set keep_parsing false\n',
                                    f'{indent}                }}\n',
                                    f'{indent}            }}\n',
                                    f'{indent}        }} else {{\n',
                                    f'{indent}            set keep_parsing false\n',
                                    f'{indent}        }}\n',
                                    f'{indent}    }}\n',
                                    f'{indent}    \n',
                                    f'{indent}    /* Now check for binary operators using p_with_postfix instead of p_primary */\n',
                                ]
                                
                                lines[m+1:m+1] = postfix_code
                                
                                # Now replace all p_primary with p_with_postfix after this point
                                for n in range(m+len(postfix_code), min(m+len(postfix_code)+50, len(lines))):
                                    lines[n] = lines[n].replace('p_primary', 'p_with_postfix')
                                
                                changes.append(f"✅ Field access postfix integrated at line {m+1}")
                                print(f"   ✅ Added field access postfix loop at line {m+1}")
                                break
                        break
                break
        break

# ============================================================================
# Write back
# ============================================================================

with open('src_nano/parser_mvp.nano', 'w') as f:
    f.writelines(lines)

print("\n" + "="*70)
print("INTEGRATION COMPLETE!")
print("="*70)
for change in changes:
    print(f"  {change}")
print(f"\nTotal integrations: {len(changes)}")
