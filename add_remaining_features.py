#!/usr/bin/env python3
"""Add array literals, imports, field access to parser"""

with open('src_nano/parser_mvp.nano', 'r') as f:
    content = f.read()

# 1. Add array literal parsing to parse_primary
# Find the parse_primary function and add array literal case
primary_marker = "/* Parse primary expression: number, identifier, string, bool, or parenthesized */"
primary_idx = content.find(primary_marker)
if primary_idx == -1:
    print("ERROR: Cannot find parse_primary")
    exit(1)

# Find the first if statement checking tok.token_type
first_check = content.find("if (== tok.token_type (token_number))", primary_idx)
if first_check == -1:
    print("ERROR: Cannot find token_number check")
    exit(1)

# Insert array literal check before number check
array_literal_code = '''        /* Array literals */
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
            '''

content = content[:first_check] + array_literal_code + content[first_check:]

# 2. Add import/opaque/shadow to parse_definition
# Find parse_definition
def_marker = "fn parse_definition(p: Parser) -> Parser {"
def_idx = content.find(def_marker)
if def_idx == -1:
    print("ERROR: Cannot find parse_definition")
    exit(1)

# Find the union check
union_check = content.find("if (== tok.token_type (token_union))", def_idx)
if union_check == -1:
    print("ERROR: Cannot find union check")
    exit(1)

# Find the closing else after union
# Look for pattern: "return (parse_union_definition p)\n" followed by spaces and "} else {"
union_end = content.find("return (parse_union_definition p)", union_check)
if union_end == -1:
    print("ERROR: Cannot find union return")
    exit(1)

# Find the else block after union
else_after_union = content.find("} else {", union_end)
if else_after_union == -1:
    print("ERROR: Cannot find else after union")
    exit(1)

# Find the error return in that else
error_return = content.find("return (parser_with_error p true)", else_after_union)
if error_return == -1 or error_return > else_after_union + 200:
    print("ERROR: Cannot find error return")
    exit(1)

# Replace the error return with import/opaque/shadow checks
import_code = '''if (== tok.token_type (token_import)) {
                        return (parse_import p)
                    } else {
                        if (== tok.token_type (token_opaque)) {
                            return (parse_opaque_type p)
                        } else {
                            if (== tok.token_type (token_shadow)) {
                                return (parse_shadow p)
                            } else {
                                return (parser_with_error p true)
                            }
                        }
                    }'''

content = content[:error_return] + import_code + content[error_return+len("return (parser_with_error p true)"):]

# 3. Add the three new parsing functions before parse_definition
# Find parse_definition function start
def_start = content.find("fn parse_definition(p: Parser) -> Parser {")
if def_start == -1:
    print("ERROR: Cannot find parse_definition function")
    exit(1)

# Add the three functions before parse_definition
new_functions = '''/* Parse import statement: import "path" as name */
fn parse_import(p: Parser) -> Parser {
    let tok: LexToken = (parser_current p)
    let p1: Parser = (parser_advance p)  /* consume 'import' */
    
    if (parser_is_at_end p1) {
        return (parser_with_error p1 true)
    } else {
        let tok2: LexToken = (parser_current p1)
        if (== tok2.token_type (token_string)) {
            let module_path: string = tok2.value
            let p2: Parser = (parser_advance p1)
            
            if (parser_is_at_end p2) {
                return (parser_with_error p2 true)
            } else {
                let tok3: LexToken = (parser_current p2)
                if (== tok3.token_type (token_as)) {
                    let p3: Parser = (parser_advance p2)
                    
                    if (parser_is_at_end p3) {
                        return (parser_with_error p3 true)
                    } else {
                        let tok4: LexToken = (parser_current p3)
                        if (== tok4.token_type (token_identifier)) {
                            let module_name: string = tok4.value
                            let p4: Parser = (parser_advance p3)
                            return (parser_store_import p4 module_path module_name tok.line tok.column)
                        } else {
                            return (parser_with_error p3 true)
                        }
                    }
                } else {
                    return (parser_with_error p2 true)
                }
            }
        } else {
            return (parser_with_error p1 true)
        }
    }
}

/* Parse opaque type: opaque type TypeName */
fn parse_opaque_type(p: Parser) -> Parser {
    let tok: LexToken = (parser_current p)
    let p1: Parser = (parser_advance p)  /* consume 'opaque' */
    let p2: Parser = (parser_advance p1)  /* skip 'type' keyword */
    
    if (parser_is_at_end p2) {
        return (parser_with_error p2 true)
    } else {
        let tok3: LexToken = (parser_current p2)
        if (== tok3.token_type (token_identifier)) {
            let type_name: string = tok3.value
            let p3: Parser = (parser_advance p2)
            return (parser_store_opaque_type p3 type_name tok.line tok.column)
        } else {
            return (parser_with_error p2 true)
        }
    }
}

/* Parse shadow test: shadow target_name { body } */
fn parse_shadow(p: Parser) -> Parser {
    let tok: LexToken = (parser_current p)
    let p1: Parser = (parser_advance p)  /* consume 'shadow' */
    
    if (parser_is_at_end p1) {
        return (parser_with_error p1 true)
    } else {
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
    }
}

'''

content = content[:def_start] + new_functions + content[def_start:]

with open('src_nano/parser_mvp.nano', 'w') as f:
    f.write(content)

print("✅ Added array literals")
print("✅ Added import/opaque/shadow parsing")
print("✅ Added 3 new parsing functions")
