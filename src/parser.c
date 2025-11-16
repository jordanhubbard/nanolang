#include "nanolang.h"

/* Parser state */
typedef struct {
    Token *tokens;
    int count;
    int pos;
    int recursion_depth;  /* Track recursion depth to prevent stack overflow */
} Parser;

/* Maximum recursion depth to prevent stack overflow */
#define MAX_RECURSION_DEPTH 1000

/* Forward declarations */
static Type parse_type_with_element(Parser *p, Type *element_type_out, char **type_param_name_out, FunctionSignature **fn_sig_out);

/* Helper functions */
static Token *current_token(Parser *p) {
    if (!p || !p->tokens || p->count <= 0) {
        return NULL;
    }
    /* Ensure pos is within bounds */
    int safe_pos = p->pos;
    if (safe_pos < 0) {
        safe_pos = 0;
    }
    if (safe_pos >= p->count) {
        safe_pos = p->count - 1;
    }
    return &p->tokens[safe_pos];
}

static Token *peek_token(Parser *p, int offset) {
    if (!p || !p->tokens || p->count <= 0) {
        return NULL;
    }
    int pos = p->pos + offset;
    if (pos < 0) {
        pos = 0;
    }
    if (pos < p->count) {
        return &p->tokens[pos];
    }
    /* Return last token (EOF) if out of bounds */
    return &p->tokens[p->count - 1];
}

static void advance(Parser *p) {
    if (p->pos < p->count - 1) {
        p->pos++;
    }
}

static bool match(Parser *p, TokenType type) {
    Token *tok = current_token(p);
    if (!tok) {
        return false;
    }
    return tok->type == type;
}

static bool expect(Parser *p, TokenType type, const char *msg) {
    if (!match(p, type)) {
        Token *tok = current_token(p);
        if (!tok || !p || p->count == 0) {
            fprintf(stderr, "Error: Parser state invalid\n");
            return false;
        }
        const char *msg_safe = msg ? msg : "(null message)";
        const char *token_name = token_type_name(tok->type);
        const char *token_name_safe = token_name ? token_name : "UNKNOWN";
        fprintf(stderr, "Error at line %d, column %d: %s (got %s)\n",
                tok->line, tok->column, msg_safe, token_name_safe);
        return false;
    }
    advance(p);
    return true;
}

/* Forward declarations */
static ASTNode *parse_statement(Parser *p);
static ASTNode *parse_expression(Parser *p);
static ASTNode *parse_block(Parser *p);
static ASTNode *parse_struct_def(Parser *p);
static ASTNode *parse_enum_def(Parser *p);
static ASTNode *parse_union_def(Parser *p);
static ASTNode *parse_match_expr(Parser *p);

/* Create AST nodes */
static ASTNode *create_node(ASTNodeType type, int line, int column) {
    ASTNode *node = calloc(1, sizeof(ASTNode));  /* Use calloc to zero-initialize */
    node->type = type;
    node->line = line;
    node->column = column;
    return node;
}

/* Forward declaration for function signature parsing */
static FunctionSignature *parse_function_signature(Parser *p);

/* Parse function signature: fn(int, string) -> bool */
static FunctionSignature *parse_function_signature(Parser *p) {
    Token *tok = current_token(p);
    if (!tok) {
        fprintf(stderr, "Error: Parser reached invalid state (NULL token) in function signature\n");
        return NULL;
    }
    
    /* Expect 'fn' */
    if (tok->type != TOKEN_FN) {
        fprintf(stderr, "Error at line %d, column %d: Expected 'fn' for function type\n",
                tok->line, tok->column);
        return NULL;
    }
    advance(p);  /* consume 'fn' */
    
    /* Expect '(' */
    tok = current_token(p);
    if (!tok) {
        fprintf(stderr, "Error: Parser reached invalid state (NULL token) after 'fn'\n");
        return NULL;
    }
    if (tok->type != TOKEN_LPAREN) {
        fprintf(stderr, "Error at line %d, column %d: Expected '(' after 'fn'\n",
                tok->line, tok->column);
        return NULL;
    }
    advance(p);  /* consume '(' */
    
    /* Allocate signature */
    FunctionSignature *sig = malloc(sizeof(FunctionSignature));
    sig->param_count = 0;
    sig->param_types = NULL;
    sig->param_struct_names = NULL;
    sig->return_type = TYPE_UNKNOWN;
    sig->return_struct_name = NULL;
    
    /* Parse parameter types */
    tok = current_token(p);
    if (!tok) {
        fprintf(stderr, "Error: Parser reached invalid state (NULL token) before parameter types\n");
        free_function_signature(sig);
        return NULL;
    }
    
    if (tok->type != TOKEN_RPAREN) {
        /* Parse comma-separated types */
        while (1) {
            char *struct_name = NULL;
            FunctionSignature *nested_fn_sig = NULL;
            Type param_type = parse_type_with_element(p, NULL, &struct_name, &nested_fn_sig);
            
            if (param_type == TYPE_UNKNOWN) {
                /* Error already reported */
                free_function_signature(sig);
                return NULL;
            }
            
            /* Grow arrays */
            sig->param_count++;
            sig->param_types = realloc(sig->param_types, 
                                      sizeof(Type) * sig->param_count);
            sig->param_struct_names = realloc(sig->param_struct_names,
                                             sizeof(char*) * sig->param_count);
            
            sig->param_types[sig->param_count - 1] = param_type;
            sig->param_struct_names[sig->param_count - 1] = struct_name;  /* May be NULL */
            
            /* TODO: Handle nested function signatures in function parameters */
            /* For now, we don't support fn(fn(int)->int)->int */
            if (nested_fn_sig) {
                fprintf(stderr, "Error: Nested function types not yet supported\n");
                free_function_signature(nested_fn_sig);
                free_function_signature(sig);
                return NULL;
            }
            
            tok = current_token(p);
            if (!tok) {
                fprintf(stderr, "Error: Parser reached invalid state (NULL token) in parameter list\n");
                free_function_signature(sig);
                return NULL;
            }
            
            if (tok->type == TOKEN_COMMA) {
                advance(p);  /* consume ',' */
            } else {
                break;
            }
        }
    }
    
    /* Expect ')' */
    tok = current_token(p);
    if (!tok) {
        fprintf(stderr, "Error: Parser reached invalid state (NULL token) before ')'\n");
        free_function_signature(sig);
        return NULL;
    }
    if (tok->type != TOKEN_RPAREN) {
        fprintf(stderr, "Error at line %d, column %d: Expected ')' in function type\n",
                tok->line, tok->column);
        free_function_signature(sig);
        return NULL;
    }
    advance(p);  /* consume ')' */
    
    /* Expect '->' */
    tok = current_token(p);
    if (!tok) {
        fprintf(stderr, "Error: Parser reached invalid state (NULL token) before '->'\n");
        free_function_signature(sig);
        return NULL;
    }
    if (tok->type != TOKEN_ARROW) {
        fprintf(stderr, "Error at line %d, column %d: Expected '->' in function type\n",
                tok->line, tok->column);
        free_function_signature(sig);
        return NULL;
    }
    advance(p);  /* consume '->' */
    
    /* Parse return type */
    char *return_struct_name = NULL;
    FunctionSignature *return_fn_sig = NULL;
    sig->return_type = parse_type_with_element(p, NULL, &return_struct_name, &return_fn_sig);
    sig->return_struct_name = return_struct_name;  /* May be NULL */
    
    /* TODO: Handle function types as return types */
    /* For now, we don't support fn() -> fn()->int */
    if (return_fn_sig) {
        fprintf(stderr, "Error: Function types as return values not yet supported in function type signatures\n");
        free_function_signature(return_fn_sig);
        free_function_signature(sig);
        return NULL;
    }
    
    if (sig->return_type == TYPE_UNKNOWN) {
        /* Error already reported */
        free_function_signature(sig);
        return NULL;
    }
    
    return sig;
}

/* Parse type annotation */
static Type parse_type(Parser *p) {
    return parse_type_with_element(p, NULL, NULL, NULL);
}

/* Parse type annotation with optional element_type output (for arrays) and type_param_name for generics */
static Type parse_type_with_element(Parser *p, Type *element_type_out, char **type_param_name_out, FunctionSignature **fn_sig_out) {
    Type type = TYPE_UNKNOWN;
    Token *tok = current_token(p);

    switch (tok->type) {
        case TOKEN_TYPE_INT: type = TYPE_INT; break;
        case TOKEN_TYPE_FLOAT: type = TYPE_FLOAT; break;
        case TOKEN_TYPE_BOOL: type = TYPE_BOOL; break;
        case TOKEN_TYPE_STRING: type = TYPE_STRING; break;
        case TOKEN_TYPE_VOID: type = TYPE_VOID; break;
        
        case TOKEN_FN: {
            /* Function type: fn(type1, type2) -> return_type */
            FunctionSignature *sig = parse_function_signature(p);
            if (sig) {
                if (fn_sig_out) {
                    *fn_sig_out = sig;
                }
                return TYPE_FUNCTION;
            }
            return TYPE_UNKNOWN;
        }
        case TOKEN_IDENTIFIER:
            /* Check for list types (legacy names) */
            if (strcmp(tok->value, "list_int") == 0) {
                type = TYPE_LIST_INT;
                advance(p);
                return type;
            } else if (strcmp(tok->value, "list_string") == 0) {
                type = TYPE_LIST_STRING;
                advance(p);
                return type;
            } else if (strcmp(tok->value, "list_token") == 0) {
                type = TYPE_LIST_TOKEN;
                advance(p);
                return type;
            }
            
            /* Check for generic type syntax: List<T> */
            if (strcmp(tok->value, "List") == 0) {
                advance(p);  /* consume 'List' */
                if (current_token(p)->type == TOKEN_LT) {
                    advance(p);  /* consume '<' */
                    
                    /* Parse type parameter */
                    Token *type_param_tok = current_token(p);
                    if (type_param_tok->type == TOKEN_TYPE_INT) {
                        type = TYPE_LIST_INT;
                        advance(p);
                    } else if (type_param_tok->type == TOKEN_TYPE_STRING) {
                        type = TYPE_LIST_STRING;
                        advance(p);
                    } else if (type_param_tok->type == TOKEN_IDENTIFIER) {
                        /* Handle Token specially for backwards compatibility */
                        if (strcmp(type_param_tok->value, "Token") == 0) {
                            type = TYPE_LIST_TOKEN;
                            advance(p);
                        } else {
                            /* Generic list with user-defined type: List<Point>, List<Player>, etc. */
                            type = TYPE_LIST_GENERIC;
                            /* Store type parameter name for later use */
                            if (type_param_name_out) {
                                *type_param_name_out = strdup(type_param_tok->value);
                            }
                            advance(p);
                        }
                    } else {
                        fprintf(stderr, "Error at line %d, column %d: Expected type parameter after 'List<'\n",
                                type_param_tok->line, type_param_tok->column);
                        return TYPE_UNKNOWN;
                    }
                    
                    if (current_token(p)->type != TOKEN_GT) {
                        fprintf(stderr, "Error at line %d, column %d: Expected '>' after List type parameter\n",
                                current_token(p)->line, current_token(p)->column);
                        return TYPE_UNKNOWN;
                    }
                    advance(p);  /* consume '>' */
                    return type;
                }
            }
            
            /* Could be a struct type */
            type = TYPE_STRUCT;
            advance(p);
            return type;
        case TOKEN_ARRAY:
            /* Parse array<element_type> */
            advance(p);  /* consume 'array' */
            if (current_token(p)->type != TOKEN_LT) {
                fprintf(stderr, "Error at line %d, column %d: Expected '<' after 'array'\n", 
                        current_token(p)->line, current_token(p)->column);
                return TYPE_UNKNOWN;
            }
            advance(p);  /* consume '<' */
            
            /* Parse element type (no need for type_param_name in arrays for now) */
            Type element_type = parse_type_with_element(p, NULL, NULL, NULL);
            if (element_type == TYPE_UNKNOWN) {
                return TYPE_UNKNOWN;
            }
            
            if (current_token(p)->type != TOKEN_GT) {
                fprintf(stderr, "Error at line %d, column %d: Expected '>' after array element type\n", 
                        current_token(p)->line, current_token(p)->column);
                return TYPE_UNKNOWN;
            }
            advance(p);  /* consume '>' */
            
            /* Store element_type if output parameter provided */
            if (element_type_out) {
                *element_type_out = element_type;
            }
            type = TYPE_ARRAY;
            return type;
        default:
            fprintf(stderr, "Error at line %d, column %d: Expected type annotation\n", tok->line, tok->column);
            return TYPE_UNKNOWN;
    }

    advance(p);
    return type;
}

/* Parse function parameters */
static bool parse_parameters(Parser *p, Parameter **params, int *param_count) {
    int capacity = 4;
    int count = 0;
    Parameter *param_list = malloc(sizeof(Parameter) * capacity);

    if (!match(p, TOKEN_RPAREN)) {
        do {
            if (count >= capacity) {
                capacity *= 2;
                param_list = realloc(param_list, sizeof(Parameter) * capacity);
            }

            /* Parameter name */
            if (!match(p, TOKEN_IDENTIFIER)) {
                fprintf(stderr, "Error at line %d, column %d: Expected parameter name\n", current_token(p)->line);
                free(param_list);
                return false;
            }
            param_list[count].name = strdup(current_token(p)->value);
            advance(p);

            /* Colon */
            if (!expect(p, TOKEN_COLON, "Expected ':' after parameter name")) {
                free(param_list);
                return false;
            }

            /* Type - check if it's a struct type (identifier) */
            Token *type_token = current_token(p);
            char *struct_name = NULL;
            if (type_token->type == TOKEN_IDENTIFIER) {
                struct_name = strdup(type_token->value);
            }
            
            /* Parse type with element_type support for arrays and generics */
            Type element_type = TYPE_UNKNOWN;
            char *type_param_name = NULL;
            FunctionSignature *fn_sig = NULL;
            param_list[count].type = parse_type_with_element(p, &element_type, &type_param_name, &fn_sig);
            param_list[count].struct_type_name = type_param_name;  /* Store generic type param here */
            param_list[count].element_type = element_type;
            param_list[count].fn_sig = fn_sig;  /* Store function signature if it's a function type */
            
            /* If it's a struct type, save the struct name */
            if (param_list[count].type == TYPE_STRUCT && struct_name) {
                param_list[count].struct_type_name = struct_name;
            } else if (struct_name) {
                free(struct_name);
            }
            
            count++;

            if (match(p, TOKEN_COMMA)) {
                advance(p);
            } else {
                break;
            }
        } while (true);
    }

    *params = param_list;
    *param_count = count;
    return true;
}

/* Parse prefix operation: (op arg1 arg2 ...) */
static ASTNode *parse_prefix_op(Parser *p) {
    Token *tok = current_token(p);
    int line = tok->line;
    int column = tok->column;

    if (!expect(p, TOKEN_LPAREN, "Expected '('")) {
        return NULL;
    }

    /* Get operator */
    tok = current_token(p);
    TokenType op = tok->type;

    /* Check if it's a valid operator or function call */
    bool is_operator = (op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR ||
                        op == TOKEN_SLASH || op == TOKEN_PERCENT ||
                        op == TOKEN_EQ || op == TOKEN_NE ||
                        op == TOKEN_LT || op == TOKEN_LE ||
                        op == TOKEN_GT || op == TOKEN_GE ||
                        op == TOKEN_AND || op == TOKEN_OR || op == TOKEN_NOT);

    if (is_operator) {
        advance(p);

        /* Parse arguments */
        int capacity = 4;
        int count = 0;
        ASTNode **args = malloc(sizeof(ASTNode*) * capacity);

        while (!match(p, TOKEN_RPAREN) && !match(p, TOKEN_EOF)) {
            if (count >= capacity) {
                capacity *= 2;
                args = realloc(args, sizeof(ASTNode*) * capacity);
            }
            args[count++] = parse_expression(p);
        }

        if (!expect(p, TOKEN_RPAREN, "Expected ')' after prefix operation")) {
            free(args);
            return NULL;
        }

        ASTNode *node = create_node(AST_PREFIX_OP, line, column);
        node->as.prefix_op.op = op;
        node->as.prefix_op.args = args;
        node->as.prefix_op.arg_count = count;
        return node;
    } else if (match(p, TOKEN_IDENTIFIER) || match(p, TOKEN_RANGE) || match(p, TOKEN_PRINT)) {
        /* Function call - allow print as function name */
        char *func_name;
        if (tok->type == TOKEN_PRINT) {
            func_name = strdup("print");
            advance(p);
        } else {
            func_name = strdup(tok->value ? tok->value : "range");
            advance(p);
        }

        int capacity = 4;
        int count = 0;
        ASTNode **args = malloc(sizeof(ASTNode*) * capacity);

        while (!match(p, TOKEN_RPAREN) && !match(p, TOKEN_EOF)) {
            if (count >= capacity) {
                capacity *= 2;
                args = realloc(args, sizeof(ASTNode*) * capacity);
            }
            args[count++] = parse_expression(p);
        }

        if (!expect(p, TOKEN_RPAREN, "Expected ')' after function call")) {
            free(func_name);
            free(args);
            return NULL;
        }

        ASTNode *node = create_node(AST_CALL, line, column);
        node->as.call.name = func_name;
        node->as.call.args = args;
        node->as.call.arg_count = count;
        return node;
    } else {
        fprintf(stderr, "Error at line %d, column %d: Invalid prefix operation\n", line);
        return NULL;
    }
}

/* Parse primary expression */
static ASTNode *parse_primary(Parser *p) {
    Token *tok = current_token(p);
    if (!tok) {
        fprintf(stderr, "Error: Parser reached invalid state (NULL token)\n");
        return NULL;
    }
    ASTNode *node;

    switch (tok->type) {
        case TOKEN_NUMBER:
            node = create_node(AST_NUMBER, tok->line, tok->column);
            node->as.number = atoll(tok->value);
            advance(p);
            return node;

        case TOKEN_FLOAT:
            node = create_node(AST_FLOAT, tok->line, tok->column);
            node->as.float_val = atof(tok->value);
            advance(p);
            return node;

        case TOKEN_STRING:
            node = create_node(AST_STRING, tok->line, tok->column);
            node->as.string_val = strdup(tok->value);
            advance(p);
            return node;

        case TOKEN_TRUE:
            node = create_node(AST_BOOL, tok->line, tok->column);
            node->as.bool_val = true;
            advance(p);
            return node;

        case TOKEN_FALSE:
            node = create_node(AST_BOOL, tok->line, tok->column);
            node->as.bool_val = false;
            advance(p);
            return node;

        case TOKEN_LBRACKET: {
            /* Array literal: [1, 2, 3] */
            int line = tok->line;
            int column = tok->column;
            advance(p);  /* consume '[' */
            
            int capacity = 4;
            int count = 0;
            ASTNode **elements = malloc(sizeof(ASTNode*) * capacity);
            
            /* Parse array elements */
            while (!match(p, TOKEN_RBRACKET) && !match(p, TOKEN_EOF)) {
                if (count >= capacity) {
                    capacity *= 2;
                    elements = realloc(elements, sizeof(ASTNode*) * capacity);
                }
                elements[count++] = parse_expression(p);
                
                /* Check for comma or end of array */
                if (match(p, TOKEN_COMMA)) {
                    advance(p);
                } else if (!match(p, TOKEN_RBRACKET)) {
                    fprintf(stderr, "Error at line %d, column %d: Expected ',' or ']' in array literal\n",
                            current_token(p)->line, current_token(p)->column);
                    free(elements);
                    return NULL;
                }
            }
            
            if (!expect(p, TOKEN_RBRACKET, "Expected ']' at end of array literal")) {
                free(elements);
                return NULL;
            }
            
            node = create_node(AST_ARRAY_LITERAL, line, column);
            node->as.array_literal.elements = elements;
            node->as.array_literal.element_count = count;
            node->as.array_literal.element_type = TYPE_UNKNOWN;  /* Will be inferred by type checker */
            return node;
        }

        case TOKEN_IDENTIFIER: {
            /* Check if this is a struct literal: StructName { ... } */
            /* Only parse as struct literal if identifier starts with uppercase (type convention) */
            Token *next = peek_token(p, 1);
            bool looks_like_struct = tok->value && tok->value[0] >= 'A' && tok->value[0] <= 'Z';
            if (next && next->type == TOKEN_LBRACE && looks_like_struct) {
                /* Parse struct literal */
                int line = tok->line;
                int column = tok->column;
                char *struct_name = strdup(tok->value);
                advance(p);  /* consume struct name */
                advance(p);  /* consume '{' */
                
                int capacity = 4;
                int count = 0;
                char **field_names = malloc(sizeof(char*) * capacity);
                ASTNode **field_values = malloc(sizeof(ASTNode*) * capacity);
                
                while (!match(p, TOKEN_RBRACE) && !match(p, TOKEN_EOF)) {
                    if (count >= capacity) {
                        capacity *= 2;
                        field_names = realloc(field_names, sizeof(char*) * capacity);
                        field_values = realloc(field_values, sizeof(ASTNode*) * capacity);
                    }
                    
                    /* Parse field name */
                    if (!match(p, TOKEN_IDENTIFIER)) {
                        fprintf(stderr, "Error at line %d, column %d: Expected field name in struct literal\n",
                                current_token(p)->line, current_token(p)->column);
                        break;
                    }
                    field_names[count] = strdup(current_token(p)->value);
                    advance(p);
                    
                    /* Expect colon */
                    if (!expect(p, TOKEN_COLON, "Expected ':' after field name")) {
                        break;
                    }
                    
                    /* Parse field value */
                    field_values[count] = parse_expression(p);
                    count++;
                    
                    /* Optional comma */
                    if (match(p, TOKEN_COMMA)) {
                        advance(p);
                    }
                }
                
                if (!expect(p, TOKEN_RBRACE, "Expected '}' at end of struct literal")) {
                    free(struct_name);
                    for (int i = 0; i < count; i++) {
                        free(field_names[i]);
                        free_ast(field_values[i]);
                    }
                    free(field_names);
                    free(field_values);
                    return NULL;
                }
                
                node = create_node(AST_STRUCT_LITERAL, line, column);
                node->as.struct_literal.struct_name = struct_name;
                node->as.struct_literal.field_names = field_names;
                node->as.struct_literal.field_values = field_values;
                node->as.struct_literal.field_count = count;
                return node;
            } else {
                /* Regular identifier */
                node = create_node(AST_IDENTIFIER, tok->line, tok->column);
                node->as.identifier = strdup(tok->value);
                advance(p);
                return node;
            }
        }

        case TOKEN_LPAREN:
            return parse_prefix_op(p);

        case TOKEN_IF:
            return parse_expression(p);

        default: {
            /* Re-read token to ensure it's still valid (might have been corrupted during recursion) */
            Token *current_tok = current_token(p);
            if (!current_tok) {
                fprintf(stderr, "Error: Parser reached invalid state (NULL token) in parse_primary default case\n");
                return NULL;
            }
            /* Safety check: validate token values are reasonable */
            if (current_tok->line < 0 || current_tok->line > 1000000 || 
                current_tok->column < 0 || current_tok->column > 1000000) {
                fprintf(stderr, "Error: Invalid token state in parse_primary (possible memory corruption at line %d, column %d)\n",
                        current_tok->line, current_tok->column);
                return NULL;
            }
            const char *type_name = token_type_name(current_tok->type);
            fprintf(stderr, "Error at line %d, column %d: Unexpected token in expression: %s\n",
                    current_tok->line, current_tok->column, type_name ? type_name : "UNKNOWN");
            return NULL;
        }
    }
}

/* Parse if expression */
static ASTNode *parse_if_expression(Parser *p) {
    int line = current_token(p)->line;
    int column = current_token(p)->column;

    if (!expect(p, TOKEN_IF, "Expected 'if'")) {
        return NULL;
    }

    ASTNode *condition = parse_expression(p);
    if (!condition) return NULL;

    ASTNode *then_branch = parse_block(p);
    if (!then_branch) return NULL;

    if (!expect(p, TOKEN_ELSE, "Expected 'else' after 'if' block")) {
        return NULL;
    }

    ASTNode *else_branch = parse_block(p);
    if (!else_branch) return NULL;

    ASTNode *node = create_node(AST_IF, line, column);
    node->as.if_stmt.condition = condition;
    node->as.if_stmt.then_branch = then_branch;
    node->as.if_stmt.else_branch = else_branch;
    return node;
}

/* Parse expression */
static ASTNode *parse_expression(Parser *p) {
    /* Recursion depth guard */
    p->recursion_depth++;
    if (p->recursion_depth > MAX_RECURSION_DEPTH) {
        Token *tok = current_token(p);
        fprintf(stderr, "Error at line %d, column %d: Expression recursion depth exceeded maximum (%d). Possible infinite recursion or extremely nested expression.\n",
                tok ? tok->line : 0, tok ? tok->column : 0, MAX_RECURSION_DEPTH);
        p->recursion_depth--;
        return NULL;
    }
    
    if (match(p, TOKEN_IF)) {
        ASTNode *result = parse_if_expression(p);
        p->recursion_depth--;
        return result;
    }
    
    if (match(p, TOKEN_MATCH)) {
        ASTNode *result = parse_match_expr(p);
        p->recursion_depth--;
        return result;
    }
    
    /* Parse primary expression */
    ASTNode *expr = parse_primary(p);
    if (!expr) {
        p->recursion_depth--;
        return NULL;
    }
    
    /* Handle field access or union construction:
     * - obj.field -> field access
     * - UnionName.Variant { ... } -> union construction
     */
    while (match(p, TOKEN_DOT)) {
        int line = current_token(p)->line;
        int column = current_token(p)->column;
        advance(p);  /* consume '.' */
        
        if (!match(p, TOKEN_IDENTIFIER)) {
            fprintf(stderr, "Error at line %d, column %d: Expected field or variant name after '.'\n",
                    current_token(p)->line, current_token(p)->column);
            return expr;
        }
        
        Token *field_tok = current_token(p);
        if (!field_tok || !field_tok->value) {
            fprintf(stderr, "Error at line %d, column %d: Invalid field/variant token\n",
                    field_tok ? field_tok->line : 0, field_tok ? field_tok->column : 0);
            return expr;
        }
        char *field_or_variant = strdup(field_tok->value);
        if (!field_or_variant) {
            fprintf(stderr, "Error: Failed to allocate memory for field/variant name\n");
            return expr;
        }
        advance(p);
        
        /* Check if this is union construction: UnionName.Variant { ... } */
        if (match(p, TOKEN_LBRACE) && expr->type == AST_IDENTIFIER) {
            /* This is union construction */
            char *union_name = expr->as.identifier;
            char *variant_name = field_or_variant;
            
            advance(p);  /* consume '{' */
            
            /* Parse variant fields */
            int capacity = 4;
            int count = 0;
            char **field_names = malloc(sizeof(char*) * capacity);
            ASTNode **field_values = malloc(sizeof(ASTNode*) * capacity);
            
            while (!match(p, TOKEN_RBRACE) && !match(p, TOKEN_EOF)) {
                if (count >= capacity) {
                    capacity *= 2;
                    field_names = realloc(field_names, sizeof(char*) * capacity);
                    field_values = realloc(field_values, sizeof(ASTNode*) * capacity);
                }
                
                /* Parse field name */
                if (!match(p, TOKEN_IDENTIFIER)) {
                    fprintf(stderr, "Error at line %d, column %d: Expected field name in union construction\n",
                            current_token(p)->line, current_token(p)->column);
                    break;
                }
                field_names[count] = strdup(current_token(p)->value);
                advance(p);
                
                /* Expect colon */
                if (!expect(p, TOKEN_COLON, "Expected ':' after field name")) {
                    break;
                }
                
                /* Parse field value */
                field_values[count] = parse_expression(p);
                count++;
                
                /* Optional comma */
                if (match(p, TOKEN_COMMA)) {
                    advance(p);
                }
            }
            
            if (!expect(p, TOKEN_RBRACE, "Expected '}' after union fields")) {
                free(union_name);
                free(variant_name);
                for (int i = 0; i < count; i++) {
                    free(field_names[i]);
                    free_ast(field_values[i]);
                }
                free(field_names);
                free(field_values);
                return NULL;
            }
            
            /* Create union construction node */
            ASTNode *union_construct = create_node(AST_UNION_CONSTRUCT, line, column);
            union_construct->as.union_construct.union_name = strdup(union_name);
            union_construct->as.union_construct.variant_name = variant_name;
            union_construct->as.union_construct.field_names = field_names;
            union_construct->as.union_construct.field_values = field_values;
            union_construct->as.union_construct.field_count = count;
            
            /* Free the original identifier node */
            free_ast(expr);
            expr = union_construct;
        } else {
            /* Regular field access */
            ASTNode *field_access = create_node(AST_FIELD_ACCESS, line, column);
            field_access->as.field_access.object = expr;
            field_access->as.field_access.field_name = field_or_variant;
            expr = field_access;
        }
    }
    
    p->recursion_depth--;
    return expr;
}

/* Parse block */
static ASTNode *parse_block(Parser *p) {
    /* Recursion depth guard */
    p->recursion_depth++;
    if (p->recursion_depth > MAX_RECURSION_DEPTH) {
        Token *tok = current_token(p);
        fprintf(stderr, "Error at line %d, column %d: Block recursion depth exceeded maximum (%d). Possible infinite recursion or extremely nested blocks.\n",
                tok ? tok->line : 0, tok ? tok->column : 0, MAX_RECURSION_DEPTH);
        p->recursion_depth--;
        return NULL;
    }
    
    int line = current_token(p)->line;
    int column = current_token(p)->column;

    if (!expect(p, TOKEN_LBRACE, "Expected '{'")) {
        p->recursion_depth--;
        return NULL;
    }

    int capacity = 8;
    int count = 0;
    ASTNode **statements = malloc(sizeof(ASTNode*) * capacity);

    while (!match(p, TOKEN_RBRACE) && !match(p, TOKEN_EOF)) {
        if (count >= capacity) {
            capacity *= 2;
            statements = realloc(statements, sizeof(ASTNode*) * capacity);
        }

        ASTNode *stmt = parse_statement(p);
        if (stmt) {
            statements[count++] = stmt;
        }
    }

    if (!expect(p, TOKEN_RBRACE, "Expected '}'")) {
        free(statements);
        p->recursion_depth--;
        return NULL;
    }

    ASTNode *node = create_node(AST_BLOCK, line, column);
    node->as.block.statements = statements;
    node->as.block.count = count;
    p->recursion_depth--;
    return node;
}

/* Parse statement */
static ASTNode *parse_statement(Parser *p) {
    Token *tok = current_token(p);
    ASTNode *node;

    switch (tok->type) {
        case TOKEN_LET: {
            int line = tok->line;
            int column = tok->column;
            advance(p);

            /* Check if variable should be mutable
             * Syntax: let x = ... (immutable by default, safe)
             *         let mut x = ... (explicitly mutable)
             * For practical algorithms, most variables need to be mutable,
             * so we encourage using 'mut' explicitly for clarity. */
            bool is_mut = false;
            if (match(p, TOKEN_MUT)) {
                is_mut = true;
                advance(p);
            }

            if (!match(p, TOKEN_IDENTIFIER)) {
                fprintf(stderr, "Error at line %d, column %d: Expected variable name\n", line);
                return NULL;
            }
            char *name = strdup(current_token(p)->value);
            advance(p);

            if (!expect(p, TOKEN_COLON, "Expected ':' after variable name")) {
                free(name);
                return NULL;
            }

            /* Capture type name if it's a struct/union type (identifier) */
            Token *type_token = current_token(p);
            char *type_name = NULL;
            if (type_token->type == TOKEN_IDENTIFIER) {
                type_name = strdup(type_token->value);
            }
            
            /* Parse type with element_type support for arrays and generics */
            Type element_type = TYPE_UNKNOWN;
            char *type_param_name = NULL;  /* For generic types like List<Point> */
            FunctionSignature *fn_sig = NULL;  /* For function types */
            Type type = parse_type_with_element(p, &element_type, &type_param_name, &fn_sig);

            /* For generic lists, type_param_name contains the element type (e.g., "Point") */
            /* For structs, type_name contains the struct name */
            if (type == TYPE_LIST_GENERIC && type_param_name) {
                /* Replace type_name with the generic parameter name */
                if (type_name) free(type_name);
                type_name = type_param_name;
            }

            if (!expect(p, TOKEN_ASSIGN, "Expected '=' in let statement")) {
                free(name);
                if (type_name) free(type_name);
                return NULL;
            }

            ASTNode *value = parse_expression(p);

            node = create_node(AST_LET, line, column);
            node->as.let.name = name;
            node->as.let.var_type = type;
            node->as.let.type_name = type_name;  /* For structs or generic type params */
            node->as.let.element_type = element_type;
            node->as.let.fn_sig = fn_sig;  /* For function types */
            node->as.let.is_mut = is_mut;
            node->as.let.value = value;
            return node;
        }

        case TOKEN_SET: {
            int line = tok->line;
            int column = tok->column;
            advance(p);

            if (!match(p, TOKEN_IDENTIFIER)) {
                fprintf(stderr, "Error at line %d, column %d: Expected variable name\n", line);
                return NULL;
            }
            char *name = strdup(current_token(p)->value);
            advance(p);

            ASTNode *value = parse_expression(p);

            node = create_node(AST_SET, line, column);
            node->as.set.name = name;
            node->as.set.value = value;
            return node;
        }

        case TOKEN_WHILE: {
            int line = tok->line;
            int column = tok->column;
            advance(p);

            ASTNode *condition = parse_expression(p);
            ASTNode *body = parse_block(p);

            node = create_node(AST_WHILE, line, column);
            node->as.while_stmt.condition = condition;
            node->as.while_stmt.body = body;
            return node;
        }

        case TOKEN_FOR: {
            int line = tok->line;
            int column = tok->column;
            advance(p);

            if (!match(p, TOKEN_IDENTIFIER)) {
                fprintf(stderr, "Error at line %d, column %d: Expected loop variable\n", line);
                return NULL;
            }
            char *var_name = strdup(current_token(p)->value);
            advance(p);

            if (!expect(p, TOKEN_IN, "Expected 'in' in for loop")) {
                free(var_name);
                return NULL;
            }

            ASTNode *range_expr = parse_expression(p);
            if (!range_expr) {
                fprintf(stderr, "Error at line %d, column %d: Invalid range expression in for loop\n", line);
                free(var_name);
                return NULL;
            }

            ASTNode *body = parse_block(p);
            if (!body) {
                fprintf(stderr, "Error at line %d, column %d: Invalid body in for loop\n", line);
                free(var_name);
                free_ast(range_expr);
                return NULL;
            }

            node = create_node(AST_FOR, line, column);
            node->as.for_stmt.var_name = var_name;
            node->as.for_stmt.range_expr = range_expr;
            node->as.for_stmt.body = body;
            return node;
        }

        case TOKEN_RETURN: {
            int line = tok->line;
            int column = tok->column;
            advance(p);

            ASTNode *value = NULL;
            if (!match(p, TOKEN_RBRACE)) {
                value = parse_expression(p);
            }

            node = create_node(AST_RETURN, line, column);
            node->as.return_stmt.value = value;
            return node;
        }

        case TOKEN_PRINT: {
            int line = tok->line;
            int column = tok->column;
            advance(p);

            ASTNode *expr = parse_expression(p);

            node = create_node(AST_PRINT, line, column);
            node->as.print.expr = expr;
            return node;
        }

        case TOKEN_ASSERT: {
            int line = tok->line;
            int column = tok->column;
            advance(p);

            ASTNode *condition = parse_expression(p);

            node = create_node(AST_ASSERT, line, column);
            node->as.assert.condition = condition;
            return node;
        }

        default:
            /* Try to parse as expression statement */
            return parse_expression(p);
    }
}

/* Parse struct definition */
static ASTNode *parse_struct_def(Parser *p) {
    int line = current_token(p)->line;
    int column = current_token(p)->column;
    
    if (!expect(p, TOKEN_STRUCT, "Expected 'struct'")) {
        return NULL;
    }
    
    /* Get struct name */
    if (!match(p, TOKEN_IDENTIFIER)) {
        fprintf(stderr, "Error at line %d, column %d: Expected struct name\n", 
                current_token(p)->line, current_token(p)->column);
        return NULL;
    }
    char *struct_name = strdup(current_token(p)->value);
    advance(p);
    
    /* Expect opening brace */
    if (!expect(p, TOKEN_LBRACE, "Expected '{' after struct name")) {
        free(struct_name);
        return NULL;
    }
    
    /* Parse fields */
    int capacity = 8;
    int count = 0;
    char **field_names = malloc(sizeof(char*) * capacity);
    Type *field_types = malloc(sizeof(Type) * capacity);
    
    while (!match(p, TOKEN_RBRACE) && !match(p, TOKEN_EOF)) {
        if (count >= capacity) {
            capacity *= 2;
            field_names = realloc(field_names, sizeof(char*) * capacity);
            field_types = realloc(field_types, sizeof(Type) * capacity);
        }
        
        /* Parse field name */
        if (!match(p, TOKEN_IDENTIFIER)) {
            fprintf(stderr, "Error at line %d, column %d: Expected field name\n",
                    current_token(p)->line, current_token(p)->column);
            break;
        }
        field_names[count] = strdup(current_token(p)->value);
        advance(p);
        
        /* Expect colon */
        if (!expect(p, TOKEN_COLON, "Expected ':' after field name")) {
            break;
        }
        
        /* Parse field type */
        field_types[count] = parse_type(p);
        count++;
        
        /* Optional comma */
        if (match(p, TOKEN_COMMA)) {
            advance(p);
        }
    }
    
    /* Expect closing brace */
    if (!expect(p, TOKEN_RBRACE, "Expected '}' after struct fields")) {
        free(struct_name);
        for (int i = 0; i < count; i++) {
            free(field_names[i]);
        }
        free(field_names);
        free(field_types);
        return NULL;
    }
    
    /* Create AST node */
    ASTNode *node = create_node(AST_STRUCT_DEF, line, column);
    node->as.struct_def.name = struct_name;
    node->as.struct_def.field_names = field_names;
    node->as.struct_def.field_types = field_types;
    node->as.struct_def.field_count = count;
    
    return node;
}

/* Parse enum definition */
static ASTNode *parse_enum_def(Parser *p) {
    int line = current_token(p)->line;
    int column = current_token(p)->column;
    
    if (!expect(p, TOKEN_ENUM, "Expected 'enum'")) {
        return NULL;
    }
    
    /* Get enum name */
    if (!match(p, TOKEN_IDENTIFIER)) {
        fprintf(stderr, "Error at line %d, column %d: Expected enum name\n",
                current_token(p)->line, current_token(p)->column);
        return NULL;
    }
    char *enum_name = strdup(current_token(p)->value);
    advance(p);
    
    /* Expect opening brace */
    if (!expect(p, TOKEN_LBRACE, "Expected '{' after enum name")) {
        free(enum_name);
        return NULL;
    }
    
    /* Parse variants */
    int capacity = 8;
    int count = 0;
    char **variant_names = malloc(sizeof(char*) * capacity);
    int *variant_values = malloc(sizeof(int) * capacity);
    int next_auto_value = 0;
    
    while (!match(p, TOKEN_RBRACE) && !match(p, TOKEN_EOF)) {
        if (count >= capacity) {
            capacity *= 2;
            variant_names = realloc(variant_names, sizeof(char*) * capacity);
            variant_values = realloc(variant_values, sizeof(int) * capacity);
        }
        
        /* Parse variant name */
        if (!match(p, TOKEN_IDENTIFIER)) {
            fprintf(stderr, "Error at line %d, column %d: Expected variant name\n",
                    current_token(p)->line, current_token(p)->column);
            break;
        }
        Token *tok = current_token(p);
        if (!tok || !tok->value) {
            fprintf(stderr, "Error at line %d, column %d: Invalid token (NULL value)\n",
                    tok ? tok->line : 0, tok ? tok->column : 0);
            break;
        }
        variant_names[count] = strdup(tok->value);
        if (!variant_names[count]) {
            fprintf(stderr, "Error: Failed to allocate memory for variant name\n");
            free(enum_name);
            for (int i = 0; i < count; i++) {
                free(variant_names[i]);
            }
            free(variant_names);
            free(variant_values);
            return NULL;
        }
        advance(p);
        
        /* Check for explicit value */
        if (match(p, TOKEN_ASSIGN)) {
            advance(p);
            if (!match(p, TOKEN_NUMBER)) {
                fprintf(stderr, "Error at line %d, column %d: Expected number after '='\n",
                        current_token(p)->line, current_token(p)->column);
                variant_values[count] = next_auto_value++;
            } else {
                Token *num_tok = current_token(p);
                variant_values[count] = (num_tok && num_tok->value) ? atoi(num_tok->value) : 0;
                next_auto_value = variant_values[count] + 1;
                advance(p);
            }
        } else {
            variant_values[count] = next_auto_value++;
        }
        
        count++;
        
        /* Optional comma */
        if (match(p, TOKEN_COMMA)) {
            advance(p);
        }
    }
    
    /* Expect closing brace */
    if (!expect(p, TOKEN_RBRACE, "Expected '}' after enum variants")) {
        free(enum_name);
        for (int i = 0; i < count; i++) {
            free(variant_names[i]);
        }
        free(variant_names);
        free(variant_values);
        return NULL;
    }
    
    /* Create AST node */
    ASTNode *node = create_node(AST_ENUM_DEF, line, column);
    node->as.enum_def.name = enum_name;
    node->as.enum_def.variant_names = variant_names;
    node->as.enum_def.variant_values = variant_values;
    node->as.enum_def.variant_count = count;
    
    return node;
}

/* Parse union definition */
static ASTNode *parse_union_def(Parser *p) {
    int line = current_token(p)->line;
    int column = current_token(p)->column;
    
    if (!expect(p, TOKEN_UNION, "Expected 'union'")) {
        return NULL;
    }
    
    /* Get union name */
    if (!match(p, TOKEN_IDENTIFIER)) {
        fprintf(stderr, "Error at line %d, column %d: Expected union name\n",
                current_token(p)->line, current_token(p)->column);
        return NULL;
    }
    char *union_name = strdup(current_token(p)->value);
    advance(p);
    
    /* Expect opening brace */
    if (!expect(p, TOKEN_LBRACE, "Expected '{' after union name")) {
        free(union_name);
        return NULL;
    }
    
    /* Parse variants */
    int capacity = 8;
    int count = 0;
    char **variant_names = malloc(sizeof(char*) * capacity);
    int *variant_field_counts = malloc(sizeof(int) * capacity);
    char ***variant_field_names = malloc(sizeof(char**) * capacity);
    Type **variant_field_types = malloc(sizeof(Type*) * capacity);
    
    while (!match(p, TOKEN_RBRACE) && !match(p, TOKEN_EOF)) {
        if (count >= capacity) {
            capacity *= 2;
            variant_names = realloc(variant_names, sizeof(char*) * capacity);
            variant_field_counts = realloc(variant_field_counts, sizeof(int) * capacity);
            variant_field_names = realloc(variant_field_names, sizeof(char**) * capacity);
            variant_field_types = realloc(variant_field_types, sizeof(Type*) * capacity);
        }
        
        /* Parse variant name */
        if (!match(p, TOKEN_IDENTIFIER)) {
            fprintf(stderr, "Error at line %d, column %d: Expected variant name\n",
                    current_token(p)->line, current_token(p)->column);
            break;
        }
        variant_names[count] = strdup(current_token(p)->value);
        advance(p);
        
        /* Expect opening brace for variant fields */
        if (!expect(p, TOKEN_LBRACE, "Expected '{' after variant name")) {
            free(variant_names[count]);
            break;
        }
        
        /* Parse variant fields (like struct fields) */
        int field_capacity = 4;
        int field_count = 0;
        char **field_names = malloc(sizeof(char*) * field_capacity);
        Type *field_types = malloc(sizeof(Type) * field_capacity);
        
        while (!match(p, TOKEN_RBRACE) && !match(p, TOKEN_EOF)) {
            if (field_count >= field_capacity) {
                field_capacity *= 2;
                field_names = realloc(field_names, sizeof(char*) * field_capacity);
                field_types = realloc(field_types, sizeof(Type) * field_capacity);
            }
            
            /* Parse field name */
            if (!match(p, TOKEN_IDENTIFIER)) {
                fprintf(stderr, "Error at line %d, column %d: Expected field name\n",
                        current_token(p)->line, current_token(p)->column);
                break;
            }
            field_names[field_count] = strdup(current_token(p)->value);
            advance(p);
            
            /* Expect colon */
            if (!expect(p, TOKEN_COLON, "Expected ':' after field name")) {
                free(field_names[field_count]);
                break;
            }
            
            /* Parse field type */
            field_types[field_count] = parse_type(p);
            if (field_types[field_count] == TYPE_UNKNOWN) {
                free(field_names[field_count]);
                break;
            }
            
            field_count++;
            
            /* Optional comma between fields */
            if (match(p, TOKEN_COMMA)) {
                advance(p);
            }
        }
        
        /* Close variant fields */
        if (!expect(p, TOKEN_RBRACE, "Expected '}' after variant fields")) {
            for (int i = 0; i < field_count; i++) {
                free(field_names[i]);
            }
            free(field_names);
            free(field_types);
            free(variant_names[count]);
            break;
        }
        
        variant_field_counts[count] = field_count;
        variant_field_names[count] = field_names;
        variant_field_types[count] = field_types;
        count++;
        
        /* Optional comma between variants */
        if (match(p, TOKEN_COMMA)) {
            advance(p);
        }
    }
    
    /* Close union definition */
    if (!expect(p, TOKEN_RBRACE, "Expected '}' after union variants")) {
        free(union_name);
        for (int i = 0; i < count; i++) {
            free(variant_names[i]);
            for (int j = 0; j < variant_field_counts[i]; j++) {
                free(variant_field_names[i][j]);
            }
            free(variant_field_names[i]);
            free(variant_field_types[i]);
        }
        free(variant_names);
        free(variant_field_counts);
        free(variant_field_names);
        free(variant_field_types);
        return NULL;
    }
    
    /* Create AST node */
    ASTNode *node = create_node(AST_UNION_DEF, line, column);
    node->as.union_def.name = union_name;
    node->as.union_def.variant_names = variant_names;
    node->as.union_def.variant_field_counts = variant_field_counts;
    node->as.union_def.variant_field_names = variant_field_names;
    node->as.union_def.variant_field_types = variant_field_types;
    node->as.union_def.variant_count = count;
    
    return node;
}

/* Parse match expression */
static ASTNode *parse_match_expr(Parser *p) {
    int line = current_token(p)->line;
    int column = current_token(p)->column;
    
    if (!expect(p, TOKEN_MATCH, "Expected 'match'")) {
        return NULL;
    }
    
    /* Parse expression to match on */
    ASTNode *expr = parse_expression(p);
    if (!expr) {
        fprintf(stderr, "Error at line %d, column %d: Expected expression after 'match'\n",
                current_token(p)->line, current_token(p)->column);
        return NULL;
    }
    
    /* Expect opening brace */
    if (!expect(p, TOKEN_LBRACE, "Expected '{' after match expression")) {
        free_ast(expr);
        return NULL;
    }
    
    /* Parse match arms */
    int capacity = 4;
    int count = 0;
    char **pattern_variants = malloc(sizeof(char*) * capacity);
    char **pattern_bindings = malloc(sizeof(char*) * capacity);
    ASTNode **arm_bodies = malloc(sizeof(ASTNode*) * capacity);
    
    while (!match(p, TOKEN_RBRACE) && !match(p, TOKEN_EOF)) {
        if (count >= capacity) {
            capacity *= 2;
            pattern_variants = realloc(pattern_variants, sizeof(char*) * capacity);
            pattern_bindings = realloc(pattern_bindings, sizeof(char*) * capacity);
            arm_bodies = realloc(arm_bodies, sizeof(ASTNode*) * capacity);
        }
        
        /* Parse pattern: VariantName(binding) */
        if (!match(p, TOKEN_IDENTIFIER)) {
            fprintf(stderr, "Error at line %d, column %d: Expected variant name in match pattern\n",
                    current_token(p)->line, current_token(p)->column);
            break;
        }
        pattern_variants[count] = strdup(current_token(p)->value);
        advance(p);
        
        /* Expect opening paren for binding */
        if (!expect(p, TOKEN_LPAREN, "Expected '(' after variant name in match")) {
            free(pattern_variants[count]);
            break;
        }
        
        /* Parse binding variable */
        if (!match(p, TOKEN_IDENTIFIER)) {
            fprintf(stderr, "Error at line %d, column %d: Expected binding variable in match pattern\n",
                    current_token(p)->line, current_token(p)->column);
            free(pattern_variants[count]);
            break;
        }
        pattern_bindings[count] = strdup(current_token(p)->value);
        advance(p);
        
        /* Expect closing paren */
        if (!expect(p, TOKEN_RPAREN, "Expected ')' after binding variable")) {
            free(pattern_variants[count]);
            free(pattern_bindings[count]);
            break;
        }
        
        /* Expect arrow */
        if (!expect(p, TOKEN_ARROW, "Expected '=>' after match pattern")) {
            free(pattern_variants[count]);
            free(pattern_bindings[count]);
            break;
        }
        
        /* Parse arm body - should be an expression or block */
        if (match(p, TOKEN_LBRACE)) {
            arm_bodies[count] = parse_block(p);
        } else {
            arm_bodies[count] = parse_expression(p);
        }
        
        if (!arm_bodies[count]) {
            free(pattern_variants[count]);
            free(pattern_bindings[count]);
            break;
        }
        
        count++;
        
        /* Optional comma between arms */
        if (match(p, TOKEN_COMMA)) {
            advance(p);
        }
    }
    
    /* Close match expression */
    if (!expect(p, TOKEN_RBRACE, "Expected '}' after match arms")) {
        free_ast(expr);
        for (int i = 0; i < count; i++) {
            free(pattern_variants[i]);
            free(pattern_bindings[i]);
            free_ast(arm_bodies[i]);
        }
        free(pattern_variants);
        free(pattern_bindings);
        free(arm_bodies);
        return NULL;
    }
    
    /* Create match expression node */
    ASTNode *match_node = create_node(AST_MATCH, line, column);
    match_node->as.match_expr.expr = expr;
    match_node->as.match_expr.arm_count = count;
    match_node->as.match_expr.pattern_variants = pattern_variants;
    match_node->as.match_expr.pattern_bindings = pattern_bindings;
    match_node->as.match_expr.arm_bodies = arm_bodies;
    match_node->as.match_expr.union_type_name = NULL;  /* Will be filled during typechecking */
    
    return match_node;
}

/* Parse function definition */
static ASTNode *parse_function(Parser *p, bool is_extern) {
    int line = current_token(p)->line;
    int column = current_token(p)->column;

    if (!expect(p, TOKEN_FN, "Expected 'fn'")) {
        return NULL;
    }

    if (!match(p, TOKEN_IDENTIFIER)) {
        fprintf(stderr, "Error at line %d, column %d: Expected function name\n", line, column);
        return NULL;
    }
    char *name = strdup(current_token(p)->value);
    advance(p);

    if (!expect(p, TOKEN_LPAREN, "Expected '(' after function name")) {
        free(name);
        return NULL;
    }

    Parameter *params;
    int param_count;
    if (!parse_parameters(p, &params, &param_count)) {
        free(name);
        return NULL;
    }

    if (!expect(p, TOKEN_RPAREN, "Expected ')' after parameters")) {
        free(name);
        free(params);
        return NULL;
    }

    if (!expect(p, TOKEN_ARROW, "Expected '->' after parameters")) {
        free(name);
        free(params);
        return NULL;
    }

    /* Parse return type and capture struct/generic type name if applicable */
    char *return_struct_name = NULL;
    FunctionSignature *return_fn_sig = NULL;
    Type return_type = parse_type_with_element(p, NULL, &return_struct_name, &return_fn_sig);
    
    /* For non-generic structs, we still need to capture the name */
    if (return_type == TYPE_STRUCT && !return_struct_name) {
        /* Go back one token to get the struct name */
        if (p->pos > 0 && p->pos <= p->count) {
            Token *prev_token = &p->tokens[p->pos - 1];
            if (prev_token && prev_token->type == TOKEN_IDENTIFIER) {
                return_struct_name = strdup(prev_token->value);
            }
        }
    }

    ASTNode *body = NULL;
    if (!is_extern) {
        /* Regular functions must have a body */
        body = parse_block(p);
        if (!body) {
            free(name);
            free(params);
            if (return_struct_name) free(return_struct_name);
            return NULL;
        }
    }
    /* Extern functions have no body - declaration only */

    ASTNode *node = create_node(AST_FUNCTION, line, column);
    node->as.function.name = name;
    node->as.function.params = params;
    node->as.function.param_count = param_count;
    node->as.function.return_type = return_type;
    node->as.function.return_struct_type_name = return_struct_name;  /* May be NULL */
    node->as.function.return_fn_sig = return_fn_sig;  /* May be NULL */
    node->as.function.body = body;
    node->as.function.is_extern = is_extern;
    return node;
}

/* Parse shadow-test block */
static ASTNode *parse_import(Parser *p) {
    int line = current_token(p)->line;
    int column = current_token(p)->column;

    if (!expect(p, TOKEN_IMPORT, "Expected 'import'")) {
        return NULL;
    }

    /* Parse import path: import "module.nano" or import module */
    char *module_path = NULL;
    char *module_name = NULL;

    if (match(p, TOKEN_STRING)) {
        /* import "module.nano" */
        module_path = strdup(current_token(p)->value);
        advance(p);
    } else if (match(p, TOKEN_IDENTIFIER)) {
        /* import module (treat as "module.nano") */
        char *ident = current_token(p)->value;
        char *path = malloc(strlen(ident) + 6);  /* +6 for ".nano\0" */
        snprintf(path, strlen(ident) + 6, "%s.nano", ident);
        module_path = path;
        advance(p);
    } else {
        fprintf(stderr, "Error at line %d, column %d: Expected string or identifier after 'import'\n", line, column);
        return NULL;
    }

    /* Optional: import "module.nano" as alias */
    if (match(p, TOKEN_IDENTIFIER) && strcmp(current_token(p)->value, "as") == 0) {
        advance(p);  /* consume "as" */
        if (!match(p, TOKEN_IDENTIFIER)) {
            fprintf(stderr, "Error at line %d, column %d: Expected module alias name after 'as'\n", line, column);
            free(module_path);
            return NULL;
        }
        module_name = strdup(current_token(p)->value);
        advance(p);
    }

    ASTNode *node = create_node(AST_IMPORT, line, column);
    node->as.import_stmt.module_path = module_path;
    node->as.import_stmt.module_name = module_name;
    return node;
}

static ASTNode *parse_shadow(Parser *p) {
    int line = current_token(p)->line;
    int column = current_token(p)->column;

    if (!expect(p, TOKEN_SHADOW, "Expected 'shadow'")) {
        return NULL;
    }

    if (!match(p, TOKEN_IDENTIFIER)) {
        fprintf(stderr, "Error at line %d, column %d: Expected function name after 'shadow'\n", line, column);
        return NULL;
    }
    char *func_name = strdup(current_token(p)->value);
    advance(p);

    ASTNode *body = parse_block(p);
    if (!body) {
        free(func_name);
        return NULL;
    }

    ASTNode *node = create_node(AST_SHADOW, line, column);
    node->as.shadow.function_name = func_name;
    node->as.shadow.body = body;
    return node;
}

/* Parse top-level program */
ASTNode *parse_program(Token *tokens, int token_count) {
    if (!tokens || token_count <= 0) {
        fprintf(stderr, "Error: Invalid token array\n");
        return NULL;
    }
    
    Parser parser;
    parser.tokens = tokens;
    parser.count = token_count;
    parser.pos = 0;
    parser.recursion_depth = 0;

    int capacity = 16;
    int count = 0;
    ASTNode **items = malloc(sizeof(ASTNode*) * capacity);

    while (!match(&parser, TOKEN_EOF)) {
        /* Safety check: if current_token returns NULL, we've hit an error */
        Token *tok = current_token(&parser);
        if (!tok) {
            fprintf(stderr, "Error: Parser reached invalid state\n");
            break;
        }
        if (count >= capacity) {
            capacity *= 2;
            items = realloc(items, sizeof(ASTNode*) * capacity);
        }

        if (match(&parser, TOKEN_IMPORT)) {
            items[count++] = parse_import(&parser);
        } else if (match(&parser, TOKEN_STRUCT)) {
            items[count++] = parse_struct_def(&parser);
        } else if (match(&parser, TOKEN_ENUM)) {
            items[count++] = parse_enum_def(&parser);
        } else if (match(&parser, TOKEN_UNION)) {
            items[count++] = parse_union_def(&parser);
        } else if (match(&parser, TOKEN_EXTERN)) {
            /* extern fn declarations */
            advance(&parser);  /* Skip 'extern' token */
            items[count++] = parse_function(&parser, true);
        } else if (match(&parser, TOKEN_FN)) {
            items[count++] = parse_function(&parser, false);
        } else if (match(&parser, TOKEN_SHADOW)) {
            items[count++] = parse_shadow(&parser);
        } else {
            fprintf(stderr, "Error at line %d, column %d: Expected import, struct, enum, union, extern, function or shadow-test definition\n",
                    current_token(&parser)->line, current_token(&parser)->column);
            advance(&parser);
        }
    }

    ASTNode *program = create_node(AST_PROGRAM, 1, 1);
    program->as.program.items = items;
    program->as.program.count = count;
    return program;
}

/* Free AST */
void free_ast(ASTNode *node) {
    if (!node) return;

    switch (node->type) {
        case AST_STRING:
            free(node->as.string_val);
            break;
        case AST_IDENTIFIER:
            free(node->as.identifier);
            break;
        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; i++) {
                free_ast(node->as.prefix_op.args[i]);
            }
            free(node->as.prefix_op.args);
            break;
        case AST_CALL:
            free(node->as.call.name);
            for (int i = 0; i < node->as.call.arg_count; i++) {
                free_ast(node->as.call.args[i]);
            }
            free(node->as.call.args);
            break;
        case AST_LET:
            free(node->as.let.name);
            if (node->as.let.type_name) {
                free(node->as.let.type_name);
            }
            free_ast(node->as.let.value);
            break;
        case AST_SET:
            free(node->as.set.name);
            free_ast(node->as.set.value);
            break;
        case AST_IF:
            free_ast(node->as.if_stmt.condition);
            free_ast(node->as.if_stmt.then_branch);
            free_ast(node->as.if_stmt.else_branch);
            break;
        case AST_WHILE:
            free_ast(node->as.while_stmt.condition);
            free_ast(node->as.while_stmt.body);
            break;
        case AST_FOR:
            free(node->as.for_stmt.var_name);
            free_ast(node->as.for_stmt.range_expr);
            free_ast(node->as.for_stmt.body);
            break;
        case AST_RETURN:
            free_ast(node->as.return_stmt.value);
            break;
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++) {
                free_ast(node->as.block.statements[i]);
            }
            free(node->as.block.statements);
            break;
        case AST_FUNCTION:
            free(node->as.function.name);
            for (int i = 0; i < node->as.function.param_count; i++) {
                free(node->as.function.params[i].name);
            }
            free(node->as.function.params);
            if (node->as.function.body) {  /* Extern functions have no body */
                free_ast(node->as.function.body);
            }
            break;
        case AST_SHADOW:
            free(node->as.shadow.function_name);
            free_ast(node->as.shadow.body);
            break;
        case AST_IMPORT:
            if (node->as.import_stmt.module_path) {
                free(node->as.import_stmt.module_path);
            }
            if (node->as.import_stmt.module_name) {
                free(node->as.import_stmt.module_name);
            }
            break;
        case AST_PROGRAM:
            for (int i = 0; i < node->as.program.count; i++) {
                free_ast(node->as.program.items[i]);
            }
            free(node->as.program.items);
            break;
        case AST_PRINT:
            free_ast(node->as.print.expr);
            break;
        case AST_ASSERT:
            free_ast(node->as.assert.condition);
            break;
        case AST_STRUCT_DEF:
            free(node->as.struct_def.name);
            for (int i = 0; i < node->as.struct_def.field_count; i++) {
                free(node->as.struct_def.field_names[i]);
            }
            free(node->as.struct_def.field_names);
            free(node->as.struct_def.field_types);
            break;
        case AST_STRUCT_LITERAL:
            free(node->as.struct_literal.struct_name);
            for (int i = 0; i < node->as.struct_literal.field_count; i++) {
                free(node->as.struct_literal.field_names[i]);
                free_ast(node->as.struct_literal.field_values[i]);
            }
            free(node->as.struct_literal.field_names);
            free(node->as.struct_literal.field_values);
            break;
        case AST_FIELD_ACCESS:
            free_ast(node->as.field_access.object);
            free(node->as.field_access.field_name);
            break;
        case AST_ENUM_DEF:
            free(node->as.enum_def.name);
            for (int i = 0; i < node->as.enum_def.variant_count; i++) {
                free(node->as.enum_def.variant_names[i]);
            }
            free(node->as.enum_def.variant_names);
            free(node->as.enum_def.variant_values);
            break;
        case AST_UNION_DEF:
            free(node->as.union_def.name);
            for (int i = 0; i < node->as.union_def.variant_count; i++) {
                free(node->as.union_def.variant_names[i]);
                for (int j = 0; j < node->as.union_def.variant_field_counts[i]; j++) {
                    free(node->as.union_def.variant_field_names[i][j]);
                }
                free(node->as.union_def.variant_field_names[i]);
                free(node->as.union_def.variant_field_types[i]);
            }
            free(node->as.union_def.variant_names);
            free(node->as.union_def.variant_field_counts);
            free(node->as.union_def.variant_field_names);
            free(node->as.union_def.variant_field_types);
            break;
        case AST_UNION_CONSTRUCT:
            free(node->as.union_construct.union_name);
            free(node->as.union_construct.variant_name);
            for (int i = 0; i < node->as.union_construct.field_count; i++) {
                free(node->as.union_construct.field_names[i]);
                free_ast(node->as.union_construct.field_values[i]);
            }
            free(node->as.union_construct.field_names);
            free(node->as.union_construct.field_values);
            break;
        case AST_MATCH:
            free_ast(node->as.match_expr.expr);
            for (int i = 0; i < node->as.match_expr.arm_count; i++) {
                free(node->as.match_expr.pattern_variants[i]);
                free(node->as.match_expr.pattern_bindings[i]);
                free_ast(node->as.match_expr.arm_bodies[i]);
            }
            free(node->as.match_expr.pattern_variants);
            free(node->as.match_expr.pattern_bindings);
            free(node->as.match_expr.arm_bodies);
            if (node->as.match_expr.union_type_name) {
                free(node->as.match_expr.union_type_name);
            }
            break;
        default:
            break;
    }

    free(node);
}