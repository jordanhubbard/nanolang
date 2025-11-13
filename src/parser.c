#include "nanolang.h"

/* Parser state */
typedef struct {
    Token *tokens;
    int count;
    int pos;
} Parser;

/* Forward declarations */
static Type parse_type_with_element(Parser *p, Type *element_type_out);

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
    int pos = p->pos + offset;
    if (pos < p->count) {
        return &p->tokens[pos];
    }
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
    ASTNode *node = malloc(sizeof(ASTNode));
    node->type = type;
    node->line = line;
    node->column = column;
    return node;
}

/* Parse type annotation */
static Type parse_type(Parser *p) {
    return parse_type_with_element(p, NULL);
}

/* Parse type annotation with optional element_type output (for arrays) */
static Type parse_type_with_element(Parser *p, Type *element_type_out) {
    Type type = TYPE_UNKNOWN;
    Token *tok = current_token(p);

    switch (tok->type) {
        case TOKEN_TYPE_INT: type = TYPE_INT; break;
        case TOKEN_TYPE_FLOAT: type = TYPE_FLOAT; break;
        case TOKEN_TYPE_BOOL: type = TYPE_BOOL; break;
        case TOKEN_TYPE_STRING: type = TYPE_STRING; break;
        case TOKEN_TYPE_VOID: type = TYPE_VOID; break;
        case TOKEN_IDENTIFIER:
            /* Check for list types */
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
            
            /* Parse element type */
            Type element_type = parse_type(p);
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
            
            /* Parse type with element_type support for arrays */
            Type element_type = TYPE_UNKNOWN;
            param_list[count].type = parse_type_with_element(p, &element_type);
            param_list[count].struct_type_name = NULL;
            param_list[count].element_type = element_type;
            
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

        default:
            fprintf(stderr, "Error at line %d, column %d: Unexpected token in expression: %s\n",
                    tok->line, token_type_name(tok->type));
            return NULL;
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
    if (match(p, TOKEN_IF)) {
        return parse_if_expression(p);
    }
    
    /* Parse primary expression */
    ASTNode *expr = parse_primary(p);
    
    /* Handle field access: obj.field or obj.field1.field2 */
    while (match(p, TOKEN_DOT)) {
        int line = current_token(p)->line;
        int column = current_token(p)->column;
        advance(p);  /* consume '.' */
        
        if (!match(p, TOKEN_IDENTIFIER)) {
            fprintf(stderr, "Error at line %d, column %d: Expected field name after '.'\n",
                    current_token(p)->line, current_token(p)->column);
            return expr;
        }
        
        char *field_name = strdup(current_token(p)->value);
        advance(p);
        
        ASTNode *field_access = create_node(AST_FIELD_ACCESS, line, column);
        field_access->as.field_access.object = expr;
        field_access->as.field_access.field_name = field_name;
        expr = field_access;
    }
    
    return expr;
}

/* Parse block */
static ASTNode *parse_block(Parser *p) {
    int line = current_token(p)->line;
    int column = current_token(p)->column;

    if (!expect(p, TOKEN_LBRACE, "Expected '{'")) {
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
        return NULL;
    }

    ASTNode *node = create_node(AST_BLOCK, line, column);
    node->as.block.statements = statements;
    node->as.block.count = count;
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

            /* Parse type with element_type support for arrays */
            Type element_type = TYPE_UNKNOWN;
            Type type = parse_type_with_element(p, &element_type);

            if (!expect(p, TOKEN_ASSIGN, "Expected '=' in let statement")) {
                free(name);
                return NULL;
            }

            ASTNode *value = parse_expression(p);

            node = create_node(AST_LET, line, column);
            node->as.let.name = name;
            node->as.let.var_type = type;
            node->as.let.element_type = element_type;
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
        variant_names[count] = strdup(current_token(p)->value);
        advance(p);
        
        /* Check for explicit value */
        if (match(p, TOKEN_ASSIGN)) {
            advance(p);
            if (!match(p, TOKEN_NUMBER)) {
                fprintf(stderr, "Error at line %d, column %d: Expected number after '='\n",
                        current_token(p)->line, current_token(p)->column);
                variant_values[count] = next_auto_value++;
            } else {
                variant_values[count] = current_token(p)->value ? atoi(current_token(p)->value) : 0;
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

    /* Capture return struct type name if it's a struct */
    Token *return_type_token = current_token(p);
    char *return_struct_name = NULL;
    if (return_type_token->type == TOKEN_IDENTIFIER) {
        return_struct_name = strdup(return_type_token->value);
    }
    
    Type return_type = parse_type(p);

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
    node->as.function.body = body;
    node->as.function.is_extern = is_extern;
    return node;
}

/* Parse shadow-test block */
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

        if (match(&parser, TOKEN_STRUCT)) {
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
            fprintf(stderr, "Error at line %d, column %d: Expected struct, enum, union, extern, function or shadow-test definition\n",
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
        default:
            break;
    }

    free(node);
}