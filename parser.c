#include "nanolang.h"

/* Parser state */
typedef struct {
    Token *tokens;
    int count;
    int pos;
} Parser;

/* Helper functions */
static Token *current_token(Parser *p) {
    if (p->pos < p->count) {
        return &p->tokens[p->pos];
    }
    return &p->tokens[p->count - 1]; /* Return EOF */
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
    return current_token(p)->type == type;
}

static bool expect(Parser *p, TokenType type, const char *msg) {
    if (!match(p, type)) {
        fprintf(stderr, "Error at line %d: %s (got %s)\n",
                current_token(p)->line, msg, token_type_name(current_token(p)->type));
        return false;
    }
    advance(p);
    return true;
}

/* Forward declarations */
static ASTNode *parse_statement(Parser *p);
static ASTNode *parse_expression(Parser *p);
static ASTNode *parse_block(Parser *p);

/* Create AST nodes */
static ASTNode *create_node(ASTNodeType type, int line) {
    ASTNode *node = malloc(sizeof(ASTNode));
    node->type = type;
    node->line = line;
    return node;
}

/* Parse type annotation */
static Type parse_type(Parser *p) {
    Type type = TYPE_UNKNOWN;
    Token *tok = current_token(p);

    switch (tok->type) {
        case TOKEN_TYPE_INT: type = TYPE_INT; break;
        case TOKEN_TYPE_FLOAT: type = TYPE_FLOAT; break;
        case TOKEN_TYPE_BOOL: type = TYPE_BOOL; break;
        case TOKEN_TYPE_STRING: type = TYPE_STRING; break;
        case TOKEN_TYPE_VOID: type = TYPE_VOID; break;
        default:
            fprintf(stderr, "Error at line %d: Expected type annotation\n", tok->line);
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
                fprintf(stderr, "Error at line %d: Expected parameter name\n", current_token(p)->line);
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

            /* Type */
            param_list[count].type = parse_type(p);
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

        ASTNode *node = create_node(AST_PREFIX_OP, line);
        node->as.prefix_op.op = op;
        node->as.prefix_op.args = args;
        node->as.prefix_op.arg_count = count;
        return node;
    } else if (match(p, TOKEN_IDENTIFIER) || match(p, TOKEN_RANGE)) {
        /* Function call */
        char *func_name = strdup(tok->value ? tok->value : "range");
        advance(p);

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

        ASTNode *node = create_node(AST_CALL, line);
        node->as.call.name = func_name;
        node->as.call.args = args;
        node->as.call.arg_count = count;
        return node;
    } else {
        fprintf(stderr, "Error at line %d: Invalid prefix operation\n", line);
        return NULL;
    }
}

/* Parse primary expression */
static ASTNode *parse_primary(Parser *p) {
    Token *tok = current_token(p);
    ASTNode *node;

    switch (tok->type) {
        case TOKEN_NUMBER:
            node = create_node(AST_NUMBER, tok->line);
            node->as.number = atoll(tok->value);
            advance(p);
            return node;

        case TOKEN_FLOAT:
            node = create_node(AST_FLOAT, tok->line);
            node->as.float_val = atof(tok->value);
            advance(p);
            return node;

        case TOKEN_STRING:
            node = create_node(AST_STRING, tok->line);
            node->as.string_val = strdup(tok->value);
            advance(p);
            return node;

        case TOKEN_TRUE:
            node = create_node(AST_BOOL, tok->line);
            node->as.bool_val = true;
            advance(p);
            return node;

        case TOKEN_FALSE:
            node = create_node(AST_BOOL, tok->line);
            node->as.bool_val = false;
            advance(p);
            return node;

        case TOKEN_IDENTIFIER:
            node = create_node(AST_IDENTIFIER, tok->line);
            node->as.identifier = strdup(tok->value);
            advance(p);
            return node;

        case TOKEN_LPAREN:
            return parse_prefix_op(p);

        case TOKEN_IF:
            return parse_expression(p);

        default:
            fprintf(stderr, "Error at line %d: Unexpected token in expression: %s\n",
                    tok->line, token_type_name(tok->type));
            return NULL;
    }
}

/* Parse if expression */
static ASTNode *parse_if_expression(Parser *p) {
    int line = current_token(p)->line;

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

    ASTNode *node = create_node(AST_IF, line);
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
    return parse_primary(p);
}

/* Parse block */
static ASTNode *parse_block(Parser *p) {
    int line = current_token(p)->line;

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

    ASTNode *node = create_node(AST_BLOCK, line);
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
            advance(p);

            bool is_mut = false;
            if (match(p, TOKEN_MUT)) {
                is_mut = true;
                advance(p);
            }

            if (!match(p, TOKEN_IDENTIFIER)) {
                fprintf(stderr, "Error at line %d: Expected variable name\n", line);
                return NULL;
            }
            char *name = strdup(current_token(p)->value);
            advance(p);

            if (!expect(p, TOKEN_COLON, "Expected ':' after variable name")) {
                free(name);
                return NULL;
            }

            Type type = parse_type(p);

            if (!expect(p, TOKEN_ASSIGN, "Expected '=' in let statement")) {
                free(name);
                return NULL;
            }

            ASTNode *value = parse_expression(p);

            node = create_node(AST_LET, line);
            node->as.let.name = name;
            node->as.let.var_type = type;
            node->as.let.is_mut = is_mut;
            node->as.let.value = value;
            return node;
        }

        case TOKEN_SET: {
            int line = tok->line;
            advance(p);

            if (!match(p, TOKEN_IDENTIFIER)) {
                fprintf(stderr, "Error at line %d: Expected variable name\n", line);
                return NULL;
            }
            char *name = strdup(current_token(p)->value);
            advance(p);

            ASTNode *value = parse_expression(p);

            node = create_node(AST_SET, line);
            node->as.set.name = name;
            node->as.set.value = value;
            return node;
        }

        case TOKEN_WHILE: {
            int line = tok->line;
            advance(p);

            ASTNode *condition = parse_expression(p);
            ASTNode *body = parse_block(p);

            node = create_node(AST_WHILE, line);
            node->as.while_stmt.condition = condition;
            node->as.while_stmt.body = body;
            return node;
        }

        case TOKEN_FOR: {
            int line = tok->line;
            advance(p);

            if (!match(p, TOKEN_IDENTIFIER)) {
                fprintf(stderr, "Error at line %d: Expected loop variable\n", line);
                return NULL;
            }
            char *var_name = strdup(current_token(p)->value);
            advance(p);

            if (!expect(p, TOKEN_IN, "Expected 'in' in for loop")) {
                free(var_name);
                return NULL;
            }

            ASTNode *range_expr = parse_expression(p);
            ASTNode *body = parse_block(p);

            node = create_node(AST_FOR, line);
            node->as.for_stmt.var_name = var_name;
            node->as.for_stmt.range_expr = range_expr;
            node->as.for_stmt.body = body;
            return node;
        }

        case TOKEN_RETURN: {
            int line = tok->line;
            advance(p);

            ASTNode *value = NULL;
            if (!match(p, TOKEN_RBRACE)) {
                value = parse_expression(p);
            }

            node = create_node(AST_RETURN, line);
            node->as.return_stmt.value = value;
            return node;
        }

        case TOKEN_PRINT: {
            int line = tok->line;
            advance(p);

            ASTNode *expr = parse_expression(p);

            node = create_node(AST_PRINT, line);
            node->as.print.expr = expr;
            return node;
        }

        case TOKEN_ASSERT: {
            int line = tok->line;
            advance(p);

            ASTNode *condition = parse_expression(p);

            node = create_node(AST_ASSERT, line);
            node->as.assert.condition = condition;
            return node;
        }

        default:
            /* Try to parse as expression statement */
            return parse_expression(p);
    }
}

/* Parse function definition */
static ASTNode *parse_function(Parser *p) {
    int line = current_token(p)->line;

    if (!expect(p, TOKEN_FN, "Expected 'fn'")) {
        return NULL;
    }

    if (!match(p, TOKEN_IDENTIFIER)) {
        fprintf(stderr, "Error at line %d: Expected function name\n", line);
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

    Type return_type = parse_type(p);

    ASTNode *body = parse_block(p);
    if (!body) {
        free(name);
        free(params);
        return NULL;
    }

    ASTNode *node = create_node(AST_FUNCTION, line);
    node->as.function.name = name;
    node->as.function.params = params;
    node->as.function.param_count = param_count;
    node->as.function.return_type = return_type;
    node->as.function.body = body;
    return node;
}

/* Parse shadow-test block */
static ASTNode *parse_shadow(Parser *p) {
    int line = current_token(p)->line;

    if (!expect(p, TOKEN_SHADOW, "Expected 'shadow'")) {
        return NULL;
    }

    if (!match(p, TOKEN_IDENTIFIER)) {
        fprintf(stderr, "Error at line %d: Expected function name after 'shadow'\n", line);
        return NULL;
    }
    char *func_name = strdup(current_token(p)->value);
    advance(p);

    ASTNode *body = parse_block(p);
    if (!body) {
        free(func_name);
        return NULL;
    }

    ASTNode *node = create_node(AST_SHADOW, line);
    node->as.shadow.function_name = func_name;
    node->as.shadow.body = body;
    return node;
}

/* Parse top-level program */
ASTNode *parse_program(Token *tokens, int token_count) {
    Parser parser;
    parser.tokens = tokens;
    parser.count = token_count;
    parser.pos = 0;

    int capacity = 16;
    int count = 0;
    ASTNode **items = malloc(sizeof(ASTNode*) * capacity);

    while (!match(&parser, TOKEN_EOF)) {
        if (count >= capacity) {
            capacity *= 2;
            items = realloc(items, sizeof(ASTNode*) * capacity);
        }

        if (match(&parser, TOKEN_FN)) {
            items[count++] = parse_function(&parser);
        } else if (match(&parser, TOKEN_SHADOW)) {
            items[count++] = parse_shadow(&parser);
        } else {
            fprintf(stderr, "Error at line %d: Expected function or shadow-test definition\n",
                    current_token(&parser)->line);
            advance(&parser);
        }
    }

    ASTNode *program = create_node(AST_PROGRAM, 1);
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
            free_ast(node->as.function.body);
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
        default:
            break;
    }

    free(node);
}