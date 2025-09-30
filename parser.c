#include "nanolang.h"

/* Helper to create AST nodes */
static ASTNode *create_node(ASTNodeType type) {
    ASTNode *node = malloc(sizeof(ASTNode));
    node->type = type;
    return node;
}

/* Forward declarations */
static ASTNode *parse_comparison(Token *tokens, int token_count, int *pos);
static ASTNode *parse_term(Token *tokens, int token_count, int *pos);
static ASTNode *parse_factor(Token *tokens, int token_count, int *pos);
static ASTNode *parse_primary(Token *tokens, int token_count, int *pos);

/* Parse primary expression (numbers, identifiers, parentheses) */
static ASTNode *parse_primary(Token *tokens, int token_count, int *pos) {
    if (*pos >= token_count) return NULL;
    
    Token current = tokens[*pos];
    
    if (current.type == TOKEN_NUMBER) {
        (*pos)++;
        ASTNode *node = create_node(AST_NUMBER);
        node->as.number = atoi(current.value);
        return node;
    }
    
    if (current.type == TOKEN_TRUE) {
        (*pos)++;
        ASTNode *node = create_node(AST_BOOL);
        node->as.boolean = true;
        return node;
    }
    
    if (current.type == TOKEN_FALSE) {
        (*pos)++;
        ASTNode *node = create_node(AST_BOOL);
        node->as.boolean = false;
        return node;
    }
    
    if (current.type == TOKEN_IDENTIFIER) {
        (*pos)++;
        ASTNode *node = create_node(AST_IDENTIFIER);
        node->as.identifier = strdup(current.value);
        return node;
    }
    
    if (current.type == TOKEN_LPAREN) {
        (*pos)++;
        ASTNode *expr = parse_expression(tokens, token_count, pos);
        if (*pos < token_count && tokens[*pos].type == TOKEN_RPAREN) {
            (*pos)++;
        }
        return expr;
    }
    
    return NULL;
}

/* Parse factor (multiplication and division) */
static ASTNode *parse_factor(Token *tokens, int token_count, int *pos) {
    ASTNode *left = parse_primary(tokens, token_count, pos);
    
    while (*pos < token_count) {
        Token current = tokens[*pos];
        if (current.type != TOKEN_STAR && current.type != TOKEN_SLASH) {
            break;
        }
        
        TokenType op = current.type;
        (*pos)++;
        ASTNode *right = parse_primary(tokens, token_count, pos);
        
        ASTNode *node = create_node(AST_BINARY_OP);
        node->as.binary.op = op;
        node->as.binary.left = left;
        node->as.binary.right = right;
        left = node;
    }
    
    return left;
}

/* Parse term (addition and subtraction) */
static ASTNode *parse_term(Token *tokens, int token_count, int *pos) {
    ASTNode *left = parse_factor(tokens, token_count, pos);
    
    while (*pos < token_count) {
        Token current = tokens[*pos];
        if (current.type != TOKEN_PLUS && current.type != TOKEN_MINUS) {
            break;
        }
        
        TokenType op = current.type;
        (*pos)++;
        ASTNode *right = parse_factor(tokens, token_count, pos);
        
        ASTNode *node = create_node(AST_BINARY_OP);
        node->as.binary.op = op;
        node->as.binary.left = left;
        node->as.binary.right = right;
        left = node;
    }
    
    return left;
}

/* Parse comparison expressions */
static ASTNode *parse_comparison(Token *tokens, int token_count, int *pos) {
    ASTNode *left = parse_term(tokens, token_count, pos);
    
    while (*pos < token_count) {
        Token current = tokens[*pos];
        if (current.type != TOKEN_EQ && current.type != TOKEN_LT && current.type != TOKEN_GT) {
            break;
        }
        
        TokenType op = current.type;
        (*pos)++;
        ASTNode *right = parse_term(tokens, token_count, pos);
        
        ASTNode *node = create_node(AST_BINARY_OP);
        node->as.binary.op = op;
        node->as.binary.left = left;
        node->as.binary.right = right;
        left = node;
    }
    
    return left;
}

/* Parse expression */
ASTNode *parse_expression(Token *tokens, int token_count, int *pos) {
    return parse_comparison(tokens, token_count, pos);
}

/* Parse statement */
ASTNode *parse_statement(Token *tokens, int token_count, int *pos) {
    if (*pos >= token_count) return NULL;
    
    Token current = tokens[*pos];
    
    /* Print statement */
    if (current.type == TOKEN_PRINT) {
        (*pos)++;
        ASTNode *expr = parse_expression(tokens, token_count, pos);
        if (*pos < token_count && tokens[*pos].type == TOKEN_SEMICOLON) {
            (*pos)++;
        }
        ASTNode *node = create_node(AST_PRINT);
        node->as.print.expr = expr;
        return node;
    }
    
    /* Let statement */
    if (current.type == TOKEN_LET) {
        (*pos)++;
        if (*pos >= token_count || tokens[*pos].type != TOKEN_IDENTIFIER) {
            fprintf(stderr, "Error: Expected identifier after 'let'\n");
            return NULL;
        }
        char *name = strdup(tokens[*pos].value);
        (*pos)++;
        
        if (*pos >= token_count || tokens[*pos].type != TOKEN_ASSIGN) {
            fprintf(stderr, "Error: Expected '=' after identifier\n");
            free(name);
            return NULL;
        }
        (*pos)++;
        
        ASTNode *value = parse_expression(tokens, token_count, pos);
        if (*pos < token_count && tokens[*pos].type == TOKEN_SEMICOLON) {
            (*pos)++;
        }
        
        ASTNode *node = create_node(AST_LET);
        node->as.let.name = name;
        node->as.let.value = value;
        return node;
    }
    
    /* If statement */
    if (current.type == TOKEN_IF) {
        (*pos)++;
        if (*pos < token_count && tokens[*pos].type == TOKEN_LPAREN) {
            (*pos)++;
        }
        ASTNode *condition = parse_expression(tokens, token_count, pos);
        if (*pos < token_count && tokens[*pos].type == TOKEN_RPAREN) {
            (*pos)++;
        }
        
        ASTNode *then_branch = parse_statement(tokens, token_count, pos);
        ASTNode *else_branch = NULL;
        
        if (*pos < token_count && tokens[*pos].type == TOKEN_ELSE) {
            (*pos)++;
            else_branch = parse_statement(tokens, token_count, pos);
        }
        
        ASTNode *node = create_node(AST_IF);
        node->as.if_stmt.condition = condition;
        node->as.if_stmt.then_branch = then_branch;
        node->as.if_stmt.else_branch = else_branch;
        return node;
    }
    
    /* While statement */
    if (current.type == TOKEN_WHILE) {
        (*pos)++;
        if (*pos < token_count && tokens[*pos].type == TOKEN_LPAREN) {
            (*pos)++;
        }
        ASTNode *condition = parse_expression(tokens, token_count, pos);
        if (*pos < token_count && tokens[*pos].type == TOKEN_RPAREN) {
            (*pos)++;
        }
        
        ASTNode *body = parse_statement(tokens, token_count, pos);
        
        ASTNode *node = create_node(AST_WHILE);
        node->as.while_stmt.condition = condition;
        node->as.while_stmt.body = body;
        return node;
    }
    
    /* Block statement */
    if (current.type == TOKEN_LBRACE) {
        (*pos)++;
        int capacity = 8;
        int count = 0;
        ASTNode **statements = malloc(sizeof(ASTNode*) * capacity);
        
        while (*pos < token_count && tokens[*pos].type != TOKEN_RBRACE) {
            if (count >= capacity) {
                capacity *= 2;
                statements = realloc(statements, sizeof(ASTNode*) * capacity);
            }
            ASTNode *stmt = parse_statement(tokens, token_count, pos);
            if (stmt) {
                statements[count++] = stmt;
            }
        }
        
        if (*pos < token_count && tokens[*pos].type == TOKEN_RBRACE) {
            (*pos)++;
        }
        
        ASTNode *node = create_node(AST_BLOCK);
        node->as.block.statements = statements;
        node->as.block.count = count;
        return node;
    }
    
    /* Assignment or expression statement */
    if (current.type == TOKEN_IDENTIFIER) {
        int saved_pos = *pos;
        (*pos)++;
        
        if (*pos < token_count && tokens[*pos].type == TOKEN_ASSIGN) {
            char *name = strdup(tokens[saved_pos].value);
            (*pos)++;
            ASTNode *value = parse_expression(tokens, token_count, pos);
            if (*pos < token_count && tokens[*pos].type == TOKEN_SEMICOLON) {
                (*pos)++;
            }
            
            ASTNode *node = create_node(AST_ASSIGN);
            node->as.assign.name = name;
            node->as.assign.value = value;
            return node;
        }
        
        /* Restore position and parse as expression */
        *pos = saved_pos;
    }
    
    /* Expression statement */
    ASTNode *expr = parse_expression(tokens, token_count, pos);
    if (*pos < token_count && tokens[*pos].type == TOKEN_SEMICOLON) {
        (*pos)++;
    }
    return expr;
}

/* Parse tokens into AST */
ASTNode *parse(Token *tokens, int token_count, int *pos) {
    return parse_statement(tokens, token_count, pos);
}

/* Free AST recursively */
void free_ast(ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_IDENTIFIER:
            free(node->as.identifier);
            break;
        case AST_BINARY_OP:
            free_ast(node->as.binary.left);
            free_ast(node->as.binary.right);
            break;
        case AST_ASSIGN:
            free(node->as.assign.name);
            free_ast(node->as.assign.value);
            break;
        case AST_PRINT:
            free_ast(node->as.print.expr);
            break;
        case AST_LET:
            free(node->as.let.name);
            free_ast(node->as.let.value);
            break;
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++) {
                free_ast(node->as.block.statements[i]);
            }
            free(node->as.block.statements);
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
        default:
            break;
    }
    
    free(node);
}
