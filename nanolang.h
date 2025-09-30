#ifndef NANOLANG_H
#define NANOLANG_H

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

/* Token types */
typedef enum {
    TOKEN_EOF,
    TOKEN_NUMBER,
    TOKEN_IDENTIFIER,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_STAR,
    TOKEN_SLASH,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_ASSIGN,
    TOKEN_SEMICOLON,
    TOKEN_PRINT,
    TOKEN_LET,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_WHILE,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_EQ,
    TOKEN_LT,
    TOKEN_GT,
    TOKEN_TRUE,
    TOKEN_FALSE
} TokenType;

/* Token structure */
typedef struct {
    TokenType type;
    char *value;
    int line;
} Token;

/* Value types */
typedef enum {
    VAL_NUMBER,
    VAL_BOOL,
    VAL_NULL
} ValueType;

/* Value structure */
typedef struct {
    ValueType type;
    union {
        int number;
        bool boolean;
    } as;
} Value;

/* AST node types */
typedef enum {
    AST_NUMBER,
    AST_BOOL,
    AST_IDENTIFIER,
    AST_BINARY_OP,
    AST_ASSIGN,
    AST_PRINT,
    AST_LET,
    AST_BLOCK,
    AST_IF,
    AST_WHILE
} ASTNodeType;

/* Forward declaration */
typedef struct ASTNode ASTNode;

/* AST node structure */
struct ASTNode {
    ASTNodeType type;
    union {
        int number;
        bool boolean;
        char *identifier;
        struct {
            TokenType op;
            ASTNode *left;
            ASTNode *right;
        } binary;
        struct {
            char *name;
            ASTNode *value;
        } assign;
        struct {
            ASTNode *expr;
        } print;
        struct {
            char *name;
            ASTNode *value;
        } let;
        struct {
            ASTNode **statements;
            int count;
        } block;
        struct {
            ASTNode *condition;
            ASTNode *then_branch;
            ASTNode *else_branch;
        } if_stmt;
        struct {
            ASTNode *condition;
            ASTNode *body;
        } while_stmt;
    } as;
};

/* Symbol table entry */
typedef struct {
    char *name;
    Value value;
} Symbol;

/* Environment for variable storage */
typedef struct {
    Symbol *symbols;
    int count;
    int capacity;
} Environment;

/* Function declarations */

/* Lexer */
Token *tokenize(const char *source, int *token_count);
void free_tokens(Token *tokens, int count);

/* Parser */
ASTNode *parse(Token *tokens, int token_count, int *pos);
ASTNode *parse_statement(Token *tokens, int token_count, int *pos);
ASTNode *parse_expression(Token *tokens, int token_count, int *pos);
void free_ast(ASTNode *node);

/* Evaluator */
Value eval(ASTNode *node, Environment *env);
void print_value(Value val);

/* Environment */
Environment *create_environment(void);
void free_environment(Environment *env);
void env_set(Environment *env, const char *name, Value value);
Value env_get(Environment *env, const char *name, bool *found);

/* REPL */
void repl(void);

#endif /* NANOLANG_H */
