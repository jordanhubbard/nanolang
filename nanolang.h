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
    TOKEN_FLOAT,
    TOKEN_STRING,
    TOKEN_IDENTIFIER,
    TOKEN_TRUE,
    TOKEN_FALSE,

    /* Delimiters */
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_COMMA,
    TOKEN_COLON,
    TOKEN_ARROW,
    TOKEN_ASSIGN,

    /* Keywords */
    TOKEN_FN,
    TOKEN_LET,
    TOKEN_MUT,
    TOKEN_SET,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_WHILE,
    TOKEN_FOR,
    TOKEN_IN,
    TOKEN_RETURN,
    TOKEN_ASSERT,
    TOKEN_SHADOW,
    TOKEN_PRINT,

    /* Types */
    TOKEN_TYPE_INT,
    TOKEN_TYPE_FLOAT,
    TOKEN_TYPE_BOOL,
    TOKEN_TYPE_STRING,
    TOKEN_TYPE_VOID,

    /* Operators (as identifiers in prefix position) */
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_STAR,
    TOKEN_SLASH,
    TOKEN_PERCENT,
    TOKEN_EQ,
    TOKEN_NE,
    TOKEN_LT,
    TOKEN_LE,
    TOKEN_GT,
    TOKEN_GE,
    TOKEN_AND,
    TOKEN_OR,
    TOKEN_NOT,
    TOKEN_RANGE
} TokenType;

/* Token structure */
typedef struct {
    TokenType type;
    char *value;
    int line;
} Token;

/* Value types */
typedef enum {
    VAL_INT,
    VAL_FLOAT,
    VAL_BOOL,
    VAL_STRING,
    VAL_VOID
} ValueType;

/* Type information */
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_BOOL,
    TYPE_STRING,
    TYPE_VOID,
    TYPE_UNKNOWN
} Type;

/* Value structure */
typedef struct {
    ValueType type;
    union {
        long long int_val;
        double float_val;
        bool bool_val;
        char *string_val;
    } as;
} Value;

/* AST node types */
typedef enum {
    AST_NUMBER,
    AST_FLOAT,
    AST_STRING,
    AST_BOOL,
    AST_IDENTIFIER,
    AST_PREFIX_OP,
    AST_CALL,
    AST_LET,
    AST_SET,
    AST_IF,
    AST_WHILE,
    AST_FOR,
    AST_RETURN,
    AST_BLOCK,
    AST_FUNCTION,
    AST_SHADOW,
    AST_PROGRAM,
    AST_PRINT,
    AST_ASSERT
} ASTNodeType;

/* Forward declaration */
typedef struct ASTNode ASTNode;

/* Parameter structure */
typedef struct {
    char *name;
    Type type;
} Parameter;

/* AST node structure */
struct ASTNode {
    ASTNodeType type;
    int line;
    union {
        long long number;
        double float_val;
        char *string_val;
        bool bool_val;
        char *identifier;
        struct {
            TokenType op;
            ASTNode **args;
            int arg_count;
        } prefix_op;
        struct {
            char *name;
            ASTNode **args;
            int arg_count;
        } call;
        struct {
            char *name;
            Type var_type;
            bool is_mut;
            ASTNode *value;
        } let;
        struct {
            char *name;
            ASTNode *value;
        } set;
        struct {
            ASTNode *condition;
            ASTNode *then_branch;
            ASTNode *else_branch;
        } if_stmt;
        struct {
            ASTNode *condition;
            ASTNode *body;
        } while_stmt;
        struct {
            char *var_name;
            ASTNode *range_expr;
            ASTNode *body;
        } for_stmt;
        struct {
            ASTNode *value;
        } return_stmt;
        struct {
            ASTNode **statements;
            int count;
        } block;
        struct {
            char *name;
            Parameter *params;
            int param_count;
            Type return_type;
            ASTNode *body;
        } function;
        struct {
            char *function_name;
            ASTNode *body;
        } shadow;
        struct {
            ASTNode **items;
            int count;
        } program;
        struct {
            ASTNode *expr;
        } print;
        struct {
            ASTNode *condition;
        } assert;
    } as;
};

/* Symbol table entry for variables */
typedef struct {
    char *name;
    Type type;
    bool is_mut;
    Value value;
} Symbol;

/* Function table entry */
typedef struct {
    char *name;
    Parameter *params;
    int param_count;
    Type return_type;
    ASTNode *body;
    ASTNode *shadow_test;
} Function;

/* Environment for variable and function storage */
typedef struct {
    Symbol *symbols;
    int symbol_count;
    int symbol_capacity;
    Function *functions;
    int function_count;
    int function_capacity;
} Environment;

/* Function declarations */

/* Lexer */
Token *tokenize(const char *source, int *token_count);
void free_tokens(Token *tokens, int count);
const char *token_type_name(TokenType type);

/* Parser */
ASTNode *parse_program(Token *tokens, int token_count);
void free_ast(ASTNode *node);

/* Type Checker */
bool type_check(ASTNode *program, Environment *env);
Type check_expression(ASTNode *expr, Environment *env);

/* Shadow-Test Runner */
bool run_shadow_tests(ASTNode *program, Environment *env);

/* C Transpiler */
char *transpile_to_c(ASTNode *program);

/* Environment */
Environment *create_environment(void);
void free_environment(Environment *env);
void env_define_var(Environment *env, const char *name, Type type, bool is_mut, Value value);
Symbol *env_get_var(Environment *env, const char *name);
void env_set_var(Environment *env, const char *name, Value value);
void env_define_function(Environment *env, Function func);
Function *env_get_function(Environment *env, const char *name);

/* Utilities */
Type token_to_type(TokenType token);
const char *type_to_string(Type type);
Value create_int(long long val);
Value create_float(double val);
Value create_bool(bool val);
Value create_string(const char *val);
Value create_void(void);

#endif /* NANOLANG_H */
