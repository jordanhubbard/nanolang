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
    TOKEN_LBRACKET,
    TOKEN_RBRACKET,
    TOKEN_COMMA,
    TOKEN_COLON,
    TOKEN_ARROW,
    TOKEN_ASSIGN,
    TOKEN_DOT,

    /* Keywords */
    TOKEN_EXTERN,
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
    TOKEN_ARRAY,
    TOKEN_STRUCT,
    TOKEN_ENUM,

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
    int column;
} Token;

/* Forward declarations */
typedef struct Value Value;

/* Value types */
typedef enum {
    VAL_INT,
    VAL_FLOAT,
    VAL_BOOL,
    VAL_STRING,
    VAL_ARRAY,
    VAL_STRUCT,  /* NEW: Struct values */
    VAL_VOID
} ValueType;

/* Array structure */
typedef struct {
    ValueType element_type;  /* Type of elements in the array */
    int length;              /* Number of elements */
    int capacity;            /* Allocated capacity */
    void *data;              /* Pointer to array data */
} Array;

/* Struct value structure */
typedef struct {
    char *struct_name;       /* Name of struct type (e.g., "Point") */
    char **field_names;      /* Array of field names */
    Value *field_values;     /* Array of field values */
    int field_count;         /* Number of fields */
} StructValue;

/* Type information */
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_BOOL,
    TYPE_STRING,
    TYPE_VOID,
    TYPE_ARRAY,
    TYPE_STRUCT,
    TYPE_ENUM,
    TYPE_LIST_INT,
    TYPE_LIST_STRING,
    TYPE_UNKNOWN
} Type;

/* Extended type information for arrays */
typedef struct TypeInfo {
    Type base_type;
    struct TypeInfo *element_type;  /* For arrays: array<int> has element_type = int */
} TypeInfo;

/* Value structure */
struct Value {
    ValueType type;
    bool is_return;  /* Flag to propagate return statements through control flow */
    union {
        long long int_val;
        double float_val;
        bool bool_val;
        char *string_val;
        Array *array_val;
        StructValue *struct_val;  /* NEW: Struct values */
    } as;
};

/* AST node types */
typedef enum {
    AST_NUMBER,
    AST_FLOAT,
    AST_STRING,
    AST_BOOL,
    AST_IDENTIFIER,
    AST_PREFIX_OP,
    AST_CALL,
    AST_ARRAY_LITERAL,
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
    AST_ASSERT,
    AST_STRUCT_DEF,
    AST_STRUCT_LITERAL,
    AST_FIELD_ACCESS,
    AST_ENUM_DEF
} ASTNodeType;

/* Forward declaration */
typedef struct ASTNode ASTNode;

/* Parameter structure */
typedef struct {
    char *name;
    Type type;
    char *struct_type_name;  /* For TYPE_STRUCT: which struct (e.g., "Point") */
} Parameter;

/* AST node structure */
struct ASTNode {
    ASTNodeType type;
    int line;
    int column;
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
            ASTNode **elements;
            int element_count;
            Type element_type;  /* Type of array elements */
        } array_literal;
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
            char *return_struct_type_name;  /* For TYPE_STRUCT returns */
            ASTNode *body;
            bool is_extern;  /* NEW: Mark external C functions */
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
        struct {
            char *name;               // Struct name
            char **field_names;       // Array of field names
            Type *field_types;        // Array of field types
            int field_count;          // Number of fields
        } struct_def;
        struct {
            char *struct_name;        // Name of struct type
            char **field_names;       // Array of field names (for initialization)
            ASTNode **field_values;   // Array of field value expressions
            int field_count;          // Number of fields
        } struct_literal;
        struct {
            ASTNode *object;          // The struct instance expression
            char *field_name;         // The field being accessed
        } field_access;
        struct {
            char *name;               // Enum name
            char **variant_names;     // Array of variant names
            int *variant_values;      // Array of variant values (or NULL for auto)
            int variant_count;        // Number of variants
        } enum_def;
    } as;
};

/* Symbol table entry for variables */
typedef struct {
    char *name;
    Type type;
    char *struct_type_name;  /* For TYPE_STRUCT: which struct (e.g., "Point", "Color") */
    Type element_type;       /* For TYPE_ARRAY: element type (e.g., TYPE_INT for array<int>) */
    bool is_mut;
    Value value;
    bool is_used;  /* Track if variable is ever used */
    int def_line;   /* Line where variable was defined */
    int def_column; /* Column where variable was defined */
} Symbol;

/* Function table entry */
typedef struct {
    char *name;
    Parameter *params;
    int param_count;
    Type return_type;
    char *return_struct_type_name;  /* For TYPE_STRUCT returns: which struct */
    ASTNode *body;
    ASTNode *shadow_test;
    bool is_extern;  /* NEW: Mark external C functions */
} Function;

/* Struct definition entry */
typedef struct {
    char *name;
    char **field_names;
    Type *field_types;
    int field_count;
} StructDef;

/* Enum definition entry */
typedef struct {
    char *name;
    char **variant_names;
    int *variant_values;
    int variant_count;
} EnumDef;

/* Environment for variable and function storage */
typedef struct {
    Symbol *symbols;
    int symbol_count;
    int symbol_capacity;
    Function *functions;
    int function_count;
    int function_capacity;
    StructDef *structs;
    int struct_count;
    int struct_capacity;
    EnumDef *enums;
    int enum_count;
    int enum_capacity;
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

/* Interpreter */
bool run_program(ASTNode *program, Environment *env);
Value call_function(const char *name, Value *args, int arg_count, Environment *env);

/* C Transpiler */
char *transpile_to_c(ASTNode *program, Environment *env);

/* Environment */
Environment *create_environment(void);
void free_environment(Environment *env);
void env_define_var(Environment *env, const char *name, Type type, bool is_mut, Value value);
void env_define_var_with_element_type(Environment *env, const char *name, Type type, Type element_type, bool is_mut, Value value);
Symbol *env_get_var(Environment *env, const char *name);
void env_set_var(Environment *env, const char *name, Value value);
void env_define_function(Environment *env, Function func);
Function *env_get_function(Environment *env, const char *name);
bool is_builtin_function(const char *name);
void env_define_struct(Environment *env, StructDef struct_def);
StructDef *env_get_struct(Environment *env, const char *name);
void env_define_enum(Environment *env, EnumDef enum_def);
EnumDef *env_get_enum(Environment *env, const char *name);
int env_get_enum_variant(Environment *env, const char *variant_name);

/* Utilities */
Type token_to_type(TokenType token);
const char *type_to_string(Type type);
Value create_int(long long val);
Value create_float(double val);
Value create_bool(bool val);
Value create_string(const char *val);
Value create_void(void);
Value create_array(ValueType elem_type, int length, int capacity);
Value create_struct(const char *struct_name, char **field_names, Value *field_values, int field_count);

#endif /* NANOLANG_H */
