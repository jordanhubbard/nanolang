#ifndef NANOLANG_H
#define NANOLANG_H

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdarg.h>
#include <assert.h>

/* Backtrace support for assertions */
#ifdef __APPLE__
#include <execinfo.h>
#include <unistd.h>
#elif defined(__linux__)
#include <execinfo.h>
#endif

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
    /* TOKEN_PRINT removed - print/println are regular built-in functions */
    TOKEN_ARRAY,
    TOKEN_STRUCT,
    TOKEN_ENUM,
    TOKEN_UNION,
    TOKEN_MATCH,
    TOKEN_IMPORT,

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
    TOKEN_NOT
    /* TOKEN_RANGE removed - range is a regular built-in function */
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
struct FunctionSignature;  /* Forward declaration - typedef defined with struct */

/* Include dynamic array and GC struct for GC support - needs to be after Value forward declaration */
#include "runtime/dyn_array.h"
#include "runtime/gc_struct.h"

/* Value types */
typedef enum {
    VAL_INT,
    VAL_FLOAT,
    VAL_BOOL,
    VAL_STRING,
    VAL_ARRAY,
    VAL_DYN_ARRAY,  /* Dynamic arrays (GC-managed) */
    VAL_STRUCT,     /* Struct values (stack-allocated) */
    VAL_GC_STRUCT,  /* GC-managed struct (heap-allocated) */
    VAL_UNION,      /* Union values (tagged unions) */
    VAL_FUNCTION,   /* Function values (for first-class functions) */
    VAL_TUPLE,      /* Tuple values: (1, "hello", true) */
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

/* Union value (tagged union) */
typedef struct {
    char *union_name;        /* Name of union type (e.g., "Status", "Result") */
    int variant_index;       /* Index of the active variant */
    char *variant_name;      /* Name of the active variant (e.g., "Ok", "Error") */
    char **field_names;      /* Array of field names for this variant */
    Value *field_values;     /* Array of field values for this variant */
    int field_count;         /* Number of fields in this variant */
} UnionValue;

/* Tuple value structure */
typedef struct {
    Value *elements;          /* Array of values */
    int element_count;        /* Number of elements */
} TupleValue;

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
    TYPE_UNION,
    TYPE_GENERIC,      /* Generic type (e.g., List<T> before instantiation) */
    TYPE_LIST_INT,
    TYPE_LIST_STRING,
    TYPE_LIST_TOKEN,
    TYPE_LIST_GENERIC, /* Generic list with user-defined type: List<Point>, List<Player>, etc. */
    TYPE_FUNCTION,     /* Function type: fn(int, int) -> int */
    TYPE_TUPLE,        /* Tuple type: (int, string, bool) */
    TYPE_UNKNOWN
} Type;

/* Extended type information for arrays and generics */
typedef struct TypeInfo {
    Type base_type;
    struct TypeInfo *element_type;  /* For arrays: array<int> has element_type = int */
    
    /* For generic types: List<int> */
    char *generic_name;              /* e.g., "List" */
    struct TypeInfo **type_params;   /* e.g., [TypeInfo{TYPE_INT}] */
    int type_param_count;            /* Number of type parameters */
    
    /* For tuple types: (int, string, bool) */
    Type *tuple_types;               /* Array of tuple element types */
    char **tuple_type_names;         /* For struct/enum/union tuple elements */
    int tuple_element_count;         /* Number of tuple elements */
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
        DynArray *dyn_array_val;    /* Dynamic array (GC-managed) */
        StructValue *struct_val;    /* Struct values (stack) */
        GCStruct *gc_struct_val;    /* GC-managed struct (heap) */
        UnionValue *union_val;      /* Union values (tagged unions) */
        struct {
            char *function_name;       /* Name of the function */
            struct FunctionSignature *signature;  /* Function signature */
        } function_val;  /* Function values */
        TupleValue *tuple_val;  /* Tuple values */
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
    AST_ENUM_DEF,
    AST_UNION_DEF,
    AST_UNION_CONSTRUCT,
    AST_MATCH,
    AST_IMPORT,
    AST_TUPLE_LITERAL,     /* Tuple literal: (1, "hello", true) */
    AST_TUPLE_INDEX        /* Tuple index access: tuple.0, tuple.1 */
} ASTNodeType;

/* Forward declaration */
typedef struct ASTNode ASTNode;

/* Function signature for function types: fn(int, string) -> bool */
typedef struct FunctionSignature {
    Type *param_types;           /* Array of parameter types */
    int param_count;             /* Number of parameters */
    char **param_struct_names;   /* For struct/enum/union parameters */
    Type return_type;            /* Return type */
    char *return_struct_name;    /* For struct/enum/union return */
} FunctionSignature;

/* Parameter structure */
typedef struct {
    char *name;
    Type type;
    char *struct_type_name;  /* For TYPE_STRUCT: which struct (e.g., "Point") */
    Type element_type;       /* For TYPE_ARRAY: element type (e.g., TYPE_INT for array<int>) */
    FunctionSignature *fn_sig;   /* For TYPE_FUNCTION: function signature */
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
            char *type_name;         /* For TYPE_STRUCT/TYPE_UNION: actual type name (e.g., "Status", "Point") */
            Type element_type;       /* For TYPE_ARRAY: element type (e.g., TYPE_INT for array<int>) */
            FunctionSignature *fn_sig;  /* For TYPE_FUNCTION: function signature */
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
            FunctionSignature *return_fn_sig;  /* For TYPE_FUNCTION returns */
            TypeInfo *return_type_info;  /* For TYPE_TUPLE returns: stores element types */
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
        
        /* Union definition: union Color { Red {}, Blue { intensity: int } } */
        struct {
            char *name;
            int variant_count;
            char **variant_names;
            int *variant_field_counts;
            char ***variant_field_names;
            Type **variant_field_types;
        } union_def;
        
        /* Union construction: Color.Red {} or Color.Blue { intensity: 5 } */
        struct {
            char *union_name;
            char *variant_name;
            int field_count;
            char **field_names;
            ASTNode **field_values;
        } union_construct;
        
        /* Match expression: match c { Red(r) => ..., Blue(b) => ... } */
        struct {
            ASTNode *expr;
            int arm_count;
            char **pattern_variants;
            char **pattern_bindings;
            ASTNode **arm_bodies;
            char *union_type_name;  /* Filled during typechecking */
        } match_expr;
        /* Import statement: import "module.nano" or import module */
        struct {
            char *module_path;  /* Path to module file (e.g., "math.nano" or "utils/math.nano") */
            char *module_name;  /* Optional module name/alias */
        } import_stmt;
        /* Tuple literal: (1, "hello", true) */
        struct {
            ASTNode **elements;    /* Array of element expressions */
            int element_count;     /* Number of elements */
            Type *element_types;   /* Types of each element (filled by type checker) */
        } tuple_literal;
        /* Tuple index access: tuple.0, tuple.1 */
        struct {
            ASTNode *tuple;        /* The tuple expression */
            int index;             /* The index being accessed (0, 1, 2, ...) */
        } tuple_index;
    } as;
};

/* Symbol table entry for variables */
typedef struct {
    char *name;
    Type type;
    char *struct_type_name;  /* For TYPE_STRUCT: which struct (e.g., "Point", "Color") */
    Type element_type;       /* For TYPE_ARRAY: element type (e.g., TYPE_INT for array<int>) */
    TypeInfo *type_info;     /* For complex types (tuples, generics, etc.) - full type information */
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
    FunctionSignature *return_fn_sig;  /* For TYPE_FUNCTION returns: function signature */
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

/* Union definition entry */
typedef struct {
    char *name;
    int variant_count;
    char **variant_names;
    int *variant_field_counts;
    char ***variant_field_names;
    Type **variant_field_types;
} UnionDef;

/* Generic type instantiation (for monomorphization) */
typedef struct {
    char *generic_name;        /* e.g., "List" */
    Type *type_args;           /* e.g., [TYPE_INT] or [TYPE_LIST_GENERIC] */
    int type_arg_count;        /* Number of type arguments */
    char **type_arg_names;     /* e.g., ["Point"] for user types, NULL for primitives */
    char *concrete_name;       /* e.g., "List_int" or "List_Point" (generated name) */
} GenericInstantiation;

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
    UnionDef *unions;
    int union_count;
    int union_capacity;
    GenericInstantiation *generic_instances;
    int generic_instance_count;
    int generic_instance_capacity;
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
bool type_check_module(ASTNode *program, Environment *env);  /* Type check without requiring main */
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
void env_define_var_with_type_info(Environment *env, const char *name, Type type, Type element_type, TypeInfo *type_info, bool is_mut, Value value);
Symbol *env_get_var(Environment *env, const char *name);
void env_set_var(Environment *env, const char *name, Value value);
void env_define_function(Environment *env, Function func);
Function *env_get_function(Environment *env, const char *name);
bool is_builtin_function(const char *name);
void env_define_struct(Environment *env, StructDef struct_def);
StructDef *env_get_struct(Environment *env, const char *name);
void env_define_enum(Environment *env, EnumDef enum_def);
EnumDef *env_get_enum(Environment *env, const char *name);
void env_register_list_instantiation(Environment *env, const char *element_type);
int env_get_enum_variant(Environment *env, const char *variant_name);
void env_define_union(Environment *env, UnionDef union_def);
UnionDef *env_get_union(Environment *env, const char *name);
int env_get_union_variant_index(Environment *env, const char *union_name, const char *variant_name);

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
Value create_union(const char *union_name, int variant_index, const char *variant_name, 
                   char **field_names, Value *field_values, int field_count);
Value create_function(const char *function_name, FunctionSignature *signature);

/* Function signature helpers */
FunctionSignature *create_function_signature(Type *param_types, int param_count, Type return_type);
void free_function_signature(FunctionSignature *sig);
bool function_signatures_equal(FunctionSignature *sig1, FunctionSignature *sig2);

/* Tuple helpers */
Value create_tuple(Value *elements, int element_count);
void free_tuple(TupleValue *tuple);

/* Module loading */
typedef struct {
    char **module_paths;
    int count;
    int capacity;
} ModuleList;

ModuleList *create_module_list(void);
void free_module_list(ModuleList *list);
void module_list_add(ModuleList *list, const char *module_path);
char *resolve_module_path(const char *module_path, const char *current_file);
char *find_module_in_paths(const char *module_name);
char *unpack_module_package(const char *package_path, char *temp_dir_out, size_t temp_dir_size);
ASTNode *load_module(const char *module_path, Environment *env);
ASTNode *load_module_from_package(const char *package_path, Environment *env, char *temp_dir_out, size_t temp_dir_size);
bool process_imports(ASTNode *program, Environment *env, ModuleList *modules, const char *current_file);
bool compile_module_to_object(const char *module_path, const char *output_obj, Environment *env, bool verbose);
bool compile_modules(ModuleList *modules, Environment *env, char *module_objs_buffer, size_t buffer_size, char *compile_flags_buffer, size_t compile_flags_buffer_size, bool verbose);

/* Module metadata for serialization */
typedef struct {
    char *module_name;
    int function_count;
    Function *functions;  /* Array of function signatures */
    int struct_count;
    StructDef *structs;
    int enum_count;
    EnumDef *enums;
    int union_count;
    UnionDef *unions;
} ModuleMetadata;

/* Module metadata serialization */
ModuleMetadata *extract_module_metadata(Environment *env, const char *module_name);
void free_module_metadata(ModuleMetadata *meta);
char *serialize_module_metadata_to_c(ModuleMetadata *meta);
bool deserialize_module_metadata_from_c(const char *c_code, ModuleMetadata **meta_out);
bool embed_metadata_in_module_c(char *c_code, ModuleMetadata *meta, size_t buffer_size);

/* Safe string utility functions - always use these instead of unsafe libc functions */
/* These functions handle NULL pointers gracefully and use bounded operations */

/* Safe strlen - returns 0 if str is NULL */
static inline size_t safe_strlen(const char *str) {
    return str ? strlen(str) : 0;
}

/* Safe strnlen - returns 0 if str is NULL, bounded by maxlen */
static inline size_t safe_strnlen(const char *str, size_t maxlen) {
    if (!str) return 0;
    size_t len = 0;
    while (len < maxlen && str[len] != '\0') len++;
    return len;
}

/* Safe strcmp - returns 0 if either string is NULL, otherwise compares */
static inline int safe_strcmp(const char *s1, const char *s2) {
    if (!s1 && !s2) return 0;
    if (!s1 || !s2) return (s1 ? 1 : -1);
    return strcmp(s1, s2);
}

/* Safe strncmp - returns 0 if either string is NULL, otherwise compares up to n chars */
static inline int safe_strncmp(const char *s1, const char *s2, size_t n) {
    if (!s1 && !s2) return 0;
    if (!s1 || !s2) return (s1 ? 1 : -1);
    return strncmp(s1, s2, n);
}

/* Safe strcpy replacement - use strncpy with explicit null termination */
static inline char *safe_strncpy(char *dest, const char *src, size_t dest_size) {
    if (!dest || dest_size == 0) return dest;
    if (!src) {
        dest[0] = '\0';
        return dest;
    }
    size_t src_len = safe_strnlen(src, dest_size - 1);
    strncpy(dest, src, src_len);
    dest[src_len] = '\0';
    return dest;
}

/* Safe strcat replacement - use strncat with bounds checking */
static inline char *safe_strncat(char *dest, const char *src, size_t dest_size) {
    if (!dest || dest_size == 0) return dest;
    if (!src) return dest;
    size_t dest_len = safe_strnlen(dest, dest_size);
    if (dest_len >= dest_size) return dest; /* No room */
    size_t src_len = safe_strnlen(src, dest_size - dest_len - 1);
    strncat(dest, src, src_len);
    return dest;
}

/* Helper to sanitize string arguments for format strings - replaces NULL with "(NULL)" */
static inline const char *safe_format_string(const char *str) {
    return str ? str : "(NULL)";
}

/* Safe fprintf wrapper - uses vsnprintf with buffer and NULL checks */
/* NOTE: For %s format specifiers, use safe_format_string() on string arguments */
static inline int safe_fprintf(FILE *stream, const char *format, ...) {
    assert(stream != NULL);
    assert(format != NULL);
    if (!stream || !format) return -1;
    
    char buffer[4096];
    va_list args;
    va_start(args, format);
    int result = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    if (result < 0) return -1;
    if (result >= (int)sizeof(buffer)) {
        /* Truncated - use a larger buffer */
        char *large_buffer = malloc(result + 1);
        if (!large_buffer) return -1;
        va_start(args, format);
        vsnprintf(large_buffer, result + 1, format, args);
        va_end(args);
        fputs(large_buffer, stream);
        free(large_buffer);
        return result;
    }
    
    fputs(buffer, stream);
    return result;
}

/* Safe sprintf replacement - always use snprintf with explicit size */
static inline int safe_snprintf(char *dest, size_t dest_size, const char *format, ...) {
    assert(dest != NULL);
    assert(format != NULL);
    if (!dest || dest_size == 0 || !format) return -1;
    
    va_list args;
    va_start(args, format);
    int result = vsnprintf(dest, dest_size, format, args);
    va_end(args);
    
    if (result >= (int)dest_size) {
        dest[dest_size - 1] = '\0'; /* Ensure null termination */
    }
    return result;
}

/* Safe printf wrapper */
static inline int safe_printf(const char *format, ...) {
    assert(format != NULL);
    if (!format) return -1;
    
    char buffer[4096];
    va_list args;
    va_start(args, format);
    int result = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    if (result < 0) return -1;
    if (result >= (int)sizeof(buffer)) {
        char *large_buffer = malloc(result + 1);
        if (!large_buffer) return -1;
        va_start(args, format);
        vsnprintf(large_buffer, result + 1, format, args);
        va_end(args);
        fputs(large_buffer, stdout);
        free(large_buffer);
        return result;
    }
    
    fputs(buffer, stdout);
    return result;
}

/* Print backtrace to stderr */
static inline void print_backtrace(void) {
#if defined(__APPLE__) || defined(__linux__)
    void *array[64];
    int size = backtrace(array, 64);
    char **symbols = backtrace_symbols(array, size);
    
    if (symbols) {
        safe_fprintf(stderr, "\n=== Backtrace ===\n");
        for (int i = 0; i < size; i++) {
            safe_fprintf(stderr, "  [%d] %s\n", i, safe_format_string(symbols[i]));
        }
        safe_fprintf(stderr, "==================\n\n");
        free(symbols);
    } else {
        safe_fprintf(stderr, "\n=== Backtrace (symbols unavailable) ===\n");
        safe_fprintf(stderr, "  %d stack frames\n", size);
        safe_fprintf(stderr, "========================================\n\n");
    }
#else
    safe_fprintf(stderr, "\n=== Backtrace not available on this platform ===\n\n");
#endif
}

/* Don't redefine assert - use standard library version */
/* Enhanced assert macro with backtrace */
#ifdef NDEBUG
/* #define assert(expr) ((void)0) */
#else
/* #define assert(expr) \
    do { \
        if (!(expr)) { \
            safe_fprintf(stderr, "\nAssertion failed: %s\n", #expr); \
            safe_fprintf(stderr, "File: %s, Line: %d\n", __FILE__, __LINE__); \
            print_backtrace(); \
            abort(); \
        } \
    } while(0) */
#endif

#endif /* NANOLANG_H */
