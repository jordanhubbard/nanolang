#include "nanolang.h"
#include "module_builder.h"
#include <stdarg.h>
#include <libgen.h>

/* String builder for C code generation */
typedef struct {
    char *buffer;
    int length;
    int capacity;
} StringBuilder;

static StringBuilder *sb_create(void) {
    StringBuilder *sb = malloc(sizeof(StringBuilder));
    sb->capacity = 1024;
    sb->length = 0;
    sb->buffer = malloc(sb->capacity);
    sb->buffer[0] = '\0';
    return sb;
}

static void sb_append(StringBuilder *sb, const char *str) {
    assert(str != NULL);
    int len = safe_strlen(str);
    while (sb->length + len >= sb->capacity) {
        /* Check for overflow before doubling capacity */
        if (sb->capacity > SIZE_MAX / 2) {
            fprintf(stderr, "Error: StringBuilder capacity overflow\n");
            exit(1);
        }
        int new_capacity = sb->capacity * 2;
        char *new_buffer = realloc(sb->buffer, new_capacity);
        if (!new_buffer) {
            fprintf(stderr, "Error: Out of memory in StringBuilder\n");
            exit(1);
        }
        sb->buffer = new_buffer;
        sb->capacity = new_capacity;
    }
    safe_strncpy(sb->buffer + sb->length, str, sb->capacity - sb->length);
    sb->length += len;
}

static void sb_appendf(StringBuilder *sb, const char *fmt, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    sb_append(sb, buffer);
}

/* Forward declarations */
static const char *type_to_c(Type type);
extern Type check_expression(ASTNode *expr, Environment *env);  /* From typechecker.c */
extern const char *get_struct_type_name(ASTNode *expr, Environment *env);  /* From typechecker.c */
extern StructDef *env_get_struct(Environment *env, const char *name);  /* From env.c */

/* Generate indentation */
static void emit_indent(StringBuilder *sb, int indent) {
    for (int i = 0; i < indent; i++) {
        sb_append(sb, "    ");
    }
}

/* Check if a struct/enum name is a runtime-provided typedef (not a user-defined type) */
static bool is_runtime_typedef(const char *name) {
    /* Runtime typedefs that don't use 'struct' keyword */
    return strcmp(name, "List_int") == 0 ||
           strcmp(name, "List_string") == 0 ||
           strcmp(name, "List_token") == 0;
}

/* Check if an enum/struct name would conflict with C runtime types */
static bool conflicts_with_runtime(const char *name) {
    /* These are defined in nanolang.h and would cause conflicts */
    return strcmp(name, "TokenType") == 0 ||
           strcmp(name, "Token") == 0;
}

/* Get prefixed type name for user-defined types */
static const char *get_prefixed_type_name(const char *name) {
    static char buffer[512];
    
    /* Runtime types: no prefix */
    if (is_runtime_typedef(name) || conflicts_with_runtime(name)) {
        return name;
    }
    
    /* User types: add nl_ prefix */
    snprintf(buffer, sizeof(buffer), "nl_%s", name);
    return buffer;
}

/* Get prefixed enum variant name */
static const char *get_prefixed_variant_name(const char *enum_name, const char *variant_name) {
    static char buffer[512];
    snprintf(buffer, sizeof(buffer), "nl_%s_%s", enum_name, variant_name);
    return buffer;
}

/* Get prefixed variant struct name for unions: UnionName.Variant -> nl_UnionName_Variant */
static const char *get_prefixed_variant_struct_name(const char *union_name, const char *variant_name) {
    static char buffer[512];
    snprintf(buffer, sizeof(buffer), "nl_%s_%s", union_name, variant_name);
    return buffer;
}

/* Function type registry for generating function pointer typedefs */
typedef struct {
    FunctionSignature **signatures;
    char **typedef_names;
    int count;
    int capacity;
} FunctionTypeRegistry;

/* Tuple type registry for generating tuple struct typedefs */
typedef struct {
    TypeInfo **tuples;
    char **typedef_names;
    int count;
    int capacity;
} TupleTypeRegistry;

/* Global tuple registry - set during transpilation */
static TupleTypeRegistry *g_tuple_registry = NULL;

/* Global header collection - gathered from imported modules */
typedef struct {
    char *name;
    int priority;  /* Higher priority = included first */
} ModuleHeader;

static ModuleHeader *g_module_headers = NULL;
static size_t g_module_headers_count = 0;
static size_t g_module_headers_capacity = 0;

static void add_module_header(const char *header, int priority) {
    if (!header) return;
    
    /* Check if already added */
    for (size_t i = 0; i < g_module_headers_count; i++) {
        if (strcmp(g_module_headers[i].name, header) == 0) {
            /* Update priority if higher */
            if (priority > g_module_headers[i].priority) {
                g_module_headers[i].priority = priority;
            }
            return;
        }
    }
    
    /* Expand capacity if needed */
    if (g_module_headers_count >= g_module_headers_capacity) {
        int new_capacity = g_module_headers_capacity == 0 ? 8 : g_module_headers_capacity * 2;
        if (new_capacity > SIZE_MAX / sizeof(ModuleHeader)) {
            fprintf(stderr, "Error: Module headers capacity overflow\n");
            exit(1);
        }
        ModuleHeader *new_headers = realloc(g_module_headers, sizeof(ModuleHeader) * new_capacity);
        if (!new_headers) {
            fprintf(stderr, "Error: Out of memory in module headers\n");
            exit(1);
        }
        g_module_headers = new_headers;
        g_module_headers_capacity = new_capacity;
    }
    
    g_module_headers[g_module_headers_count].name = strdup(header);
    g_module_headers[g_module_headers_count].priority = priority;
    g_module_headers_count++;
}

/* Compare function for qsort - higher priority first */
static int compare_headers_by_priority(const void *a, const void *b) {
    const ModuleHeader *ha = (const ModuleHeader *)a;
    const ModuleHeader *hb = (const ModuleHeader *)b;
    return hb->priority - ha->priority;  /* Descending order */
}

static void clear_module_headers(void) {
    for (size_t i = 0; i < g_module_headers_count; i++) {
        free(g_module_headers[i].name);
    }
    free(g_module_headers);
    g_module_headers = NULL;
    g_module_headers_count = 0;
    g_module_headers_capacity = 0;
}

static void collect_headers_from_module(const char *module_path) {
    if (!module_path) return;
    
    /* Extract module directory from path */
    char *path_copy = strdup(module_path);
    char *dir = dirname(path_copy);
    
    /* Load module metadata */
    ModuleBuildMetadata *meta = module_load_metadata(dir);
    if (meta && meta->headers) {
        for (size_t i = 0; i < meta->headers_count; i++) {
            add_module_header(meta->headers[i], meta->header_priority);
        }
        module_metadata_free(meta);
    }
    
    free(path_copy);
}

static FunctionTypeRegistry *create_fn_type_registry(void) {
    FunctionTypeRegistry *reg = malloc(sizeof(FunctionTypeRegistry));
    reg->signatures = malloc(sizeof(FunctionSignature*) * 16);
    reg->typedef_names = malloc(sizeof(char*) * 16);
    reg->count = 0;
    reg->capacity = 16;
    return reg;
}

static void free_fn_type_registry(FunctionTypeRegistry *reg) {
    if (!reg) return;
    if (reg->typedef_names) {
        for (int i = 0; i < reg->count; i++) {
            free(reg->typedef_names[i]);
        }
        free(reg->typedef_names);
    }
    /* Free each FunctionSignature struct, not just the array of pointers */
    if (reg->signatures) {
        for (int i = 0; i < reg->count; i++) {
            free_function_signature(reg->signatures[i]);
        }
        free(reg->signatures);
    }
    free(reg);
}

/* Tuple type registry functions */
static TupleTypeRegistry *create_tuple_type_registry(void) {
    TupleTypeRegistry *reg = malloc(sizeof(TupleTypeRegistry));
    reg->tuples = malloc(sizeof(TypeInfo*) * 16);
    reg->typedef_names = malloc(sizeof(char*) * 16);
    reg->count = 0;
    reg->capacity = 16;
    return reg;
}

static void free_tuple_type_registry(TupleTypeRegistry *reg) {
    if (!reg) return;
    if (reg->typedef_names) {
        for (int i = 0; i < reg->count; i++) {
            free(reg->typedef_names[i]);
        }
        free(reg->typedef_names);
    }
    /* Free each TypeInfo struct and its tuple_types array, not just the array of pointers */
    if (reg->tuples) {
        for (int i = 0; i < reg->count; i++) {
            if (reg->tuples[i]) {
                if (reg->tuples[i]->tuple_types) {
                    free(reg->tuples[i]->tuple_types);
                }
                free(reg->tuples[i]);
            }
        }
        free(reg->tuples);
    }
    free(reg);
}

/* Check if two tuple types are equal */
static bool tuple_types_equal(TypeInfo *a, TypeInfo *b) {
    if (!a || !b) return false;
    if (a->tuple_element_count != b->tuple_element_count) return false;
    
    for (int i = 0; i < a->tuple_element_count; i++) {
        if (a->tuple_types[i] != b->tuple_types[i]) return false;
    }
    
    return true;
}

/* Generate typedef name for a tuple type */
static char *get_tuple_typedef_name(TypeInfo *info, int index) {
    char *name = malloc(256);
    StringBuilder *sb = sb_create();
    
    sb_append(sb, "Tuple");
    for (int i = 0; i < info->tuple_element_count; i++) {
        sb_append(sb, "_");
        switch (info->tuple_types[i]) {
            case TYPE_INT: sb_append(sb, "int"); break;
            case TYPE_FLOAT: sb_append(sb, "float"); break;
            case TYPE_BOOL: sb_append(sb, "bool"); break;
            case TYPE_STRING: sb_append(sb, "string"); break;
            case TYPE_BSTRING: sb_append(sb, "bstring"); break;
            default: sb_appendf(sb, "t%d", i); break;
        }
    }
    sb_appendf(sb, "_%d", index);
    
    snprintf(name, 256, "%s", sb->buffer);
    free(sb->buffer);
    free(sb);
    return name;
}

/* Register a tuple type and get its typedef name */
static const char *register_tuple_type(TupleTypeRegistry *reg, TypeInfo *info) {
    /* Check if already registered */
    for (int i = 0; i < reg->count; i++) {
        if (tuple_types_equal(reg->tuples[i], info)) {
            return reg->typedef_names[i];
        }
    }
    
    /* Register new tuple type */
    if (reg->count >= reg->capacity) {
        if (reg->capacity > SIZE_MAX / 2) {
            fprintf(stderr, "Error: Tuple registry capacity overflow\n");
            exit(1);
        }
        int new_capacity = reg->capacity * 2;
        TypeInfo **new_tuples = realloc(reg->tuples, sizeof(TypeInfo*) * new_capacity);
        char **new_names = realloc(reg->typedef_names, sizeof(char*) * new_capacity);
        if (!new_tuples || !new_names) {
            fprintf(stderr, "Error: Out of memory in tuple registry\n");
            exit(1);
        }
        reg->tuples = new_tuples;
        reg->typedef_names = new_names;
        reg->capacity = new_capacity;
    }
    
    reg->tuples[reg->count] = info;
    reg->typedef_names[reg->count] = get_tuple_typedef_name(info, reg->count);
    reg->count++;
    
    return reg->typedef_names[reg->count - 1];
}

/* Generate C typedef for a tuple type */
static void generate_tuple_typedef(StringBuilder *sb, TypeInfo *info, const char *typedef_name) {
    sb_appendf(sb, "typedef struct { ");
    for (int i = 0; i < info->tuple_element_count; i++) {
        if (i > 0) sb_append(sb, "; ");
        sb_appendf(sb, "%s _%d", type_to_c(info->tuple_types[i]), i);
    }
    sb_appendf(sb, "; } %s;\n", typedef_name);
}

/* Generate unique typedef name for a function signature */
static char *get_function_typedef_name(FunctionSignature *sig, int index) {
    char *name = malloc(64);
    
    /* Generate descriptive name based on signature pattern */
    if (sig->param_count == 1 && sig->return_type == TYPE_BOOL) {
        /* Predicate: fn(T) -> bool */
        snprintf(name, 64, "Predicate_%d", index);
    } else if (sig->param_count == 2 && 
               sig->param_types[0] == sig->param_types[1] &&
               sig->return_type == sig->param_types[0]) {
        /* Binary op: fn(T, T) -> T */
        snprintf(name, 64, "BinaryOp_%d", index);
    } else {
        /* Generic: FnType_N */
        snprintf(name, 64, "FnType_%d", index);
    }
    
    return name;
}

/* Register a function signature and get its typedef name */
static const char *register_function_signature(FunctionTypeRegistry *reg, FunctionSignature *sig) {
    /* Check if already registered */
    for (int i = 0; i < reg->count; i++) {
        if (function_signatures_equal(reg->signatures[i], sig)) {
            return reg->typedef_names[i];
        }
    }
    
    /* Register new signature */
    if (reg->count >= reg->capacity) {
        if (reg->capacity > SIZE_MAX / 2) {
            fprintf(stderr, "Error: Function registry capacity overflow\n");
            exit(1);
        }
        int new_capacity = reg->capacity * 2;
        FunctionSignature **new_sigs = realloc(reg->signatures,
                                               sizeof(FunctionSignature*) * new_capacity);
        char **new_names = realloc(reg->typedef_names,
                                   sizeof(char*) * new_capacity);
        if (!new_sigs || !new_names) {
            fprintf(stderr, "Error: Out of memory in function registry\n");
            exit(1);
        }
        reg->signatures = new_sigs;
        reg->typedef_names = new_names;
        reg->capacity = new_capacity;
    }
    
    reg->signatures[reg->count] = sig;
    reg->typedef_names[reg->count] = get_function_typedef_name(sig, reg->count);
    reg->count++;
    
    return reg->typedef_names[reg->count - 1];
}

/* Generate C typedef for a function signature */
static void generate_function_typedef(StringBuilder *sb, FunctionSignature *sig,
                                     const char *typedef_name) {
    sb_append(sb, "typedef ");
    
    /* Return type */
    if (sig->return_type == TYPE_FUNCTION && sig->return_fn_sig) {
        /* Nested function return type: fn() -> fn(int, int) -> int
         * C syntax: typedef int64_t (*(*FnType_N)())(int64_t, int64_t);
         */
        /* Return type of nested function */
        sb_appendf(sb, "%s ", type_to_c(sig->return_fn_sig->return_type));
        /* Outer function pointer: (*(*typedef_name)()) */
        sb_appendf(sb, "(*(*%s)(", typedef_name);
        /* Parameters for outer function */
        for (int i = 0; i < sig->param_count; i++) {
            if (i > 0) sb_append(sb, ", ");
            
            if (sig->param_types[i] == TYPE_STRUCT && sig->param_struct_names[i]) {
                sb_appendf(sb, "struct %s", sig->param_struct_names[i]);
            } else {
                sb_append(sb, type_to_c(sig->param_types[i]));
            }
        }
        sb_append(sb, "))(");
        /* Parameters for nested function */
        for (int i = 0; i < sig->return_fn_sig->param_count; i++) {
            if (i > 0) sb_append(sb, ", ");
            
            if (sig->return_fn_sig->param_types[i] == TYPE_STRUCT && sig->return_fn_sig->param_struct_names[i]) {
                sb_appendf(sb, "struct %s", sig->return_fn_sig->param_struct_names[i]);
            } else {
                sb_append(sb, type_to_c(sig->return_fn_sig->param_types[i]));
            }
        }
        sb_append(sb, ");\n");
    } else if (sig->return_type == TYPE_STRUCT && sig->return_struct_name) {
        sb_appendf(sb, "struct %s ", sig->return_struct_name);
        /* Function pointer syntax: (*typedef_name) */
        sb_appendf(sb, "(*%s)", typedef_name);
        /* Parameters */
        sb_append(sb, "(");
        for (int i = 0; i < sig->param_count; i++) {
            if (i > 0) sb_append(sb, ", ");
            
            if (sig->param_types[i] == TYPE_STRUCT && sig->param_struct_names[i]) {
                sb_appendf(sb, "struct %s", sig->param_struct_names[i]);
            } else {
                sb_append(sb, type_to_c(sig->param_types[i]));
            }
        }
        sb_append(sb, ");\n");
    } else {
        sb_appendf(sb, "%s ", type_to_c(sig->return_type));
        /* Function pointer syntax: (*typedef_name) */
        sb_appendf(sb, "(*%s)", typedef_name);
        /* Parameters */
        sb_append(sb, "(");
        for (int i = 0; i < sig->param_count; i++) {
            if (i > 0) sb_append(sb, ", ");
            
            if (sig->param_types[i] == TYPE_STRUCT && sig->param_struct_names[i]) {
                sb_appendf(sb, "struct %s", sig->param_struct_names[i]);
            } else {
                sb_append(sb, type_to_c(sig->param_types[i]));
            }
        }
        sb_append(sb, ");\n");
    }
}

/* SDL-specific scalar type mapping for FFI
 * 
 * NOTE: Opaque pointer types (SDL_Window*, SDL_Renderer*, Mix_Chunk*, TTF_Font*, etc.)
 * are now handled by the generic opaque type system. This function only handles
 * SDL-specific scalar types (Uint32, Uint8) and non-opaque struct types
 * (SDL_Rect*, SDL_Event*) that can't be represented as opaque types.
 * 
 * This is a minimal legacy compatibility function. Once SDL scalar types are
 * properly represented in the type system, this function can be removed entirely.
 */
static const char *get_sdl_c_type(const char *func_name, int param_index, bool is_return) {
    if (!func_name) return NULL;
    
    /* SDL/TTF functions only */
    if (strncmp(func_name, "SDL_", 4) != 0 && strncmp(func_name, "TTF_", 4) != 0) {
        return NULL;
    }
    
    if (is_return) {
        /* Return scalar types (opaque pointers now handled by opaque type system) */
        if (strstr(func_name, "GetTicks")) return "Uint32";
        return NULL;
    } else {
        /* Parameter types - only SDL-specific scalars and non-opaque structs */
        
        /* SDL_Rect* - struct pointers (data structures, not opaque resources) */
        if (strstr(func_name, "RenderFillRect") && param_index == 1) return "const SDL_Rect*";
        if (strstr(func_name, "RenderCopy")) {
            if (param_index == 2 || param_index == 3) return "const SDL_Rect*";
        }
        
        /* SDL_Event* - struct pointer (data structure, not opaque resource) */
        if (strstr(func_name, "PollEvent") && param_index == 0) return "SDL_Event*";
        
        /* int* for out parameters */
        if (strstr(func_name, "QueryTexture") && param_index >= 2) return "int*";
        
        /* SDL scalar types (Uint32, Uint8) - these should eventually become proper types */
        if (strstr(func_name, "Init") && param_index == 0) return "Uint32";
        if (strstr(func_name, "Delay") && param_index == 0) return "Uint32";
        if (strstr(func_name, "CreateWindow") && param_index == 5) return "Uint32";
        if (strstr(func_name, "CreateRenderer") && param_index == 2) return "Uint32";
        if (strstr(func_name, "SetRenderDrawColor") && param_index >= 1 && param_index <= 4) return "Uint8";
    }
    return NULL;
}

/* Transpile type to C type */
static const char *type_to_c(Type type) {
    switch (type) {
        case TYPE_INT: return "int64_t";
        case TYPE_FLOAT: return "double";
        case TYPE_BOOL: return "bool";
        case TYPE_STRING: return "const char*";
        case TYPE_BSTRING: return "nl_string_t*";
        case TYPE_VOID: return "void";
        case TYPE_ARRAY: return "DynArray*";  /* All arrays are now dynamic arrays with GC */
        case TYPE_STRUCT: return "struct"; /* Will be extended with struct name */
        case TYPE_ENUM: return ""; /* Enum names are used directly (typedef'd) */
        case TYPE_UNION: return ""; /* Union names are used directly (typedef'd) */
        case TYPE_FUNCTION: return ""; /* Will be handled with typedef */
        case TYPE_LIST_INT: return "List_int*";
        case TYPE_LIST_STRING: return "List_string*";
        case TYPE_LIST_TOKEN: return "List_token*";
        case TYPE_LIST_GENERIC: return ""; /* Will be handled specially with type_name */
        case TYPE_OPAQUE: return "void*"; /* Opaque pointers stored as void* */
        default: return "void";
    }
}

/* Get C function name with prefix to avoid conflicts with standard library */
static const char *get_c_func_name(const char *nano_name) {
    /* Note: main() now gets nl_ prefix to support library mode (Stage 1.5+) */
    /* Standalone programs use --entry-point to call nl_main() */
    
    /* Don't prefix list runtime functions */
    if (strncmp(nano_name, "list_int_", 9) == 0 || 
        strncmp(nano_name, "list_string_", 12) == 0 ||
        strncmp(nano_name, "list_token_", 11) == 0) {
        return nano_name;
    }
    
    /* Don't prefix advanced string operations (generated inline) */
    if (strcmp(nano_name, "char_at") == 0 ||
        strcmp(nano_name, "string_from_char") == 0 ||
        strcmp(nano_name, "is_digit") == 0 ||
        strcmp(nano_name, "is_alpha") == 0 ||
        strcmp(nano_name, "is_alnum") == 0 ||
        strcmp(nano_name, "is_whitespace") == 0 ||
        strcmp(nano_name, "is_upper") == 0 ||
        strcmp(nano_name, "is_lower") == 0 ||
        strcmp(nano_name, "int_to_string") == 0 ||
        strcmp(nano_name, "string_to_int") == 0 ||
        strcmp(nano_name, "digit_value") == 0 ||
        strcmp(nano_name, "char_to_lower") == 0 ||
        strcmp(nano_name, "char_to_upper") == 0) {
        return nano_name;
    }

    /* Prefix user functions to avoid conflicts with C stdlib (abs, min, max, etc.) */
    static char buffer[256];
    snprintf(buffer, sizeof(buffer), "nl_%s", nano_name);
    return buffer;
}

/* ============================================================================
 * ITERATIVE TRANSPILER IMPLEMENTATION
 * Two-pass architecture: clean and simple!
 * ============================================================================ */
#include "transpiler_iterative_v3_twopass.c"


/* ============================================================================
 * TRANSPILER DISPATCHER
 * Calls the iterative two-pass transpiler implementation
 * ============================================================================ */

/* Iterative versions are defined in transpiler_iterative_v3_twopass.c */
/* They are named transpile_expression_iterative and transpile_statement_iterative */
/* Create wrapper functions that call them */
static void transpile_expression_wrapper(StringBuilder *sb, ASTNode *expr, Environment *env) {
    transpile_expression_iterative(sb, expr, env);
}
static void transpile_statement_wrapper(StringBuilder *sb, ASTNode *stmt, int indent, Environment *env, FunctionTypeRegistry *fn_registry) {
    transpile_statement_iterative(sb, stmt, indent, env, fn_registry);
}
#define transpile_expression transpile_expression_wrapper
#define transpile_statement transpile_statement_wrapper

/* Helper to recursively collect function signatures from statements */
/* Collect tuple types from expressions */
static void collect_tuple_types_from_expr(ASTNode *expr, TupleTypeRegistry *reg) {
    if (!expr) return;
    
    switch (expr->type) {
        case AST_TUPLE_LITERAL:
            if (expr->as.tuple_literal.element_count > 0) {
                if (expr->as.tuple_literal.element_types) {
                    /* Element types are set - register directly */
                    TypeInfo *temp_info = malloc(sizeof(TypeInfo));
                    temp_info->tuple_element_count = expr->as.tuple_literal.element_count;
                    temp_info->tuple_types = malloc(sizeof(Type) * expr->as.tuple_literal.element_count);
                    for (int i = 0; i < expr->as.tuple_literal.element_count; i++) {
                        temp_info->tuple_types[i] = expr->as.tuple_literal.element_types[i];
                    }
                    temp_info->tuple_type_names = NULL;
                    register_tuple_type(reg, temp_info);
                } else {
                    /* Element types not set - infer from elements */
                    TypeInfo *temp_info = malloc(sizeof(TypeInfo));
                    temp_info->tuple_element_count = expr->as.tuple_literal.element_count;
                    temp_info->tuple_types = malloc(sizeof(Type) * expr->as.tuple_literal.element_count);
                    for (int i = 0; i < expr->as.tuple_literal.element_count; i++) {
                        /* Try to infer type from expression */
                        Type elem_type = TYPE_INT;  /* Default to int */
                        ASTNode *elem = expr->as.tuple_literal.elements[i];
                        if (elem) {
                            if (elem->type == AST_NUMBER) elem_type = TYPE_INT;
                            else if (elem->type == AST_STRING) elem_type = TYPE_STRING;
                            else if (elem->type == AST_BOOL) elem_type = TYPE_BOOL;
                            else if (elem->type == AST_FLOAT) elem_type = TYPE_FLOAT;
                            else if (elem->type == AST_IDENTIFIER) elem_type = TYPE_INT;  /* Assume int for vars */
                        }
                        temp_info->tuple_types[i] = elem_type;
                    }
                    temp_info->tuple_type_names = NULL;
                    register_tuple_type(reg, temp_info);
                }
            }
            /* Also collect from tuple elements */
            for (int i = 0; i < expr->as.tuple_literal.element_count; i++) {
                collect_tuple_types_from_expr(expr->as.tuple_literal.elements[i], reg);
            }
            break;
        case AST_PREFIX_OP:
            for (int i = 0; i < expr->as.prefix_op.arg_count; i++) {
                collect_tuple_types_from_expr(expr->as.prefix_op.args[i], reg);
            }
            break;
        case AST_CALL:
            for (int i = 0; i < expr->as.call.arg_count; i++) {
                collect_tuple_types_from_expr(expr->as.call.args[i], reg);
            }
            break;
        case AST_IF:
            if (expr->as.if_stmt.condition) {
                collect_tuple_types_from_expr(expr->as.if_stmt.condition, reg);
            }
            if (expr->as.if_stmt.then_branch) {
                /* If expressions can have tuple literals as branches */
                collect_tuple_types_from_expr(expr->as.if_stmt.then_branch, reg);
            }
            if (expr->as.if_stmt.else_branch) {
                collect_tuple_types_from_expr(expr->as.if_stmt.else_branch, reg);
            }
            break;

        default:
            break;
    }
}

/* Collect tuple types from statements */
static void collect_tuple_types_from_stmt(ASTNode *stmt, TupleTypeRegistry *reg) {
    if (!stmt) return;
    
    switch (stmt->type) {
        case AST_LET:
            if (stmt->as.let.value) {
                collect_tuple_types_from_expr(stmt->as.let.value, reg);
            }
            break;
        case AST_RETURN:
            if (stmt->as.return_stmt.value) {
                collect_tuple_types_from_expr(stmt->as.return_stmt.value, reg);
            }
            break;
        case AST_BLOCK:
            for (int i = 0; i < stmt->as.block.count; i++) {
                collect_tuple_types_from_stmt(stmt->as.block.statements[i], reg);
            }
            break;
        case AST_IF:
            if (stmt->as.if_stmt.condition) {
                collect_tuple_types_from_expr(stmt->as.if_stmt.condition, reg);
            }
            collect_tuple_types_from_stmt(stmt->as.if_stmt.then_branch, reg);
            if (stmt->as.if_stmt.else_branch) {
                collect_tuple_types_from_stmt(stmt->as.if_stmt.else_branch, reg);
            }
            break;
        case AST_WHILE:
            if (stmt->as.while_stmt.condition) {
                collect_tuple_types_from_expr(stmt->as.while_stmt.condition, reg);
            }
            collect_tuple_types_from_stmt(stmt->as.while_stmt.body, reg);
            break;
        case AST_FOR:
            collect_tuple_types_from_stmt(stmt->as.for_stmt.body, reg);
            break;
        default:
            break;
    }
}

static void collect_fn_sigs(ASTNode *stmt, FunctionTypeRegistry *reg) {
    if (!stmt) return;
    
    switch (stmt->type) {
        case AST_LET:
            if (stmt->as.let.var_type == TYPE_FUNCTION && stmt->as.let.fn_sig) {
                register_function_signature(reg, stmt->as.let.fn_sig);
            }
            break;
        case AST_BLOCK:
            for (int i = 0; i < stmt->as.block.count; i++) {
                collect_fn_sigs(stmt->as.block.statements[i], reg);
            }
            break;
        case AST_IF:
            collect_fn_sigs(stmt->as.if_stmt.then_branch, reg);
            if (stmt->as.if_stmt.else_branch) {
                collect_fn_sigs(stmt->as.if_stmt.else_branch, reg);
            }
            break;
        case AST_WHILE:
            collect_fn_sigs(stmt->as.while_stmt.body, reg);
            break;
        case AST_FOR:
            collect_fn_sigs(stmt->as.for_stmt.body, reg);
            break;
        default:
            break;
    }
}

/* Transpile program to C */
char *transpile_to_c(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        return NULL;
    }
    
    if (!env) {
        fprintf(stderr, "Error: Environment is NULL in transpile_to_c\n");
        return NULL;
    }

    /* Clear and collect headers from imported modules */
    clear_module_headers();
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_IMPORT) {
            /* Resolve module path and collect headers */
            char *module_path = resolve_module_path(item->as.import_stmt.module_path, NULL);
            if (module_path) {
                collect_headers_from_module(module_path);
                free(module_path);
            }
        }
    }

    StringBuilder *sb = sb_create();

    /* POSIX feature macro for strdup, strnlen, etc. */
    sb_append(sb, "#define _POSIX_C_SOURCE 200809L\n\n");

    /* C includes and headers */
    sb_append(sb, "#include <stdio.h>\n");
    sb_append(sb, "#include <stdint.h>\n");
    sb_append(sb, "#include <stdbool.h>\n");
    sb_append(sb, "#include <string.h>\n");
    sb_append(sb, "#include <stdlib.h>\n");
    sb_append(sb, "#include <time.h>\n");
    sb_append(sb, "#include <stdarg.h>\n");
    sb_append(sb, "#include <math.h>\n");
    sb_append(sb, "#include \"runtime/nl_string.h\"\n");
    
    /* Include headers from imported modules (generic C library support) */
    if (g_module_headers_count > 0) {
        /* Sort headers by priority (highest first) */
        qsort(g_module_headers, g_module_headers_count, sizeof(ModuleHeader), compare_headers_by_priority);
        
        sb_append(sb, "\n/* Headers from imported modules (sorted by priority) */\n");
        for (size_t i = 0; i < g_module_headers_count; i++) {
            sb_appendf(sb, "#include <%s>  /* priority: %d */\n", 
                       g_module_headers[i].name, 
                       g_module_headers[i].priority);
        }
    }
    
    sb_append(sb, "\n/* nanolang runtime */\n");
    sb_append(sb, "#include \"runtime/list_int.h\"\n");
    sb_append(sb, "#include \"runtime/list_string.h\"\n");
    sb_append(sb, "#include \"runtime/list_token.h\"\n");
    sb_append(sb, "#include \"runtime/token_helpers.h\"\n");
    sb_append(sb, "#include <sys/stat.h>\n");
    sb_append(sb, "#include <sys/types.h>\n");
    sb_append(sb, "#include <dirent.h>\n");
    sb_append(sb, "#include <unistd.h>\n");
    sb_append(sb, "#include <libgen.h>\n");
    sb_append(sb, "\n");

    /* OS stdlib runtime library */
    sb_append(sb, "/* ========== OS Standard Library ========== */\n\n");

    /* File operations */
    sb_append(sb, "static char* nl_os_file_read(const char* path) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"rb\");  /* Binary mode for MOD files */\n");
    sb_append(sb, "    if (!f) return \"\";\n");
    sb_append(sb, "    fseek(f, 0, SEEK_END);\n");
    sb_append(sb, "    long size = ftell(f);\n");
    sb_append(sb, "    fseek(f, 0, SEEK_SET);\n");
    sb_append(sb, "    char* buffer = malloc(size + 1);\n");
    sb_append(sb, "    fread(buffer, 1, size, f);\n");
    sb_append(sb, "    buffer[size] = '\\0';\n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    /* Binary file reading - returns DynArray of bytes (0-255) */
    sb_append(sb, "static DynArray* nl_os_file_read_bytes(const char* path) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"rb\");\n");
    sb_append(sb, "    if (!f) {\n");
    sb_append(sb, "        /* Return empty array on error */\n");
    sb_append(sb, "        return dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    fseek(f, 0, SEEK_END);\n");
    sb_append(sb, "    long size = ftell(f);\n");
    sb_append(sb, "    fseek(f, 0, SEEK_SET);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Create dynamic array for bytes */\n");
    sb_append(sb, "    DynArray* bytes = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    /* Read bytes and add to array */\n");
    sb_append(sb, "    for (long i = 0; i < size; i++) {\n");
    sb_append(sb, "        int c = fgetc(f);\n");
    sb_append(sb, "        if (c == EOF) break;\n");
    sb_append(sb, "        int64_t byte_val = (int64_t)(unsigned char)c;\n");
    sb_append(sb, "        dyn_array_push_int(bytes, byte_val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    \n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return bytes;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_write(const char* path, const char* content) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"w\");\n");
    sb_append(sb, "    if (!f) return -1;\n");
    sb_append(sb, "    fputs(content, f);\n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_append(const char* path, const char* content) {\n");
    sb_append(sb, "    FILE* f = fopen(path, \"a\");\n");
    sb_append(sb, "    if (!f) return -1;\n");
    sb_append(sb, "    fputs(content, f);\n");
    sb_append(sb, "    fclose(f);\n");
    sb_append(sb, "    return 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_remove(const char* path) {\n");
    sb_append(sb, "    return remove(path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_rename(const char* old_path, const char* new_path) {\n");
    sb_append(sb, "    return rename(old_path, new_path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool nl_os_file_exists(const char* path) {\n");
    sb_append(sb, "    return access(path, F_OK) == 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_file_size(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return -1;\n");
    sb_append(sb, "    return st.st_size;\n");
    sb_append(sb, "}\n\n");

    /* Directory operations */
    sb_append(sb, "static int64_t nl_os_dir_create(const char* path) {\n");
    sb_append(sb, "    return mkdir(path, 0755) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_dir_remove(const char* path) {\n");
    sb_append(sb, "    return rmdir(path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_dir_list(const char* path) {\n");
    sb_append(sb, "    DIR* dir = opendir(path);\n");
    sb_append(sb, "    if (!dir) return \"\";\n");
    sb_append(sb, "    size_t capacity = 4096;\n");
    sb_append(sb, "    size_t used = 0;\n");
    sb_append(sb, "    char* buffer = malloc(capacity);\n");
    sb_append(sb, "    if (!buffer) { closedir(dir); return \"\"; }\n");
    sb_append(sb, "    buffer[0] = '\\0';\n");
    sb_append(sb, "    struct dirent* entry;\n");
    sb_append(sb, "    while ((entry = readdir(dir)) != NULL) {\n");
    sb_append(sb, "        if (strcmp(entry->d_name, \".\") == 0 || strcmp(entry->d_name, \"..\") == 0) continue;\n");
    sb_append(sb, "        size_t name_len = strlen(entry->d_name);\n");
    sb_append(sb, "        size_t needed = used + name_len + 2; /* +1 for newline, +1 for null */\n");
    sb_append(sb, "        if (needed > capacity) {\n");
    sb_append(sb, "            capacity = needed * 2;\n");
    sb_append(sb, "            char* new_buffer = realloc(buffer, capacity);\n");
    sb_append(sb, "            if (!new_buffer) { free(buffer); closedir(dir); return \"\"; }\n");
    sb_append(sb, "            buffer = new_buffer;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        memcpy(buffer + used, entry->d_name, name_len);\n");
    sb_append(sb, "        used += name_len;\n");
    sb_append(sb, "        buffer[used++] = '\\n';\n");
    sb_append(sb, "        buffer[used] = '\\0';\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    closedir(dir);\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool nl_os_dir_exists(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return false;\n");
    sb_append(sb, "    return S_ISDIR(st.st_mode);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_getcwd(void) {\n");
    sb_append(sb, "    char* buffer = malloc(1024);\n");
    sb_append(sb, "    if (getcwd(buffer, 1024) == NULL) {\n");
    sb_append(sb, "        buffer[0] = '\\0';\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static int64_t nl_os_chdir(const char* path) {\n");
    sb_append(sb, "    return chdir(path) == 0 ? 0 : -1;\n");
    sb_append(sb, "}\n\n");

    /* Path operations */
    sb_append(sb, "static bool nl_os_path_isfile(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return false;\n");
    sb_append(sb, "    return S_ISREG(st.st_mode);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static bool nl_os_path_isdir(const char* path) {\n");
    sb_append(sb, "    struct stat st;\n");
    sb_append(sb, "    if (stat(path, &st) != 0) return false;\n");
    sb_append(sb, "    return S_ISDIR(st.st_mode);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_join(const char* a, const char* b) {\n");
    sb_append(sb, "    char* buffer = malloc(2048);\n");
    sb_append(sb, "    if (strlen(a) == 0) {\n");
    sb_append(sb, "        snprintf(buffer, 2048, \"%s\", b);\n");
    sb_append(sb, "    } else if (a[strlen(a) - 1] == '/') {\n");
    sb_append(sb, "        snprintf(buffer, 2048, \"%s%s\", a, b);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        snprintf(buffer, 2048, \"%s/%s\", a, b);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_basename(const char* path) {\n");
    sb_append(sb, "    char* path_copy = strdup(path);\n");
    sb_append(sb, "    char* base = basename(path_copy);\n");
    sb_append(sb, "    char* result = strdup(base);\n");
    sb_append(sb, "    free(path_copy);\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_path_dirname(const char* path) {\n");
    sb_append(sb, "    char* path_copy = strdup(path);\n");
    sb_append(sb, "    char* dir = dirname(path_copy);\n");
    sb_append(sb, "    char* result = strdup(dir);\n");
    sb_append(sb, "    free(path_copy);\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");

    /* Process operations */
    sb_append(sb, "static int64_t nl_os_system(const char* command) {\n");
    sb_append(sb, "    return system(command);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_os_exit(int64_t code) {\n");
    sb_append(sb, "    exit((int)code);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_getenv(const char* name) {\n");
    sb_append(sb, "    const char* value = getenv(name);\n");
    sb_append(sb, "    return value ? (char*)value : \"\";\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* ========== End OS Standard Library ========== */\n\n");

    sb_append(sb, "/* ========== Advanced String Operations ========== */\n\n");
    
    /* char_at - use strnlen for safety */
    sb_append(sb, "static int64_t char_at(const char* s, int64_t index) {\n");
    sb_append(sb, "    /* Safety: Bound string scan to reasonable size (1MB) */\n");
    sb_append(sb, "    int len = strnlen(s, 1024*1024);\n");
    sb_append(sb, "    if (index < 0 || index >= len) {\n");
    sb_append(sb, "        fprintf(stderr, \"Error: Index %lld out of bounds (string length %d)\\n\", (long long)index, len);\n");
    sb_append(sb, "        return 0;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return (unsigned char)s[index];\n");
    sb_append(sb, "}\n\n");
    
    /* string_from_char */
    sb_append(sb, "static char* string_from_char(int64_t c) {\n");
    sb_append(sb, "    char* buffer = malloc(2);\n");
    sb_append(sb, "    buffer[0] = (char)c;\n");
    sb_append(sb, "    buffer[1] = '\\0';\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");
    
    /* Character classification */
    sb_append(sb, "static bool is_digit(int64_t c) {\n");
    sb_append(sb, "    return c >= '0' && c <= '9';\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_alpha(int64_t c) {\n");
    sb_append(sb, "    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_alnum(int64_t c) {\n");
    sb_append(sb, "    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_whitespace(int64_t c) {\n");
    sb_append(sb, "    return c == ' ' || c == '\\t' || c == '\\n' || c == '\\r';\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_upper(int64_t c) {\n");
    sb_append(sb, "    return c >= 'A' && c <= 'Z';\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool is_lower(int64_t c) {\n");
    sb_append(sb, "    return c >= 'a' && c <= 'z';\n");
    sb_append(sb, "}\n\n");
    
    /* Type conversions */
    sb_append(sb, "static char* int_to_string(int64_t n) {\n");
    sb_append(sb, "    char* buffer = malloc(32);\n");
    sb_append(sb, "    snprintf(buffer, 32, \"%lld\", (long long)n);\n");
    sb_append(sb, "    return buffer;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t string_to_int(const char* s) {\n");
    sb_append(sb, "    return strtoll(s, NULL, 10);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t digit_value(int64_t c) {\n");
    sb_append(sb, "    if (c >= '0' && c <= '9') {\n");
    sb_append(sb, "        return c - '0';\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return -1;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t char_to_lower(int64_t c) {\n");
    sb_append(sb, "    if (c >= 'A' && c <= 'Z') {\n");
    sb_append(sb, "        return c + 32;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return c;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t char_to_upper(int64_t c) {\n");
    sb_append(sb, "    if (c >= 'a' && c <= 'z') {\n");
    sb_append(sb, "        return c - 32;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return c;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* ========== End Advanced String Operations ========== */\n\n");

    sb_append(sb, "/* ========== Math and Utility Built-in Functions ========== */\n\n");

    /* abs function - works with int and float via macro */
    sb_append(sb, "#define nl_abs(x) _Generic((x), \\\n");
    sb_append(sb, "    double: (double)((x) < 0.0 ? -(x) : (x)), \\\n");
    sb_append(sb, "    default: (int64_t)((x) < 0 ? -(x) : (x)))\n\n");

    /* min function */
    sb_append(sb, "#define nl_min(a, b) _Generic((a), \\\n");
    sb_append(sb, "    double: (double)((a) < (b) ? (a) : (b)), \\\n");
    sb_append(sb, "    default: (int64_t)((a) < (b) ? (a) : (b)))\n\n");

    /* max function */
    sb_append(sb, "#define nl_max(a, b) _Generic((a), \\\n");
    sb_append(sb, "    double: (double)((a) > (b) ? (a) : (b)), \\\n");
    sb_append(sb, "    default: (int64_t)((a) > (b) ? (a) : (b)))\n\n");

    /* Type casting functions */
    sb_append(sb, "static int64_t nl_cast_int(double x) { return (int64_t)x; }\n");
    sb_append(sb, "static int64_t nl_cast_int_from_int(int64_t x) { return x; }\n");
    sb_append(sb, "static double nl_cast_float(int64_t x) { return (double)x; }\n");
    sb_append(sb, "static double nl_cast_float_from_float(double x) { return x; }\n");
    sb_append(sb, "static int64_t nl_cast_bool_to_int(bool x) { return x ? 1 : 0; }\n");
    sb_append(sb, "static bool nl_cast_bool(int64_t x) { return x != 0; }\n\n");

    /* println function - uses _Generic for type dispatch */
    sb_append(sb, "static void nl_println(void* value_ptr) {\n");
    sb_append(sb, "    /* This is a placeholder - actual implementation uses type info from checker */\n");
    sb_append(sb, "}\n\n");

    /* Specialized print functions for each type (no newline) */
    sb_append(sb, "static void nl_print_int(int64_t value) {\n");
    sb_append(sb, "    printf(\"%lld\", (long long)value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_print_float(double value) {\n");
    sb_append(sb, "    printf(\"%g\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_print_string(const char* value) {\n");
    sb_append(sb, "    printf(\"%s\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_print_bool(bool value) {\n");
    sb_append(sb, "    printf(value ? \"true\" : \"false\");\n");
    sb_append(sb, "}\n\n");

    /* Specialized println functions for each type */
    sb_append(sb, "static void nl_println_int(int64_t value) {\n");
    sb_append(sb, "    printf(\"%lld\\n\", (long long)value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_println_float(double value) {\n");
    sb_append(sb, "    printf(\"%g\\n\", value);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_println_string(const char* value) {\n");
    sb_append(sb, "    printf(\"%s\\n\", value);\n");
    sb_append(sb, "}\n\n");
    
    /* Dynamic array runtime - LEGACY (for old array<T> type) */
    sb_append(sb, "/* Dynamic array runtime (using GC) - LEGACY */\n");
    sb_append(sb, "#include \"runtime/gc.h\"\n");
    sb_append(sb, "#include \"runtime/dyn_array.h\"\n\n");
    
    /* Array literals create dynamic arrays - renamed to avoid conflicts */
    sb_append(sb, "static DynArray* dynarray_literal_int(int count, ...) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    va_list args;\n");
    sb_append(sb, "    va_start(args, count);\n");
    sb_append(sb, "    for (int i = 0; i < count; i++) {\n");
    sb_append(sb, "        int64_t val = va_arg(args, int64_t);\n");
    sb_append(sb, "        dyn_array_push_int(arr, val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    va_end(args);\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static DynArray* dynarray_literal_float(int count, ...) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_FLOAT);\n");
    sb_append(sb, "    va_list args;\n");
    sb_append(sb, "    va_start(args, count);\n");
    sb_append(sb, "    for (int i = 0; i < count; i++) {\n");
    sb_append(sb, "        double val = va_arg(args, double);\n");
    sb_append(sb, "        dyn_array_push_float(arr, val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    va_end(args);\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    /* Array operations - renamed to avoid conflicts */
    sb_append(sb, "static DynArray* dynarray_push(DynArray* arr, double val) {\n");
    sb_append(sb, "    if (arr->elem_type == ELEM_INT) {\n");
    sb_append(sb, "        return dyn_array_push_int(arr, (int64_t)val);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        return dyn_array_push_float(arr, val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "}\n\n");
    
    /* Wrappers for array push/pop that work with GC dynamic arrays */
    sb_append(sb, "static DynArray* nl_array_push(DynArray* arr, double val) {\n");
    sb_append(sb, "    if (arr->elem_type == ELEM_INT) {\n");
    sb_append(sb, "        return dyn_array_push_int(arr, (int64_t)val);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        return dyn_array_push_float(arr, val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static double nl_array_pop(DynArray* arr) {\n");
    sb_append(sb, "    bool success = false;\n");
    sb_append(sb, "    if (arr->elem_type == ELEM_INT) {\n");
    sb_append(sb, "        return (double)dyn_array_pop_int(arr, &success);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        return dyn_array_pop_float(arr, &success);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "}\n\n");
    
    /* Array length wrapper */
    sb_append(sb, "static int64_t nl_array_length(DynArray* arr) {\n");
    sb_append(sb, "    return dyn_array_length(arr);\n");
    sb_append(sb, "}\n\n");
    
    /* Array remove_at wrapper */
    sb_append(sb, "static DynArray* nl_array_remove_at(DynArray* arr, int64_t index) {\n");
    sb_append(sb, "    return dyn_array_remove_at(arr, index);\n");
    sb_append(sb, "}\n\n");
    
    /* Array get (at) wrapper */
    sb_append(sb, "static int64_t nl_array_at_int(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_int(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static double nl_array_at_float(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_float(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static const char* nl_array_at_string(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_string(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static bool nl_array_at_bool(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_bool(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    /* Array set wrapper */
    sb_append(sb, "static void nl_array_set_int(DynArray* arr, int64_t idx, int64_t val) {\n");
    sb_append(sb, "    dyn_array_set_int(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_array_set_float(DynArray* arr, int64_t idx, double val) {\n");
    sb_append(sb, "    dyn_array_set_float(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_array_set_string(DynArray* arr, int64_t idx, const char* val) {\n");
    sb_append(sb, "    dyn_array_set_string(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_array_set_bool(DynArray* arr, int64_t idx, bool val) {\n");
    sb_append(sb, "    dyn_array_set_bool(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    /* Nested array support */
    sb_append(sb, "static DynArray* nl_array_at_array(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_array(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_array_set_array(DynArray* arr, int64_t idx, DynArray* val) {\n");
    sb_append(sb, "    dyn_array_set_array(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    /* Array new wrapper - creates DynArray with specified size and default value */
    sb_append(sb, "static DynArray* nl_array_new_int(int64_t size, int64_t default_val) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i = 0; i < size; i++) {\n");
    sb_append(sb, "        dyn_array_push_int(arr, default_val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static DynArray* nl_array_new_float(int64_t size, double default_val) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_FLOAT);\n");
    sb_append(sb, "    for (int64_t i = 0; i < size; i++) {\n");
    sb_append(sb, "        dyn_array_push_float(arr, default_val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static DynArray* nl_array_new_string(int64_t size, const char* default_val) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_STRING);\n");
    sb_append(sb, "    for (int64_t i = 0; i < size; i++) {\n");
    sb_append(sb, "        dyn_array_push_string(arr, default_val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static DynArray* nl_array_new_bool(int64_t size, bool default_val) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_BOOL);\n");
    sb_append(sb, "    for (int64_t i = 0; i < size; i++) {\n");
    sb_append(sb, "        dyn_array_push_bool(arr, default_val);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    return arr;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static int64_t dynarray_length(DynArray* arr) {\n");
    sb_append(sb, "    return dyn_array_length(arr);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static double dynarray_at_for_transpiler(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    if (arr->elem_type == ELEM_INT) {\n");
    sb_append(sb, "        return (double)dyn_array_get_int(arr, idx);\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        return dyn_array_get_float(arr, idx);\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "}\n\n");
    
    /* String operations */
    sb_append(sb, "/* String concatenation - use strnlen for safety */\n");
    sb_append(sb, "static const char* nl_str_concat(const char* s1, const char* s2) {\n");
    sb_append(sb, "    /* Safety: Bound string scan to 1MB */\n");
    sb_append(sb, "    size_t len1 = strnlen(s1, 1024*1024);\n");
    sb_append(sb, "    size_t len2 = strnlen(s2, 1024*1024);\n");
    sb_append(sb, "    char* result = malloc(len1 + len2 + 1);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    memcpy(result, s1, len1);\n");
    sb_append(sb, "    memcpy(result + len1, s2, len2);\n");
    sb_append(sb, "    result[len1 + len2] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* String substring - use strnlen for safety */\n");
    sb_append(sb, "static const char* nl_str_substring(const char* str, int64_t start, int64_t length) {\n");
    sb_append(sb, "    /* Safety: Bound string scan to 1MB */\n");
    sb_append(sb, "    int64_t str_len = strnlen(str, 1024*1024);\n");
    sb_append(sb, "    if (start < 0 || start >= str_len || length < 0) return \"\";\n");
    sb_append(sb, "    if (start + length > str_len) length = str_len - start;\n");
    sb_append(sb, "    char* result = malloc(length + 1);\n");
    sb_append(sb, "    if (!result) return \"\";\n");
    sb_append(sb, "    strncpy(result, str + start, length);\n");
    sb_append(sb, "    result[length] = '\\0';\n");
    sb_append(sb, "    return result;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* String contains */\n");
    sb_append(sb, "static bool nl_str_contains(const char* str, const char* substr) {\n");
    sb_append(sb, "    return strstr(str, substr) != NULL;\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "/* String equals */\n");
    sb_append(sb, "static bool nl_str_equals(const char* s1, const char* s2) {\n");
    sb_append(sb, "    return strcmp(s1, s2) == 0;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_println_bool(bool value) {\n");
    sb_append(sb, "    printf(\"%s\\n\", value ? \"true\" : \"false\");\n");
    sb_append(sb, "}\n\n");

    /* Array printing - supports DynArray */
    sb_append(sb, "static void nl_print_array(DynArray* arr) {\n");
    sb_append(sb, "    printf(\"[\");\n");
    sb_append(sb, "    for (int i = 0; i < arr->length; i++) {\n");
    sb_append(sb, "        if (i > 0) printf(\", \");\n");
    sb_append(sb, "        switch (arr->elem_type) {\n");
    sb_append(sb, "            case ELEM_INT:\n");
    sb_append(sb, "                printf(\"%lld\", (long long)((int64_t*)arr->data)[i]);\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            case ELEM_FLOAT:\n");
    sb_append(sb, "                printf(\"%g\", ((double*)arr->data)[i]);\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "            default:\n");
    sb_append(sb, "                printf(\"?\");\n");
    sb_append(sb, "                break;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    printf(\"]\");\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_println_array(DynArray* arr) {\n");
    sb_append(sb, "    nl_print_array(arr);\n");
    sb_append(sb, "    printf(\"\\n\");\n");
    sb_append(sb, "}\n\n");

    /* Array operations */
    sb_append(sb, "/* ========== Array Operations (With Bounds Checking!) ========== */\n\n");
    
    sb_append(sb, "/* Array struct */\n");
    /* Old nl_array operations removed - now using DynArray exclusively */
    
    sb_append(sb, "/* ========== End Array Operations ========== */\n\n");

    sb_append(sb, "/* ========== End Math and Utility Built-in Functions ========== */\n\n");

    /* Generate enum typedefs first (before structs, since structs may use enums) */
    sb_append(sb, "/* ========== Enum Definitions ========== */\n\n");
    for (int i = 0; i < env->enum_count; i++) {
        EnumDef *edef = &env->enums[i];
        
        /* Skip runtime-provided enums - they're already defined in nanolang.h */
        if (is_runtime_typedef(edef->name)) {
            continue;
        }
        
        /* Get prefixed enum name */
        const char *prefixed_enum = get_prefixed_type_name(edef->name);
        
        /* Generate typedef enum with prefixed variants */
        sb_appendf(sb, "typedef enum {\n");
        for (int j = 0; j < edef->variant_count; j++) {
            /* Prefix variants: nl_EnumName_VARIANT */
            const char *prefixed_variant = get_prefixed_variant_name(edef->name, edef->variant_names[j]);
            sb_appendf(sb, "    %s = %d",
                      prefixed_variant,
                      edef->variant_values[j]);
            if (j < edef->variant_count - 1) sb_append(sb, ",\n");
            else sb_append(sb, "\n");
        }
        sb_appendf(sb, "} %s;\n\n", prefixed_enum);
    }
    sb_append(sb, "/* ========== End Enum Definitions ========== */\n\n");

    /* Forward declare List types BEFORE structs (in case structs contain List fields) */
    char *detected_list_types_early[32];
    int detected_list_count_early = 0;
    
    if (env && env->generic_instances) {
        for (int i = 0; i < env->generic_instance_count && i < 1000 && detected_list_count_early < 32; i++) {
            GenericInstantiation *inst = &env->generic_instances[i];
            if (inst && strcmp(inst->generic_name, "List") == 0 && inst->type_arg_names && inst->type_arg_names[0]) {
                const char *elem_type = inst->type_arg_names[0];
                bool found = false;
                for (int j = 0; j < detected_list_count_early; j++) {
                    if (strcmp(detected_list_types_early[j], elem_type) == 0) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    detected_list_types_early[detected_list_count_early++] = (char*)elem_type;
                }
            }
        }
    }
    
    if (detected_list_count_early > 0) {
        sb_append(sb, "/* ========== Generic List Forward Declarations ========== */\n");
        for (int i = 0; i < detected_list_count_early; i++) {
            sb_appendf(sb, "typedef struct List_%s List_%s;\n", detected_list_types_early[i], detected_list_types_early[i]);
        }
        sb_append(sb, "/* ========== End Generic List Forward Declarations ========== */\n\n");
    }

    /* Generate struct typedefs */
    sb_append(sb, "/* ========== Struct Definitions ========== */\n\n");
    for (int i = 0; i < env->struct_count; i++) {
        StructDef *sdef = &env->structs[i];
        
        /* Get prefixed name (adds nl_ for user types, keeps runtime types as-is) */
        /* IMPORTANT: Save a copy since get_prefixed_type_name uses a static buffer */
        const char *prefixed_name = strdup(get_prefixed_type_name(sdef->name));
        
        /* Generate typedef struct */
        sb_appendf(sb, "typedef struct %s {\n", prefixed_name);
        for (int j = 0; j < sdef->field_count; j++) {
            sb_append(sb, "    ");
            if (sdef->field_types[j] == TYPE_LIST_GENERIC) {
                /* Generic list field: List<TypeName> -> List_TypeName* */
                if (sdef->field_type_names && sdef->field_type_names[j]) {
                    sb_appendf(sb, "List_%s*", sdef->field_type_names[j]);
                } else {
                    /* Fallback if type name not captured */
                    sb_append(sb, "void* /* List field */");
                }
            } else if (sdef->field_types[j] == TYPE_STRUCT || sdef->field_types[j] == TYPE_UNION || sdef->field_types[j] == TYPE_ENUM) {
                /* Use the actual struct/union/enum type name if available */
                if (sdef->field_type_names && sdef->field_type_names[j]) {
                    const char *field_type_name = get_prefixed_type_name(sdef->field_type_names[j]);
                    sb_append(sb, field_type_name);
                } else {
                    /* Fallback to void* if type name not captured */
                    sb_append(sb, "void* /* composite type field */");
                }
            } else {
                sb_append(sb, type_to_c(sdef->field_types[j]));
            }
            sb_appendf(sb, " %s;\n", sdef->field_names[j]);
        }
        sb_appendf(sb, "} %s;\n\n", prefixed_name);
        free((void*)prefixed_name);  /* Free the duplicated name */
    }
    sb_append(sb, "/* ========== End Struct Definitions ========== */\n\n");

    /* Detect generic list usage BEFORE emitting includes */
    char *detected_list_types[32];
    int detected_list_count = 0;
    
    /* Scan generic instantiations for List<Type> usage */
    if (env && env->generic_instances) {
        for (int i = 0; i < env->generic_instance_count && i < 1000 && detected_list_count < 32; i++) {
            GenericInstantiation *inst = &env->generic_instances[i];
            if (inst && strcmp(inst->generic_name, "List") == 0 && inst->type_arg_names && inst->type_arg_names[0]) {
                const char *elem_type = inst->type_arg_names[0];
                /* Check if already detected */
                bool found = false;
                for (int j = 0; j < detected_list_count; j++) {
                    if (strcmp(detected_list_types[j], elem_type) == 0) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    detected_list_types[detected_list_count++] = (char*)elem_type;
                }
            }
        }
    }
    
    /* Emit includes and function forward declarations */
    if (detected_list_count > 0) {
        /* Emit includes */
        sb_append(sb, "/* ========== Generic List Includes (Auto-Generated) ========== */\n");
        for (int i = 0; i < detected_list_count; i++) {
            sb_appendf(sb, "#include \"/tmp/list_%s.h\"\n", detected_list_types[i]);
        }
        sb_append(sb, "\n/* Function forward declarations */\n");
        for (int i = 0; i < detected_list_count; i++) {
            const char *type_name = detected_list_types[i];
            sb_appendf(sb, "List_%s* nl_list_%s_new(void);\n", type_name, type_name);
            sb_appendf(sb, "void nl_list_%s_push(List_%s *list, nl_%s value);\n", type_name, type_name, type_name);
            sb_appendf(sb, "nl_%s nl_list_%s_get(List_%s *list, int index);\n", type_name, type_name, type_name);
            sb_appendf(sb, "int nl_list_%s_length(List_%s *list);\n", type_name, type_name);
        }
        sb_append(sb, "/* ========== End Generic List Includes ========== */\n\n");
    }

    /* Generate specialized generic list types (skip if using external files) */
    if (detected_list_count == 0) {
        /* Only generate inline if no external list files are being used */
        sb_append(sb, "/* ========== Generic List Specializations ========== */\n\n");
        for (int i = 0; i < env->generic_instance_count; i++) {
            GenericInstantiation *inst = &env->generic_instances[i];
            if (strcmp(inst->generic_name, "List") == 0 && inst->type_arg_names) {
            const char *elem_type = inst->type_arg_names[0];
            const char *specialized_name = inst->concrete_name;
            
            /* Get prefixed struct name (adds nl_ prefix for user types) */
            const char *prefixed_elem_type = get_prefixed_type_name(elem_type);
            
            /* Generate struct definition */
            sb_appendf(sb, "typedef struct {\n");
            sb_appendf(sb, "    %s *data;\n", prefixed_elem_type);
            sb_appendf(sb, "    int count;\n");
            sb_appendf(sb, "    int capacity;\n");
            sb_appendf(sb, "} %s;\n\n", specialized_name);
            
            /* Generate constructor */
            sb_appendf(sb, "%s* %s_new() {\n", specialized_name, specialized_name);
            sb_appendf(sb, "    %s *list = malloc(sizeof(%s));\n", specialized_name, specialized_name);
            sb_appendf(sb, "    list->data = malloc(sizeof(%s) * 4);\n", prefixed_elem_type);
            sb_appendf(sb, "    list->count = 0;\n");
            sb_appendf(sb, "    list->capacity = 4;\n");
            sb_appendf(sb, "    return list;\n");
            sb_appendf(sb, "}\n\n");
            
            /* Generate push function */
            sb_appendf(sb, "void %s_push(%s *list, %s value) {\n",
                      specialized_name, specialized_name, prefixed_elem_type);
            sb_appendf(sb, "    if (list->count >= list->capacity) {\n");
            sb_appendf(sb, "        list->capacity *= 2;\n");
            sb_appendf(sb, "        list->data = realloc(list->data, sizeof(%s) * list->capacity);\n",
                      prefixed_elem_type);
            sb_appendf(sb, "    }\n");
            sb_appendf(sb, "    list->data[list->count++] = value;\n");
            sb_appendf(sb, "}\n\n");
            
            /* Generate get function */
            sb_appendf(sb, "%s %s_get(%s *list, int index) {\n",
                      prefixed_elem_type, specialized_name, specialized_name);
            sb_appendf(sb, "    return list->data[index];\n");
            sb_appendf(sb, "}\n\n");
            
            /* Generate length function */
            sb_appendf(sb, "int %s_length(%s *list) {\n", specialized_name, specialized_name);
            sb_appendf(sb, "    return list->count;\n");
            sb_appendf(sb, "}\n\n");
            }
        }
        sb_append(sb, "/* ========== End Generic List Specializations ========== */\n\n");
    } else {
        /* Using external list implementations */
        sb_append(sb, "/* ========== Generic List Specializations ========== */\n");
        sb_append(sb, "/* (Using external implementations from /tmp/list_*.h) */\n");
        sb_append(sb, "/* ========== End Generic List Specializations ========== */\n\n");
    }

    /* Generate union definitions */
    sb_append(sb, "/* ========== Union Definitions ========== */\n\n");
    for (int i = 0; i < env->union_count; i++) {
        UnionDef *udef = &env->unions[i];
        
        /* Get prefixed union name */
        const char *prefixed_union = get_prefixed_type_name(udef->name);
        
        /* First, generate typedef struct for each variant (so they can be used as types in match) */
        for (int j = 0; j < udef->variant_count; j++) {
            if (udef->variant_field_counts[j] > 0) {
                /* Variant has fields - create typedef struct */
                const char *variant_struct = get_prefixed_variant_struct_name(udef->name, udef->variant_names[j]);
                sb_appendf(sb, "typedef struct {\n");
                for (int k = 0; k < udef->variant_field_counts[j]; k++) {
                    sb_append(sb, "    ");
                    sb_append(sb, type_to_c(udef->variant_field_types[j][k]));
                    sb_appendf(sb, " %s;\n", udef->variant_field_names[j][k]);
                }
                sb_appendf(sb, "} %s;\n\n", variant_struct);
            }
        }
        
        /* Generate tag enum with prefixed name */
        sb_appendf(sb, "typedef enum {\n");
        for (int j = 0; j < udef->variant_count; j++) {
            /* Prefix tag enum variants: nl_UnionName_TAG_VARIANT */
            sb_appendf(sb, "    nl_%s_TAG_%s = %d",
                      udef->name,
                      udef->variant_names[j],
                      j);
            if (j < udef->variant_count - 1) sb_append(sb, ",\n");
            else sb_append(sb, "\n");
        }
        sb_appendf(sb, "} %s_Tag;\n\n", prefixed_union);
        
        /* Generate tagged union struct with prefixed name */
        sb_appendf(sb, "typedef struct %s {\n", prefixed_union);
        sb_appendf(sb, "    %s_Tag tag;\n", prefixed_union);
        sb_append(sb, "    union {\n");
        
        for (int j = 0; j < udef->variant_count; j++) {
            if (udef->variant_field_counts[j] > 0) {
                /* Use the typedef'd variant struct */
                const char *variant_struct = get_prefixed_variant_struct_name(udef->name, udef->variant_names[j]);
                sb_appendf(sb, "        %s %s;\n", variant_struct, udef->variant_names[j]);
            } else {
                /* Variant has no fields - use dummy int */
                sb_appendf(sb, "        int %s; /* empty variant */\n", udef->variant_names[j]);
            }
        }
        
        sb_append(sb, "    } data;\n");
        sb_appendf(sb, "} %s;\n\n", prefixed_union);
    }
    sb_append(sb, "/* ========== End Union Definitions ========== */\n\n");

    /* ========== Function Type Typedefs ========== */
    /* Collect all function signatures used in the program */
    FunctionTypeRegistry *fn_registry = create_fn_type_registry();
    
    /* Create tuple type registry for tuple return types */
    TupleTypeRegistry *tuple_registry = create_tuple_type_registry();
    g_tuple_registry = tuple_registry;  /* Set global registry for expression transpilation */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        if (item->type == AST_FUNCTION) {
            /* Check parameters for function types */
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (item->as.function.params[j].type == TYPE_FUNCTION && 
                    item->as.function.params[j].fn_sig) {
                    register_function_signature(fn_registry, item->as.function.params[j].fn_sig);
                }
            }
            
            /* Check return type for function type */
            if (item->as.function.return_type == TYPE_FUNCTION && 
                item->as.function.return_fn_sig) {
                /* Register the nested function signature */
                register_function_signature(fn_registry, item->as.function.return_fn_sig);
                /* Note: We used to also register an outer signature, but it caused
                 * a double-free bug since the inner signature would be freed twice.
                 * The typedef for the return function signature is sufficient. */
            }
            
            /* Check return type for tuple type */
            if (item->as.function.return_type == TYPE_TUPLE && 
                item->as.function.return_type_info) {
                register_tuple_type(tuple_registry, item->as.function.return_type_info);
            }
            
            /* Collect from function body */
            collect_fn_sigs(item->as.function.body, fn_registry);
            collect_tuple_types_from_stmt(item->as.function.body, tuple_registry);
        }
    }
    
    /* Generate typedef declarations */
    if (fn_registry->count > 0) {
        sb_append(sb, "/* Function Type Typedefs */\n");
        for (int i = 0; i < fn_registry->count; i++) {
            generate_function_typedef(sb, fn_registry->signatures[i],
                                    fn_registry->typedef_names[i]);
        }
        sb_append(sb, "\n");
    }
    
    /* Generate tuple type typedefs */
    if (tuple_registry->count > 0) {
        sb_appendf(sb, "/* Tuple Type Typedefs (found %d types) */\n", tuple_registry->count);
        for (int i = 0; i < tuple_registry->count; i++) {
            generate_tuple_typedef(sb, tuple_registry->tuples[i],
                                 tuple_registry->typedef_names[i]);
        }
        sb_append(sb, "\n");
    }

    /* Generate extern function declarations */
    sb_append(sb, "/* External C function declarations */\n");
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION && item->as.function.is_extern) {
            const char *func_name = item->as.function.name;
            
            /* Skip generic list functions - they're generated by the compiler */
            if (strncmp(func_name, "List_", 5) == 0) {
                continue;
            }
            
            /* Skip runtime list functions - they're declared in runtime headers */
            if (strncmp(func_name, "list_int_", 9) == 0 ||
                strncmp(func_name, "list_string_", 12) == 0 ||
                strncmp(func_name, "list_token_", 11) == 0) {
                continue;
            }
            
            /* Skip standard C library functions - they're already declared in system headers */
            if (strcmp(func_name, "rand") == 0 || strcmp(func_name, "srand") == 0 ||
                strcmp(func_name, "time") == 0 || strcmp(func_name, "malloc") == 0 ||
                strcmp(func_name, "free") == 0 || strcmp(func_name, "printf") == 0 ||
                strcmp(func_name, "fprintf") == 0 || strcmp(func_name, "sprintf") == 0 ||
                strcmp(func_name, "strlen") == 0 || strcmp(func_name, "strcmp") == 0 ||
                strcmp(func_name, "strncmp") == 0 || strcmp(func_name, "strchr") == 0 ||
                strcmp(func_name, "getchar") == 0 || strcmp(func_name, "putchar") == 0 ||
                strcmp(func_name, "isalpha") == 0 || strcmp(func_name, "isdigit") == 0 ||
                strcmp(func_name, "isalnum") == 0 || strcmp(func_name, "islower") == 0 ||
                strcmp(func_name, "isupper") == 0 || strcmp(func_name, "tolower") == 0 ||
                strcmp(func_name, "toupper") == 0 || strcmp(func_name, "isspace") == 0 ||
                strcmp(func_name, "isprint") == 0 || strcmp(func_name, "ispunct") == 0 ||
                strcmp(func_name, "asin") == 0 || strcmp(func_name, "acos") == 0 ||
                strcmp(func_name, "atan") == 0 || strcmp(func_name, "atan2") == 0 ||
                strcmp(func_name, "exp") == 0 || strcmp(func_name, "exp2") == 0 ||
                strcmp(func_name, "log") == 0 || strcmp(func_name, "log10") == 0 ||
                strcmp(func_name, "log2") == 0 || strcmp(func_name, "cbrt") == 0 ||
                strcmp(func_name, "hypot") == 0 || strcmp(func_name, "sinh") == 0 ||
                strcmp(func_name, "cosh") == 0 || strcmp(func_name, "tanh") == 0 ||
                strcmp(func_name, "fmod") == 0 || strcmp(func_name, "fabs") == 0) {
                continue;  /* Skip - already in C stdlib */
            }
            
            /* Generate extern declaration with proper SDL types */
            sb_append(sb, "extern ");
            
            /* Handle return type */
            const char *sdl_ret_type = get_sdl_c_type(func_name, -1, true);
            if (sdl_ret_type) {
                sb_append(sb, sdl_ret_type);
            } else if (item->as.function.return_type == TYPE_STRUCT && item->as.function.return_struct_type_name) {
                /* Check if this is an opaque type */
                OpaqueTypeDef *opaque = env_get_opaque_type(env, item->as.function.return_struct_type_name);
                if (opaque) {
                    /* Opaque types are stored as void* */
                    sb_append(sb, "void*");
                } else {
                    /* Struct return type: use prefixed name */
                    const char *prefixed_name = get_prefixed_type_name(item->as.function.return_struct_type_name);
                    sb_append(sb, prefixed_name);
                }
            } else if (item->as.function.return_type == TYPE_UNION && item->as.function.return_struct_type_name) {
                /* Union return type: use prefixed name */
                const char *prefixed_name = get_prefixed_type_name(item->as.function.return_struct_type_name);
                sb_append(sb, prefixed_name);
            } else if (item->as.function.return_type == TYPE_LIST_GENERIC && item->as.function.return_struct_type_name) {
                /* Generic list return type: List<ElementType> -> List_ElementType* */
                sb_appendf(sb, "List_%s*", item->as.function.return_struct_type_name);
            } else if (item->as.function.return_type == TYPE_INT && 
                      (strncmp(func_name, "SDL_", 4) == 0 || strncmp(func_name, "TTF_", 4) == 0)) {
                /* For SDL functions returning int that might be Uint32, check */
                if (strstr(func_name, "GetTicks")) {
                    sb_append(sb, "Uint32");
                } else {
                    sb_append(sb, type_to_c(item->as.function.return_type));
                }
            } else {
                sb_append(sb, type_to_c(item->as.function.return_type));
            }
            
            sb_appendf(sb, " %s(", func_name);
            
            /* Handle parameters */
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                
                /* Check for SDL-specific parameter types */
                const char *sdl_param_type = get_sdl_c_type(func_name, j, false);
                if (sdl_param_type) {
                    sb_append(sb, sdl_param_type);
                } else if (item->as.function.params[j].type == TYPE_STRUCT && item->as.function.params[j].struct_type_name) {
                    /* Check if this is an opaque type */
                    OpaqueTypeDef *opaque = env_get_opaque_type(env, item->as.function.params[j].struct_type_name);
                    if (opaque) {
                        /* Opaque types are stored as void* */
                        sb_append(sb, "void*");
                    } else {
                        /* Struct parameter: use prefixed name */
                        const char *prefixed_name = get_prefixed_type_name(item->as.function.params[j].struct_type_name);
                        sb_append(sb, prefixed_name);
                    }
                } else if (item->as.function.params[j].type == TYPE_UNION && item->as.function.params[j].struct_type_name) {
                    /* Union parameter: use prefixed name */
                    const char *prefixed_name = get_prefixed_type_name(item->as.function.params[j].struct_type_name);
                    sb_append(sb, prefixed_name);
                } else if (item->as.function.params[j].type == TYPE_LIST_GENERIC && item->as.function.params[j].struct_type_name) {
                    /* Generic list parameter: List<ElementType> -> List_ElementType* */
                    sb_appendf(sb, "List_%s*", item->as.function.params[j].struct_type_name);
                } else {
                    sb_append(sb, type_to_c(item->as.function.params[j].type));
                }
                
                sb_appendf(sb, " %s", item->as.function.params[j].name);
            }
            sb_append(sb, ");\n");
        }
    }
    
    /* Also generate extern declarations for extern functions from imported modules */
    /* BUT: If any modules provide C headers, skip this entirely - the headers declare the functions */
    if (g_module_headers_count == 0 && env && env->functions && env->function_count > 0) {
        for (int i = 0; i < env->function_count; i++) {
            Function *func = &env->functions[i];
            if (!func || !func->name || !func->is_extern) continue;
            
            /* Skip generic list functions - they're generated by the compiler, not extern */
            if (strncmp(func->name, "List_", 5) == 0) {
                continue;
            }
            
            /* Check if this extern function is already in the program AST (declared above) */
            bool in_program = false;
            for (int j = 0; j < program->as.program.count; j++) {
                ASTNode *item = program->as.program.items[j];
                if (item->type == AST_FUNCTION && item->as.function.is_extern &&
                    strcmp(item->as.function.name, func->name) == 0) {
                    in_program = true;
                    break;
                }
            }
            if (in_program) continue;  /* Already declared above */
            
            /* Generate extern declaration for this module extern function */
            sb_append(sb, "extern ");
            
            const char *ret_type_c = type_to_c(func->return_type);
            const char *sdl_ret_type = get_sdl_c_type(func->name, -1, true);
            if (sdl_ret_type) {
                ret_type_c = sdl_ret_type;
            }
            
            sb_append(sb, ret_type_c);
            sb_appendf(sb, " %s(", func->name);
            
            for (int j = 0; j < func->param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                const char *param_type_c = type_to_c(func->params[j].type);
                
                const char *sdl_param_type = get_sdl_c_type(func->name, j, false);
                if (sdl_param_type) {
                    param_type_c = sdl_param_type;
                }
                
                sb_appendf(sb, "%s %s", param_type_c, func->params[j].name);
            }
            sb_append(sb, ");\n");
        }
    }
    
    sb_append(sb, "\n");
    
    /* Forward declare all functions (including module functions) */
    /* First, collect function names from current program AST */
    bool *program_functions = NULL;
    if (env && env->functions && env->function_count > 0) {
        program_functions = calloc(env->function_count, sizeof(bool));
        if (!program_functions) {
            /* Out of memory - skip module function declarations */
            program_functions = NULL;
        } else {
            for (int i = 0; i < program->as.program.count; i++) {
                ASTNode *item = program->as.program.items[i];
                if (item && item->type == AST_FUNCTION) {
                    /* Find function in environment and mark it */
                    for (int j = 0; j < env->function_count; j++) {
                        if (env->functions[j].name && 
                            strcmp(env->functions[j].name, item->as.function.name) == 0) {
                            program_functions[j] = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    /* Forward declare module functions (functions in env but not in program AST) */
    if (env && env->functions && env->function_count > 0 && env->function_count < 10000) {
        sb_append(sb, "/* Forward declarations for module functions */\n");
        for (int i = 0; i < env->function_count; i++) {
            /* Bounds check */
            if (i >= env->function_capacity) break;
            
            Function *func = &env->functions[i];
            if (!func) continue;
            if (!func->name) continue;
            
            /* Skip extern functions - they're declared above */
            if (func->is_extern) continue;
            
            /* Skip built-in functions - they're macros or already declared */
            if (is_builtin_function(func->name)) {
                continue;
            }
            
            /* Skip generic list functions - they're generated by the compiler */
            if (strncmp(func->name, "List_", 5) == 0) {
                continue;
            }
            
            /* Skip functions that are in the current program (they'll be declared below) */
            bool is_program_function = false;
            if (program_functions && i < env->function_count) {
                is_program_function = program_functions[i];
            }
            if (is_program_function) {
                continue;
            }
            
            /* Skip complex return types for now to avoid bus errors */
            /* TODO: Fix struct/union return type handling */
            if (func->return_type == TYPE_STRUCT || func->return_type == TYPE_UNION || func->return_type == TYPE_LIST_GENERIC || func->return_type == TYPE_FUNCTION) {
                continue;  /* Skip complex types for now */
            }
            
            /* Simple forward declaration for basic types only */
            sb_append(sb, type_to_c(func->return_type));
            sb_appendf(sb, " nl_%s(", func->name);
            
            /* Function parameters */
            if (func->params && func->param_count > 0 && func->param_count <= 16) {
                for (int j = 0; j < func->param_count; j++) {
                    if (j > 0) sb_append(sb, ", ");
                    /* Handle struct parameter types */
                    if (func->params[j].type == TYPE_STRUCT && func->params[j].struct_type_name) {
                        /* Check if this is an opaque type */
                        OpaqueTypeDef *opaque = env_get_opaque_type(env, func->params[j].struct_type_name);
                        if (opaque) {
                            /* Opaque types are stored as void* */
                            sb_append(sb, "void*");
                        } else {
                            const char *prefixed_name = get_prefixed_type_name(func->params[j].struct_type_name);
                            if (prefixed_name) {
                                sb_append(sb, prefixed_name);
                            } else {
                                sb_append(sb, type_to_c(func->params[j].type));
                            }
                        }
                    } else if (func->params[j].type == TYPE_LIST_GENERIC && func->params[j].struct_type_name) {
                        sb_appendf(sb, "List_%s*", func->params[j].struct_type_name);
                    } else {
                        sb_append(sb, type_to_c(func->params[j].type));
                    }
                    if (func->params[j].name) {
                        sb_appendf(sb, " %s", func->params[j].name);
                    } else {
                        sb_appendf(sb, " param%d", j);
                    }
                }
            }
            sb_append(sb, ");\n");
        }
        sb_append(sb, "\n");
    }
    
    if (program_functions) {
        free(program_functions);
    }
    
    /* Emit top-level constants */
    sb_append(sb, "/* Top-level constants */\n");
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_LET && !item->as.let.is_mut) {
            /* Skip constants that come from C headers - they're already defined in the headers */
            Symbol *sym = env_get_var(env, item->as.let.name);
            if (sym && sym->from_c_header) {
                continue;  /* Skip - defined in C header */
            }
            
            /* Emit as C constant */
            sb_append(sb, "static const ");
            sb_append(sb, type_to_c(item->as.let.var_type));
            sb_appendf(sb, " %s = ", item->as.let.name);
            transpile_expression(sb, item->as.let.value, env);
            sb_append(sb, ";\n");
        }
    }
    sb_append(sb, "\n");
    
    /* Forward declare functions from current program */
    sb_append(sb, "/* Forward declarations for program functions */\n");
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            /* Skip extern functions - they're declared above */
            if (item->as.function.is_extern) {
                continue;
            }
            
            /* Regular functions - forward declare with nl_ prefix */
            /* Function return type */
            if (item->as.function.return_type == TYPE_FUNCTION && item->as.function.return_fn_sig) {
                /* Function return type: use typedef */
                const char *typedef_name = register_function_signature(fn_registry, 
                                                                      item->as.function.return_fn_sig);
                sb_append(sb, typedef_name);
            } else if (item->as.function.return_type == TYPE_LIST_GENERIC && item->as.function.return_struct_type_name) {
                /* Generic list return type: List<ElementType> -> List_ElementType* */
                sb_appendf(sb, "List_%s*", item->as.function.return_struct_type_name);
            } else if (item->as.function.return_type == TYPE_STRUCT && item->as.function.return_struct_type_name) {
                /* Check if this is an opaque type */
                OpaqueTypeDef *opaque = env_get_opaque_type(env, item->as.function.return_struct_type_name);
                if (opaque) {
                    /* Opaque types are stored as void* */
                    sb_append(sb, "void*");
                } else {
                    /* Use prefixed type name */
                    const char *prefixed_name = get_prefixed_type_name(item->as.function.return_struct_type_name);
                    sb_append(sb, prefixed_name);
                }
            } else if (item->as.function.return_type == TYPE_UNION && item->as.function.return_struct_type_name) {
                /* Use prefixed union name */
                const char *prefixed_name = get_prefixed_type_name(item->as.function.return_struct_type_name);
                sb_append(sb, prefixed_name);
            } else if (item->as.function.return_type == TYPE_TUPLE && item->as.function.return_type_info) {
                /* Use typedef name for tuple return type */
                const char *typedef_name = register_tuple_type(tuple_registry, 
                                                              item->as.function.return_type_info);
                sb_append(sb, typedef_name);
            } else {
                sb_append(sb, type_to_c(item->as.function.return_type));
            }
            
            const char *c_func_name = get_c_func_name(item->as.function.name);
            sb_appendf(sb, " %s(", c_func_name);
            
            /* Function parameters */
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                
                if (item->as.function.params[j].type == TYPE_FUNCTION && item->as.function.params[j].fn_sig) {
                    /* Function parameter: use typedef */
                    const char *typedef_name = register_function_signature(fn_registry, 
                                                                          item->as.function.params[j].fn_sig);
                    sb_appendf(sb, "%s %s", typedef_name, item->as.function.params[j].name);
                } else if (item->as.function.params[j].type == TYPE_LIST_GENERIC && item->as.function.params[j].struct_type_name) {
                    /* Generic list parameter: List<ElementType> -> List_ElementType* */
                    sb_appendf(sb, "List_%s* %s",
                              item->as.function.params[j].struct_type_name,
                              item->as.function.params[j].name);
                } else if (item->as.function.params[j].type == TYPE_STRUCT && item->as.function.params[j].struct_type_name) {
                    /* Check if this is an opaque type */
                    OpaqueTypeDef *opaque = env_get_opaque_type(env, item->as.function.params[j].struct_type_name);
                    if (opaque) {
                        /* Opaque types are stored as void* */
                        sb_appendf(sb, "void* %s", item->as.function.params[j].name);
                    } else {
                        /* Use prefixed type name for regular structs */
                        const char *prefixed_name = get_prefixed_type_name(item->as.function.params[j].struct_type_name);
                        sb_appendf(sb, "%s %s", prefixed_name, item->as.function.params[j].name);
                    }
                } else if (item->as.function.params[j].type == TYPE_UNION && item->as.function.params[j].struct_type_name) {
                    /* Use prefixed union name */
                    const char *prefixed_name = get_prefixed_type_name(item->as.function.params[j].struct_type_name);
                    sb_appendf(sb, "%s %s", prefixed_name, item->as.function.params[j].name);
                } else {
                    sb_appendf(sb, "%s %s",
                              type_to_c(item->as.function.params[j].type),
                              item->as.function.params[j].name);
                }
            }
            sb_append(sb, ");\n");
        }
    }
    sb_append(sb, "\n");

    /* Transpile all functions (skip shadow tests and extern functions) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            /* Skip extern functions - they're declared only, no implementation */
            if (item->as.function.is_extern) {
                continue;
            }
            
            /* Function return type */
            if (item->as.function.return_type == TYPE_FUNCTION && item->as.function.return_fn_sig) {
                /* Function return type: use typedef */
                const char *typedef_name = register_function_signature(fn_registry, 
                                                                      item->as.function.return_fn_sig);
                sb_append(sb, typedef_name);
            } else if (item->as.function.return_type == TYPE_LIST_GENERIC && item->as.function.return_struct_type_name) {
                /* Generic list return type: List<ElementType> -> List_ElementType* */
                sb_appendf(sb, "List_%s*", item->as.function.return_struct_type_name);
            } else if (item->as.function.return_type == TYPE_STRUCT && item->as.function.return_struct_type_name) {
                /* Check if this is an opaque type */
                OpaqueTypeDef *opaque = env_get_opaque_type(env, item->as.function.return_struct_type_name);
                if (opaque) {
                    /* Opaque types are stored as void* */
                    sb_append(sb, "void*");
                } else {
                    /* Use prefixed type name */
                    const char *prefixed_name = get_prefixed_type_name(item->as.function.return_struct_type_name);
                    sb_append(sb, prefixed_name);
                }
            } else if (item->as.function.return_type == TYPE_UNION && item->as.function.return_struct_type_name) {
                /* Use prefixed union name */
                const char *prefixed_name = get_prefixed_type_name(item->as.function.return_struct_type_name);
                sb_append(sb, prefixed_name);
            } else if (item->as.function.return_type == TYPE_TUPLE && item->as.function.return_type_info) {
                /* Use typedef name for tuple return type */
                const char *typedef_name = register_tuple_type(tuple_registry, 
                                                              item->as.function.return_type_info);
                sb_append(sb, typedef_name);
            } else {
                sb_append(sb, type_to_c(item->as.function.return_type));
            }
            
            const char *c_func_name = get_c_func_name(item->as.function.name);
            sb_appendf(sb, " %s(", c_func_name);
            
            /* Function parameters */
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                
                if (item->as.function.params[j].type == TYPE_FUNCTION && item->as.function.params[j].fn_sig) {
                    /* Function parameter: use typedef */
                    const char *typedef_name = register_function_signature(fn_registry, 
                                                                          item->as.function.params[j].fn_sig);
                    sb_appendf(sb, "%s %s", typedef_name, item->as.function.params[j].name);
                } else if (item->as.function.params[j].type == TYPE_LIST_GENERIC && item->as.function.params[j].struct_type_name) {
                    /* Generic list parameter: List<ElementType> -> List_ElementType* */
                    sb_appendf(sb, "List_%s* %s",
                              item->as.function.params[j].struct_type_name,
                              item->as.function.params[j].name);
                } else if (item->as.function.params[j].type == TYPE_STRUCT && item->as.function.params[j].struct_type_name) {
                    /* Check if this is an opaque type */
                    OpaqueTypeDef *opaque = env_get_opaque_type(env, item->as.function.params[j].struct_type_name);
                    if (opaque) {
                        /* Opaque types are stored as void* */
                        sb_appendf(sb, "void* %s", item->as.function.params[j].name);
                    } else {
                        /* Use prefixed type name for regular structs */
                        const char *prefixed_name = get_prefixed_type_name(item->as.function.params[j].struct_type_name);
                        sb_appendf(sb, "%s %s", prefixed_name, item->as.function.params[j].name);
                    }
                } else if (item->as.function.params[j].type == TYPE_UNION && item->as.function.params[j].struct_type_name) {
                    /* Use prefixed union name */
                    const char *prefixed_name = get_prefixed_type_name(item->as.function.params[j].struct_type_name);
                    sb_appendf(sb, "%s %s", prefixed_name, item->as.function.params[j].name);
                } else {
                    sb_appendf(sb, "%s %s",
                              type_to_c(item->as.function.params[j].type),
                              item->as.function.params[j].name);
                }
            }
            sb_append(sb, ") ");

            /* Add parameters to environment for type checking during transpilation */
            int saved_symbol_count = env->symbol_count;
            for (int j = 0; j < item->as.function.param_count; j++) {
                Value dummy_val = create_void();
                /* Preserve element_type for array parameters */
                env_define_var_with_type_info(env, item->as.function.params[j].name,
                             item->as.function.params[j].type, item->as.function.params[j].element_type, 
                             NULL, false, dummy_val);
                
                /* For array<Struct> parameters, set struct_type_name */
                if (item->as.function.params[j].type == TYPE_ARRAY && 
                    item->as.function.params[j].element_type == TYPE_STRUCT &&
                    item->as.function.params[j].struct_type_name) {
                    Symbol *param_sym = env_get_var(env, item->as.function.params[j].name);
                    if (param_sym) {
                        param_sym->struct_type_name = strdup(item->as.function.params[j].struct_type_name);
                    }
                }
            }

            /* Function body */
            transpile_statement(sb, item->as.function.body, 0, env, fn_registry);
            sb_append(sb, "\n");

            /* Restore environment (remove parameters) */
            /* Note: We intentionally don't free the symbol names here because:
             * 1. They will be freed by free_environment() at program end
             * 2. Some symbol metadata might still be referenced during transpilation
             * 3. This is a short-lived compiler process, not a long-running server
             * The memory leak is acceptable for this use case. */
            env->symbol_count = saved_symbol_count;
        }
    }

    /* Add C main() wrapper that calls nl_main() for standalone executables */
    /* Only add if there's a main function (modules don't have main) */
    Function *main_func = env_get_function(env, "main");
    if (main_func && !main_func->is_extern) {
    sb_append(sb, "\n/* C main() entry point - calls nanolang main (nl_main) */\n");
    sb_append(sb, "/* Global argc/argv for CLI runtime support */\n");
    sb_append(sb, "int g_argc = 0;\n");
    sb_append(sb, "char **g_argv = NULL;\n\n");
    sb_append(sb, "int main(int argc, char **argv) {\n");
    sb_append(sb, "    g_argc = argc;\n");
    sb_append(sb, "    g_argv = argv;\n");
    sb_append(sb, "    return (int)nl_main();\n");
    sb_append(sb, "}\n");
    }

    /* Cleanup */
    free_fn_type_registry(fn_registry);
    free_tuple_type_registry(tuple_registry);
    g_tuple_registry = NULL;  /* Clear global registry */
    clear_module_headers();  /* Clear collected headers */

    char *result = sb->buffer;
    free(sb);
    return result;
}