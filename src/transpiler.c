#include "nanolang.h"
#include "module_builder.h"
#include "stdlib_runtime.h"
#include <stdarg.h>
#include <libgen.h>

/* String builder for C code generation - now defined in stdlib_runtime.h */

StringBuilder *sb_create(void) {
    StringBuilder *sb = malloc(sizeof(StringBuilder));
    if (!sb) {
        fprintf(stderr, "Error: Out of memory allocating StringBuilder\n");
        exit(1);
    }
    sb->capacity = 1024;
    sb->length = 0;
    sb->buffer = malloc(sb->capacity);
    if (!sb->buffer) {
        fprintf(stderr, "Error: Out of memory allocating StringBuilder buffer\n");
        free(sb);
        exit(1);
    }
    sb->buffer[0] = '\0';
    return sb;
}

void sb_append(StringBuilder *sb, const char *str) {
    assert(str != NULL);
    int len = safe_strlen(str);
    while (sb->length + len >= sb->capacity) {
        /* Check for overflow before doubling capacity */
        if ((size_t)sb->capacity > SIZE_MAX / 2) {
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

/* Safe helper to build monomorphized type names with bounds checking
 * Returns true on success, false if buffer would overflow */
static bool build_monomorphized_name(char *dest, size_t dest_size, 
                                     const char *base_name, 
                                     const char **type_args, int type_arg_count) {
    if (!dest || !base_name || dest_size == 0) return false;
    
    /* Start with base name */
    size_t pos = 0;
    int written = snprintf(dest + pos, dest_size - pos, "%s", base_name);
    if (written < 0 || (size_t)written >= dest_size - pos) {
        return false;  /* Base name too long */
    }
    pos += written;
    
    /* Append each type argument with underscore separator */
    for (int i = 0; i < type_arg_count; i++) {
        if (!type_args[i]) continue;
        
        /* Append underscore */
        if (pos + 1 >= dest_size) return false;
        dest[pos++] = '_';
        dest[pos] = '\0';
        
        /* Append type arg name */
        written = snprintf(dest + pos, dest_size - pos, "%s", type_args[i]);
        if (written < 0 || (size_t)written >= dest_size - pos) {
            return false;  /* Type arg name too long */
        }
        pos += written;
    }
    
    return true;
}

/* Helper to build monomorphized name from TypeInfo parameters */
static bool build_monomorphized_name_from_typeinfo(char *dest, size_t dest_size,
                                                   const char *base_name,
                                                   TypeInfo **type_params, 
                                                   int type_param_count) {
    if (!dest || !base_name || dest_size == 0) return false;
    if (type_param_count == 0) {
        return snprintf(dest, dest_size, "%s", base_name) < (int)dest_size;
    }
    
    /* Extract type names from TypeInfo structures */
    const char *type_names[32];  /* Max 32 type parameters */
    char tmp_names[32][128];
    if (type_param_count > 32) return false;
    
    for (int i = 0; i < type_param_count; i++) {
        TypeInfo *param = type_params[i];
        if (!param) {
            type_names[i] = "unknown";
            continue;
        }
        
        if (param->base_type == TYPE_INT) {
            type_names[i] = "int";
        } else if (param->base_type == TYPE_U8) {
            type_names[i] = "u8";
        } else if (param->base_type == TYPE_STRING) {
            type_names[i] = "string";
        } else if (param->base_type == TYPE_BOOL) {
            type_names[i] = "bool";
        } else if (param->base_type == TYPE_FLOAT) {
            type_names[i] = "float";
        } else if (param->base_type == TYPE_ARRAY) {
            /* Name arrays as array_<elem>, e.g. array_int, array_u8, array_Point */
            const char *elem = "unknown";
            if (param->element_type) {
                TypeInfo *et = param->element_type;
                if (et->base_type == TYPE_INT) elem = "int";
                else if (et->base_type == TYPE_U8) elem = "u8";
                else if (et->base_type == TYPE_STRING) elem = "string";
                else if (et->base_type == TYPE_BOOL) elem = "bool";
                else if (et->base_type == TYPE_FLOAT) elem = "float";
                else if ((et->base_type == TYPE_STRUCT || et->base_type == TYPE_UNION || et->base_type == TYPE_ENUM) && et->generic_name) elem = et->generic_name;
            }
            snprintf(tmp_names[i], sizeof(tmp_names[i]), "array_%s", elem);
            type_names[i] = tmp_names[i];
        } else if (param->base_type == TYPE_STRUCT && param->generic_name) {
            type_names[i] = param->generic_name;
        } else {
            type_names[i] = "unknown";
        }
    }
    
    return build_monomorphized_name(dest, dest_size, base_name, type_names, type_param_count);
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
    if (strncmp(name, "List_", 5) == 0) {
        return true;
    }
    
    if (strcmp(name, "LexerToken") == 0) {
        return true;
    }
    
    /* Schema types defined in compiler_schema.h are also considered runtime typedefs
     * to avoid redefinition errors when self-hosting. */
    if (strncmp(name, "AST", 3) == 0 ||
        strcmp(name, "ParseNode") == 0 ||
        strcmp(name, "Parser") == 0 ||
        strcmp(name, "CompilerPhase") == 0 ||
        strncmp(name, "Compiler", 8) == 0 ||
        strcmp(name, "Token") == 0 ||
        strcmp(name, "Type") == 0 ||
        strcmp(name, "NSType") == 0 ||
        strcmp(name, "DiagnosticSeverity") == 0 ||
        strcmp(name, "OptionType") == 0 ||
        strstr(name, "PhaseOutput") != NULL) {
        return true;
    }
    
    return false;
}

/* Schema-defined list element types have dedicated runtime list implementations */
static bool is_schema_list_type(const char *name) {
    if (!name) return false;
    if (strncmp(name, "AST", 3) == 0) {
        return true;
    }
    if (strcmp(name, "LexerToken") == 0) {
        return true;
    }
    if (strcmp(name, "CompilerDiagnostic") == 0) {
        return true;
    }
    return false;
}

/* Check if an enum/struct name would conflict with C runtime types */
static bool conflicts_with_runtime(const char *name) {
    /* These are defined in nanolang.h and would cause conflicts */
    if (strcmp(name, "TokenType") == 0 ||
        strcmp(name, "Token") == 0) {
        return true;
    }
    
    /* Schema types should also avoid nl_ prefix to match compiler_schema.h */
    return is_runtime_typedef(name);
}

/* Get prefixed type name for user-defined types */
/* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
static const char *get_prefixed_type_name(const char *name) {
    static _Thread_local char buffer[512];
    
    /* Native types */
    if (strcmp(name, "int") == 0) return "int64_t";
    if (strcmp(name, "u8") == 0) return "uint8_t";
    if (strcmp(name, "float") == 0) return "double";
    if (strcmp(name, "bool") == 0) return "bool";
    if (strcmp(name, "string") == 0) return "const char *";
    if (strcmp(name, "void") == 0) return "void";
    
    /* Special mappings for runtime types */
    if (strcmp(name, "Token") == 0) return "Token";
    if (strcmp(name, "NSType") == 0) return "NSType";
    
    /* Runtime types: no prefix */
    if (is_runtime_typedef(name) || conflicts_with_runtime(name)) {
        return name;
    }
    
    /* User types: add nl_ prefix */
    snprintf(buffer, sizeof(buffer), "nl_%s", name);
    return buffer;
}

/* Get prefixed enum variant name */
/* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
static const char *get_prefixed_variant_name(const char *enum_name, const char *variant_name) {
    static _Thread_local char buffer[512];
    if (is_runtime_typedef(enum_name)) {
        snprintf(buffer, sizeof(buffer), "%s_%s", enum_name, variant_name);
    } else {
        snprintf(buffer, sizeof(buffer), "nl_%s_%s", enum_name, variant_name);
    }
    return buffer;
}

/* Get prefixed variant struct name for unions: UnionName.Variant -> nl_UnionName_Variant */
/* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
static const char *get_prefixed_variant_struct_name(const char *union_name, const char *variant_name) {
    static _Thread_local char buffer[512];
    snprintf(buffer, sizeof(buffer), "nl_%s_%s", union_name, variant_name);
    return buffer;
}

/* Get prefixed union tag name: nl_UnionName_TAG_Variant */
/* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
static const char *get_prefixed_tag_name(const char *union_name, const char *variant_name) {
    static _Thread_local char buffer[512];
    if (is_runtime_typedef(union_name)) {
        snprintf(buffer, sizeof(buffer), "%s_TAG_%s", union_name, variant_name);
    } else {
        snprintf(buffer, sizeof(buffer), "nl_%s_TAG_%s", union_name, variant_name);
    }
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
        if ((size_t)new_capacity > SIZE_MAX / sizeof(ModuleHeader)) {
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
    if (!g_module_headers[g_module_headers_count].name) {
        fprintf(stderr, "Error: Out of memory duplicating module header name\n");
        exit(1);
    }
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
    if (!path_copy) {
        fprintf(stderr, "Error: Out of memory duplicating module path\n");
        exit(1);
    }
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
    if (!reg) {
        fprintf(stderr, "Error: Out of memory allocating FunctionTypeRegistry\n");
        exit(1);
    }
    reg->signatures = malloc(sizeof(FunctionSignature*) * 16);
    if (!reg->signatures) {
        fprintf(stderr, "Error: Out of memory allocating function signatures array\n");
        free(reg);
        exit(1);
    }
    reg->typedef_names = malloc(sizeof(char*) * 16);
    if (!reg->typedef_names) {
        fprintf(stderr, "Error: Out of memory allocating typedef names array\n");
        free(reg->signatures);
        free(reg);
        exit(1);
    }
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
    if (!reg) {
        fprintf(stderr, "Error: Out of memory allocating TupleTypeRegistry\n");
        exit(1);
    }
    reg->tuples = malloc(sizeof(TypeInfo*) * 16);
    if (!reg->tuples) {
        fprintf(stderr, "Error: Out of memory allocating tuples array\n");
        free(reg);
        exit(1);
    }
    reg->typedef_names = malloc(sizeof(char*) * 16);
    if (!reg->typedef_names) {
        fprintf(stderr, "Error: Out of memory allocating tuple typedef names\n");
        free(reg->tuples);
        free(reg);
        exit(1);
    }
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
    
    /* Only allocate what's needed using strdup */
    char *name = strdup(sb->buffer);
    if (!name) {
        fprintf(stderr, "Error: Out of memory duplicating tuple typedef name\n");
        exit(1);
    }
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
        if ((size_t)reg->capacity > SIZE_MAX / 2) {
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
        Type t = info->tuple_types[i];
        if (t == TYPE_STRUCT || t == TYPE_UNION || t == TYPE_ENUM) {
            if (info->tuple_type_names && info->tuple_type_names[i]) {
                const char *prefixed = get_prefixed_type_name(info->tuple_type_names[i]);
                sb_appendf(sb, "%s _%d", prefixed, i);
            } else {
                sb_appendf(sb, "void* /* tuple composite */ _%d", i);
            }
        } else {
            sb_appendf(sb, "%s _%d", type_to_c(t), i);
        }
    }
    sb_appendf(sb, "; } %s;\n", typedef_name);
}

/* Generate unique typedef name for a function signature */
static char *get_function_typedef_name(FunctionSignature *sig, int index) {
    char *name = malloc(64);
    if (!name) {
        fprintf(stderr, "Error: Out of memory allocating function typedef name\n");
        exit(1);
    }
    
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
        if ((size_t)reg->capacity > SIZE_MAX / 2) {
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
        case TYPE_U8: return "uint8_t";
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
        case TYPE_LIST_TOKEN: return "List_Token*";
        case TYPE_LIST_GENERIC: return ""; /* Will be handled specially with type_name */
        case TYPE_OPAQUE: return "void*"; /* Opaque pointers stored as void* */
        default: return "void";
    }
}

/* Get C function name with prefix to avoid conflicts with standard library */
/* Helper: Mangle module name for C identifier (module -> module, std::io -> std__io) */
static void mangle_module_name(char *dest, size_t dest_size, const char *module_name) {
    /* Safety check: ensure valid input */
    if (!dest || dest_size == 0 || !module_name || module_name[0] == '\0') {
        if (dest && dest_size > 0) {
            dest[0] = '\0';
        }
        return;
    }
    
    size_t i = 0, j = 0;
    while (module_name[i] && j < dest_size - 1) {
        /* Only copy valid ASCII/UTF-8 characters */
        unsigned char c = (unsigned char)module_name[i];
        
        if (module_name[i] == ':' && module_name[i+1] == ':') {
            /* Replace :: with __ */
            dest[j++] = '_';
            if (j < dest_size - 1) {
                dest[j++] = '_';
            }
            i += 2;
        } else if (c >= 32 && c < 127) {
            /* Only copy printable ASCII characters */
            dest[j++] = module_name[i++];
        } else {
            /* Skip invalid/non-ASCII characters */
            i++;
        }
    }
    dest[j] = '\0';
}

/* Helper: Get C function name with namespace mangling support */
static const char *get_c_func_name_with_module(const char *nano_name, const char *module_name, bool is_extern) {
    /* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
    static _Thread_local char buffer[512];
    
    /* Extern functions use their original name without any mangling or nl_ prefix */
    if (is_extern) {
        return nano_name;
    }
    
    /* Don't prefix list runtime functions */
    if (strncmp(nano_name, "list_int_", 9) == 0 || 
        strncmp(nano_name, "list_string_", 12) == 0 ||
        strncmp(nano_name, "nl_list_Token_", 11) == 0) {
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
    
    /* If module_name is provided, use namespace mangling: module__func */
    if (module_name && module_name[0] != '\0') {
        /* Validate module_name contains only printable ASCII before using */
        bool valid_module_name = true;
        for (const char *p = module_name; *p; p++) {
            unsigned char c = (unsigned char)*p;
            if (c < 32 || c >= 127) {
                valid_module_name = false;
                break;
            }
        }
        
        if (valid_module_name) {
            char mangled_module[256];
            mangle_module_name(mangled_module, sizeof(mangled_module), module_name);
            /* Only use mangled name if it's non-empty */
            if (mangled_module[0] != '\0') {
                snprintf(buffer, sizeof(buffer), "%s__%s", mangled_module, nano_name);
                return buffer;
            }
        }
    }
    
    /* Legacy: prefix with nl_ for global scope */
    snprintf(buffer, sizeof(buffer), "nl_%s", nano_name);
    return buffer;
}

/* Legacy wrapper - now we always use get_c_func_name_with_module directly
 * NOTE: Kept for reference but disabled to avoid unused function warnings
 */
#if 0  /* Disabled - use get_c_func_name_with_module directly */
static const char *get_c_func_name(const char *nano_name) {
    /* Note: main() now gets nl_ prefix to support library mode (Stage 1.5+) */
    /* Standalone programs use --entry-point to call nl_main() */
    return get_c_func_name_with_module(nano_name, NULL);
}
#endif  /* Disabled - use get_c_func_name_with_module directly */

/* ============================================================================
 * ITERATIVE TRANSPILER IMPLEMENTATION
 * Two-pass architecture: clean and simple!
 * ============================================================================ */

/* Global context for the iterative transpiler */
ASTNode *g_current_function = NULL;

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
                    if (!temp_info) {
                        fprintf(stderr, "Error: Out of memory allocating tuple TypeInfo\n");
                        exit(1);
                    }
                    temp_info->tuple_element_count = expr->as.tuple_literal.element_count;
                    temp_info->tuple_types = malloc(sizeof(Type) * expr->as.tuple_literal.element_count);
                    if (!temp_info->tuple_types) {
                        fprintf(stderr, "Error: Out of memory allocating tuple types array\n");
                        free(temp_info);
                        exit(1);
                    }
                    for (int i = 0; i < expr->as.tuple_literal.element_count; i++) {
                        temp_info->tuple_types[i] = expr->as.tuple_literal.element_types[i];
                    }
                    temp_info->tuple_type_names = NULL;
                    register_tuple_type(reg, temp_info);
                } else {
                    /* Element types not set - infer from elements */
                    TypeInfo *temp_info = malloc(sizeof(TypeInfo));
                    if (!temp_info) {
                        fprintf(stderr, "Error: Out of memory allocating tuple TypeInfo\n");
                        exit(1);
                    }
                    temp_info->tuple_element_count = expr->as.tuple_literal.element_count;
                    temp_info->tuple_types = malloc(sizeof(Type) * expr->as.tuple_literal.element_count);
                    if (!temp_info->tuple_types) {
                        fprintf(stderr, "Error: Out of memory allocating tuple types array\n");
                        free(temp_info);
                        exit(1);
                    }
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

        case AST_COND:
            for (int i = 0; i < expr->as.cond_expr.clause_count; i++) {
                collect_tuple_types_from_expr(expr->as.cond_expr.conditions[i], reg);
                collect_tuple_types_from_expr(expr->as.cond_expr.values[i], reg);
            }
            collect_tuple_types_from_expr(expr->as.cond_expr.else_value, reg);
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
        case AST_COND:
            for (int i = 0; i < stmt->as.cond_expr.clause_count; i++) {
                collect_tuple_types_from_expr(stmt->as.cond_expr.conditions[i], reg);
                collect_tuple_types_from_expr(stmt->as.cond_expr.values[i], reg);
            }
            collect_tuple_types_from_expr(stmt->as.cond_expr.else_value, reg);
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
        case AST_COND:
            /* Cond values are expressions, not statements, so no function signatures to collect */
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

/* Generate C headers and includes */
static void generate_c_headers(StringBuilder *sb) {
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
    sb_append(sb, "#include \"runtime/gc.h\"\n");
    sb_append(sb, "#include \"runtime/dyn_array.h\"\n");
    sb_append(sb, "#include \"nanolang.h\"\n");
    
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
    sb_append(sb, "#include <sys/wait.h>\n");
    sb_append(sb, "#include <spawn.h>\n");
    sb_append(sb, "#include <fcntl.h>\n");
    sb_append(sb, "\n");
}

/* Generate List<T> specializations and forward declarations */
static void generate_list_specializations(Environment *env, StringBuilder *sb) {
    /* Forward declare List types BEFORE structs (in case structs contain List fields) */
    int capacity_early = 32;
    char **detected_list_types_early = malloc(sizeof(char*) * capacity_early);
    if (!detected_list_types_early) {
        fprintf(stderr, "Error: Out of memory allocating list types array\n");
        exit(1);
    }
    int detected_list_count_early = 0;
    
    if (env && env->generic_instances) {
        for (int i = 0; i < env->generic_instance_count && i < 1000; i++) {
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
                    /* Grow array if needed */
                    if (detected_list_count_early >= capacity_early) {
                        capacity_early *= 2;
                        char **new_array = realloc(detected_list_types_early, sizeof(char*) * capacity_early);
                        if (!new_array) {
                            fprintf(stderr, "Error: Out of memory growing list types array to %d\n", capacity_early);
                            free(detected_list_types_early);
                            exit(1);
                        }
                        detected_list_types_early = new_array;
                    }
                    detected_list_types_early[detected_list_count_early++] = (char*)elem_type;
                }
            }
        }
    }
    
    if (detected_list_count_early > 0) {
        sb_append(sb, "/* ========== Generic List Forward Declarations ========== */\n");
        for (int i = 0; i < detected_list_count_early; i++) {
            sb_appendf(sb, "#ifndef FORWARD_DEFINED_List_%s\n", detected_list_types_early[i]);
            sb_appendf(sb, "#define FORWARD_DEFINED_List_%s\n", detected_list_types_early[i]);
            sb_appendf(sb, "typedef struct List_%s List_%s;\n", detected_list_types_early[i], detected_list_types_early[i]);
            sb_append(sb, "#endif\n");
        }
        sb_append(sb, "/* ========== End Generic List Forward Declarations ========== */\n\n");
    }
    
    free(detected_list_types_early);
}

/* Generate List<T> includes and implementations */
static void generate_list_implementations(Environment *env, StringBuilder *sb) {
    /* Detect generic list usage BEFORE emitting includes */
    int capacity = 32;
    char **detected_list_types = malloc(sizeof(char*) * capacity);
    if (!detected_list_types) {
        fprintf(stderr, "Error: Out of memory allocating list types array\n");
        exit(1);
    }
    int detected_list_count = 0;
    
    /* Scan generic instantiations for List<Type> usage */
    if (env && env->generic_instances) {
        for (int i = 0; i < env->generic_instance_count && i < 1000; i++) {
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
                    /* Grow array if needed */
                    if (detected_list_count >= capacity) {
                        capacity *= 2;
                        char **new_array = realloc(detected_list_types, sizeof(char*) * capacity);
                        if (!new_array) {
                            fprintf(stderr, "Error: Out of memory growing list types array to %d\n", capacity);
                            free(detected_list_types);
                            exit(1);
                        }
                        detected_list_types = new_array;
                    }
                    detected_list_types[detected_list_count++] = (char*)elem_type;
                }
            }
        }
    }
    
    if (detected_list_count > 0) {
        bool emitted_runtime_includes = false;
        for (int i = 0; i < detected_list_count; i++) {
            const char *type_name = detected_list_types[i];
            if (is_schema_list_type(type_name)) {
                if (!emitted_runtime_includes) {
                    sb_append(sb, "/* ========== Schema List Runtime Includes ========== */\n");
                    emitted_runtime_includes = true;
                }
                sb_appendf(sb, "#include \"runtime/list_%s.h\"\n", type_name);
            }
        }
        if (emitted_runtime_includes) {
            sb_append(sb, "/* ========== End Schema List Runtime Includes ========== */\n\n");
        }

        bool emitted_specializations = false;
        for (int i = 0; i < detected_list_count; i++) {
            const char *type_name = detected_list_types[i];
            if (is_schema_list_type(type_name)) {
                continue;
            }

            if (!emitted_specializations) {
                sb_append(sb, "/* ========== Generic List Specializations ========== */\n\n");
                emitted_specializations = true;
            }

            const char *prefixed = get_prefixed_type_name(type_name);
            char *prefixed_elem_type = prefixed ? strdup(prefixed) : NULL;
            if (!prefixed_elem_type) {
                fprintf(stderr, "Error: Out of memory duplicating prefixed list type for %s\n", type_name);
                exit(1);
            }
            char specialized_name[256];
            snprintf(specialized_name, sizeof(specialized_name), "List_%s", type_name);

            sb_appendf(sb, "struct %s {\n", specialized_name);
            sb_appendf(sb, "    %s *data;\n", prefixed_elem_type);
            sb_appendf(sb, "    int count;\n");
            sb_appendf(sb, "    int capacity;\n");
            sb_appendf(sb, "};\n\n");

            sb_appendf(sb, "List_%s* nl_list_%s_new(void) {\n", type_name, type_name);
            sb_appendf(sb, "    %s *list = malloc(sizeof(%s));\n", specialized_name, specialized_name);
            sb_appendf(sb, "    if (!list) return NULL;\n");
            sb_appendf(sb, "    list->capacity = 4;\n");
            sb_appendf(sb, "    list->count = 0;\n");
            sb_appendf(sb, "    list->data = malloc(sizeof(%s) * list->capacity);\n", prefixed_elem_type);
            sb_appendf(sb, "    if (!list->data) { free(list); return NULL; }\n");
            sb_appendf(sb, "    return list;\n");
            sb_appendf(sb, "}\n\n");

            sb_appendf(sb, "void nl_list_%s_push(List_%s *list, %s value) {\n",
                      type_name, type_name, prefixed_elem_type);
            sb_appendf(sb, "    if (!list) return;\n");
            sb_appendf(sb, "    if (list->count >= list->capacity) {\n");
            sb_appendf(sb, "        int new_capacity = list->capacity * 2;\n");
            sb_appendf(sb, "        %s *new_data = realloc(list->data, sizeof(%s) * new_capacity);\n",
                      prefixed_elem_type, prefixed_elem_type);
            sb_appendf(sb, "        if (!new_data) return;\n");
            sb_appendf(sb, "        list->data = new_data;\n");
            sb_appendf(sb, "        list->capacity = new_capacity;\n");
            sb_appendf(sb, "    }\n");
            sb_appendf(sb, "    list->data[list->count++] = value;\n");
            sb_appendf(sb, "}\n\n");

            sb_appendf(sb, "%s nl_list_%s_get(List_%s *list, int index) {\n",
                      prefixed_elem_type, type_name, type_name);
            sb_appendf(sb, "    return list->data[index];\n");
            sb_appendf(sb, "}\n\n");

            sb_appendf(sb, "void nl_list_%s_set(List_%s *list, int index, %s value) {\n",
                      type_name, type_name, prefixed_elem_type);
            sb_appendf(sb, "    if (!list) return;\n");
            sb_appendf(sb, "    if (index < 0 || index >= list->count) return;\n");
            sb_appendf(sb, "    list->data[index] = value;\n");
            sb_appendf(sb, "}\n\n");

            sb_appendf(sb, "int nl_list_%s_length(List_%s *list) {\n", type_name, type_name);
            sb_appendf(sb, "    return list ? list->count : 0;\n");
            sb_appendf(sb, "}\n\n");

            free(prefixed_elem_type);
        }

        if (emitted_specializations) {
            sb_append(sb, "/* ========== End Generic List Specializations ========== */\n\n");
        }
    }
    
    free(detected_list_types);
}

/* Generate enum definitions */
static void generate_enum_definitions(Environment *env, StringBuilder *sb) {
    sb_append(sb, "/* ========== Enum Definitions ========== */\n\n");
    for (int i = 0; i < env->enum_count; i++) {
        EnumDef *edef = &env->enums[i];
        
        /* Skip external C types */
        if (edef->is_extern) {
            continue;
        }
        
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
}

/* Generate struct definitions */
static void __attribute__((unused)) generate_struct_definitions(Environment *env, StringBuilder *sb) {
    sb_append(sb, "/* ========== Struct Definitions ========== */\n\n");
    for (int i = 0; i < env->struct_count; i++) {
        StructDef *sdef = &env->structs[i];
        
        /* Skip external C types */
        if (sdef->is_extern) {
            printf("Skipping extern struct: %s\n", sdef->name);
            continue;
        }
        printf("Emitting struct: %s (is_extern=%d)\n", sdef->name, sdef->is_extern);
        
        /* Get prefixed name (adds nl_ for user types, keeps runtime types as-is) */
        /* IMPORTANT: Save a copy since get_prefixed_type_name uses a static buffer */
        const char *prefixed_name = strdup(get_prefixed_type_name(sdef->name));
        if (!prefixed_name) {
            fprintf(stderr, "Error: Out of memory duplicating struct name\n");
            exit(1);
        }
        
        /* Generate typedef struct with guards to prevent redefinition errors with compiler_schema.h */
        sb_appendf(sb, "#ifndef DEFINED_%s\n", prefixed_name);
        sb_appendf(sb, "#define DEFINED_%s\n", prefixed_name);
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
        sb_appendf(sb, "} %s;\n", prefixed_name);
        sb_append(sb, "#endif\n\n");
        free((void*)prefixed_name);  /* Free the duplicated name */
    }
    sb_append(sb, "/* ========== End Struct Definitions ========== */\n\n");
}

/* Generate union definitions */
static void __attribute__((unused)) generate_union_definitions(Environment *env, StringBuilder *sb) {
    sb_append(sb, "/* ========== Union Definitions ========== */\n\n");
    for (int i = 0; i < env->union_count; i++) {
        UnionDef *udef = &env->unions[i];
        
        /* Skip external C types */
        if (udef->is_extern) {
            continue;
        }
        
        /* Skip generic union definitions - they'll be generated as instantiations */
        if (udef->generic_param_count > 0) {
            continue;
        }
        
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
                    Type ft = udef->variant_field_types[j][k];
                    if (ft == TYPE_STRUCT || ft == TYPE_UNION || ft == TYPE_ENUM) {
                        if (udef->variant_field_type_names && udef->variant_field_type_names[j] &&
                            udef->variant_field_type_names[j][k]) {
                            const char *field_type_name = get_prefixed_type_name(udef->variant_field_type_names[j][k]);
                            sb_append(sb, field_type_name);
                        } else {
                            sb_append(sb, "void* /* composite type field */");
                        }
                    } else {
                        sb_append(sb, type_to_c(ft));
                    }
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
    
    /* Generate generic union instantiations */
    if (env && env->generic_instances) {
        sb_append(sb, "/* ========== Generic Union Instantiations ========== */\n\n");
        
        /* Track generated instantiations to avoid duplicates */
        char **generated = malloc(sizeof(char*) * env->generic_instance_count);
        int generated_count = 0;
        
        for (int i = 0; i < env->generic_instance_count && i < 1000; i++) {
            GenericInstantiation *inst = &env->generic_instances[i];
            if (!inst || !inst->generic_name || !inst->type_arg_names) continue;
            
            /* Skip List instantiations (handled separately) */
            if (strcmp(inst->generic_name, "List") == 0) continue;
            
            /* Look up the generic union definition */
            UnionDef *udef = env_get_union(env, inst->generic_name);
            if (!udef || udef->generic_param_count == 0) continue;
            
            /* Verify type arg count matches */
            if (inst->type_arg_count != udef->generic_param_count) continue;
            
            /* Generate monomorphized union name: Result_int_string */
            char monomorphized_name[256];
            if (!build_monomorphized_name(monomorphized_name, sizeof(monomorphized_name),
                                          inst->generic_name, 
                                          (const char **)inst->type_arg_names, 
                                          inst->type_arg_count)) {
                fprintf(stderr, "Warning: Monomorphized type name too long for %s, skipping\n", 
                        inst->generic_name);
                continue;
            }
            
            /* Check if already generated */
            bool already_generated = false;
            for (int j = 0; j < generated_count; j++) {
                if (strcmp(generated[j], monomorphized_name) == 0) {
                    already_generated = true;
                    break;
                }
            }
            if (already_generated) continue;
            
            /* Mark as generated */
            generated[generated_count++] = strdup(monomorphized_name);
            
            const char *prefixed_union = get_prefixed_type_name(monomorphized_name);
            
            /* Generate variant structs with type substitution */
            for (int j = 0; j < udef->variant_count; j++) {
                if (udef->variant_field_counts[j] > 0) {
                    const char *variant_struct = get_prefixed_variant_struct_name(monomorphized_name, udef->variant_names[j]);
                    sb_appendf(sb, "typedef struct {\n");
                    
                    for (int k = 0; k < udef->variant_field_counts[j]; k++) {
                        sb_append(sb, "    ");
                        
                        /* Check if field type is a generic parameter */
                        Type field_type = udef->variant_field_types[j][k];
                        if (field_type == TYPE_GENERIC || field_type == TYPE_STRUCT) {
                            /* Look up which generic parameter this is */
                            bool substituted = false;
                            if (udef->variant_field_type_names && udef->variant_field_type_names[j] &&
                                udef->variant_field_type_names[j][k]) {
                                const char *type_name = udef->variant_field_type_names[j][k];
                                
                                /* Check if it matches a generic parameter */
                                for (int p = 0; p < udef->generic_param_count; p++) {
                                    if (strcmp(type_name, udef->generic_params[p]) == 0) {
                                        /* Substitute with concrete type */
                                        const char *concrete_type = inst->type_arg_names[p];
                                        
                                        /* Map to C type */
                                        if (strcmp(concrete_type, "int") == 0) {
                                            sb_append(sb, "int64_t");
                                        } else if (strcmp(concrete_type, "u8") == 0) {
                                            sb_append(sb, "uint8_t");
                                        } else if (strcmp(concrete_type, "string") == 0) {
                                            sb_append(sb, "const char*");
                                        } else if (strcmp(concrete_type, "bool") == 0) {
                                            sb_append(sb, "bool");
                                        } else if (strcmp(concrete_type, "float") == 0) {
                                            sb_append(sb, "double");
                                        } else if (strcmp(concrete_type, "array") == 0 || strncmp(concrete_type, "array_", 6) == 0) {
                                            sb_append(sb, "DynArray*");
                                        } else {
                                            /* User-defined type */
                                            const char *prefixed_type = get_prefixed_type_name(concrete_type);
                                            sb_append(sb, prefixed_type);
                                        }
                                        substituted = true;
                                        break;
                                    }
                                }
                            }
                            
                            if (!substituted) {
                                /* Fallback: use original type */
                                if (field_type == TYPE_STRUCT || field_type == TYPE_UNION || field_type == TYPE_ENUM) {
                                    if (udef->variant_field_type_names && udef->variant_field_type_names[j] &&
                                        udef->variant_field_type_names[j][k]) {
                                        const char *prefixed_type = get_prefixed_type_name(udef->variant_field_type_names[j][k]);
                                        sb_append(sb, prefixed_type);
                                    } else {
                                        sb_append(sb, "void* /* composite type field */");
                                    }
                                } else if (field_type == TYPE_GENERIC) {
                                    sb_append(sb, "void* /* generic type field */");
                                } else {
                                    sb_append(sb, type_to_c(field_type));
                                }
                            }
                        } else {
                            /* Non-generic field type */
                            if (field_type == TYPE_STRUCT || field_type == TYPE_UNION || field_type == TYPE_ENUM) {
                                if (udef->variant_field_type_names && udef->variant_field_type_names[j] &&
                                    udef->variant_field_type_names[j][k]) {
                                    const char *prefixed_type = get_prefixed_type_name(udef->variant_field_type_names[j][k]);
                                    sb_append(sb, prefixed_type);
                                } else {
                                    sb_append(sb, "void* /* composite type field */");
                                }
                            } else {
                                sb_append(sb, type_to_c(field_type));
                            }
                        }
                        
                        sb_appendf(sb, " %s;\n", udef->variant_field_names[j][k]);
                    }
                    sb_appendf(sb, "} %s;\n\n", variant_struct);
                }
            }
            
            /* Generate tag enum */
            sb_appendf(sb, "typedef enum {\n");
            for (int j = 0; j < udef->variant_count; j++) {
                sb_appendf(sb, "    nl_%s_TAG_%s = %d",
                          monomorphized_name,
                          udef->variant_names[j],
                          j);
                if (j < udef->variant_count - 1) sb_append(sb, ",\n");
                else sb_append(sb, "\n");
            }
            sb_appendf(sb, "} %s_Tag;\n\n", prefixed_union);
            
            /* Generate tagged union struct */
            sb_appendf(sb, "typedef struct %s {\n", prefixed_union);
            sb_appendf(sb, "    %s_Tag tag;\n", prefixed_union);
            sb_append(sb, "    union {\n");
            
            for (int j = 0; j < udef->variant_count; j++) {
                if (udef->variant_field_counts[j] > 0) {
                    const char *variant_struct = get_prefixed_variant_struct_name(monomorphized_name, udef->variant_names[j]);
                    sb_appendf(sb, "        %s %s;\n", variant_struct, udef->variant_names[j]);
                } else {
                    sb_appendf(sb, "        int %s; /* empty variant */\n", udef->variant_names[j]);
                }
            }
            
            sb_append(sb, "    } data;\n");
            sb_appendf(sb, "} %s;\n\n", prefixed_union);
        }
        
        /* Free generated names tracking */
        for (int i = 0; i < generated_count; i++) {
            free(generated[i]);
        }
        free(generated);
        
        sb_append(sb, "/* ========== End Generic Union Instantiations ========== */\n\n");
    }
}

typedef enum {
    NL_CT_STRUCT = 0,
    NL_CT_UNION = 1,
    NL_CT_GENERIC_UNION_INSTANTIATION = 2
} NLCompositeTypeKind;

typedef struct {
    NLCompositeTypeKind kind;
    const char *name;                /* nanolang type name (unprefixed) */
    int index;                       /* env->structs/env->unions/env->generic_instances index */
    UnionDef *generic_union_def;     /* only for instantiations */
} NLCompositeTypeItem;

static int find_composite_type_item(NLCompositeTypeItem *items, int count, const char *name) {
    if (!name) return -1;
    for (int i = 0; i < count; i++) {
        if (items[i].name && strcmp(items[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static void emit_struct_definition_single(Environment *env, StringBuilder *sb, StructDef *sdef) {
    (void)env;
    if (!sdef || !sdef->name) return;

    const char *prefixed_name_dup = strdup(get_prefixed_type_name(sdef->name));
    if (!prefixed_name_dup) {
        fprintf(stderr, "Error: Out of memory duplicating struct name\n");
        exit(1);
    }

    /* Generate typedef struct with guards to prevent redefinition errors */
    sb_appendf(sb, "#ifndef DEFINED_%s\n", prefixed_name_dup);
    sb_appendf(sb, "#define DEFINED_%s\n", prefixed_name_dup);
    
    /* For runtime types, use the name without struct keyword if possible, 
     * but we need to define it if it's not already defined. */
    sb_appendf(sb, "typedef struct %s {\n", prefixed_name_dup);
    for (int j = 0; j < sdef->field_count; j++) {
        sb_append(sb, "    ");

        /* Opaque types are represented as TYPE_STRUCT with a registered opaque type name */
        if (sdef->field_types[j] == TYPE_STRUCT && sdef->field_type_names && sdef->field_type_names[j] &&
            env_get_opaque_type(env, sdef->field_type_names[j])) {
            sb_append(sb, "void*");
        } else if (sdef->field_types[j] == TYPE_LIST_GENERIC) {
            if (sdef->field_type_names && sdef->field_type_names[j]) {
                sb_appendf(sb, "List_%s*", sdef->field_type_names[j]);
            } else {
                sb_append(sb, "void* /* List field */");
            }
        } else if (sdef->field_types[j] == TYPE_STRUCT || sdef->field_types[j] == TYPE_UNION || sdef->field_types[j] == TYPE_ENUM) {
            if (sdef->field_type_names && sdef->field_type_names[j]) {
                const char *field_type_name = get_prefixed_type_name(sdef->field_type_names[j]);
                sb_append(sb, field_type_name);
            } else {
                sb_append(sb, "void* /* composite type field */");
            }
        } else {
            sb_append(sb, type_to_c(sdef->field_types[j]));
        }
        sb_appendf(sb, " %s;\n", sdef->field_names[j]);
    }
    sb_appendf(sb, "} %s;\n", prefixed_name_dup);
    sb_append(sb, "#endif\n\n");
    free((void*)prefixed_name_dup);
}

static void emit_union_definition_single(Environment *env, StringBuilder *sb, UnionDef *udef) {
    (void)env;
    if (!udef || !udef->name) return;
    if (udef->generic_param_count > 0) return;

    const char *prefixed_union = strdup(get_prefixed_type_name(udef->name));
    if (!prefixed_union) {
        fprintf(stderr, "Error: Out of memory duplicating union name\n");
        exit(1);
    }

    for (int j = 0; j < udef->variant_count; j++) {
        if (udef->variant_field_counts[j] > 0) {
            const char *variant_struct = get_prefixed_variant_struct_name(udef->name, udef->variant_names[j]);
            sb_appendf(sb, "typedef struct {\n");
            for (int k = 0; k < udef->variant_field_counts[j]; k++) {
                sb_append(sb, "    ");
                Type ft = udef->variant_field_types[j][k];

                if (ft == TYPE_STRUCT && udef->variant_field_type_names && udef->variant_field_type_names[j] &&
                    udef->variant_field_type_names[j][k] &&
                    env_get_opaque_type(env, udef->variant_field_type_names[j][k])) {
                    sb_append(sb, "void*");
                } else if (ft == TYPE_STRUCT || ft == TYPE_UNION || ft == TYPE_ENUM) {
                    if (udef->variant_field_type_names && udef->variant_field_type_names[j] &&
                        udef->variant_field_type_names[j][k]) {
                        const char *field_type_name = get_prefixed_type_name(udef->variant_field_type_names[j][k]);
                        sb_append(sb, field_type_name);
                    } else {
                        sb_append(sb, "void* /* composite type field */");
                    }
                } else {
                    sb_append(sb, type_to_c(ft));
                }
                sb_appendf(sb, " %s;\n", udef->variant_field_names[j][k]);
            }
            sb_appendf(sb, "} %s;\n\n", variant_struct);
        }
    }

    sb_appendf(sb, "typedef enum {\n");
    for (int j = 0; j < udef->variant_count; j++) {
        sb_appendf(sb, "    nl_%s_TAG_%s = %d", udef->name, udef->variant_names[j], j);
        if (j < udef->variant_count - 1) sb_append(sb, ",\n");
        else sb_append(sb, "\n");
    }
    sb_appendf(sb, "} %s_Tag;\n\n", prefixed_union);

    sb_appendf(sb, "typedef struct %s {\n", prefixed_union);
    sb_appendf(sb, "    %s_Tag tag;\n", prefixed_union);
    sb_append(sb, "    union {\n");
    for (int j = 0; j < udef->variant_count; j++) {
        if (udef->variant_field_counts[j] > 0) {
            const char *variant_struct = get_prefixed_variant_struct_name(udef->name, udef->variant_names[j]);
            sb_appendf(sb, "        %s %s;\n", variant_struct, udef->variant_names[j]);
        } else {
            sb_appendf(sb, "        int %s; /* empty variant */\n", udef->variant_names[j]);
        }
    }
    sb_append(sb, "    } data;\n");
    sb_appendf(sb, "} %s;\n\n", prefixed_union);

    free((void*)prefixed_union);
}

static void emit_generic_union_instantiation(Environment *env, StringBuilder *sb, UnionDef *udef, GenericInstantiation *inst, const char *monomorphized_name) {
    if (!env || !sb || !udef || !inst || !monomorphized_name) return;

    const char *prefixed_union = strdup(get_prefixed_type_name(monomorphized_name));
    if (!prefixed_union) {
        fprintf(stderr, "Error: Out of memory duplicating union name\n");
        exit(1);
    }

    for (int j = 0; j < udef->variant_count; j++) {
        if (udef->variant_field_counts[j] <= 0) continue;

        const char *variant_struct = get_prefixed_variant_struct_name(monomorphized_name, udef->variant_names[j]);
        sb_appendf(sb, "typedef struct {\n");

        for (int k = 0; k < udef->variant_field_counts[j]; k++) {
            sb_append(sb, "    ");

            Type field_type = udef->variant_field_types[j][k];

            /* Check if field type is a generic parameter and substitute */
            bool substituted = false;
            if (field_type == TYPE_GENERIC || field_type == TYPE_STRUCT || field_type == TYPE_UNION || field_type == TYPE_ENUM) {
                if (udef->variant_field_type_names && udef->variant_field_type_names[j] &&
                    udef->variant_field_type_names[j][k]) {
                    const char *type_name = udef->variant_field_type_names[j][k];
                    for (int p = 0; p < udef->generic_param_count; p++) {
                        if (strcmp(type_name, udef->generic_params[p]) == 0) {
                            const char *concrete_type = inst->type_arg_names[p];
                            if (strcmp(concrete_type, "int") == 0) {
                                sb_append(sb, "int64_t");
                            } else if (strcmp(concrete_type, "u8") == 0) {
                                sb_append(sb, "uint8_t");
                            } else if (strcmp(concrete_type, "string") == 0) {
                                sb_append(sb, "const char*");
                            } else if (strcmp(concrete_type, "bool") == 0) {
                                sb_append(sb, "bool");
                            } else if (strcmp(concrete_type, "float") == 0) {
                                sb_append(sb, "double");
                            } else if (strcmp(concrete_type, "array") == 0 || strncmp(concrete_type, "array_", 6) == 0) {
                                sb_append(sb, "DynArray*");
                            } else {
                                const char *prefixed_type = get_prefixed_type_name(concrete_type);
                                sb_append(sb, prefixed_type);
                            }
                            substituted = true;
                            break;
                        }
                    }
                }
            }

            if (!substituted) {
                if ((field_type == TYPE_STRUCT || field_type == TYPE_UNION || field_type == TYPE_ENUM) &&
                    udef->variant_field_type_names && udef->variant_field_type_names[j] &&
                    udef->variant_field_type_names[j][k]) {
                    const char *prefixed_type = get_prefixed_type_name(udef->variant_field_type_names[j][k]);
                    sb_append(sb, prefixed_type);
                } else if (field_type == TYPE_GENERIC) {
                    sb_append(sb, "void* /* generic type field */");
                } else {
                    sb_append(sb, type_to_c(field_type));
                }
            }

            sb_appendf(sb, " %s;\n", udef->variant_field_names[j][k]);
        }

        sb_appendf(sb, "} %s;\n\n", variant_struct);
    }

    sb_appendf(sb, "typedef enum {\n");
    for (int j = 0; j < udef->variant_count; j++) {
        sb_appendf(sb, "    nl_%s_TAG_%s = %d", monomorphized_name, udef->variant_names[j], j);
        if (j < udef->variant_count - 1) sb_append(sb, ",\n");
        else sb_append(sb, "\n");
    }
    sb_appendf(sb, "} %s_Tag;\n\n", prefixed_union);

    sb_appendf(sb, "typedef struct %s {\n", prefixed_union);
    sb_appendf(sb, "    %s_Tag tag;\n", prefixed_union);
    sb_append(sb, "    union {\n");

    for (int j = 0; j < udef->variant_count; j++) {
        if (udef->variant_field_counts[j] > 0) {
            const char *variant_struct = get_prefixed_variant_struct_name(monomorphized_name, udef->variant_names[j]);
            sb_appendf(sb, "        %s %s;\n", variant_struct, udef->variant_names[j]);
        } else {
            sb_appendf(sb, "        int %s; /* empty variant */\n", udef->variant_names[j]);
        }
    }

    sb_append(sb, "    } data;\n");
    sb_appendf(sb, "} %s;\n\n", prefixed_union);

    free((void*)prefixed_union);
}

static void generate_struct_and_union_definitions_ordered(Environment *env, StringBuilder *sb) {
    if (!env || !sb) return;

    sb_append(sb, "/* ========== Struct and Union Definitions ========== */\n\n");

    int capacity = env->struct_count + env->union_count + (env->generic_instance_count > 0 ? env->generic_instance_count : 0);
    NLCompositeTypeItem *items = malloc(sizeof(NLCompositeTypeItem) * capacity);
    if (!items) {
        fprintf(stderr, "Error: Out of memory allocating composite type list\n");
        exit(1);
    }
    int count = 0;

    for (int i = 0; i < env->struct_count; i++) {
        if (!env->structs[i].name) continue;
        if (env->structs[i].is_extern) continue;
        items[count++] = (NLCompositeTypeItem){ .kind = NL_CT_STRUCT, .name = env->structs[i].name, .index = i, .generic_union_def = NULL };
    }

    for (int i = 0; i < env->union_count; i++) {
        UnionDef *udef = &env->unions[i];
        if (!udef || !udef->name) continue;
        if (udef->generic_param_count > 0) continue;
        if (udef->is_extern) continue;
        items[count++] = (NLCompositeTypeItem){ .kind = NL_CT_UNION, .name = udef->name, .index = i, .generic_union_def = NULL };
    }

    /* Add generic union instantiations (monomorphized names) */
    char **generated = NULL;
    int generated_count = 0;
    if (env->generic_instances && env->generic_instance_count > 0) {
        generated = malloc(sizeof(char*) * env->generic_instance_count);
        if (!generated) {
            fprintf(stderr, "Error: Out of memory allocating generic instantiation set\n");
            exit(1);
        }

        for (int i = 0; i < env->generic_instance_count && i < 1000; i++) {
            GenericInstantiation *inst = &env->generic_instances[i];
            if (!inst || !inst->generic_name || !inst->type_arg_names) continue;
            if (strcmp(inst->generic_name, "List") == 0) continue;

            UnionDef *udef = env_get_union(env, inst->generic_name);
            if (!udef || udef->generic_param_count == 0) continue;
            if (inst->type_arg_count != udef->generic_param_count) continue;

            char monomorphized_name_buf[256];
            if (!build_monomorphized_name(monomorphized_name_buf, sizeof(monomorphized_name_buf),
                                          inst->generic_name,
                                          (const char **)inst->type_arg_names,
                                          inst->type_arg_count)) {
                continue;
            }

            bool already = false;
            for (int j = 0; j < generated_count; j++) {
                if (strcmp(generated[j], monomorphized_name_buf) == 0) {
                    already = true;
                    break;
                }
            }
            if (already) continue;

            generated[generated_count++] = strdup(monomorphized_name_buf);
            items[count++] = (NLCompositeTypeItem){
                .kind = NL_CT_GENERIC_UNION_INSTANTIATION,
                .name = generated[generated_count - 1],
                .index = i,
                .generic_union_def = udef
            };
        }
    }

    /* Dependency edges: edge[from][to] means 'from' must be defined before 'to' */
    bool *edges = calloc((size_t)count * (size_t)count, sizeof(bool));
    int *indegree = calloc((size_t)count, sizeof(int));
    bool *emitted = calloc((size_t)count, sizeof(bool));
    if (!edges || !indegree || !emitted) {
        fprintf(stderr, "Error: Out of memory allocating composite type graph\n");
        exit(1);
    }

    for (int i = 0; i < count; i++) {
        NLCompositeTypeItem *it = &items[i];
        if (it->kind == NL_CT_STRUCT) {
            StructDef *sdef = &env->structs[it->index];
            for (int f = 0; f < sdef->field_count; f++) {
                if (sdef->field_types[f] == TYPE_STRUCT || sdef->field_types[f] == TYPE_UNION || sdef->field_types[f] == TYPE_ENUM) {
                    if (!sdef->field_type_names || !sdef->field_type_names[f]) continue;
                    int dep = find_composite_type_item(items, count, sdef->field_type_names[f]);
                    if (dep >= 0 && dep != i && !edges[(size_t)dep * (size_t)count + (size_t)i]) {
                        edges[(size_t)dep * (size_t)count + (size_t)i] = true;
                        indegree[i]++;
                    }
                }
            }
        } else if (it->kind == NL_CT_UNION) {
            UnionDef *udef = &env->unions[it->index];
            for (int v = 0; v < udef->variant_count; v++) {
                for (int f = 0; f < udef->variant_field_counts[v]; f++) {
                    Type ft = udef->variant_field_types[v][f];
                    if (ft != TYPE_STRUCT && ft != TYPE_UNION && ft != TYPE_ENUM) continue;
                    if (!udef->variant_field_type_names || !udef->variant_field_type_names[v] ||
                        !udef->variant_field_type_names[v][f]) {
                        continue;
                    }
                    int dep = find_composite_type_item(items, count, udef->variant_field_type_names[v][f]);
                    if (dep >= 0 && dep != i && !edges[(size_t)dep * (size_t)count + (size_t)i]) {
                        edges[(size_t)dep * (size_t)count + (size_t)i] = true;
                        indegree[i]++;
                    }
                }
            }
        } else {
            /* Generic union instantiation depends on any composite type used as a concrete type argument */
            GenericInstantiation *inst = &env->generic_instances[it->index];
            UnionDef *udef = it->generic_union_def;
            if (!inst || !udef) continue;

            for (int v = 0; v < udef->variant_count; v++) {
                for (int f = 0; f < udef->variant_field_counts[v]; f++) {
                    Type ft = udef->variant_field_types[v][f];
                    if (!udef->variant_field_type_names || !udef->variant_field_type_names[v] ||
                        !udef->variant_field_type_names[v][f]) {
                        continue;
                    }

                    const char *type_name = udef->variant_field_type_names[v][f];
                    const char *concrete_type = NULL;
                    if (ft == TYPE_GENERIC || ft == TYPE_STRUCT || ft == TYPE_UNION || ft == TYPE_ENUM) {
                        for (int p = 0; p < udef->generic_param_count; p++) {
                            if (strcmp(type_name, udef->generic_params[p]) == 0) {
                                concrete_type = inst->type_arg_names[p];
                                break;
                            }
                        }
                    }
                    if (!concrete_type) {
                        /* Non-generic composite field */
                        if (ft == TYPE_STRUCT || ft == TYPE_UNION || ft == TYPE_ENUM) {
                            concrete_type = type_name;
                        }
                    }

                    if (!concrete_type) continue;
                    if (strcmp(concrete_type, "int") == 0 || strcmp(concrete_type, "u8") == 0 ||
                        strcmp(concrete_type, "float") == 0 || strcmp(concrete_type, "bool") == 0 ||
                        strcmp(concrete_type, "string") == 0 || strcmp(concrete_type, "array") == 0 ||
                        strncmp(concrete_type, "array_", 6) == 0) {
                        continue;
                    }

                    int dep = find_composite_type_item(items, count, concrete_type);
                    if (dep >= 0 && dep != i && !edges[(size_t)dep * (size_t)count + (size_t)i]) {
                        edges[(size_t)dep * (size_t)count + (size_t)i] = true;
                        indegree[i]++;
                    }
                }
            }
        }
    }

    /* Emit in topological order (stable: first match wins) */
    for (int step = 0; step < count; step++) {
        int pick = -1;
        for (int i = 0; i < count; i++) {
            if (!emitted[i] && indegree[i] == 0) {
                pick = i;
                break;
            }
        }

        if (pick < 0) {
            /* Cycle or missing type info; emit remaining in original order */
            for (int i = 0; i < count; i++) {
                if (emitted[i]) continue;
                NLCompositeTypeItem *it = &items[i];
                if (it->kind == NL_CT_STRUCT) {
                    emit_struct_definition_single(env, sb, &env->structs[it->index]);
                } else if (it->kind == NL_CT_UNION) {
                    emit_union_definition_single(env, sb, &env->unions[it->index]);
                } else {
                    GenericInstantiation *inst = &env->generic_instances[it->index];
                    emit_generic_union_instantiation(env, sb, it->generic_union_def, inst, it->name);
                }
                emitted[i] = true;
            }
            break;
        }

        NLCompositeTypeItem *it = &items[pick];
        if (it->kind == NL_CT_STRUCT) {
            emit_struct_definition_single(env, sb, &env->structs[it->index]);
        } else if (it->kind == NL_CT_UNION) {
            emit_union_definition_single(env, sb, &env->unions[it->index]);
        } else {
            GenericInstantiation *inst = &env->generic_instances[it->index];
            emit_generic_union_instantiation(env, sb, it->generic_union_def, inst, it->name);
        }

        emitted[pick] = true;
        for (int j = 0; j < count; j++) {
            if (edges[(size_t)pick * (size_t)count + (size_t)j]) {
                indegree[j]--;
            }
        }
    }

    sb_append(sb, "/* ========== End Struct and Union Definitions ========== */\n\n");

    /* Clean up */
    if (generated) {
        for (int i = 0; i < generated_count; i++) {
            free(generated[i]);
        }
        free(generated);
    }
    free(items);
    free(edges);
    free(indegree);
    free(emitted);
}

static void generate_to_string_helpers(Environment *env, StringBuilder *sb) {
    sb_append(sb, "/* ========== To-String Helpers ========== */\n\n");

    /* Forward declarations (structs/unions can reference each other in to_string) */
    sb_append(sb, "/* To-String forward declarations */\n");

    for (int i = 0; i < env->enum_count; i++) {
        EnumDef *edef = &env->enums[i];
        if (!edef || !edef->name) continue;
        const char *prefixed_enum = get_prefixed_type_name(edef->name);
        sb_appendf(sb, "static const char* nl_to_string_%s(%s v);\n", edef->name, prefixed_enum);
    }

    for (int i = 0; i < env->struct_count; i++) {
        StructDef *sdef = &env->structs[i];
        if (!sdef || !sdef->name) continue;
        const char *prefixed_struct = get_prefixed_type_name(sdef->name);
        sb_appendf(sb, "static const char* nl_to_string_%s(%s v);\n", sdef->name, prefixed_struct);
    }

    for (int i = 0; i < env->union_count; i++) {
        UnionDef *udef = &env->unions[i];
        if (!udef || !udef->name) continue;
        if (udef->generic_param_count > 0) continue;
        const char *prefixed_union = get_prefixed_type_name(udef->name);
        sb_appendf(sb, "static const char* nl_to_string_%s(%s u);\n", udef->name, prefixed_union);
    }

    if (env && env->generic_instances) {
        for (int i = 0; i < env->generic_instance_count && i < 1000; i++) {
            GenericInstantiation *inst = &env->generic_instances[i];
            if (!inst || !inst->generic_name || !inst->type_arg_names) continue;

            UnionDef *udef = env_get_union(env, inst->generic_name);
            if (!udef || udef->generic_param_count == 0) continue;
            if (inst->type_arg_count != udef->generic_param_count) continue;

            char monomorphized_name[256];
            if (!build_monomorphized_name(monomorphized_name, sizeof(monomorphized_name),
                                          inst->generic_name,
                                          (const char **)inst->type_arg_names,
                                          inst->type_arg_count)) {
                continue;
            }

            const char *prefixed_union = get_prefixed_type_name(monomorphized_name);
            sb_appendf(sb, "static const char* nl_to_string_%s(%s u);\n", monomorphized_name, prefixed_union);
        }
    }

    sb_append(sb, "\n");

    /* Enums */
    for (int i = 0; i < env->enum_count; i++) {
        EnumDef *edef = &env->enums[i];
        if (!edef || !edef->name) continue;

        const char *prefixed_enum = get_prefixed_type_name(edef->name);
        sb_appendf(sb, "static const char* nl_to_string_%s(%s v) {\n", edef->name, prefixed_enum);
        sb_append(sb, "    switch (v) {\n");
        for (int j = 0; j < edef->variant_count; j++) {
            const char *prefixed_variant = get_prefixed_variant_name(edef->name, edef->variant_names[j]);
            sb_appendf(sb, "        case %s: return \"%s.%s\";\n",
                       prefixed_variant, edef->name, edef->variant_names[j]);
        }
        sb_appendf(sb, "        default: return \"%s.<unknown>\";\n", edef->name);
        sb_append(sb, "    }\n");
        sb_append(sb, "}\n\n");
    }

    /* Structs */
    for (int i = 0; i < env->struct_count; i++) {
        StructDef *sdef = &env->structs[i];
        if (!sdef || !sdef->name) continue;

        const char *prefixed_struct = get_prefixed_type_name(sdef->name);
        sb_appendf(sb, "static const char* nl_to_string_%s(%s v) {\n", sdef->name, prefixed_struct);
        sb_append(sb, "    nl_fmt_sb_t sb = nl_fmt_sb_new(256);\n");
        sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, \"%s { \");\n", sdef->name);

        for (int j = 0; j < sdef->field_count; j++) {
            if (j > 0) {
                sb_append(sb, "    nl_fmt_sb_append_cstr(&sb, \", \");\n");
            }

            sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, \"%s: \");\n", sdef->field_names[j]);

            Type ft = sdef->field_types[j];
            if (ft == TYPE_INT) {
                sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, nl_to_string_int(v.%s));\n", sdef->field_names[j]);
            } else if (ft == TYPE_FLOAT) {
                sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, nl_to_string_float(v.%s));\n", sdef->field_names[j]);
            } else if (ft == TYPE_BOOL) {
                sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, nl_to_string_bool(v.%s));\n", sdef->field_names[j]);
            } else if (ft == TYPE_STRING) {
                sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, nl_to_string_string(v.%s));\n", sdef->field_names[j]);
            } else if (ft == TYPE_ARRAY) {
                sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, nl_to_string_array(v.%s));\n", sdef->field_names[j]);
            } else if (ft == TYPE_ENUM) {
                if (sdef->field_type_names && sdef->field_type_names[j]) {
                    sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, nl_to_string_%s(v.%s));\n",
                               sdef->field_type_names[j], sdef->field_names[j]);
                } else {
                    sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, nl_to_string_int(v.%s));\n", sdef->field_names[j]);
                }
            } else if (ft == TYPE_STRUCT || ft == TYPE_UNION) {
                if (sdef->field_type_names && sdef->field_type_names[j]) {
                    OpaqueTypeDef *opaque = env_get_opaque_type(env, sdef->field_type_names[j]);
                    if (opaque) {
                        sb_append(sb, "    nl_fmt_sb_append_cstr(&sb, \"<opaque>\");\n");
                    } else {
                        sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, nl_to_string_%s(v.%s));\n",
                                   sdef->field_type_names[j], sdef->field_names[j]);
                    }
                } else {
                    sb_append(sb, "    nl_fmt_sb_append_cstr(&sb, \"<struct>\");\n");
                }
            } else {
                sb_append(sb, "    nl_fmt_sb_append_cstr(&sb, \"?\");\n");
            }
        }

        sb_append(sb, "    nl_fmt_sb_append_cstr(&sb, \" }\");\n");
        sb_append(sb, "    return nl_fmt_sb_build(&sb);\n");
        sb_append(sb, "}\n\n");
    }

    /* Unions (non-generic) */
    for (int i = 0; i < env->union_count; i++) {
        UnionDef *udef = &env->unions[i];
        if (!udef || !udef->name) continue;
        if (udef->generic_param_count > 0) continue;

        const char *prefixed_union = get_prefixed_type_name(udef->name);
        sb_appendf(sb, "static const char* nl_to_string_%s(%s u) {\n", udef->name, prefixed_union);
        sb_append(sb, "    nl_fmt_sb_t sb = nl_fmt_sb_new(256);\n");
        sb_append(sb, "    switch (u.tag) {\n");

        for (int j = 0; j < udef->variant_count; j++) {
            const char *prefixed_tag = get_prefixed_tag_name(udef->name, udef->variant_names[j]);
            sb_appendf(sb, "        case %s: {\n", prefixed_tag);
            sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, \"%s.%s\");\n", udef->name, udef->variant_names[j]);
            if (udef->variant_field_counts[j] > 0) {
                sb_append(sb, "            nl_fmt_sb_append_cstr(&sb, \" { \");\n");
                for (int k = 0; k < udef->variant_field_counts[j]; k++) {
                    if (k > 0) sb_append(sb, "            nl_fmt_sb_append_cstr(&sb, \", \");\n");
                    sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, \"%s: \");\n",
                               udef->variant_field_names[j][k]);

                    Type ft = udef->variant_field_types[j][k];
                    if (ft == TYPE_INT) {
                        sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, nl_to_string_int(u.data.%s.%s));\n",
                                   udef->variant_names[j], udef->variant_field_names[j][k]);
                    } else if (ft == TYPE_FLOAT) {
                        sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, nl_to_string_float(u.data.%s.%s));\n",
                                   udef->variant_names[j], udef->variant_field_names[j][k]);
                    } else if (ft == TYPE_BOOL) {
                        sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, nl_to_string_bool(u.data.%s.%s));\n",
                                   udef->variant_names[j], udef->variant_field_names[j][k]);
                    } else if (ft == TYPE_STRING) {
                        sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, nl_to_string_string(u.data.%s.%s));\n",
                                   udef->variant_names[j], udef->variant_field_names[j][k]);
                    } else if (ft == TYPE_ARRAY) {
                        sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, nl_to_string_array(u.data.%s.%s));\n",
                                   udef->variant_names[j], udef->variant_field_names[j][k]);
                    } else if (ft == TYPE_ENUM) {
                        if (udef->variant_field_type_names && udef->variant_field_type_names[j] &&
                            udef->variant_field_type_names[j][k]) {
                            sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, nl_to_string_%s(u.data.%s.%s));\n",
                                       udef->variant_field_type_names[j][k],
                                       udef->variant_names[j],
                                       udef->variant_field_names[j][k]);
                        } else {
                            sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, nl_to_string_int(u.data.%s.%s));\n",
                                       udef->variant_names[j],
                                       udef->variant_field_names[j][k]);
                        }
                    } else if (ft == TYPE_STRUCT || ft == TYPE_UNION) {
                        if (udef->variant_field_type_names && udef->variant_field_type_names[j] &&
                            udef->variant_field_type_names[j][k]) {
                            OpaqueTypeDef *opaque = env_get_opaque_type(env, udef->variant_field_type_names[j][k]);
                            if (opaque) {
                                sb_append(sb, "            nl_fmt_sb_append_cstr(&sb, \"<opaque>\");\n");
                            } else {
                                sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, nl_to_string_%s(u.data.%s.%s));\n",
                                           udef->variant_field_type_names[j][k],
                                           udef->variant_names[j],
                                           udef->variant_field_names[j][k]);
                            }
                        } else {
                            sb_append(sb, "            nl_fmt_sb_append_cstr(&sb, \"<struct>\");\n");
                        }
                    } else {
                        sb_append(sb, "            nl_fmt_sb_append_cstr(&sb, \"?\");\n");
                    }
                }
                sb_append(sb, "            nl_fmt_sb_append_cstr(&sb, \" }\");\n");
            }
            sb_append(sb, "            break;\n");
            sb_append(sb, "        }\n");
        }
        sb_append(sb, "        default: nl_fmt_sb_append_cstr(&sb, \"<union>\");\n");
        sb_append(sb, "    }\n");
        sb_append(sb, "    return nl_fmt_sb_build(&sb);\n");
        sb_append(sb, "}\n\n");
    }

    /* Generic union instantiations (tag-only) */
    if (env && env->generic_instances) {
        for (int i = 0; i < env->generic_instance_count && i < 1000; i++) {
            GenericInstantiation *inst = &env->generic_instances[i];
            if (!inst || !inst->generic_name || !inst->type_arg_names) continue;

            UnionDef *udef = env_get_union(env, inst->generic_name);
            if (!udef || udef->generic_param_count == 0) continue;
            if (inst->type_arg_count != udef->generic_param_count) continue;

            char monomorphized_name[256];
            if (!build_monomorphized_name(monomorphized_name, sizeof(monomorphized_name),
                                          inst->generic_name,
                                          (const char **)inst->type_arg_names,
                                          inst->type_arg_count)) {
                continue;
            }

            const char *prefixed_union = get_prefixed_type_name(monomorphized_name);
            sb_appendf(sb, "static const char* nl_to_string_%s(%s u) {\n", monomorphized_name, prefixed_union);
            sb_append(sb, "    switch (u.tag) {\n");
            for (int j = 0; j < udef->variant_count; j++) {
                sb_appendf(sb, "        case nl_%s_TAG_%s: return \"%s.%s\";\n",
                           monomorphized_name,
                           udef->variant_names[j],
                           inst->generic_name,
                           udef->variant_names[j]);
            }
            sb_appendf(sb, "        default: return \"%s.<unknown>\";\n", inst->generic_name);
            sb_append(sb, "    }\n");
            sb_append(sb, "}\n\n");
        }
    }

    sb_append(sb, "/* ========== End To-String Helpers ========== */\n\n");
}

/* Generate forward declarations for functions defined in imported nanolang modules.
 * These modules are compiled into separate .o files (see compile_modules()), so the
 * main translation unit needs prototypes to avoid implicit-declaration errors. */
static void generate_module_function_declarations(StringBuilder *sb, ASTNode *program, Environment *env, const char *current_file, FunctionTypeRegistry *fn_registry) {
    if (!program || program->type != AST_PROGRAM) return;

    sb_append(sb, "/* Forward declarations for imported module functions */\n");

    /* Track already-emitted declarations to avoid duplicates across repeated imports */
    int emitted_count = 0;
    int emitted_capacity = 64;
    char **emitted = malloc(sizeof(char*) * emitted_capacity);
    if (!emitted) {
        fprintf(stderr, "Error: Out of memory allocating module decl set\n");
        exit(1);
    }

    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (!item || item->type != AST_IMPORT) continue;

        const char *resolved = resolve_module_path(item->as.import_stmt.module_path, current_file);
        if (!resolved) continue;

        ASTNode *module_ast = get_cached_module_ast(resolved);
        free((char*)resolved);  /* Cast away const for free() */
        if (!module_ast || module_ast->type != AST_PROGRAM) continue;

        const char *module_name = NULL;
        for (int j = 0; j < module_ast->as.program.count; j++) {
            ASTNode *mi = module_ast->as.program.items[j];
            if (mi && mi->type == AST_MODULE_DECL && mi->as.module_decl.name) {
                module_name = mi->as.module_decl.name;
                break;
            }
        }

        for (int j = 0; j < module_ast->as.program.count; j++) {
            ASTNode *mi = module_ast->as.program.items[j];
            if (!mi || mi->type != AST_FUNCTION) continue;
            if (!mi->as.function.is_pub) continue;
            /* main is always entry point, not a module function to be imported */
            if (strcmp(mi->as.function.name, "main") == 0) continue;

            const char *c_name = NULL;
            char c_name_buf[512];
            if (mi->as.function.is_extern) {
                /* Extern functions use their literal name */
                c_name = mi->as.function.name;
            } else if (module_name) {
                snprintf(c_name_buf, sizeof(c_name_buf), "%s__%s", module_name, mi->as.function.name);
                c_name = c_name_buf;
            } else {
                c_name = get_c_func_name_with_module(mi->as.function.name, NULL, mi->as.function.is_extern);
            }

            /* De-dupe */
            bool seen = false;
            for (int k = 0; k < emitted_count; k++) {
                if (strcmp(emitted[k], c_name) == 0) { seen = true; break; }
            }
            if (seen) continue;

            if (emitted_count >= emitted_capacity) {
                emitted_capacity *= 2;
                char **new_emitted = realloc(emitted, sizeof(char*) * emitted_capacity);
                if (!new_emitted) {
                    fprintf(stderr, "Error: Out of memory growing module decl set\n");
                    exit(1);
                }
                emitted = new_emitted;
            }
            emitted[emitted_count++] = strdup(c_name);

            sb_append(sb, "extern ");

            /* Return type */
            if (mi->as.function.return_type == TYPE_STRUCT && mi->as.function.return_struct_type_name) {
                OpaqueTypeDef *opaque = env ? env_get_opaque_type(env, mi->as.function.return_struct_type_name) : NULL;
                if (opaque) {
                    sb_append(sb, "void*");
                } else {
                    sb_append(sb, get_prefixed_type_name(mi->as.function.return_struct_type_name));
                }
            } else if (mi->as.function.return_type == TYPE_UNION) {
                if (mi->as.function.return_type_info &&
                    mi->as.function.return_type_info->generic_name &&
                    mi->as.function.return_type_info->type_param_count > 0) {
                    char monomorphized_name[256];
                    if (!build_monomorphized_name_from_typeinfo(
                            monomorphized_name, sizeof(monomorphized_name),
                            mi->as.function.return_type_info->generic_name,
                            mi->as.function.return_type_info->type_params,
                            mi->as.function.return_type_info->type_param_count)) {
                        sb_append(sb, type_to_c(mi->as.function.return_type));
                    } else {
                        sb_append(sb, get_prefixed_type_name(monomorphized_name));
                    }
                } else if (mi->as.function.return_struct_type_name) {
                    sb_append(sb, get_prefixed_type_name(mi->as.function.return_struct_type_name));
                } else {
                    sb_append(sb, type_to_c(mi->as.function.return_type));
                }
            } else if (mi->as.function.return_type == TYPE_LIST_GENERIC && mi->as.function.return_struct_type_name) {
                sb_appendf(sb, "List_%s*", mi->as.function.return_struct_type_name);
            } else if (mi->as.function.return_type == TYPE_FUNCTION) {
                /* Avoid emitting incorrect prototypes (not needed for current stdlib) */
                continue;
            } else {
                sb_append(sb, type_to_c(mi->as.function.return_type));
            }

            sb_appendf(sb, " %s(", c_name);

            /* Parameters */
            for (int p = 0; p < mi->as.function.param_count; p++) {
                if (p > 0) sb_append(sb, ", ");
                Parameter *param = &mi->as.function.params[p];
                if (param->type == TYPE_STRUCT && param->struct_type_name) {
                    OpaqueTypeDef *opaque = env ? env_get_opaque_type(env, param->struct_type_name) : NULL;
                    if (opaque) {
                        sb_append(sb, "void*");
                    } else {
                        sb_append(sb, get_prefixed_type_name(param->struct_type_name));
                    }
                } else if (param->type == TYPE_UNION && param->struct_type_name) {
                    sb_append(sb, get_prefixed_type_name(param->struct_type_name));
                } else if (param->type == TYPE_LIST_GENERIC && param->struct_type_name) {
                    sb_appendf(sb, "List_%s*", param->struct_type_name);
                } else if (param->type == TYPE_FUNCTION && param->fn_sig && fn_registry) {
                    const char *typedef_name = register_function_signature(fn_registry, param->fn_sig);
                    sb_append(sb, typedef_name);
                } else {
                    sb_append(sb, type_to_c(param->type));
                }
                if (param->name) {
                    sb_appendf(sb, " %s", param->name);
                } else {
                    sb_appendf(sb, " param%d", p);
                }
            }
            sb_append(sb, ");\n");
        }
    }

    for (int i = 0; i < emitted_count; i++) {
        free(emitted[i]);
    }
    free(emitted);

    sb_append(sb, "\n");
}

/* Generate forward declarations for program functions */
static void generate_program_function_declarations(StringBuilder *sb, ASTNode *program, 
                                                    Environment *env,
                                                    FunctionTypeRegistry *fn_registry,
                                                    TupleTypeRegistry *tuple_registry) {
    /* Forward declare functions from current program */
    sb_append(sb, "/* Forward declarations for program functions */\n");
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            /* Skip extern functions - they're declared above */
            if (item->as.function.is_extern) {
                continue;
            }
            
            /* Add static for private functions */
            if (!item->as.function.is_pub) {
                sb_append(sb, "static ");
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
            } else if (item->as.function.return_type == TYPE_UNION) {
                /* Check if this is a generic union instantiation */
                if (item->as.function.return_type_info &&
                    item->as.function.return_type_info->generic_name &&
                    item->as.function.return_type_info->type_param_count > 0) {
                    /* Build monomorphized name: Result<int, string> -> Result_int_string */
                    char monomorphized_name[256];
                    if (!build_monomorphized_name_from_typeinfo(
                            monomorphized_name, sizeof(monomorphized_name),
                            item->as.function.return_type_info->generic_name,
                            item->as.function.return_type_info->type_params,
                            item->as.function.return_type_info->type_param_count)) {
                        fprintf(stderr, "Warning: Monomorphized type name too long, using fallback\n");
                        sb_append(sb, type_to_c(item->as.function.return_type));
                    } else {
                        const char *prefixed_name = get_prefixed_type_name(monomorphized_name);
                        sb_append(sb, prefixed_name);
                    }
                } else if (item->as.function.return_struct_type_name) {
                    /* Non-generic union - use prefixed union name */
                const char *prefixed_name = get_prefixed_type_name(item->as.function.return_struct_type_name);
                sb_append(sb, prefixed_name);
                } else {
                    /* Fallback */
                    sb_append(sb, type_to_c(item->as.function.return_type));
                }
            } else if (item->as.function.return_type == TYPE_TUPLE && item->as.function.return_type_info) {
                /* Use typedef name for tuple return type */
                const char *typedef_name = register_tuple_type(tuple_registry, 
                                                              item->as.function.return_type_info);
                sb_append(sb, typedef_name);
            } else {
                sb_append(sb, type_to_c(item->as.function.return_type));
            }
            
            /* Get module name from environment for namespace-aware mangling */
            const char *module_name = NULL;
            Function *func = env_get_function(env, item->as.function.name);
            if (func) {
                module_name = func->module_name;
            }
            /* Use namespace-aware function name (handles module::function -> module__function) */
            const char *c_func_name = get_c_func_name_with_module(item->as.function.name, module_name, item->as.function.is_extern);
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
}

/* Functions for stdlib runtime generation moved to stdlib_runtime.c */
/* Generate function implementations from program AST */
static void generate_function_implementations(StringBuilder *sb, ASTNode *program, Environment *env,
                                              FunctionTypeRegistry *fn_registry, TupleTypeRegistry *tuple_registry) {
    /* Transpile all functions (skip shadow tests and extern functions) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            /* Skip extern functions - they're declared only, no implementation */
            if (item->as.function.is_extern) {
                continue;
            }
            
            /* Add static for private functions */
            if (!item->as.function.is_pub) {
                sb_append(sb, "static ");
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
            } else if (item->as.function.return_type == TYPE_UNION) {
                /* Check if this is a generic union instantiation */
                if (item->as.function.return_type_info &&
                    item->as.function.return_type_info->generic_name &&
                    item->as.function.return_type_info->type_param_count > 0) {
                    /* Build monomorphized name: Result<int, string> -> Result_int_string */
                    char monomorphized_name[256];
                    if (!build_monomorphized_name_from_typeinfo(
                            monomorphized_name, sizeof(monomorphized_name),
                            item->as.function.return_type_info->generic_name,
                            item->as.function.return_type_info->type_params,
                            item->as.function.return_type_info->type_param_count)) {
                        fprintf(stderr, "Warning: Monomorphized type name too long, using fallback\n");
                        sb_append(sb, type_to_c(item->as.function.return_type));
                    } else {
                        const char *prefixed_name = get_prefixed_type_name(monomorphized_name);
                        sb_append(sb, prefixed_name);
                    }
                } else if (item->as.function.return_struct_type_name) {
                    /* Non-generic union - use prefixed union name */
                const char *prefixed_name = get_prefixed_type_name(item->as.function.return_struct_type_name);
                sb_append(sb, prefixed_name);
                } else {
                    /* Fallback */
                    sb_append(sb, type_to_c(item->as.function.return_type));
                }
            } else if (item->as.function.return_type == TYPE_TUPLE && item->as.function.return_type_info) {
                /* Use typedef name for tuple return type */
                const char *typedef_name = register_tuple_type(tuple_registry, 
                                                              item->as.function.return_type_info);
                sb_append(sb, typedef_name);
            } else {
                sb_append(sb, type_to_c(item->as.function.return_type));
            }
            
            /* Get module name from environment for namespace-aware mangling */
            const char *module_name = NULL;
            Function *func = env_get_function(env, item->as.function.name);
            if (func) {
                module_name = func->module_name;
            }
            /* Use namespace-aware function name (handles module::function -> module__function) */
            const char *c_func_name = get_c_func_name_with_module(item->as.function.name, module_name, item->as.function.is_extern);
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
                        if (!param_sym->struct_type_name) {
                            fprintf(stderr, "Error: Out of memory duplicating param struct type name\n");
                            exit(1);
                        }
                    }
                }
            }

            /* Function body */
            g_current_function = item;  /* Set context for union construction in returns */
            transpile_statement(sb, item->as.function.body, 0, env, fn_registry);
            g_current_function = NULL;  /* Clear context */
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
}

/* Generate OS process operations (system, exit, getenv) */
static void generate_process_operations(StringBuilder *sb) {
    sb_append(sb, "static int64_t nl_os_system(const char* command) {\n");
    sb_append(sb, "    return system(command);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static void nl_os_exit(int64_t code) {\n");
    sb_append(sb, "    exit((int)code);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static const char* nl_os_getenv(const char* name) {\n");
    sb_append(sb, "    const char* value = getenv(name);\n");
    sb_append(sb, "    return value ? value : \"\";\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "/* system() wrapper - stdlib system() available via stdlib.h */\n");
    sb_append(sb, "static inline int64_t nl_exec_shell(const char* cmd) {\n");
    sb_append(sb, "    return (int64_t)system(cmd);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static char* nl_os_read_all_fd(int fd) {\n");
    sb_append(sb, "    size_t cap = 4096;\n");
    sb_append(sb, "    size_t len = 0;\n");
    sb_append(sb, "    char* buf = malloc(cap);\n");
    sb_append(sb, "    if (!buf) return strdup(\"\");\n");
    sb_append(sb, "    while (1) {\n");
    sb_append(sb, "        if (len + 1 >= cap) {\n");
    sb_append(sb, "            cap *= 2;\n");
    sb_append(sb, "            char* n = realloc(buf, cap);\n");
    sb_append(sb, "            if (!n) { free(buf); return strdup(\"\"); }\n");
    sb_append(sb, "            buf = n;\n");
    sb_append(sb, "        }\n");
    sb_append(sb, "        ssize_t r = read(fd, buf + len, cap - len - 1);\n");
    sb_append(sb, "        if (r <= 0) break;\n");
    sb_append(sb, "        len += (size_t)r;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "    buf[len] = '\\0';\n");
    sb_append(sb, "    return buf;\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "static DynArray* nl_os_process_run(const char* command) {\n");
    sb_append(sb, "    DynArray* out = dyn_array_new(ELEM_STRING);\n");
    sb_append(sb, "    if (!command) {\n");
    sb_append(sb, "        dyn_array_push_string(out, strdup(\"-1\"));\n");
    sb_append(sb, "        dyn_array_push_string(out, strdup(\"\"));\n");
    sb_append(sb, "        dyn_array_push_string(out, strdup(\"\"));\n");
    sb_append(sb, "        return out;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "\n");
    sb_append(sb, "    int out_pipe[2];\n");
    sb_append(sb, "    int err_pipe[2];\n");
    sb_append(sb, "    if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {\n");
    sb_append(sb, "        dyn_array_push_string(out, strdup(\"-1\"));\n");
    sb_append(sb, "        dyn_array_push_string(out, strdup(\"\"));\n");
    sb_append(sb, "        dyn_array_push_string(out, strdup(\"\"));\n");
    sb_append(sb, "        return out;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "\n");
    sb_append(sb, "    posix_spawn_file_actions_t actions;\n");
    sb_append(sb, "    posix_spawn_file_actions_init(&actions);\n");
    sb_append(sb, "    posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);\n");
    sb_append(sb, "    posix_spawn_file_actions_adddup2(&actions, err_pipe[1], STDERR_FILENO);\n");
    sb_append(sb, "    posix_spawn_file_actions_addclose(&actions, out_pipe[0]);\n");
    sb_append(sb, "    posix_spawn_file_actions_addclose(&actions, err_pipe[0]);\n");
    sb_append(sb, "\n");
    sb_append(sb, "    pid_t pid = 0;\n");
    sb_append(sb, "    char* argv[] = { \"sh\", \"-c\", (char*)command, NULL };\n");
    sb_append(sb, "    extern char **environ;\n");
    sb_append(sb, "    int rc = posix_spawn(&pid, \"/bin/sh\", &actions, NULL, argv, environ);\n");
    sb_append(sb, "    posix_spawn_file_actions_destroy(&actions);\n");
    sb_append(sb, "\n");
    sb_append(sb, "    close(out_pipe[1]);\n");
    sb_append(sb, "    close(err_pipe[1]);\n");
    sb_append(sb, "\n");
    sb_append(sb, "    char* out_s = nl_os_read_all_fd(out_pipe[0]);\n");
    sb_append(sb, "    char* err_s = nl_os_read_all_fd(err_pipe[0]);\n");
    sb_append(sb, "    close(out_pipe[0]);\n");
    sb_append(sb, "    close(err_pipe[0]);\n");
    sb_append(sb, "\n");
    sb_append(sb, "    int code = -1;\n");
    sb_append(sb, "    if (rc != 0) {\n");
    sb_append(sb, "        code = rc;\n");
    sb_append(sb, "    } else {\n");
    sb_append(sb, "        int status = 0;\n");
    sb_append(sb, "        (void)waitpid(pid, &status, 0);\n");
    sb_append(sb, "        if (WIFEXITED(status)) code = WEXITSTATUS(status);\n");
    sb_append(sb, "        else if (WIFSIGNALED(status)) code = 128 + WTERMSIG(status);\n");
    sb_append(sb, "        else code = -1;\n");
    sb_append(sb, "    }\n");
    sb_append(sb, "\n");
    sb_append(sb, "    char code_buf[64];\n");
    sb_append(sb, "    snprintf(code_buf, sizeof(code_buf), \"%d\", code);\n");
    sb_append(sb, "    dyn_array_push_string(out, strdup(code_buf));\n");
    sb_append(sb, "    dyn_array_push_string(out, out_s);\n");
    sb_append(sb, "    dyn_array_push_string(out, err_s);\n");
    sb_append(sb, "    return out;\n");
    sb_append(sb, "}\n\n");
}

/* Generate C main() wrapper that calls nanolang main() */
static void generate_main_wrapper(StringBuilder *sb, Environment *env) {
    /* Only generate if there's a non-extern main function */
    Function *main_func = env_get_function(env, "main");
    if (!main_func || main_func->is_extern) {
        return;
    }
    
    /* Get the mangled name for main (could be module__main) */
    const char *c_main_name = get_c_func_name_with_module("main", main_func->module_name, main_func->is_extern);
    
    sb_append(sb, "\n/* C main() entry point - calls nanolang main */\n");
    sb_append(sb, "/* Global argc/argv for CLI runtime support */\n");
    sb_append(sb, "int g_argc = 0;\n");
    sb_append(sb, "char **g_argv = NULL;\n\n");
    sb_append(sb, "int main(int argc, char **argv) {\n");
    sb_append(sb, "    g_argc = argc;\n");
    sb_append(sb, "    g_argv = argv;\n");
    sb_appendf(sb, "    return (int)%s();\n", c_main_name);
    sb_append(sb, "}\n");
}

/* Generate top-level constants */
static void generate_toplevel_constants(StringBuilder *sb, ASTNode *program, Environment *env) {
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
}

/* Collect module headers from all import statements */
static void collect_module_headers_from_imports(ASTNode *program) {
    clear_module_headers();
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_IMPORT) {
            /* Resolve module path and collect headers */
            const char *module_path = resolve_module_path(item->as.import_stmt.module_path, NULL);
            if (module_path) {
                collect_headers_from_module(module_path);
                free((char*)module_path);  /* Cast away const for free() */
            }
        }
    }
}

/* Generate typedef declarations for function and tuple types */
static void generate_type_typedefs(StringBuilder *sb, FunctionTypeRegistry *fn_registry, 
                                     TupleTypeRegistry *tuple_registry) {
    /* Generate function type typedefs */
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
}

/* Collect function signatures and tuple types from program AST */
static void collect_function_and_tuple_types(ASTNode *program, FunctionTypeRegistry *fn_registry,
                                               TupleTypeRegistry *tuple_registry) {
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
}

/* Collect function type signatures referenced by imported modules so their typedefs
 * are available before emitting module forward declarations. */
static void collect_module_function_types(ASTNode *program, FunctionTypeRegistry *fn_registry, const char *current_file) {
    if (!program || !fn_registry) return;
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (!item || item->type != AST_IMPORT) continue;

        const char *resolved = resolve_module_path(item->as.import_stmt.module_path, current_file);
        if (!resolved) continue;

        ASTNode *module_ast = get_cached_module_ast(resolved);
        free((char*)resolved);  /* Cast away const for free() */
        if (!module_ast || module_ast->type != AST_PROGRAM) continue;

        for (int j = 0; j < module_ast->as.program.count; j++) {
            ASTNode *mi = module_ast->as.program.items[j];
            if (!mi || mi->type != AST_FUNCTION) continue;
            if (!mi->as.function.is_pub) continue;

            for (int p = 0; p < mi->as.function.param_count; p++) {
                if (mi->as.function.params[p].type == TYPE_FUNCTION && mi->as.function.params[p].fn_sig) {
                    register_function_signature(fn_registry, mi->as.function.params[p].fn_sig);
                }
            }

            if (mi->as.function.return_type == TYPE_FUNCTION && mi->as.function.return_fn_sig) {
                register_function_signature(fn_registry, mi->as.function.return_fn_sig);
            }
        }
    }
}

/* Generate module extern declarations (extern functions from imported modules) */
static void generate_module_extern_declarations(StringBuilder *sb, ASTNode *program, Environment *env) {
    /* Generate extern declarations for module wrapper functions (e.g., nl_sqlite3_*)
     * Note: System library functions (e.g., SDL_*, sqlite3_*) are declared in module headers,
     * but module wrapper functions need explicit extern declarations */
    if (env && env->functions && env->function_count > 0) {
        for (int i = 0; i < env->function_count; i++) {
            Function *func = &env->functions[i];
            if (!func || !func->name || !func->is_extern) continue;

            if (strcmp(func->name, "main") == 0) {
                continue;
            }
            
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
            
            /* If modules provide C headers AND this is a system library function (not a wrapper),
             * skip the declaration - it's already in the system header */
            if (g_module_headers_count > 0) {
                /* Check for known system library prefixes */
                bool is_system_function = (
                    strncmp(func->name, "SDL_", 4) == 0 ||
                    strncmp(func->name, "TTF_", 4) == 0 ||
                    strncmp(func->name, "IMG_", 4) == 0 ||
                    strncmp(func->name, "Mix_", 4) == 0 ||
                    strncmp(func->name, "sqlite3_", 8) == 0 ||
                    strncmp(func->name, "curl_", 5) == 0 ||
                    strncmp(func->name, "glfwInit", 8) == 0 ||
                    strncmp(func->name, "glfw", 4) == 0
                );
                
                /* Check for wrapper functions from modules with custom headers
                 * All module wrapper functions start with "nl_" prefix
                 * These are declared in their module headers (e.g., sdl_helpers.h, ui_widgets.h) */
                bool is_module_with_header = (strncmp(func->name, "nl_", 3) == 0);
                
                if (is_system_function || is_module_with_header) {
                    continue;  /* Skip - already in system/module header */
                }
            }
            
            /* Generate extern declaration for this module extern function */
            sb_append(sb, "extern ");
            
            /* Handle return type - check for opaque types */
            const char *sdl_ret_type = get_sdl_c_type(func->name, -1, true);
            if (sdl_ret_type) {
                sb_append(sb, sdl_ret_type);
            } else if (func->return_type == TYPE_STRUCT && func->return_struct_type_name) {
                /* Check if this is an opaque type */
                OpaqueTypeDef *opaque = env_get_opaque_type(env, func->return_struct_type_name);
                if (opaque) {
                    sb_append(sb, "void*");
                } else {
                    const char *prefixed = get_prefixed_type_name(func->return_struct_type_name);
                    sb_append(sb, prefixed);
                }
            } else {
                sb_append(sb, type_to_c(func->return_type));
            }
            
            sb_appendf(sb, " %s(", func->name);
            
            /* Handle parameters - check for opaque types */
            for (int j = 0; j < func->param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                
                const char *sdl_param_type = get_sdl_c_type(func->name, j, false);
                if (sdl_param_type) {
                    sb_append(sb, sdl_param_type);
                } else if (func->params[j].type == TYPE_STRUCT && func->params[j].struct_type_name) {
                    /* Check if this is an opaque type */
                    OpaqueTypeDef *opaque = env_get_opaque_type(env, func->params[j].struct_type_name);
                    if (opaque) {
                        sb_append(sb, "void*");
                    } else {
                        const char *prefixed = get_prefixed_type_name(func->params[j].struct_type_name);
                        sb_append(sb, prefixed);
                    }
                } else {
                    sb_append(sb, type_to_c(func->params[j].type));
                }
                
                sb_appendf(sb, " %s", func->params[j].name);
            }
            sb_append(sb, ");\n");
        }
    }
    
        sb_append(sb, "\n");
    }

/* Generate extern function declarations from program AST */
static bool extern_decl_set_contains(char **set, int count, const char *name) {
    if (!set || !name) return false;
    for (int i = 0; i < count; i++) {
        if (set[i] && strcmp(set[i], name) == 0) {
            return true;
        }
    }
    return false;
}

static void extern_decl_set_add(char ***set, int *count, int *capacity, const char *name) {
    if (!set || !count || !capacity || !name) return;
    if (*count >= *capacity) {
        *capacity *= 2;
        char **new_set = realloc(*set, sizeof(char*) * (*capacity));
        if (!new_set) {
            fprintf(stderr, "Error: Out of memory growing extern decl set\n");
            exit(1);
        }
        *set = new_set;
    }
    (*set)[(*count)++] = strdup(name);
}

static void generate_extern_declarations(StringBuilder *sb, ASTNode *program, Environment *env) {
    sb_append(sb, "/* External C function declarations */\n");

    /* Track what we've already emitted so env-scanned externs don't duplicate AST externs */
    int emitted_count = 0;
    int emitted_capacity = 64;
    char **emitted = malloc(sizeof(char*) * emitted_capacity);
    if (!emitted) {
        fprintf(stderr, "Error: Out of memory allocating extern decl set\n");
        exit(1);
    }

    /* Emit a single extern decl given a Function signature */
    #define EMIT_EXTERN_DECL(_name, _return_type, _return_struct_name, _return_type_info, _params, _param_count) do { \
        const char *func_name = (_name); \
        if (strcmp(func_name, "main") == 0) break; \
        \
        /* Skip generic list functions - they're generated by the compiler */ \
        if (strncmp(func_name, "List_", 5) == 0) break; \
        \
        /* Skip runtime list functions - they're declared in runtime headers */ \
        if (strncmp(func_name, "list_int_", 9) == 0 || \
            strncmp(func_name, "list_string_", 12) == 0 || \
            strncmp(func_name, "nl_list_Token_", 11) == 0) { \
            break; \
        } \
        \
        /* Skip standard C library functions - they're already declared in system headers */ \
        if (strcmp(func_name, "rand") == 0 || strcmp(func_name, "srand") == 0 || \
            strcmp(func_name, "time") == 0 || strcmp(func_name, "malloc") == 0 || \
            strcmp(func_name, "free") == 0 || strcmp(func_name, "printf") == 0 || \
            strcmp(func_name, "fprintf") == 0 || strcmp(func_name, "sprintf") == 0 || \
            strcmp(func_name, "strlen") == 0 || strcmp(func_name, "strcmp") == 0 || \
            strcmp(func_name, "strncmp") == 0 || strcmp(func_name, "strchr") == 0 || \
            strcmp(func_name, "getchar") == 0 || strcmp(func_name, "putchar") == 0 || \
            strcmp(func_name, "isalpha") == 0 || strcmp(func_name, "isdigit") == 0 || \
            strcmp(func_name, "isalnum") == 0 || strcmp(func_name, "islower") == 0 || \
            strcmp(func_name, "isupper") == 0 || strcmp(func_name, "tolower") == 0 || \
            strcmp(func_name, "toupper") == 0 || strcmp(func_name, "isspace") == 0 || \
            strcmp(func_name, "isprint") == 0 || strcmp(func_name, "ispunct") == 0 || \
            strcmp(func_name, "asin") == 0 || strcmp(func_name, "acos") == 0 || \
            strcmp(func_name, "atan") == 0 || strcmp(func_name, "atan2") == 0 || \
            strcmp(func_name, "exp") == 0 || strcmp(func_name, "exp2") == 0 || \
            strcmp(func_name, "log") == 0 || strcmp(func_name, "log10") == 0 || \
            strcmp(func_name, "log2") == 0 || strcmp(func_name, "cbrt") == 0 || \
            strcmp(func_name, "hypot") == 0 || strcmp(func_name, "sinh") == 0 || \
            strcmp(func_name, "cosh") == 0 || strcmp(func_name, "tanh") == 0 || \
            strcmp(func_name, "fmod") == 0 || strcmp(func_name, "fabs") == 0) { \
            break; \
        } \
        \
        if (extern_decl_set_contains(emitted, emitted_count, func_name)) break; \
        \
        sb_append(sb, "extern "); \
        \
        const char *sdl_ret_type = get_sdl_c_type(func_name, -1, true); \
        if (sdl_ret_type) { \
            sb_append(sb, sdl_ret_type); \
        } else if ((_return_type) == TYPE_STRUCT && (_return_struct_name)) { \
            OpaqueTypeDef *opaque = env_get_opaque_type(env, (_return_struct_name)); \
            if (opaque) { \
                sb_append(sb, "void*"); \
            } else { \
                const char *prefixed_name = get_prefixed_type_name((_return_struct_name)); \
                sb_append(sb, prefixed_name); \
            } \
        } else if ((_return_type) == TYPE_UNION) { \
            if ((_return_type_info) && (_return_type_info)->generic_name && (_return_type_info)->type_param_count > 0) { \
                char monomorphized_name[256]; \
                if (!build_monomorphized_name_from_typeinfo( \
                        monomorphized_name, sizeof(monomorphized_name), \
                        (_return_type_info)->generic_name, \
                        (_return_type_info)->type_params, \
                        (_return_type_info)->type_param_count)) { \
                    sb_append(sb, type_to_c((_return_type))); \
                } else { \
                    const char *prefixed_name = get_prefixed_type_name(monomorphized_name); \
                    sb_append(sb, prefixed_name); \
                } \
            } else if ((_return_struct_name)) { \
                const char *prefixed_name = get_prefixed_type_name((_return_struct_name)); \
                sb_append(sb, prefixed_name); \
            } else { \
                sb_append(sb, type_to_c((_return_type))); \
            } \
        } else if ((_return_type) == TYPE_LIST_GENERIC && (_return_struct_name)) { \
            sb_appendf(sb, "List_%s*", (_return_struct_name)); \
        } else if ((_return_type) == TYPE_INT && \
                   (strncmp(func_name, "SDL_", 4) == 0 || strncmp(func_name, "TTF_", 4) == 0)) { \
            if (strstr(func_name, "GetTicks")) { \
                sb_append(sb, "Uint32"); \
            } else { \
                sb_append(sb, type_to_c((_return_type))); \
            } \
        } else { \
            sb_append(sb, type_to_c((_return_type))); \
        } \
        \
        sb_appendf(sb, " %s(", func_name); \
        for (int j = 0; j < (_param_count); j++) { \
            if (j > 0) sb_append(sb, ", "); \
            const char *sdl_param_type = get_sdl_c_type(func_name, j, false); \
            if (sdl_param_type) { \
                sb_append(sb, sdl_param_type); \
            } else if ((_params)[j].type == TYPE_STRUCT && (_params)[j].struct_type_name) { \
                OpaqueTypeDef *opaque = env_get_opaque_type(env, (_params)[j].struct_type_name); \
                if (opaque) { \
                    sb_append(sb, "void*"); \
                } else { \
                    const char *prefixed_name = get_prefixed_type_name((_params)[j].struct_type_name); \
                    sb_append(sb, prefixed_name); \
                } \
            } else if ((_params)[j].type == TYPE_UNION && (_params)[j].struct_type_name) { \
                const char *prefixed_name = get_prefixed_type_name((_params)[j].struct_type_name); \
                sb_append(sb, prefixed_name); \
            } else if ((_params)[j].type == TYPE_LIST_GENERIC && (_params)[j].struct_type_name) { \
                sb_appendf(sb, "List_%s*", (_params)[j].struct_type_name); \
            } else { \
                sb_append(sb, type_to_c((_params)[j].type)); \
            } \
            sb_appendf(sb, " %s", (_params)[j].name); \
        } \
        sb_append(sb, ");\n"); \
        extern_decl_set_add(&emitted, &emitted_count, &emitted_capacity, func_name); \
    } while(0)
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION && item->as.function.is_extern) {
            EMIT_EXTERN_DECL(item->as.function.name,
                             item->as.function.return_type,
                             item->as.function.return_struct_type_name,
                             item->as.function.return_type_info,
                             item->as.function.params,
                             item->as.function.param_count);
        }
    }

    for (int i = 0; i < emitted_count; i++) {
        free(emitted[i]);
    }
    free(emitted);

    #undef EMIT_EXTERN_DECL
}

/* Transpile program to C */
char *transpile_to_c(ASTNode *program, Environment *env, const char *input_file) {
    if (!program || program->type != AST_PROGRAM) {
        return NULL;
    }
    
    if (!env) {
        fprintf(stderr, "Error: Environment is NULL in transpile_to_c\n");
        return NULL;
    }

    /* Clear and collect headers from imported modules */
    collect_module_headers_from_imports(program);

    StringBuilder *sb = sb_create();

    /* POSIX feature macro for strdup, strnlen, etc. */
    sb_append(sb, "#define _POSIX_C_SOURCE 200809L\n\n");

    /* Generate headers */
    generate_c_headers(sb);

    /* OS stdlib runtime library */
    sb_append(sb, "/* ========== OS Standard Library ========== */\n\n");

    /* File operations */
    generate_file_operations(sb);

    /* Directory operations */
    generate_dir_operations(sb);

    /* Path operations */
    generate_path_operations(sb);

    /* Process operations */
    generate_process_operations(sb);

    sb_append(sb, "/* ========== End OS Standard Library ========== */\n\n");

    /* String operations */
    generate_string_operations(sb);

    /* Math and utility built-in functions */
    generate_math_utility_builtins(sb);

    /* Generate enum typedefs first (before structs, since structs may use enums) */
    generate_enum_definitions(env, sb);

    /* Forward declare List types BEFORE structs */
    generate_list_specializations(env, sb);

    /* Generate struct + union definitions in dependency-safe order */
    generate_struct_and_union_definitions_ordered(env, sb);

    /* Generate List implementations and includes */
    generate_list_implementations(env, sb);

    /* Generate to_string helpers for user-defined types */
    generate_to_string_helpers(env, sb);

    /* ========== Function Type Typedefs ========== */
    /* Collect all function signatures and tuple types used in the program */
    FunctionTypeRegistry *fn_registry = create_fn_type_registry();
    TupleTypeRegistry *tuple_registry = create_tuple_type_registry();
    g_tuple_registry = tuple_registry;  /* Set global registry for expression transpilation */
    
    collect_module_function_types(program, fn_registry, input_file);
    collect_function_and_tuple_types(program, fn_registry, tuple_registry);
    
    /* Generate typedef declarations */
    generate_type_typedefs(sb, fn_registry, tuple_registry);

    /* Generate extern function declarations */
    generate_extern_declarations(sb, program, env);
    
    /* Also generate extern declarations for extern functions from imported modules */
    generate_module_extern_declarations(sb, program, env);

    /* Forward declare imported module functions */
    generate_module_function_declarations(sb, program, env, input_file, fn_registry);
    
    /* Emit top-level constants */
    generate_toplevel_constants(sb, program, env);
    
    /* Forward declare functions from current program */
    generate_program_function_declarations(sb, program, env, fn_registry, tuple_registry);

    /* Generate function implementations */
    generate_function_implementations(sb, program, env, fn_registry, tuple_registry);

    /* Add C main() wrapper that calls nl_main() for standalone executables */
    generate_main_wrapper(sb, env);

    /* Cleanup */
    free_fn_type_registry(fn_registry);
    free_tuple_type_registry(tuple_registry);
    g_tuple_registry = NULL;  /* Clear global registry */
    clear_module_headers();  /* Clear collected headers */

    char *result = sb->buffer;
    free(sb);
    return result;
}
