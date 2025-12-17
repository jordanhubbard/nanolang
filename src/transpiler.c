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
/* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
static const char *get_prefixed_type_name(const char *name) {
    static _Thread_local char buffer[512];
    
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
    snprintf(buffer, sizeof(buffer), "nl_%s_%s", enum_name, variant_name);
    return buffer;
}

/* Get prefixed variant struct name for unions: UnionName.Variant -> nl_UnionName_Variant */
/* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
static const char *get_prefixed_variant_struct_name(const char *union_name, const char *variant_name) {
    static _Thread_local char buffer[512];
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
        sb_appendf(sb, "%s _%d", type_to_c(info->tuple_types[i]), i);
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
static const char *get_c_func_name_with_module(const char *nano_name, const char *module_name) {
    /* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
    static _Thread_local char buffer[512];
    
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

static const char *get_c_func_name(const char *nano_name) {
    /* Note: main() now gets nl_ prefix to support library mode (Stage 1.5+) */
    /* Standalone programs use --entry-point to call nl_main() */
    return get_c_func_name_with_module(nano_name, NULL);
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
            sb_appendf(sb, "typedef struct List_%s List_%s;\n", detected_list_types_early[i], detected_list_types_early[i]);
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
            sb_appendf(sb, "    if (!list) return NULL;\n");
            sb_appendf(sb, "    list->data = malloc(sizeof(%s) * 4);\n", prefixed_elem_type);
            sb_appendf(sb, "    if (!list->data) { free(list); return NULL; }\n");
            sb_appendf(sb, "    list->count = 0;\n");
            sb_appendf(sb, "    list->capacity = 4;\n");
            sb_appendf(sb, "    return list;\n");
            sb_appendf(sb, "}\n\n");
            
            /* Generate push function */
            sb_appendf(sb, "void %s_push(%s *list, %s value) {\n",
                      specialized_name, specialized_name, prefixed_elem_type);
            sb_appendf(sb, "    if (list->count >= list->capacity) {\n");
            sb_appendf(sb, "        list->capacity *= 2;\n");
            sb_appendf(sb, "        %s *new_data = realloc(list->data, sizeof(%s) * list->capacity);\n",
                      prefixed_elem_type, prefixed_elem_type);
            sb_appendf(sb, "        if (!new_data) return; /* Out of memory */\n");
            sb_appendf(sb, "        list->data = new_data;\n");
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
    
    free(detected_list_types);
}

/* Generate enum definitions */
static void generate_enum_definitions(Environment *env, StringBuilder *sb) {
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
}

/* Generate struct definitions */
static void generate_struct_definitions(Environment *env, StringBuilder *sb) {
    sb_append(sb, "/* ========== Struct Definitions ========== */\n\n");
    for (int i = 0; i < env->struct_count; i++) {
        StructDef *sdef = &env->structs[i];
        
        /* Get prefixed name (adds nl_ for user types, keeps runtime types as-is) */
        /* IMPORTANT: Save a copy since get_prefixed_type_name uses a static buffer */
        const char *prefixed_name = strdup(get_prefixed_type_name(sdef->name));
        if (!prefixed_name) {
            fprintf(stderr, "Error: Out of memory duplicating struct name\n");
            exit(1);
        }
        
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
}

/* Generate union definitions */
static void generate_union_definitions(Environment *env, StringBuilder *sb) {
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
}

/* Generate forward declarations for module functions */
static void generate_module_function_declarations(StringBuilder *sb, Environment *env, 
                                                   bool *program_functions) {
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
            
            /* Handle return type - List type enabled, others still skipped (incremental rollout) */
            if (func->return_type == TYPE_LIST_GENERIC) {
                if (!func->return_struct_type_name) {
                    continue;  /* Skip - missing element type metadata */
                }
                sb_appendf(sb, "List_%s*", func->return_struct_type_name);
            } else if (func->return_type == TYPE_STRUCT || func->return_type == TYPE_UNION || func->return_type == TYPE_FUNCTION) {
                /* Skip Struct/Union/Function - causes 48 test failures
                 * Root cause: Module functions with complex return types conflict with program declarations
                 * Needs deeper investigation of declaration ordering/visibility (nanolang-cv7)
                 */
                continue;
            } else {
                /* Simple types */
                sb_append(sb, type_to_c(func->return_type));
            }
            
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
            
            /* Get module name from environment for namespace-aware mangling */
            const char *module_name = NULL;
            Function *func = env_get_function(env, item->as.function.name);
            if (func) {
                module_name = func->module_name;
            }
            /* Use namespace-aware function name (handles module::function -> module__function) */
            const char *c_func_name = get_c_func_name_with_module(item->as.function.name, module_name);
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
            
            /* Get module name from environment for namespace-aware mangling */
            const char *module_name = NULL;
            Function *func = env_get_function(env, item->as.function.name);
            if (func) {
                module_name = func->module_name;
            }
            /* Use namespace-aware function name (handles module::function -> module__function) */
            const char *c_func_name = get_c_func_name_with_module(item->as.function.name, module_name);
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

    /* String operations */
    generate_string_operations(sb);

    /* Math and utility built-in functions */
    generate_math_utility_builtins(sb);

    /* Generate enum typedefs first (before structs, since structs may use enums) */
    generate_enum_definitions(env, sb);

    /* Forward declare List types BEFORE structs */
    generate_list_specializations(env, sb);

    /* Generate struct typedefs */
    generate_struct_definitions(env, sb);

    /* Generate List implementations and includes */
    generate_list_implementations(env, sb);

    /* Generate union definitions */
    generate_union_definitions(env, sb);

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
    
    /* Forward declare module functions */
    generate_module_function_declarations(sb, env, program_functions);
    
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
    generate_program_function_declarations(sb, program, env, fn_registry, tuple_registry);

    /* Generate function implementations */
    generate_function_implementations(sb, program, env, fn_registry, tuple_registry);

    /* Add C main() wrapper that calls nl_main() for standalone executables */
    /* Only add if there's a main function (modules don't have main) */
    Function *main_func = env_get_function(env, "main");
    if (main_func && !main_func->is_extern) {
    /* Get the mangled name for main (could be module__main) */
    const char *c_main_name = get_c_func_name_with_module("main", main_func->module_name);
    
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

    /* Cleanup */
    free_fn_type_registry(fn_registry);
    free_tuple_type_registry(tuple_registry);
    g_tuple_registry = NULL;  /* Clear global registry */
    clear_module_headers();  /* Clear collected headers */

    char *result = sb->buffer;
    free(sb);
    return result;
}