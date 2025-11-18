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
        sb->capacity *= 2;
        sb->buffer = realloc(sb->buffer, sb->capacity);
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
static void transpile_expression(StringBuilder *sb, ASTNode *expr, Environment *env);
static const char *type_to_c(Type type);

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

/* Forward declarations */
static void transpile_statement(StringBuilder *sb, ASTNode *stmt, int indent, Environment *env, FunctionTypeRegistry *fn_registry);
static void transpile_expression(StringBuilder *sb, ASTNode *expr, Environment *env);

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
        g_module_headers_capacity = g_module_headers_capacity == 0 ? 8 : g_module_headers_capacity * 2;
        g_module_headers = realloc(g_module_headers, sizeof(ModuleHeader) * g_module_headers_capacity);
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
    free(reg->signatures);
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
    free(reg->tuples);
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
        reg->capacity *= 2;
        reg->tuples = realloc(reg->tuples, sizeof(TypeInfo*) * reg->capacity);
        reg->typedef_names = realloc(reg->typedef_names, sizeof(char*) * reg->capacity);
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
        reg->capacity *= 2;
        reg->signatures = realloc(reg->signatures,
                                 sizeof(FunctionSignature*) * reg->capacity);
        reg->typedef_names = realloc(reg->typedef_names,
                                    sizeof(char*) * reg->capacity);
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
    if (sig->return_type == TYPE_STRUCT && sig->return_struct_name) {
        sb_appendf(sb, "struct %s ", sig->return_struct_name);
    } else {
        sb_appendf(sb, "%s ", type_to_c(sig->return_type));
    }
    
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

/* Get SDL C type for a function parameter/return based on function name and position
 * 
 * This function provides SDL-specific type mapping for FFI. SDL uses custom C types
 * (SDL_Window*, SDL_Renderer*, Uint32, Uint8, etc.) that don't exist in nanolang's
 * type system. nanolang represents these as `int`, but the transpiler needs to know
 * the correct C types for code generation.
 * 
 * This is a feature of the FFI system, not a hack. It enables seamless integration
 * with C libraries that use custom types.
 */
static const char *get_sdl_c_type(const char *func_name, int param_index, bool is_return) {
    if (!func_name || strncmp(func_name, "SDL_", 4) != 0) {
        if (func_name && strncmp(func_name, "TTF_", 4) == 0) {
            /* TTF functions */
            if (is_return) {
                if (strstr(func_name, "OpenFont")) return "TTF_Font*";
                if (strstr(func_name, "RenderText")) return "SDL_Surface*";
            }
            if (param_index == 0 && strstr(func_name, "CloseFont")) return "TTF_Font*";
            if (param_index == 0 && strstr(func_name, "RenderText")) return "TTF_Font*";
        }
        return NULL;
    }
    
    if (is_return) {
        /* Return types */
        if (strstr(func_name, "CreateWindow")) return "SDL_Window*";
        if (strstr(func_name, "CreateRenderer")) return "SDL_Renderer*";
        if (strstr(func_name, "CreateTexture")) return "SDL_Texture*";
        if (strstr(func_name, "GetError")) return "const char*";
        if (strstr(func_name, "GetTicks")) return "Uint32";
        if (strstr(func_name, "PollEvent")) return "int";
        if (strstr(func_name, "Init")) return "int";
        if (strstr(func_name, "RenderClear")) return "int";
        if (strstr(func_name, "SetRenderDrawColor")) return "int";
        return NULL;
    } else {
        /* Parameter types */
        if (param_index == 0) {
            /* First parameter */
            if (strstr(func_name, "DestroyWindow")) return "SDL_Window*";
            if (strstr(func_name, "DestroyRenderer")) return "SDL_Renderer*";
            if (strstr(func_name, "DestroyTexture")) return "SDL_Texture*";
            if (strstr(func_name, "CreateRenderer")) return "SDL_Window*";
            if (strstr(func_name, "RenderClear") || strstr(func_name, "RenderPresent") ||
                strstr(func_name, "SetRenderDrawColor") || strstr(func_name, "RenderFillRect") ||
                strstr(func_name, "RenderDrawPoint") || strstr(func_name, "RenderDrawLine") ||
                strstr(func_name, "SetRenderDrawBlendMode") || strstr(func_name, "RenderCopy") ||
                strstr(func_name, "CreateTextureFromSurface")) return "SDL_Renderer*";
            if (strstr(func_name, "QueryTexture")) return "SDL_Texture*";
            if (strstr(func_name, "PollEvent")) return "SDL_Event*";
            if (strstr(func_name, "FreeSurface")) return "SDL_Surface*";
        }
        if (strstr(func_name, "RenderFillRect") && param_index == 1) return "const SDL_Rect*";
        if (strstr(func_name, "RenderCopy")) {
            if (param_index == 2) return "const SDL_Rect*";
            if (param_index == 3) return "const SDL_Rect*";
        }
        if (strstr(func_name, "QueryTexture") && param_index >= 2) return "int*";
        if (strstr(func_name, "RenderText_Solid") && param_index == 2) return "SDL_Color";
        /* SDL functions that take Uint32 for flags/delays */
        if (strstr(func_name, "Init") && param_index == 0) return "Uint32";
        if (strstr(func_name, "Delay") && param_index == 0) return "Uint32";
        if (strstr(func_name, "CreateWindow")) {
            if (param_index == 1 || param_index == 2 || param_index == 3 || param_index == 4) return "int";
            if (param_index == 5) return "Uint32";
        }
        if (strstr(func_name, "CreateRenderer")) {
            if (param_index == 1) return "int";
            if (param_index == 2) return "Uint32";
        }
        if (strstr(func_name, "SetRenderDrawColor")) {
            if (param_index >= 1 && param_index <= 4) return "Uint8";
        }
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
        case TYPE_VOID: return "void";
        case TYPE_ARRAY: return "DynArray*";  /* All arrays are now dynamic arrays with GC */
        case TYPE_STRUCT: return "struct"; /* Will be extended with struct name */
        case TYPE_UNION: return ""; /* Union names are used directly (typedef'd) */
        case TYPE_FUNCTION: return ""; /* Will be handled with typedef */
        case TYPE_LIST_INT: return "List_int*";
        case TYPE_LIST_STRING: return "List_string*";
        case TYPE_LIST_TOKEN: return "List_token*";
        case TYPE_LIST_GENERIC: return ""; /* Will be handled specially with type_name */
        default: return "void";
    }
}

/* Get C type for struct (returns "struct StructName") */
/* Extract struct type name from an expression (returns NULL if not a struct) */
static const char *get_struct_type_from_expr(ASTNode *expr) {
    if (!expr) return NULL;
    
    if (expr->type == AST_STRUCT_LITERAL) {
        return expr->as.struct_literal.struct_name;
    }
    
    /* For identifiers, calls, field access - would need type environment lookup */
    /* For now, return NULL and let caller handle it */
    return NULL;
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

/* Transpile expression to C */
static void transpile_expression(StringBuilder *sb, ASTNode *expr, Environment *env) {
    if (!expr) return;

    switch (expr->type) {
        case AST_NUMBER:
            sb_appendf(sb, "%lldLL", expr->as.number);
            break;

        case AST_FLOAT:
            // Always output at least one decimal place to ensure float type in C
            if (expr->as.float_val == (double)(int64_t)expr->as.float_val) {
                // It's a whole number like 0.0 or 1.0 - add .0
                sb_appendf(sb, "%.1f", expr->as.float_val);
            } else {
                // Has decimal places - use %g
                sb_appendf(sb, "%g", expr->as.float_val);
            }
            break;

        case AST_STRING:
            sb_appendf(sb, "\"%s\"", expr->as.string_val);
            break;

        case AST_BOOL:
            sb_append(sb, expr->as.bool_val ? "true" : "false");
            break;

        case AST_IDENTIFIER: {
            /* Check if this identifier is a function name */
            Function *func_def = env_get_function(env, expr->as.identifier);
            if (func_def && !func_def->is_extern && func_def->body != NULL) {
                /* User-defined function (has body) - add nl_ prefix */
                sb_append(sb, get_c_func_name(expr->as.identifier));
            } else {
                /* Check if this is an immutable constant that should be inlined */
                /* This handles imported constants from other modules */
                int var_index = -1;
                for (int i = env->symbol_count - 1; i >= 0; i--) {
                    if (strcmp(env->symbols[i].name, expr->as.identifier) == 0) {
                        var_index = i;
                        break;
                    }
                }
                
                if (var_index >= 0 && !env->symbols[var_index].is_mut) {
                    /* This is an immutable variable - inline its value if it's a constant */
                    Value val = env->symbols[var_index].value;
                    if (val.type == VAL_INT) {
                        char num_buf[64];
                        snprintf(num_buf, sizeof(num_buf), "%lld", val.as.int_val);
                        sb_append(sb, num_buf);
                        break;
                    } else if (val.type == VAL_FLOAT) {
                        char num_buf[64];
                        snprintf(num_buf, sizeof(num_buf), "%g", val.as.float_val);
                        sb_append(sb, num_buf);
                        break;
                    } else if (val.type == VAL_BOOL) {
                        sb_append(sb, val.as.bool_val ? "true" : "false");
                        break;
                    }
                    /* For other types, fall through to normal variable reference */
                }
                
                /* Variable, builtin function, or extern function - use as-is */
                sb_append(sb, expr->as.identifier);
            }
            break;
        }

        case AST_PREFIX_OP: {
            TokenType op = expr->as.prefix_op.op;
            int arg_count = expr->as.prefix_op.arg_count;

            if (arg_count == 2) {
                /* Check if this is a string comparison - use strcmp instead of == */
                bool is_string_comparison = false;
                if (op == TOKEN_EQ || op == TOKEN_NE) {
                    Type arg1_type = check_expression(expr->as.prefix_op.args[0], env);
                    Type arg2_type = check_expression(expr->as.prefix_op.args[1], env);
                    if (arg1_type == TYPE_STRING && arg2_type == TYPE_STRING) {
                        is_string_comparison = true;
                    }
                }

                if (is_string_comparison) {
                    /* String comparison: use strcmp */
                    sb_append(sb, "(strcmp(");
                    transpile_expression(sb, expr->as.prefix_op.args[0], env);
                    sb_append(sb, ", ");
                    transpile_expression(sb, expr->as.prefix_op.args[1], env);
                    if (op == TOKEN_EQ) {
                        sb_append(sb, ") == 0)");
                    } else {  /* TOKEN_NE */
                        sb_append(sb, ") != 0)");
                    }
                } else {
                    /* Regular binary operation */
                    /* Only add parentheses for arithmetic to ensure correct precedence */
                    bool needs_parens = (op == TOKEN_PLUS || op == TOKEN_MINUS || 
                                       op == TOKEN_STAR || op == TOKEN_SLASH || op == TOKEN_PERCENT);
                    
                    if (needs_parens) sb_append(sb, "(");
                    transpile_expression(sb, expr->as.prefix_op.args[0], env);
                    switch (op) {
                        case TOKEN_PLUS: sb_append(sb, " + "); break;
                        case TOKEN_MINUS: sb_append(sb, " - "); break;
                        case TOKEN_STAR: sb_append(sb, " * "); break;
                        case TOKEN_SLASH: sb_append(sb, " / "); break;
                        case TOKEN_PERCENT: sb_append(sb, " % "); break;
                        case TOKEN_EQ: sb_append(sb, " == "); break;
                        case TOKEN_NE: sb_append(sb, " != "); break;
                        case TOKEN_LT: sb_append(sb, " < "); break;
                        case TOKEN_LE: sb_append(sb, " <= "); break;
                        case TOKEN_GT: sb_append(sb, " > "); break;
                        case TOKEN_GE: sb_append(sb, " >= "); break;
                        case TOKEN_AND: sb_append(sb, " && "); break;
                        case TOKEN_OR: sb_append(sb, " || "); break;
                        default: sb_append(sb, " OP "); break;
                    }
                    transpile_expression(sb, expr->as.prefix_op.args[1], env);
                    if (needs_parens) sb_append(sb, ")");
                }
            } else if (arg_count == 1 && op == TOKEN_NOT) {
                sb_append(sb, "(!");
                transpile_expression(sb, expr->as.prefix_op.args[0], env);
                sb_append(sb, ")");
            } else {
                sb_append(sb, "(/* unknown op */)");
            }

            break;
        }

        case AST_CALL: {
            /* Map nanolang OS function names to C implementation names */
            const char *func_name = expr->as.call.name;
            const char *original_func_name = func_name;  /* Save original for struct array detection */

            /* File operations */
            if (strcmp(func_name, "file_read") == 0) {
                func_name = "nl_os_file_read";
            } else if (strcmp(func_name, "file_read_binary") == 0) {
                func_name = "nl_os_file_read_binary";
            } else if (strcmp(func_name, "file_size") == 0) {
                func_name = "nl_os_file_size";
            } else if (strcmp(func_name, "file_read_bytes") == 0) {
                func_name = "nl_os_file_read_bytes";
            } else if (strcmp(func_name, "file_write") == 0) {
                func_name = "nl_os_file_write";
            } else if (strcmp(func_name, "file_append") == 0) {
                func_name = "nl_os_file_append";
            } else if (strcmp(func_name, "file_remove") == 0) {
                func_name = "nl_os_file_remove";
            } else if (strcmp(func_name, "file_rename") == 0) {
                func_name = "nl_os_file_rename";
            } else if (strcmp(func_name, "file_exists") == 0) {
                func_name = "nl_os_file_exists";
            } else if (strcmp(func_name, "file_size") == 0) {
                func_name = "nl_os_file_size";

            /* Directory operations */
            } else if (strcmp(func_name, "dir_create") == 0) {
                func_name = "nl_os_dir_create";
            } else if (strcmp(func_name, "dir_remove") == 0) {
                func_name = "nl_os_dir_remove";
            } else if (strcmp(func_name, "dir_list") == 0) {
                func_name = "nl_os_dir_list";
            } else if (strcmp(func_name, "dir_exists") == 0) {
                func_name = "nl_os_dir_exists";

            /* Working directory operations */
            } else if (strcmp(func_name, "getcwd") == 0) {
                func_name = "nl_os_getcwd";
            } else if (strcmp(func_name, "chdir") == 0) {
                func_name = "nl_os_chdir";

            /* Path operations */
            } else if (strcmp(func_name, "path_isfile") == 0) {
                func_name = "nl_os_path_isfile";
            } else if (strcmp(func_name, "path_isdir") == 0) {
                func_name = "nl_os_path_isdir";
            } else if (strcmp(func_name, "path_join") == 0) {
                func_name = "nl_os_path_join";
            } else if (strcmp(func_name, "path_basename") == 0) {
                func_name = "nl_os_path_basename";
            } else if (strcmp(func_name, "path_dirname") == 0) {
                func_name = "nl_os_path_dirname";

            /* Process operations */
            } else if (strcmp(func_name, "system") == 0) {
                func_name = "nl_os_system";
            } else if (strcmp(func_name, "exit") == 0) {
                func_name = "nl_os_exit";
            } else if (strcmp(func_name, "getenv") == 0) {
                func_name = "nl_os_getenv";

            /* Math and utility functions */
            } else if (strcmp(func_name, "abs") == 0) {
                func_name = "nl_abs";
            } else if (strcmp(func_name, "min") == 0) {
                func_name = "nl_min";
            } else if (strcmp(func_name, "max") == 0) {
                func_name = "nl_max";
            } else if (strcmp(func_name, "sqrt") == 0) {
                func_name = "sqrt";  /* Use C math library directly */
            } else if (strcmp(func_name, "pow") == 0) {
                func_name = "pow";  /* Use C math library directly */
            } else if (strcmp(func_name, "floor") == 0) {
                func_name = "floor";  /* Use C math library directly */
            } else if (strcmp(func_name, "ceil") == 0) {
                func_name = "ceil";  /* Use C math library directly */
            } else if (strcmp(func_name, "round") == 0) {
                func_name = "round";  /* Use C math library directly */
            } else if (strcmp(func_name, "sin") == 0) {
                func_name = "sin";  /* Use C math library directly */
            } else if (strcmp(func_name, "cos") == 0) {
                func_name = "cos";  /* Use C math library directly */
            } else if (strcmp(func_name, "tan") == 0) {
                func_name = "tan";  /* Use C math library directly */
            
            /* String operations */
            } else if (strcmp(func_name, "str_length") == 0) {
                func_name = "strlen";  /* Use C string library directly */
            } else if (strcmp(func_name, "array_length") == 0) {
                /* array_length - get DynArray size */
                func_name = "dyn_array_length";
            } else if (strcmp(func_name, "str_concat") == 0) {
                func_name = "nl_str_concat";
            } else if (strcmp(func_name, "str_substring") == 0) {
                func_name = "nl_str_substring";
            } else if (strcmp(func_name, "str_contains") == 0) {
                func_name = "nl_str_contains";
            } else if (strcmp(func_name, "str_equals") == 0) {
                func_name = "nl_str_equals";
            
            /* Array operations */
            } else if (strcmp(func_name, "array_length") == 0) {
                func_name = "nl_array_length";
            } else if (strcmp(func_name, "array_new") == 0) {
                func_name = "nl_array_new_int";  /* For now, assume int arrays */
            } else if (strcmp(func_name, "array_set") == 0) {
                /* array_set needs type detection - will handle specially below */
                func_name = "nl_array_set";  /* Placeholder, will be replaced */
            } else if (strcmp(func_name, "array_push") == 0) {
                /* array_push needs type detection - will handle specially below */
                func_name = "nl_array_push";  /* Placeholder, will be replaced */
            } else if (strcmp(func_name, "array_pop") == 0) {
                /* array_pop needs type detection - will handle specially below */
                func_name = "nl_array_pop";  /* Placeholder, will be replaced */
            } else if (strcmp(func_name, "at") == 0 || strcmp(func_name, "array_get") == 0) {
                /* at() and array_get() need type detection - will handle specially below */
                func_name = "nl_array_at";  /* Placeholder, will be replaced */
            } else if (strcmp(func_name, "print") == 0) {
                /* Special handling for print - dispatch based on argument type */
                Type arg_type = check_expression(expr->as.call.args[0], env);
                switch (arg_type) {
                    case TYPE_INT:
                        func_name = "nl_print_int";
                        break;
                    case TYPE_FLOAT:
                        func_name = "nl_print_float";
                        break;
                    case TYPE_STRING:
                        func_name = "nl_print_string";
                        break;
                    case TYPE_BOOL:
                        func_name = "nl_print_bool";
                        break;
                    case TYPE_ARRAY:
                        func_name = "nl_print_array";
                        break;
                    default:
                        func_name = "nl_print_int";  /* Fallback */
                        break;
                }
            } else if (strcmp(func_name, "println") == 0) {
                /* Special handling for println - dispatch based on argument type */
                Type arg_type = check_expression(expr->as.call.args[0], env);
                switch (arg_type) {
                    case TYPE_INT:
                        func_name = "nl_println_int";
                        break;
                    case TYPE_FLOAT:
                        func_name = "nl_println_float";
                        break;
                    case TYPE_STRING:
                        func_name = "nl_println_string";
                        break;
                    case TYPE_BOOL:
                        func_name = "nl_println_bool";
                        break;
                    case TYPE_ARRAY:
                        func_name = "nl_println_array";
                        break;
                    default:
                        func_name = "nl_println_int";  /* Fallback */
                        break;
                }
            } else {
                /* Check if this is an extern function */
                Function *func_def = env_get_function(env, func_name);
                if (func_def && func_def->is_extern) {
                    /* Extern functions - call directly with original name (no change) */
                } else if (!func_def) {
                    /* Not a function definition - check if it's a function parameter */
                    Symbol *sym = env_get_var(env, func_name);
                    if (sym && sym->type == TYPE_FUNCTION) {
                        /* Function parameter - use as-is (no nl_ prefix) */
                    } else {
                        /* Unknown - add prefix anyway (will fail at C compilation) */
                        func_name = get_c_func_name(func_name);
                    }
                } else {
                    /* User-defined functions get nl_ prefix to avoid C stdlib conflicts */
                    func_name = get_c_func_name(func_name);
                }
            }

            /* Special handling for struct array operations */
            bool is_struct_array_op = false;
            const char *struct_type_name = NULL;
            
            if ((strcmp(original_func_name, "array_push") == 0 || strcmp(original_func_name, "at") == 0 || 
                 strcmp(original_func_name, "array_set") == 0) && expr->as.call.arg_count >= 1) {
                /* Check the ARRAY's element type (first argument) */
                if (expr->as.call.args[0]->type == AST_IDENTIFIER) {
                    const char *arr_name = expr->as.call.args[0]->as.identifier;
                    Symbol *arr_sym = env_get_var(env, arr_name);
                    if (arr_sym) {
                        /* Enums are int64_t, treat as int arrays */
                        if (arr_sym->type == TYPE_ARRAY && arr_sym->element_type == TYPE_ENUM) {
                            /* Enum arrays use int array functions (enums are int64_t) */
                            /* Let it fall through to int array handling below */
                        }
                        /* Check if this is a struct array */
                        else if (arr_sym->type == TYPE_ARRAY && arr_sym->element_type == TYPE_STRUCT) {
                            if (arr_sym->struct_type_name) {
                                struct_type_name = arr_sym->struct_type_name;
                                is_struct_array_op = true;
                            } else {
                                /* Try to get from type annotation */
                                /* For array_push: check the value being pushed */
                                if (strcmp(original_func_name, "array_push") == 0 && expr->as.call.arg_count >= 2) {
                                    if (expr->as.call.args[1]->type == AST_STRUCT_LITERAL) {
                                        struct_type_name = expr->as.call.args[1]->as.struct_literal.struct_name;
                                        is_struct_array_op = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            if (is_struct_array_op && struct_type_name) {
                /* Generate struct-specific array operation */
                const char *prefixed_struct = get_prefixed_type_name(struct_type_name);
                
                if (strcmp(original_func_name, "array_push") == 0) {
                    /* dyn_array_push_struct(arr, &value, sizeof(nl_StructName)) */
                    sb_append(sb, "dyn_array_push_struct(");
                    transpile_expression(sb, expr->as.call.args[0], env);  /* array */
                    sb_append(sb, ", &(");
                    transpile_expression(sb, expr->as.call.args[1], env);  /* value */
                    sb_appendf(sb, "), sizeof(%s))", prefixed_struct);
                } else if (strcmp(original_func_name, "at") == 0) {
                    /* *((nl_StructName*)dyn_array_get_struct(arr, index)) */
                    sb_appendf(sb, "*((%s*)dyn_array_get_struct(", prefixed_struct);
                    transpile_expression(sb, expr->as.call.args[0], env);  /* array */
                    sb_append(sb, ", ");
                    transpile_expression(sb, expr->as.call.args[1], env);  /* index */
                    sb_append(sb, "))");
                } else if (strcmp(original_func_name, "array_set") == 0) {
                    /* dyn_array_set_struct(arr, index, &value, sizeof(nl_StructName)) */
                    sb_append(sb, "dyn_array_set_struct(");
                    transpile_expression(sb, expr->as.call.args[0], env);  /* array */
                    sb_append(sb, ", ");
                    transpile_expression(sb, expr->as.call.args[1], env);  /* index */
                    sb_append(sb, ", &(");
                    transpile_expression(sb, expr->as.call.args[2], env);  /* value */
                    sb_appendf(sb, "), sizeof(%s))", prefixed_struct);
                }
            } else if (strcmp(original_func_name, "at") == 0 || strcmp(original_func_name, "array_get") == 0 || strcmp(original_func_name, "array_set") == 0) {
                /* Non-struct arrays: use type-specific int/float versions */
                /* For at() and array_get(): nl_array_at_int or nl_array_at_float */
                /* For array_set(): nl_array_set_int or nl_array_set_float */
                if (strcmp(original_func_name, "at") == 0 || strcmp(original_func_name, "array_get") == 0) {
                    func_name = "nl_array_at_int";  /* Default to int */
                } else if (strcmp(original_func_name, "array_set") == 0) {
                    func_name = "nl_array_set_int";  /* Default to int */
                }
                
                /* Normal function call */
                sb_appendf(sb, "%s(", func_name);
                
                for (int i = 0; i < expr->as.call.arg_count; i++) {
                    if (i > 0) sb_append(sb, ", ");
                    transpile_expression(sb, expr->as.call.args[i], env);
                }
                sb_append(sb, ")");
            } else {
                /* Normal function call */
                
                /* Check if this is an extern function that needs return type casting */
                Function *func_def = env_get_function(env, func_name);
                bool needs_return_cast = false;
                
                /* If this is an extern function returning int (which represents a pointer),
                 * and we have headers included, cast the return value to avoid type errors.
                 * Don't cast void returns! */
                if (func_def && func_def->is_extern && func_def->return_type == TYPE_INT) {
                    /* Extern functions that return int likely return pointers
                     * When C headers are included, we need to cast pointer returns to int64_t */
                    needs_return_cast = true;
                } else if (func_def && func_def->is_extern && func_def->return_type == TYPE_VOID) {
                    /* Void returns - don't cast */
                    needs_return_cast = false;
                }
                
                /* Add cast for pointer-returning extern functions */
                if (needs_return_cast) {
                    sb_append(sb, "(int64_t)");
                }
                
                sb_appendf(sb, "%s(", func_name);
                
                /* Check if this is an SDL/Mix function that needs pointer casts */
                bool is_sdl_func = func_def && func_def->is_extern && 
                                  (strncmp(func_name, "SDL_", 4) == 0 || strncmp(func_name, "TTF_", 4) == 0);
                bool is_mix_func = func_def && func_def->is_extern && strncmp(func_name, "Mix_", 4) == 0;
                bool is_nl_sdl_helper = func_def && func_def->is_extern && 
                                       strncmp(func_name, "nl_sdl_", 7) == 0;
                
                for (int i = 0; i < expr->as.call.arg_count; i++) {
                    if (i > 0) sb_append(sb, ", ");
                    
                    /* Check if this parameter needs a cast for SDL functions */
                    if (is_sdl_func && func_def && i < func_def->param_count) {
                        const char *sdl_param_type = get_sdl_c_type(func_name, i, false);
                        if (sdl_param_type && func_def->params[i].type == TYPE_INT) {
                            /* Cast int to pointer type */
                            sb_appendf(sb, "(%s)", sdl_param_type);
                        }
                    } else if (is_mix_func && func_def && i < func_def->param_count) {
                        /* SDL_mixer functions: cast int64_t to pointer types */
                        if (func_def->params[i].type == TYPE_INT) {
                            /* Common SDL_mixer pointer types */
                            if (strcmp(func_name, "Mix_PlayMusic") == 0 && i == 0) {
                                sb_append(sb, "(Mix_Music*)");
                            } else if (strcmp(func_name, "Mix_FreeMusic") == 0 && i == 0) {
                                sb_append(sb, "(Mix_Music*)");
                            } else if (strcmp(func_name, "Mix_PlayChannel") == 0 && i == 1) {
                                sb_append(sb, "(Mix_Chunk*)");
                            } else if (strcmp(func_name, "Mix_FreeChunk") == 0 && i == 0) {
                                sb_append(sb, "(Mix_Chunk*)");
                            }
                        }
                    } else if (is_nl_sdl_helper && func_def && i < func_def->param_count) {
                        /* nl_sdl_ helpers: first param is SDL_Renderer* passed as int64_t */
                        /* Don't cast - the helper function accepts int64_t and casts internally */
                    }
                    
                    transpile_expression(sb, expr->as.call.args[i], env);
                }
                sb_append(sb, ")");
            }
            break;
        }

        case AST_ARRAY_LITERAL: {
            /* Transpile array literal */
            int count = expr->as.array_literal.element_count;
            Type elem_type = expr->as.array_literal.element_type;
            
            if (elem_type == TYPE_STRUCT) {
                /* Struct array literal - create empty ELEM_STRUCT array */
                /* The array will be populated by subsequent array_push operations if non-empty */
                sb_append(sb, "dyn_array_new(ELEM_STRUCT)");
            } else if (elem_type == TYPE_FLOAT) {
                sb_appendf(sb, "dynarray_literal_float(%d", count);
                for (int i = 0; i < count; i++) {
                    sb_append(sb, ", ");
                    transpile_expression(sb, expr->as.array_literal.elements[i], env);
                }
                sb_append(sb, ")");
            } else {
                /* Default to int */
                sb_appendf(sb, "dynarray_literal_int(%d", count);
                for (int i = 0; i < count; i++) {
                    sb_append(sb, ", ");
                    transpile_expression(sb, expr->as.array_literal.elements[i], env);
                }
                sb_append(sb, ")");
            }
            break;
        }

        case AST_IF:
            sb_append(sb, "(");
            transpile_expression(sb, expr->as.if_stmt.condition, env);
            sb_append(sb, " ? ");
            /* For if expressions, we need to handle block expressions */
            /* Simplified: just handle single expressions */
            if (expr->as.if_stmt.then_branch->type == AST_BLOCK) {
                sb_append(sb, "/* block */ 0");
            } else {
                transpile_expression(sb, expr->as.if_stmt.then_branch, env);
            }
            sb_append(sb, " : ");
            if (expr->as.if_stmt.else_branch->type == AST_BLOCK) {
                sb_append(sb, "/* block */ 0");
            } else {
                transpile_expression(sb, expr->as.if_stmt.else_branch, env);
            }
            sb_append(sb, ")");
            break;

        case AST_STRUCT_LITERAL: {
            /* Transpile struct literal: Point { x: 10, y: 20 } -> (nl_Point){.x = 10, .y = 20} */
            const char *struct_name = expr->as.struct_literal.struct_name;
            const char *prefixed_name = get_prefixed_type_name(struct_name);
            
            /* Use prefixed type name */
            sb_appendf(sb, "(%s){", prefixed_name);
            for (int i = 0; i < expr->as.struct_literal.field_count; i++) {
                if (i > 0) sb_append(sb, ", ");
                sb_append(sb, ".");
                sb_append(sb, expr->as.struct_literal.field_names[i]);
                sb_append(sb, " = ");
                transpile_expression(sb, expr->as.struct_literal.field_values[i], env);
            }
            sb_append(sb, "}");
            break;
        }

        case AST_UNION_CONSTRUCT: {
            /* Transpile union construction: Status.Ok {} -> (nl_Status){.tag = nl_Status_TAG_Ok, .data.Ok = {}} */
            const char *union_name = expr->as.union_construct.union_name;
            const char *variant_name = expr->as.union_construct.variant_name;
            const char *prefixed_union = get_prefixed_type_name(union_name);
            
            /* Use prefixed union name and tag */
            sb_appendf(sb, "(%s){.tag = nl_%s_TAG_%s", 
                      prefixed_union, union_name, variant_name);
            
            /* If the variant has fields, initialize them */
            if (expr->as.union_construct.field_count > 0) {
                sb_appendf(sb, ", .data.%s = {", variant_name);
                for (int i = 0; i < expr->as.union_construct.field_count; i++) {
                    if (i > 0) sb_append(sb, ", ");
                    sb_append(sb, ".");
                    sb_append(sb, expr->as.union_construct.field_names[i]);
                    sb_append(sb, " = ");
                    transpile_expression(sb, expr->as.union_construct.field_values[i], env);
                }
                sb_append(sb, "}");
            }
            
            sb_append(sb, "}");
            break;
        }
        
        case AST_MATCH: {
            /* Transpile match expression as a statement expression with switch
             * match status { Ok(x) => x, Error(msg) => ... }
             * becomes:
             * ({
             *   Status _match_tmp = status;
             *   __typeof__(_match_tmp) _match_result;
             *   switch(_match_tmp.tag) {
             *     case Status_TAG_Ok: {
             *       int x = _match_tmp.data.Ok.value;  // bind pattern variables
             *       _match_result = x;
             *     } break;
             *     case Status_TAG_Error: {
             *       int msg = _match_tmp.data.Error.code;
             *       _match_result = ...;
             *     } break;
             *   }
             *   _match_result;
             * })
             */
            
            /* Get union name from AST (filled during typechecking) */
            const char *union_name = expr->as.match_expr.union_type_name;
            
            if (!union_name) {
                fprintf(stderr, "Error: Union type name not set for match expression at line %d\n", expr->line);
                union_name = "UnknownUnion";
            }
            
            /* Get union definition for field information */
            UnionDef *udef = env_get_union(env, union_name);
            const char *prefixed_union = get_prefixed_type_name(union_name);
            
            /* Generate statement expression with temp variable using prefixed name */
            sb_append(sb, "({ ");
            sb_appendf(sb, "%s _match_tmp = ", prefixed_union);
            transpile_expression(sb, expr->as.match_expr.expr, env);
            sb_append(sb, "; ");
            
            /* Determine result type from first arm (simplification: assume int for now) */
            sb_append(sb, "int _match_result; ");
            
            /* Generate switch on tag */
            sb_append(sb, "switch(_match_tmp.tag) { ");
            
            /* Emit each arm as a case */
            for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                const char *variant = expr->as.match_expr.pattern_variants[i];
                const char *binding = expr->as.match_expr.pattern_bindings[i];
                
                /* Use prefixed tag name */
                sb_appendf(sb, "case nl_%s_TAG_%s: { ", union_name, variant);
                
                /* Bind pattern variable to the variant data
                 * For a variant with fields: Type binding = _match_tmp.data.VariantName;
                 * For a variant without fields: just create a dummy binding
                 */
                if (udef) {
                    int variant_idx = env_get_union_variant_index(env, union_name, variant);
                    if (variant_idx >= 0 && udef->variant_field_counts[variant_idx] > 0) {
                        /* Variant has fields - bind the entire struct
                         * Use the variant struct type name directly instead of typeof
                         */
                        const char *variant_type = get_prefixed_variant_struct_name(union_name, variant);
                        sb_appendf(sb, "%s %s = _match_tmp.data.%s; ",
                                  variant_type, binding, variant);
                    } else {
                        /* Variant has no fields - create a dummy int binding */
                        sb_appendf(sb, "int %s __attribute__((unused)) = 0; ", binding);
                    }
                }
                
                /* Transpile arm body */
                sb_append(sb, "_match_result = ");
                transpile_expression(sb, expr->as.match_expr.arm_bodies[i], env);
                sb_append(sb, "; } break; ");
            }
            
            sb_append(sb, "} _match_result; })");
            break;
        }

        case AST_FIELD_ACCESS: {
            /* Check if this is an enum variant access */
            if (expr->as.field_access.object->type == AST_IDENTIFIER) {
                const char *enum_name = expr->as.field_access.object->as.identifier;
                if (env_get_enum(env, enum_name)) {
                    /* For runtime enums, just use variant name (e.g., TOKEN_RETURN)
                     * For enums that conflict with runtime types, use TOKEN_ prefix
                     * For user enums, use prefixed nl_EnumName_VARIANT */
                    if (is_runtime_typedef(enum_name)) {
                        sb_append(sb, expr->as.field_access.field_name);
                    } else if (conflicts_with_runtime(enum_name)) {
                        /* Use runtime enum variant naming (TOKEN_ prefix) */
                        sb_appendf(sb, "TOKEN_%s", expr->as.field_access.field_name);
                    } else {
                        /* User-defined enum - use prefixed variant name */
                        const char *prefixed_variant = get_prefixed_variant_name(enum_name, expr->as.field_access.field_name);
                        sb_append(sb, prefixed_variant);
                    }
                    break;
                }
            }
            
            /* Regular struct field access: point.x -> point.x */
            transpile_expression(sb, expr->as.field_access.object, env);
            sb_append(sb, ".");
            sb_append(sb, expr->as.field_access.field_name);
            break;
        }

        case AST_BLOCK: {
            /* Blocks can be used as expressions in match arms
             * Transpile as a statement expression: ({ statements... last_value })
             */
            sb_append(sb, "({ ");
            for (int i = 0; i < expr->as.block.count; i++) {
                ASTNode *stmt = expr->as.block.statements[i];
                if (stmt->type == AST_RETURN && stmt->as.return_stmt.value) {
                    /* For return statements in blocks-as-expressions,
                     * just evaluate the return value without the return keyword
                     */
                    transpile_expression(sb, stmt->as.return_stmt.value, env);
                    sb_append(sb, "; ");
                } else {
                    /* Regular statement - transpile normally */
                    transpile_statement(sb, stmt, 0, env, NULL);
                }
            }
            sb_append(sb, "})");
            break;
        }

        case AST_RETURN: {
            /* Return statements can appear in blocks that are used as expressions
             * Just transpile the value being returned
             */
            if (expr->as.return_stmt.value) {
                transpile_expression(sb, expr->as.return_stmt.value, env);
            }
            break;
        }

        case AST_TUPLE_LITERAL: {
            /* Transpile tuple literal to C compound literal
             * (int, string) literal becomes: (Tuple_int_string){._0 = val1, ._1 = val2}
             * For now, we'll use a simpler approach with struct initialization
             */
            int element_count = expr->as.tuple_literal.element_count;
            
            /* Empty tuple */
            if (element_count == 0) {
                sb_append(sb, "(struct {int _dummy;}){0}");
                break;
            }
            
            /* Try to use typedef if we have a tuple registry and element types */
            const char *typedef_name = NULL;
            if (g_tuple_registry && expr->as.tuple_literal.element_types) {
                /* Create temporary TypeInfo to look up typedef */
                TypeInfo temp_info;
                temp_info.tuple_types = expr->as.tuple_literal.element_types;
                temp_info.tuple_type_names = NULL;
                temp_info.tuple_element_count = element_count;
                
                /* Look up existing typedef */
                for (int i = 0; i < g_tuple_registry->count; i++) {
                    if (tuple_types_equal(g_tuple_registry->tuples[i], &temp_info)) {
                        typedef_name = g_tuple_registry->typedef_names[i];
                        break;
                    }
                }
            }
            
            if (typedef_name) {
                /* Use typedef name */
                sb_appendf(sb, "(%s){", typedef_name);
            } else {
                /* Fall back to inline struct for tuple */
                sb_append(sb, "(struct { ");
                for (int i = 0; i < element_count; i++) {
                    /* Determine C type for each element */
                    Type elem_type = expr->as.tuple_literal.element_types ? 
                                    expr->as.tuple_literal.element_types[i] : TYPE_INT;
                    const char *c_type = type_to_c(elem_type);
                    sb_appendf(sb, "%s _%d; ", c_type, i);
                }
                sb_append(sb, "}){");
            }
            
            /* Initialize fields */
            for (int i = 0; i < element_count; i++) {
                if (i > 0) sb_append(sb, ", ");
                sb_appendf(sb, "._%d = ", i);
                transpile_expression(sb, expr->as.tuple_literal.elements[i], env);
            }
            sb_append(sb, "}");
            break;
        }

        case AST_TUPLE_INDEX: {
            /* Transpile tuple index access: tuple.0 becomes tuple._0 */
            transpile_expression(sb, expr->as.tuple_index.tuple, env);
            sb_appendf(sb, "._%d", expr->as.tuple_index.index);
            break;
        }

        default:
            sb_append(sb, "/* unknown expr */");
            break;
    }
}

/* Transpile statement to C */
static void transpile_statement(StringBuilder *sb, ASTNode *stmt, int indent, Environment *env, FunctionTypeRegistry *fn_registry) {
    if (!stmt) return;

    switch (stmt->type) {
        case AST_LET: {
            emit_indent(sb, indent);
            
            /* Check if type_name refers to a union */
            bool is_union = false;
            if (stmt->as.let.type_name) {
                is_union = (env_get_union(env, stmt->as.let.type_name) != NULL);
            }
            
            /* Handle union types */
            if (is_union) {
                const char *prefixed_name = get_prefixed_type_name(stmt->as.let.type_name);
                sb_appendf(sb, "%s %s = ", prefixed_name, stmt->as.let.name);
            }
            /* Handle generic lists: List<UserType> */
            else if (stmt->as.let.var_type == TYPE_LIST_GENERIC && stmt->as.let.type_name) {
                /* Generate specialized list pointer type: List_Point* */
                sb_appendf(sb, "List_%s* %s = ", stmt->as.let.type_name, stmt->as.let.name);
            }
            /* For struct/union types that might be enums */
            else if (stmt->as.let.var_type == TYPE_STRUCT || stmt->as.let.var_type == TYPE_UNION) {
                /* Check if value is an enum variant access */
                bool is_enum = false;
                if (stmt->as.let.value->type == AST_FIELD_ACCESS &&
                    stmt->as.let.value->as.field_access.object->type == AST_IDENTIFIER) {
                    const char *enum_name = stmt->as.let.value->as.field_access.object->as.identifier;
                    if (env_get_enum(env, enum_name)) {
                        is_enum = true;
                    }
                }
                
                if (is_enum) {
                    /* This is an enum, use prefixed enum type */
                    const char *enum_name = stmt->as.let.value->as.field_access.object->as.identifier;
                    const char *prefixed_enum = get_prefixed_type_name(enum_name);
                    sb_appendf(sb, "%s %s = ", prefixed_enum, stmt->as.let.name);
                } else {
                    /* Regular struct/union type - use prefixed name */
                    const char *type_name = stmt->as.let.type_name ? stmt->as.let.type_name : get_struct_type_from_expr(stmt->as.let.value);
                    if (type_name) {
                        const char *prefixed_name = get_prefixed_type_name(type_name);
                        sb_appendf(sb, "%s %s = ", prefixed_name, stmt->as.let.name);
                    } else {
                        /* Fallback if we can't determine struct type */
                        sb_appendf(sb, "/* struct */ void* %s = ", stmt->as.let.name);
                    }
                }
            }
            /* Handle function types */
            else if (stmt->as.let.var_type == TYPE_FUNCTION && stmt->as.let.fn_sig) {
                /* Get or register the typedef for this function signature */
                const char *typedef_name = register_function_signature(fn_registry, stmt->as.let.fn_sig);
                sb_appendf(sb, "%s %s = ", typedef_name, stmt->as.let.name);
            }
            /* Handle tuple types */
            else if (stmt->as.let.var_type == TYPE_TUPLE && stmt->as.let.value && 
                     stmt->as.let.value->type == AST_TUPLE_LITERAL) {
                /* Generate inline struct type for tuple variables
                 * let t: (int, string) = (10, "hello")
                 * becomes: struct { int64_t _0; const char* _1; } t = {._0 = 10LL, ._1 = "hello"};
                 * Note: We use designated initializer syntax instead of compound literal
                 * to avoid C's anonymous struct type incompatibility issues.
                 */
                ASTNode *tuple_literal = stmt->as.let.value;
                int element_count = tuple_literal->as.tuple_literal.element_count;
                
                /* Empty tuple */
                if (element_count == 0) {
                    sb_appendf(sb, "struct {int _dummy;} %s = {0};\n", stmt->as.let.name);
                } else {
                    /* Generate inline struct with element types */
                    sb_append(sb, "struct { ");
                    for (int i = 0; i < element_count; i++) {
                        Type elem_type = tuple_literal->as.tuple_literal.element_types ? 
                                        tuple_literal->as.tuple_literal.element_types[i] : TYPE_INT;
                        const char *c_type = type_to_c(elem_type);
                        sb_appendf(sb, "%s _%d; ", c_type, i);
                    }
                    sb_appendf(sb, "} %s = {", stmt->as.let.name);
                    
                    /* Generate designated initializers for each field */
                    for (int i = 0; i < element_count; i++) {
                        if (i > 0) sb_append(sb, ", ");
                        sb_appendf(sb, "._%d = ", i);
                        transpile_expression(sb, tuple_literal->as.tuple_literal.elements[i], env);
                    }
                    sb_append(sb, "};\n");
                }
                /* Skip the generic value transpilation since we handled it above */
                goto skip_value_transpile;
            } else {
                sb_appendf(sb, "%s %s = ", type_to_c(stmt->as.let.var_type), stmt->as.let.name);
            }
            
            /* Check if we need to cast pointer return values to int64_t */
            bool needs_cast = false;
            if (stmt->as.let.var_type == TYPE_INT && stmt->as.let.value->type == AST_CALL) {
                const char *func_name = stmt->as.let.value->as.call.name;
                Function *func_def = env_get_function(env, func_name);
                if (func_def && func_def->is_extern && 
                    (strncmp(func_name, "SDL_", 4) == 0 || strncmp(func_name, "TTF_", 4) == 0)) {
                    const char *sdl_ret_type = get_sdl_c_type(func_name, -1, true);
                    if (sdl_ret_type && strstr(sdl_ret_type, "*")) {
                        /* Function returns a pointer, but we're assigning to int64_t */
                        needs_cast = true;
                    }
                }
            }
            
            if (needs_cast) {
                sb_append(sb, "(int64_t)");
            }
            transpile_expression(sb, stmt->as.let.value, env);
            sb_append(sb, ";\n");
            
        skip_value_transpile:
            /* Register variable in environment for type tracking during transpilation */
            /* Preserve element_type and type_name for arrays and structs */
            env_define_var_with_type_info(env, stmt->as.let.name, stmt->as.let.var_type, 
                                         stmt->as.let.element_type, NULL, stmt->as.let.is_mut, create_void());
            
            /* For arrays of structs, set struct_type_name */
            if (stmt->as.let.var_type == TYPE_ARRAY && stmt->as.let.element_type == TYPE_STRUCT && stmt->as.let.type_name) {
                Symbol *sym = env_get_var(env, stmt->as.let.name);
                if (sym) {
                    sym->struct_type_name = strdup(stmt->as.let.type_name);
                }
            }
            break;
        }

        case AST_SET:
            emit_indent(sb, indent);
            sb_appendf(sb, "%s = ", stmt->as.set.name);
            transpile_expression(sb, stmt->as.set.value, env);
            sb_append(sb, ";\n");
            break;

        case AST_WHILE:
            emit_indent(sb, indent);
            sb_append(sb, "while (");
            transpile_expression(sb, stmt->as.while_stmt.condition, env);
            sb_append(sb, ") ");
            transpile_statement(sb, stmt->as.while_stmt.body, indent, env, fn_registry);
            break;

        case AST_FOR: {
            emit_indent(sb, indent);
            /* Extract range bounds */
            ASTNode *range = stmt->as.for_stmt.range_expr;
            sb_appendf(sb, "for (int64_t %s = ", stmt->as.for_stmt.var_name);
            if (range && range->type == AST_CALL && range->as.call.arg_count == 2) {
                transpile_expression(sb, range->as.call.args[0], env);
                sb_appendf(sb, "; %s < ", stmt->as.for_stmt.var_name);
                transpile_expression(sb, range->as.call.args[1], env);
                sb_appendf(sb, "; %s++) ", stmt->as.for_stmt.var_name);
            } else {
                sb_append(sb, "0; 0; ) ");
            }
            transpile_statement(sb, stmt->as.for_stmt.body, indent, env, fn_registry);
            break;
        }

        case AST_RETURN:
            emit_indent(sb, indent);
            sb_append(sb, "return");
            if (stmt->as.return_stmt.value) {
                sb_append(sb, " ");
                transpile_expression(sb, stmt->as.return_stmt.value, env);
            }
            sb_append(sb, ";\n");
            break;

        case AST_BLOCK:
            sb_append(sb, "{\n");
            for (int i = 0; i < stmt->as.block.count; i++) {
                transpile_statement(sb, stmt->as.block.statements[i], indent + 1, env, fn_registry);
            }
            emit_indent(sb, indent);
            sb_append(sb, "}\n");
            break;

        case AST_PRINT:
            emit_indent(sb, indent);
            /* Use semantic type from type checker instead of AST node type */
            Type expr_type = check_expression(stmt->as.print.expr, env);
            if (expr_type == TYPE_STRING) {
                sb_append(sb, "printf(\"%s\\n\", ");
                transpile_expression(sb, stmt->as.print.expr, env);
                sb_append(sb, ");\n");
            } else if (expr_type == TYPE_BOOL) {
                sb_append(sb, "printf(\"%s\\n\", ");
                transpile_expression(sb, stmt->as.print.expr, env);
                sb_append(sb, " ? \"true\" : \"false\");\n");
            } else if (expr_type == TYPE_FLOAT) {
                sb_append(sb, "printf(\"%g\\n\", ");
                transpile_expression(sb, stmt->as.print.expr, env);
                sb_append(sb, ");\n");
            } else {
                sb_append(sb, "printf(\"%lld\\n\", (long long)");
                transpile_expression(sb, stmt->as.print.expr, env);
                sb_append(sb, ");\n");
            }
            break;

        case AST_IF:
            emit_indent(sb, indent);
            sb_append(sb, "if (");
            transpile_expression(sb, stmt->as.if_stmt.condition, env);
            sb_append(sb, ") ");
            transpile_statement(sb, stmt->as.if_stmt.then_branch, indent, env, fn_registry);
            if (stmt->as.if_stmt.else_branch) {
                emit_indent(sb, indent);
                sb_append(sb, "else ");
                transpile_statement(sb, stmt->as.if_stmt.else_branch, indent, env, fn_registry);
            }
            break;

        default:
            /* Expression statements */
            emit_indent(sb, indent);
            transpile_expression(sb, stmt, env);
            sb_append(sb, ";\n");
            break;
    }
}

/* Helper to recursively collect function signatures from statements */
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
            collect_fn_sigs(stmt->as.if_stmt.else_branch, reg);
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

    /* C includes and headers */
    sb_append(sb, "#include <stdio.h>\n");
    sb_append(sb, "#include <stdint.h>\n");
    sb_append(sb, "#include <stdbool.h>\n");
    sb_append(sb, "#include <string.h>\n");
    sb_append(sb, "#include <stdlib.h>\n");
    sb_append(sb, "#include <time.h>\n");
    sb_append(sb, "#include <stdarg.h>\n");
    sb_append(sb, "#include <math.h>\n");
    
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
    sb_append(sb, "    char* buffer = malloc(4096);\n");
    sb_append(sb, "    buffer[0] = '\\0';\n");
    sb_append(sb, "    struct dirent* entry;\n");
    sb_append(sb, "    while ((entry = readdir(dir)) != NULL) {\n");
    sb_append(sb, "        if (strcmp(entry->d_name, \".\") == 0 || strcmp(entry->d_name, \"..\") == 0) continue;\n");
    sb_append(sb, "        strcat(buffer, entry->d_name);\n");
    sb_append(sb, "        strcat(buffer, \"\\n\");\n");
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
    sb_append(sb, "        fprintf(stderr, \"Error: Index %lld out of bounds (string length %d)\\n\", index, len);\n");
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
    sb_append(sb, "    snprintf(buffer, 32, \"%lld\", n);\n");
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
    sb_append(sb, "    int64_t: (int64_t)((x) < 0 ? -(x) : (x)), \\\n");
    sb_append(sb, "    double: (double)((x) < 0.0 ? -(x) : (x)))\n\n");

    /* min function */
    sb_append(sb, "#define nl_min(a, b) _Generic((a), \\\n");
    sb_append(sb, "    int64_t: (int64_t)((a) < (b) ? (a) : (b)), \\\n");
    sb_append(sb, "    double: (double)((a) < (b) ? (a) : (b)))\n\n");

    /* max function */
    sb_append(sb, "#define nl_max(a, b) _Generic((a), \\\n");
    sb_append(sb, "    int64_t: (int64_t)((a) > (b) ? (a) : (b)), \\\n");
    sb_append(sb, "    double: (double)((a) > (b) ? (a) : (b)))\n\n");

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
    sb_append(sb, "    printf(\"%lld\", value);\n");
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
    sb_append(sb, "    printf(\"%lld\\n\", value);\n");
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
    
    /* Array get (at) wrapper */
    sb_append(sb, "static int64_t nl_array_at_int(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_int(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static double nl_array_at_float(DynArray* arr, int64_t idx) {\n");
    sb_append(sb, "    return dyn_array_get_float(arr, idx);\n");
    sb_append(sb, "}\n\n");
    
    /* Array set wrapper */
    sb_append(sb, "static void nl_array_set_int(DynArray* arr, int64_t idx, int64_t val) {\n");
    sb_append(sb, "    dyn_array_set_int(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    sb_append(sb, "static void nl_array_set_float(DynArray* arr, int64_t idx, double val) {\n");
    sb_append(sb, "    dyn_array_set_float(arr, idx, val);\n");
    sb_append(sb, "}\n\n");
    
    /* Array new wrapper - creates DynArray with specified size and default value */
    sb_append(sb, "static DynArray* nl_array_new_int(int64_t size, int64_t default_val) {\n");
    sb_append(sb, "    DynArray* arr = dyn_array_new(ELEM_INT);\n");
    sb_append(sb, "    for (int64_t i = 0; i < size; i++) {\n");
    sb_append(sb, "        dyn_array_push_int(arr, default_val);\n");
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
    sb_append(sb, "    strcpy(result, s1);\n");
    sb_append(sb, "    strcat(result, s2);\n");
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
    sb_append(sb, "                printf(\"%lld\", ((int64_t*)arr->data)[i]);\n");
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

    /* Generate struct typedefs */
    sb_append(sb, "/* ========== Struct Definitions ========== */\n\n");
    for (int i = 0; i < env->struct_count; i++) {
        StructDef *sdef = &env->structs[i];
        
        /* Get prefixed name (adds nl_ for user types, keeps runtime types as-is) */
        const char *prefixed_name = get_prefixed_type_name(sdef->name);
        
        /* Generate typedef struct */
        sb_appendf(sb, "typedef struct %s {\n", prefixed_name);
        for (int j = 0; j < sdef->field_count; j++) {
            sb_append(sb, "    ");
            if (sdef->field_types[j] == TYPE_STRUCT) {
                /* For struct fields, we'd need the struct type name - for now use void* */
                sb_append(sb, "void* /* struct field */");
            } else {
                sb_append(sb, type_to_c(sdef->field_types[j]));
            }
            sb_appendf(sb, " %s;\n", sdef->field_names[j]);
        }
        sb_appendf(sb, "} %s;\n\n", prefixed_name);
    }
    sb_append(sb, "/* ========== End Struct Definitions ========== */\n\n");

    /* Generate enum typedefs */
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

    /* Generate specialized generic list types */
    sb_append(sb, "/* ========== Generic List Specializations ========== */\n\n");
    for (int i = 0; i < env->generic_instance_count; i++) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (strcmp(inst->generic_name, "List") == 0 && inst->type_arg_names) {
            const char *elem_type = inst->type_arg_names[0];
            const char *specialized_name = inst->concrete_name;
            
            /* Generate struct definition */
            sb_appendf(sb, "typedef struct {\n");
            sb_appendf(sb, "    struct %s *data;\n", elem_type);
            sb_appendf(sb, "    int count;\n");
            sb_appendf(sb, "    int capacity;\n");
            sb_appendf(sb, "} %s;\n\n", specialized_name);
            
            /* Generate constructor */
            sb_appendf(sb, "%s* %s_new() {\n", specialized_name, specialized_name);
            sb_appendf(sb, "    %s *list = malloc(sizeof(%s));\n", specialized_name, specialized_name);
            sb_appendf(sb, "    list->data = malloc(sizeof(struct %s) * 4);\n", elem_type);
            sb_appendf(sb, "    list->count = 0;\n");
            sb_appendf(sb, "    list->capacity = 4;\n");
            sb_appendf(sb, "    return list;\n");
            sb_appendf(sb, "}\n\n");
            
            /* Generate push function */
            sb_appendf(sb, "void %s_push(%s *list, struct %s value) {\n",
                      specialized_name, specialized_name, elem_type);
            sb_appendf(sb, "    if (list->count >= list->capacity) {\n");
            sb_appendf(sb, "        list->capacity *= 2;\n");
            sb_appendf(sb, "        list->data = realloc(list->data, sizeof(struct %s) * list->capacity);\n",
                      elem_type);
            sb_appendf(sb, "    }\n");
            sb_appendf(sb, "    list->data[list->count++] = value;\n");
            sb_appendf(sb, "}\n\n");
            
            /* Generate get function */
            sb_appendf(sb, "struct %s %s_get(%s *list, int index) {\n",
                      elem_type, specialized_name, specialized_name);
            sb_appendf(sb, "    return list->data[index];\n");
            sb_appendf(sb, "}\n\n");
            
            /* Generate length function */
            sb_appendf(sb, "int %s_length(%s *list) {\n", specialized_name, specialized_name);
            sb_appendf(sb, "    return list->count;\n");
            sb_appendf(sb, "}\n\n");
        }
    }
    sb_append(sb, "/* ========== End Generic List Specializations ========== */\n\n");

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
                register_function_signature(fn_registry, item->as.function.return_fn_sig);
            }
            
            /* Check return type for tuple type */
            if (item->as.function.return_type == TYPE_TUPLE && 
                item->as.function.return_type_info) {
                register_tuple_type(tuple_registry, item->as.function.return_type_info);
            }
            
            /* Collect from function body */
            collect_fn_sigs(item->as.function.body, fn_registry);
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
        sb_append(sb, "/* Tuple Type Typedefs */\n");
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
            /* Generate extern declaration with proper SDL types */
            sb_append(sb, "extern ");
            
            const char *func_name = item->as.function.name;
            const char *ret_type_c = type_to_c(item->as.function.return_type);
            
            /* Check for SDL-specific types */
            const char *sdl_ret_type = get_sdl_c_type(func_name, -1, true);
            if (sdl_ret_type) {
                ret_type_c = sdl_ret_type;
            } else if (item->as.function.return_type == TYPE_INT && 
                      (strncmp(func_name, "SDL_", 4) == 0 || strncmp(func_name, "TTF_", 4) == 0)) {
                /* For SDL functions returning int that might be Uint32, check */
                if (strstr(func_name, "GetTicks")) {
                    ret_type_c = "Uint32";
                }
            }
            
            sb_append(sb, ret_type_c);
            sb_appendf(sb, " %s(", func_name);
            
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (j > 0) sb_append(sb, ", ");
                const char *param_type_c = type_to_c(item->as.function.params[j].type);
                
                /* Check for SDL-specific parameter types */
                const char *sdl_param_type = get_sdl_c_type(func_name, j, false);
                if (sdl_param_type) {
                    param_type_c = sdl_param_type;
                } else if (item->as.function.params[j].type == TYPE_INT && 
                          (strncmp(func_name, "SDL_", 4) == 0 || strncmp(func_name, "TTF_", 4) == 0)) {
                    /* Keep as int64_t for non-pointer int parameters */
                }
                
                sb_appendf(sb, "%s %s", param_type_c, item->as.function.params[j].name);
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
                        const char *prefixed_name = get_prefixed_type_name(func->params[j].struct_type_name);
                        if (prefixed_name) {
                            sb_append(sb, prefixed_name);
                        } else {
                            sb_append(sb, type_to_c(func->params[j].type));
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
                /* Use prefixed type name */
                const char *prefixed_name = get_prefixed_type_name(item->as.function.return_struct_type_name);
                sb_append(sb, prefixed_name);
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
                    /* Use prefixed type name */
                    const char *prefixed_name = get_prefixed_type_name(item->as.function.params[j].struct_type_name);
                    sb_appendf(sb, "%s %s", prefixed_name, item->as.function.params[j].name);
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
                /* Use prefixed type name */
                const char *prefixed_name = get_prefixed_type_name(item->as.function.return_struct_type_name);
                sb_append(sb, prefixed_name);
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
                    /* Use prefixed type name */
                    const char *prefixed_name = get_prefixed_type_name(item->as.function.params[j].struct_type_name);
                    sb_appendf(sb, "%s %s", prefixed_name, item->as.function.params[j].name);
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
            env->symbol_count = saved_symbol_count;
        }
    }

    /* Add C main() wrapper that calls nl_main() for standalone executables */
    /* Only add if there's a main function (modules don't have main) */
    Function *main_func = env_get_function(env, "main");
    if (main_func && !main_func->is_extern) {
    sb_append(sb, "\n/* C main() entry point - calls nanolang main (nl_main) */\n");
    sb_append(sb, "int main() {\n");
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