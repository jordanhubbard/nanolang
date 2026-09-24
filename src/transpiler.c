#include "nanolang.h"
#include "module_symbol.h"
#include "module_builder.h"
#include "stdlib_runtime.h"
#include <stdarg.h>
#include <libgen.h>
#include <limits.h>
#include "transpiler_opaque_names.inc"
static const char *native_derived_type_name(const TypeInfo *);
static bool native_derived_forwarded(const char *);
static bool native_derived_emitted(const TypeInfo *);
static void emit_native_type_info(Environment *, StringBuilder *, TypeInfo *);


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
    int length = vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    if (length < 0) {
        fprintf(stderr, "I cannot format native output\n");
        exit(1);
    }
    if ((size_t)length < sizeof(buffer)) {
        sb_append(sb, buffer);
        return;
    }
    char *complete = malloc((size_t)length + 1);
    if (!complete) {
        fprintf(stderr, "I cannot allocate native formatted output\n");
        exit(1);
    }
    va_start(args, fmt);
    vsnprintf(complete, (size_t)length + 1, fmt, args);
    va_end(args);
    sb_append(sb, complete);
    free(complete);
}

/* Safe helper to build monomorphized type names with bounds checking
 * Returns true on success, false if buffer would overflow */
static bool build_monomorphized_name(char *dest, size_t dest_size, 
                                     const char *base_name, 
                                     const char **type_args, int type_arg_count, const TypeInfo *complete) {
    if (!dest || !base_name || dest_size == 0) return false;
    
    if (opaque_type_info_present(complete)) {
        char *key = opaque_type_info_key(complete);
        if (!key) native_opaque_name_failure();
        const char *name = native_opaque_projection(key);
        int written = snprintf(dest, dest_size, "%s", name);
        free(key);
        if (written < 0 || (size_t)written >= dest_size) native_opaque_name_failure();
        return true;
    }
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
    TypeInfo info = { .base_type = TYPE_UNION, .generic_name = (char *)base_name,
                      .type_params = type_params, .type_param_count = type_param_count };
    char *name = typeinfo_to_generic_arg_name(&info);
    if (!name) return false;
    int written = snprintf(dest, dest_size, "%s", native_opaque_projection(name));
    free(name);
    return written >= 0 && (size_t)written < dest_size;
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
    static const char *types[] = {
        "ASTArrayLiteral",
        "ASTAssert",
        "ASTBinaryOp",
        "ASTBlock",
        "ASTBool",
        "ASTCall",
        "ASTEnum",
        "ASTFieldAccess",
        "ASTFloat",
        "ASTFor",
        "ASTFunction",
        "ASTIdentifier",
        "ASTIf",
        "ASTImport",
        "ASTLet",
        "ASTMatch",
        "ASTModuleQualifiedCall",
        "ASTNumber",
        "ASTOpaqueType",
        "ASTServiceDecl",
        "ASTPrint",
        "ASTReturn",
        "ASTSet",
        "ASTShadow",
        "ASTStmtRef",
        "ASTString",
        "ASTStruct",
        "ASTStructLiteral",
        "ASTTupleIndex",
        "ASTTupleLiteral",
        "ASTUnion",
        "ASTUnionConstruct",
        "ASTUnsafeBlock",
        "ASTWhile",
        "CompilerDiagnostic",
        "CompilerSourceLocation",
        "LexerToken",
    };
    if (!name) return false;
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i)
        if (!strcmp(name, types[i])) return true;
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

/* I snapshot declared one-letter unions and enums for this emission only. */
static _Thread_local uint32_t native_declared_letters;

/* Get prefixed type name for user-defined types */
/* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
static const char *get_prefixed_type_name(const char *name) {
    name = native_opaque_projection(name);
    static _Thread_local char *buffer;
    static _Thread_local size_t capacity;
    
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

    /* Free type variables (single-letter uppercase, e.g. T, U, V, K, E):
     * These are bare generic type parameters not yet monomorphized.
     * Emit void* so the generated C compiles; correctness is tested
     * via shadow tests (interpreter path) which handle generics natively. */
    if (name[0] >= 'A' && name[0] <= 'Z' && name[1] == '\0' &&
        !(native_declared_letters & (UINT32_C(1) << (name[0] - 'A')))) {
        return "void*";
    }

    /* User types: add nl_ prefix */
    size_t length = strlen(name);
    if (length > SIZE_MAX - 4) {
        fprintf(stderr, "I cannot represent this native type name\n");
        exit(1);
    }
    if (capacity < length + 4) {
        char *grown = realloc(buffer, length + 4);
        if (!grown) {
            fprintf(stderr, "I cannot allocate a native type name\n");
            exit(1);
        }
        buffer = grown;
        capacity = length + 4;
    }
    memcpy(buffer, "nl_", 3);
    memcpy(buffer + 3, name, length + 1);
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
    union_name = native_opaque_projection(union_name);
    if (!union_name || !variant_name) {
        fprintf(stderr, "I cannot emit a union variant without both name components\n");
        exit(1);
    }
    static _Thread_local char buffer[512];
    snprintf(buffer, sizeof(buffer), "nl_%s_%s", union_name, variant_name);
    return buffer;
}

/* Get prefixed union tag name: nl_UnionName_TAG_Variant */
/* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
static const char *get_prefixed_tag_name(const char *union_name, const char *variant_name) {
    union_name = native_opaque_projection(union_name);
    if (!union_name || !variant_name) {
        fprintf(stderr, "I cannot emit a union variant without both name components\n");
        exit(1);
    }
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
    bool *owned;
    const Environment *env;
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

static bool module_headers_contain(const char *substr) {
    if (!substr || substr[0] == '\0') return false;
    for (size_t i = 0; i < g_module_headers_count; i++) {
        if (g_module_headers[i].name && strstr(g_module_headers[i].name, substr)) {
            return true;
        }
    }
    return false;
}

static bool module_header_declares_nl_wrapper(const char *func_name) {
    if (!func_name || strncmp(func_name, "nl_", 3) != 0) return false;

    static const struct {
        const char *prefix;
        const char *header;
    } header_wrappers[] = {
        {"nl_audio_viz_", "audio_viz.h"},
        {"nl_bullet_", "bullet_bindings.h"},
        {"nl_event_", "event_helpers.h"},
        {"nl_evtimer_", "event_helpers.h"},
        {"nl_examples_", "examples_io.h"},
        {"nl_forth_see", "forth_see.h"},
        {"nl_fs_", "filesystem.h"},
        {"nl_github_", "github.h"},
        {"nl_gl", "glew_wrappers.h"},
        {"nl_group_", "dispatch.h"},
        {"nl_hm_", "collections.h"},
        {"nl_img_", "sdl_image_helpers.h"},
        {"nl_json_", "json.h"},
        {"nl_keypad", "ncurses_helpers.h"},
        {"nl_log_", "log.h"},
        {"nl_nodelay", "ncurses_helpers.h"},
        {"nl_open_font_portable", "sdl_ttf_helpers.h"},
        {"nl_openai_", "openai.h"},
        {"nl_os_fd_", "process.h"},
        {"nl_os_process_", "process.h"},
        {"nl_peg2_ffi_", "peg2.h"},
        {"nl_peg_", "peg.h"},
        {"nl_prefs_", "preferences.h"},
        {"nl_pybridge_", "pybridge.h"},
        {"nl_queue_", "dispatch.h"},
        {"nl_render_text_", "sdl_ttf_helpers.h"},
        {"nl_draw_text_", "sdl_ttf_helpers.h"},
        {"nl_sb_", "collections.h"},
        {"nl_sdl_term_", "sdl_term.h"},
        {"nl_sdl_", "sdl_helpers.h"},
        {"nl_set_", "collections.h"},
        {"nl_term_", "sdl_term.h"},
        {"nl_ui_", "ui_widgets.h"},
        {"nl_uv_", "uv_helpers.h"},
        {"nl_ws_", "websocket_helpers.h"},
        {NULL, NULL}
    };

    for (int i = 0; header_wrappers[i].prefix; i++) {
        size_t len = strlen(header_wrappers[i].prefix);
        if (strncmp(func_name, header_wrappers[i].prefix, len) == 0 &&
            module_headers_contain(header_wrappers[i].header)) {
            return true;
        }
    }

    if ((strcmp(func_name, "nl_system") == 0 ||
         strcmp(func_name, "nl_flush_stdout") == 0) &&
        module_headers_contain("sdl_helpers.h")) {
        return true;
    }

    return false;
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
    }
    module_metadata_free(meta);
    
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
    /* Function registry does not own signatures (AST owns them). */
    if (reg->signatures) {
        free(reg->signatures);
    }
    free(reg);
}

/* Tuple type registry functions */
static TupleTypeRegistry *create_tuple_type_registry(const Environment *env) {
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
    reg->owned = calloc(16, sizeof(bool));
    if (!reg->owned) { free(reg->tuples); free(reg->typedef_names); free(reg); fprintf(stderr, "I cannot retain a complete native tuple registry.\n"); exit(1); }
    reg->env = env;
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
    for (int i = 0; i < reg->count; ++i)
        if (reg->owned[i]) free_payload_type_info(reg->tuples[i]);
    free(reg->owned);
    /* I borrow AST/context rows and own explicitly transferred temporaries. */
    if (reg->tuples) {
        free(reg->tuples);
    }
    free(reg);
}

/* Check if two tuple types are equal */
static bool tuple_types_equal(TypeInfo *a, TypeInfo *b) {
    return a && b && type_infos_equal(a, b);
}

/* Generate typedef name for a tuple type */
static char *get_tuple_typedef_name(TypeInfo *info, int index) {
    if (opaque_type_info_present(info)) {
        char *key = opaque_type_info_key(info);
        if (!key) native_opaque_name_failure();
        char *name = strdup(native_opaque_projection(key));
        free(key);
        if (!name) native_opaque_name_failure();
        return name;
    }
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
static const char *register_tuple_type_mode(TupleTypeRegistry *reg, TypeInfo *info, bool owned) {
    /* Check if already registered */
    for (int i = 0; i < reg->count; i++) {
        if (tuple_types_equal(reg->tuples[i], info)) {
            if (owned) free_payload_type_info(info);
            return reg->typedef_names[i];
        }
    }
    
    /* Register new tuple type */
    if (reg->count >= reg->capacity) {
        if (reg->capacity > INT_MAX / 2 || (size_t)reg->capacity > SIZE_MAX / (2 * sizeof(TypeInfo *))) {
            fprintf(stderr, "Error: Tuple registry capacity overflow\n");
            exit(1);
        }
        int new_capacity = reg->capacity * 2;
        TypeInfo **new_tuples = malloc(sizeof(TypeInfo*) * (size_t)new_capacity);
        char **new_names = malloc(sizeof(char*) * (size_t)new_capacity);
        bool *new_owned = calloc((size_t)new_capacity, sizeof(bool));
        if (!new_tuples || !new_names || !new_owned) {
            free(new_tuples); free(new_names); free(new_owned);
            if (owned) free_payload_type_info(info);
            fprintf(stderr, "I cannot retain a complete native tuple registry.\n"); exit(1);
        }
        memcpy(new_tuples, reg->tuples, (size_t)reg->count * sizeof *new_tuples);
        memcpy(new_names, reg->typedef_names, (size_t)reg->count * sizeof *new_names);
        memcpy(new_owned, reg->owned, (size_t)reg->count * sizeof *new_owned);
        free(reg->tuples); free(reg->typedef_names); free(reg->owned);
        reg->tuples = new_tuples; reg->typedef_names = new_names; reg->owned = new_owned;
        reg->capacity = new_capacity;
    }
    
    reg->owned[reg->count] = owned;
    reg->tuples[reg->count] = info;
    reg->typedef_names[reg->count] = get_tuple_typedef_name(info, reg->count);
    reg->count++;
    
    return reg->typedef_names[reg->count - 1];
}

static const char *register_tuple_type(TupleTypeRegistry *reg, TypeInfo *info) {
    return register_tuple_type_mode(reg, info, false);
}

/* Generate C typedef for a tuple type */
static void generate_tuple_typedef(StringBuilder *sb, TypeInfo *info, const char *typedef_name, Environment *env) {
    if (native_derived_forwarded(typedef_name)) sb_appendf(sb, "struct %s { ", typedef_name);
    else sb_appendf(sb, "typedef struct { ");
    if (!info->tuple_element_count) sb_append(sb, "int _placeholder");
    for (int i = 0; i < info->tuple_element_count; i++) {
        if (i > 0) sb_append(sb, "; ");
        TypeInfo flat;
        const TypeInfo *child = type_info_tuple_element(info, i, &flat);
        if (!child) native_opaque_name_failure();
        if (info->type_param_count) {
            emit_native_type_info(env, sb, (TypeInfo *)child);
            sb_appendf(sb, " _%d", i); continue;
        }
        Type t = info->tuple_types[i];
        if (t == TYPE_STRUCT || t == TYPE_UNION || t == TYPE_ENUM) {
            if (info->tuple_type_names && info->tuple_type_names[i]) {
                const char *prefixed = env_get_opaque_type(env, info->tuple_type_names[i])
                    ? "void*" : get_prefixed_type_name(info->tuple_type_names[i]);
                sb_appendf(sb, "%s _%d", prefixed, i);
            } else {
                sb_appendf(sb, "void* /* tuple composite */ _%d", i);
            }
        } else {
            sb_appendf(sb, "%s _%d", type_to_c(t), i);
        }
    }
    if (native_derived_forwarded(typedef_name)) sb_append(sb, "; };\n");
    else sb_appendf(sb, "; } %s;\n", typedef_name);
}

/* Generate unique typedef name for a function signature */
static char *get_function_typedef_name(FunctionSignature *sig, int index) {
    TypeInfo view = {.base_type = TYPE_FUNCTION, .fn_sig = sig};
    if (opaque_type_info_present(&view)) {
        char *key = opaque_type_info_key(&view);
        if (!key) native_opaque_name_failure();
        char *name = strdup(native_opaque_projection(key));
        free(key);
        if (!name) native_opaque_name_failure();
        return name;
    }
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

static void emit_native_type_info(Environment *env, StringBuilder *sb, TypeInfo *info);

static void emit_signature_type(StringBuilder *sb, Environment *env, Type type,
                                const char *name, TypeInfo *info) {
    if (info && native_derived_type_name(info) && (type == TYPE_TUPLE || type == TYPE_FUNCTION)) {
        emit_native_type_info(env, sb, info);
    } else if (info && (type == TYPE_STRUCT || type == TYPE_UNION || type == TYPE_ENUM)) {
        emit_native_type_info(env, sb, info);
    } else if (info && (type == TYPE_LIST_GENERIC || type == TYPE_HASHMAP)) {
        char *concrete = typeinfo_to_generic_arg_name(info);
        if (!concrete) { fprintf(stderr, "I cannot allocate a callback type name\n"); exit(1); }
        sb_appendf(sb, "%s*", concrete);
        free(concrete);
    } else if (name && (type == TYPE_STRUCT || type == TYPE_UNION || type == TYPE_ENUM)) {
        sb_append(sb, env_get_opaque_type(env, name) ? "void*" : get_prefixed_type_name(name));
    } else if (name && type == TYPE_LIST_GENERIC) {
        sb_appendf(sb, "List_%s*", name);
    } else {
        sb_append(sb, type_to_c(type));
    }
}
static void emit_signature_parameters(StringBuilder *sb, Environment *env, FunctionSignature *sig) {
    if (!sig->param_count) sb_append(sb, "void");
    for (int i = 0; i < sig->param_count; ++i) {
        if (i) sb_append(sb, ", ");
        emit_signature_type(sb, env, sig->param_types[i],
            sig->param_struct_names ? sig->param_struct_names[i] : NULL,
            sig->param_type_info ? sig->param_type_info[i] : NULL);
    }
}
/* I use complete annotation trees for each native callback boundary. */
static void generate_function_typedef(StringBuilder *sb, FunctionSignature *sig,
                                     const char *typedef_name, Environment *env) {
    TypeInfo function = {.base_type = TYPE_FUNCTION, .fn_sig = sig};
    if (native_derived_type_name(&function)) {
        TypeInfo result = {.base_type = sig->return_type, .generic_name = sig->return_struct_name, .fn_sig = sig->return_fn_sig};
        if (sig->return_type_info) result = *sig->return_type_info;
        if (!result.fn_sig) result.fn_sig = sig->return_fn_sig;
        sb_append(sb, "typedef ");
        emit_signature_type(sb, env, result.base_type, result.generic_name, &result);
        sb_appendf(sb, " (*%s)(", typedef_name);
        emit_signature_parameters(sb, env, sig);
        sb_append(sb, ");\n"); return;
    }
    sb_append(sb, "typedef ");
    if (sig->return_type == TYPE_FUNCTION && sig->return_fn_sig) {
        FunctionSignature *inner = sig->return_fn_sig;
        emit_signature_type(sb, env, inner->return_type, inner->return_struct_name, inner->return_type_info);
        sb_appendf(sb, " (*(*%s)(", typedef_name);
        emit_signature_parameters(sb, env, sig);
        sb_append(sb, "))(");
        emit_signature_parameters(sb, env, inner);
    } else {
        emit_signature_type(sb, env, sig->return_type, sig->return_struct_name, sig->return_type_info);
        sb_appendf(sb, " (*%s)(", typedef_name);
        emit_signature_parameters(sb, env, sig);
    }
    sb_append(sb, ");\n");
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

/* Returns true if name is a single uppercase letter (generic type variable) */
static bool is_type_var(const char *name) {
    return name && name[0] != '\0' && name[1] == '\0' && (name[0] >= 'A' && name[0] <= 'Z');
}

/* Returns true if an AST function node has any type-variable parameters */
static bool func_node_is_generic(const ASTNode *item) {
    if (!item->as.function.params) return false;
    for (int i = 0; i < item->as.function.param_count; i++) {
        if (item->as.function.params[i].type == TYPE_STRUCT &&
            is_type_var(item->as.function.params[i].struct_type_name))
            return true;
    }
    return false;
}

/* Resolve a type variable to its concrete bound type; returns the original type if not a var */
static const char *resolve_generic_param_c(Type type, const char *struct_name,
                                             const GenericFuncInstance *inst) {
    if (type == TYPE_STRUCT && is_type_var(struct_name)) {
        for (int i = 0; i < inst->binding_count; i++) {
            if (strcmp(inst->var_names[i], struct_name) == 0) {
                /* Primitive bound type */
                if (inst->bound_types[i] == TYPE_INT)    return "int64_t";
                if (inst->bound_types[i] == TYPE_FLOAT)  return "double";
                if (inst->bound_types[i] == TYPE_BOOL)   return "bool";
                if (inst->bound_types[i] == TYPE_STRING) return "const char*";
                if (inst->bound_types[i] == TYPE_STRUCT && inst->bound_type_names[i]) {
                    static _Thread_local char buf[256];
                    snprintf(buf, sizeof(buf), "nl_%s", inst->bound_type_names[i]);
                    return buf;
                }
                return "int64_t";  /* fallback */
            }
        }
    }
    /* Not a type variable: fall through to regular type_to_c */
    if (type == TYPE_STRUCT && struct_name) {
        static _Thread_local char sbuf[256];
        snprintf(sbuf, sizeof(sbuf), "nl_%s", struct_name);
        return sbuf;
    }
    return NULL;  /* caller should use type_to_c() */
}

/* Emit forward declaration for one generic function instance */
static void emit_generic_forward_decl(StringBuilder *sb, const ASTNode *orig,
                                        const GenericFuncInstance *inst,
                                        Environment *env, bool is_module) {
    if (!orig->as.function.is_pub && !is_module) sb_append(sb, "static ");

    /* Return type */
    Type rt = orig->as.function.return_type;
    const char *rs = orig->as.function.return_struct_type_name;
    const char *rname = resolve_generic_param_c(rt, rs, inst);
    if (rname) sb_append(sb, rname);
    else        sb_append(sb, type_to_c(rt));

    sb_appendf(sb, " nl_%s(", inst->mono_name);
    if (orig->as.function.param_count == 0) sb_append(sb, "void");

    /* Parameters */
    for (int j = 0; j < orig->as.function.param_count; j++) {
        if (j > 0) sb_append(sb, ", ");
        Type pt = orig->as.function.params[j].type;
        const char *ps = orig->as.function.params[j].struct_type_name;
        const char *pname = resolve_generic_param_c(pt, ps, inst);
        if (pname) sb_appendf(sb, "%s %s", pname, orig->as.function.params[j].name);
        else        sb_appendf(sb, "%s %s", type_to_c(pt), orig->as.function.params[j].name);
    }
    sb_append(sb, ");\n");
    (void)env;
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
        case TYPE_HASHMAP: return "void*"; /* Specialized as HashMap_K_V* when TypeInfo is available */
        case TYPE_OPAQUE: return "void*"; /* Opaque pointers stored as void* */
        case TYPE_OPEN_RECORD: return "int64_t"; /* row-poly: interpreter-only stub */
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
    const char *helper_name = module_helper_c_name(nano_name);
    if (helper_name != nano_name) return helper_name;
    /* WARNING: Returns pointer to thread-local static storage. Valid until next call. */
    static _Thread_local char buffer[512];
    
    /* Extern functions use their original name without any mangling or nl_ prefix */
    if (is_extern) {
        return nano_name;
    }
    
    /* Don't prefix list runtime functions */
    if (strncmp(nano_name, "nl_list_Token_", 11) == 0) {
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
    
    /* I keep an allowed declaration separate from my retained builtin helper. */
    if (env_native_array_operation(nano_name)) {
        snprintf(buffer, sizeof(buffer), "__nl_declared_%s", nano_name);
        return buffer;
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
const char *g_source_file_for_line_directives = NULL;
bool g_profile_mode = false;          /* --profile: emit timing guard in next function body block */
const char *g_profile_func_name = NULL; /* name of function being profiled */
bool g_trace_mode = false;            /* --trace: emit trace guard in next function body block */
const char *g_trace_func_name = NULL; /* name of function being traced */

#define TRANSPILER_INTERNAL_TYPES_DEFINED
static const char *native_list_element(Environment *env, const GenericInstantiation *inst);
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
        case AST_TUPLE_LITERAL: {
            const TypeInfo *complete = checked_expression_type_info(expr, (Environment *)reg->env);
            if (!complete || !type_info_tuple_valid(complete)) native_opaque_name_failure();
            register_tuple_type(reg, (TypeInfo *)complete);
            /* Also collect from tuple elements */
            for (int i = 0; i < expr->as.tuple_literal.element_count; i++) {
                collect_tuple_types_from_expr(expr->as.tuple_literal.elements[i], reg);
            }
            break;
        }
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
        case AST_UNSAFE_BLOCK:
            for (int i = 0; i < stmt->as.unsafe_block.count; ++i)
                collect_fn_sigs(stmt->as.unsafe_block.statements[i], reg);
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
    /* WASM target: use refcount_gc instead of the interpreter GC.
     * When compiled with clang --target wasm32-unknown-unknown, __wasm__
     * is defined; we switch to the bump-pointer reference-counting GC. */
    sb_append(sb, "#ifdef __wasm__\n");
    sb_append(sb, "#  include \"runtime/refcount_gc.h\"\n");
    sb_append(sb, "#  define gc_alloc_string(len)   nl_rc_str_new(NULL, (len))\n");
    sb_append(sb, "#  define gc_retain(p)           nl_rc_retain(p)\n");
    sb_append(sb, "#  define gc_release(p)          nl_rc_release(p)\n");
    sb_append(sb, "#else\n");
    sb_append(sb, "#  include \"runtime/gc.h\"\n");
    sb_append(sb, "#endif\n");
    sb_append(sb, "#include \"runtime/dyn_array.h\"\n");
    sb_append(sb, "#include \"runtime/native_array_abi.h\"\n");
    sb_append(sb, "#ifndef __wasm__\n");
    sb_append(sb, "#  include \"nanolang.h\"\n");
    sb_append(sb, "#endif\n");
    
    /* Include headers from imported modules (generic C library support) */
    if (g_module_headers_count > 0) {
        /* Sort headers by priority (highest first) */
        qsort(g_module_headers, g_module_headers_count, sizeof(ModuleHeader), compare_headers_by_priority);
        
        sb_append(sb, "\n/* Headers from imported modules (sorted by priority) */\n");
        for (size_t i = 0; i < g_module_headers_count; i++) {
            /* Local module headers (no subdirectory) need "..." quotes so the
             * compiler searches -I paths (module directories).  System headers
             * with a path separator (e.g. SDL2/SDL.h) keep angle brackets. */
            if (strchr(g_module_headers[i].name, '/') != NULL) {
                /* System-style header: #include <SDL2/SDL.h> */
                sb_appendf(sb, "#include <%s>  /* priority: %d */\n",
                           g_module_headers[i].name,
                           g_module_headers[i].priority);
            } else {
                /* Local module header: #include "sdl_helpers.h" */
                sb_appendf(sb, "#include \"%s\"  /* priority: %d */\n",
                           g_module_headers[i].name,
                           g_module_headers[i].priority);
            }
        }
    }
    
    sb_append(sb, "\n/* nanolang runtime */\n");
    sb_append(sb, "#include \"runtime/list_int.h\"\n");
    sb_append(sb, "#include \"runtime/native_record_list.h\"\n");
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
    /* std module filesystem helpers: forward-declare directly in the
     * generated C so the intermediate C compiler always sees the correct
     * signatures regardless of the -I search path.  Signatures are kept in
     * sync with modules/std/fs.h. */
    sb_append(sb, "/* fs.h forward declarations */\n");
    sb_append(sb, "DynArray* fs_walkdir(const char* root);\n");
    sb_append(sb, "const char* path_normalize(const char* path);\n");
    sb_append(sb, "const char* path_join(const char* a, const char* b);\n");
    sb_append(sb, "const char* path_basename(const char* path);\n");
    sb_append(sb, "const char* path_dirname(const char* path);\n");
    sb_append(sb, "int64_t  fs_mkdir_p(const char* path);\n");
    sb_append(sb, "const char* path_relpath(const char* target, const char* base);\n");
    sb_append(sb, "const char* file_read(const char* path);\n");
    sb_append(sb, "int64_t  file_write(const char* path, const char* content);\n");
    sb_append(sb, "int64_t  file_append(const char* path, const char* content);\n");
    sb_append(sb, "bool     file_exists(const char* path);\n");
    sb_append(sb, "int64_t  file_delete(const char* path);\n");
    sb_append(sb, "int64_t  file_copy(const char* src, const char* dst);\n");
    sb_append(sb, "int64_t  dir_copy(const char* src, const char* dst);\n");
    /* std module process helpers: forward-declare directly so generated C
     * compiles even when no module include path is present. Signatures are
     * kept in sync with modules/std/process.h. */
    sb_append(sb, "int64_t  nl_os_process_spawn(const char* command);\n");
    sb_append(sb, "int64_t  nl_os_process_is_running(int64_t pid);\n");
    sb_append(sb, "int64_t  nl_os_process_wait(int64_t pid);\n");
    sb_append(sb, "DynArray* nl_os_process_spawn_with_pipes(const char* command);\n");
    sb_append(sb, "const char* nl_os_fd_read_available(int64_t fd);\n");
    sb_append(sb, "int64_t  nl_os_fd_close(int64_t fd);\n");
    sb_append(sb, "\n");
}

/* I consume the exact registered record, never a spelling-only specialization. */
static const char *native_list_element(Environment *env, const GenericInstantiation *inst) {
    if (!inst->generic_name || strcmp(inst->generic_name, "List")) return NULL;
    const char *name = env_nominal_name(env, inst->list_element);
    if (inst->list_element.kind != TYPE_STRUCT || !inst->list_element.ordinal ||
        !name || inst->type_arg_count != 1 || !inst->type_arg_names ||
        !inst->type_arg_names[0] || strcmp(name, inst->type_arg_names[0])) {
        fprintf(stderr, "I cannot emit a list without its exact registered record.\n");
        exit(1);
    }
    return name;
}

/* Generate List<T> forward declarations before complete record definitions. */
static void generate_list_specializations(Environment *env, StringBuilder *sb) {
    if (!env) return;
    for (int i = 0; i < env->generic_instance_count; ++i) {
        const char *name = native_list_element(env, &env->generic_instances[i]);
        if (!name) continue;
        sb_appendf(sb, "#ifndef FORWARD_DEFINED_List_%s\n", name);
        sb_appendf(sb, "#define FORWARD_DEFINED_List_%s\n", name);
        sb_appendf(sb, "typedef struct List_%s List_%s;\n#endif\n", name, name);
    }
}

/* Generate List<T> providers after complete record definitions. */
static void generate_list_implementations(Environment *env, StringBuilder *sb) {
    if (!env) return;
    for (int i = 0; i < env->generic_instance_count; ++i) {
        GenericInstantiation *inst = &env->generic_instances[i];
        const char *name = native_list_element(env, inst);
        if (!name) continue;
        bool duplicate = false;
        for (int j = 0; j < i; ++j) {
            GenericInstantiation *prior = &env->generic_instances[j];
            if (prior->list_element.kind == inst->list_element.kind &&
                prior->list_element.ordinal == inst->list_element.ordinal) duplicate = true;
        }
        if (duplicate) continue;
        StructDef *record = &env->structs[inst->list_element.ordinal - 1];
        if (record->is_extern && is_schema_list_type(name)) {
            sb_appendf(sb, "#include \"runtime/list_%s.h\"\n", name);
        } else {
            static const char *operations[] = {"new", "push", "get", "set", "insert", "remove",
                "pop", "length", "capacity", "is_empty", "clear", "free", "validate", "index", "reserve"};
            for (size_t op = 0; op < sizeof(operations) / sizeof(operations[0]); ++op) {
                char generated[512];
                snprintf(generated, sizeof(generated), "nl_list_%s_%s", name, operations[op]);
                for (int f = 0; f < env->function_count; ++f) {
                    Function *function = &env->functions[f];
                    if (!function->name || env_generated_list_element(env, function).ordinal) continue;
                    const char *actual = get_c_func_name_with_module(function->alias_of ? function->alias_of : function->name,
                                                                    function->module_name, function->is_extern);
                    if (!strcmp(actual, generated)) {
                        fprintf(stderr, "I cannot share a native list provider symbol with a declaration.\n");
                        exit(1);
                    }
                }
            }
            const char *c_type = record->is_extern ? name : get_prefixed_type_name(name);
            sb_append(sb, "#include \"runtime/native_record_list.h\"\n");
            sb_appendf(sb, "NL_DEFINE_RECORD_LIST(%s, %s)\n\n", name, c_type);
        }
    }
}

static void generate_hashmap_specializations(Environment *env, StringBuilder *sb) {
    if (!env || !env->generic_instances) return;

    bool emitted = false;
    for (int i = 0; i < env->generic_instance_count && i < 1000; i++) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (!inst || !inst->generic_name) continue;
        if (strcmp(inst->generic_name, "HashMap") != 0) continue;
        if (!inst->concrete_name || inst->type_arg_count != 2) continue;

        if (!emitted) {
            sb_append(sb, "/* ========== HashMap Forward Declarations ========== */\n");
            emitted = true;
        }
        sb_appendf(sb, "typedef struct %s %s;\n", inst->concrete_name, inst->concrete_name);
    }

    if (emitted) {
        sb_append(sb, "/* ========== End HashMap Forward Declarations ========== */\n\n");
    }
}

static void generate_hashmap_implementations(Environment *env, StringBuilder *sb) {
    if (!env || !env->generic_instances) return;

    bool emitted_any = false;

    /* Shared helpers */
    sb_append(sb, "/* ========== HashMap Runtime (Generated) ========== */\n\n");
    sb_append(sb, "static uint64_t nl_hashmap_hash_string(const char *s) {\n");
    sb_append(sb, "    if (!s) return 0;\n");
    sb_append(sb, "    uint64_t hash = 1469598103934665603ULL;\n");
    sb_append(sb, "    while (*s) { hash ^= (uint8_t)(*s++); hash *= 1099511628211ULL; }\n");
    sb_append(sb, "    return hash;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static uint64_t nl_hashmap_hash_int(int64_t x) {\n");
    sb_append(sb, "    uint64_t z = (uint64_t)x;\n");
    sb_append(sb, "    z ^= z >> 33;\n");
    sb_append(sb, "    z *= 0xff51afd7ed558ccdULL;\n");
    sb_append(sb, "    z ^= z >> 33;\n");
    sb_append(sb, "    z *= 0xc4ceb9fe1a85ec53ULL;\n");
    sb_append(sb, "    z ^= z >> 33;\n");
    sb_append(sb, "    return z;\n");
    sb_append(sb, "}\n\n");
    sb_append(sb, "static bool nl_hashmap_key_eq_string(const char *a, const char *b) {\n");
    sb_append(sb, "    if (a == b) return true;\n");
    sb_append(sb, "    if (!a || !b) return false;\n");
    sb_append(sb, "    return strcmp(a, b) == 0;\n");
    sb_append(sb, "}\n\n");

    for (int i = 0; i < env->generic_instance_count && i < 1000; i++) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (!inst || !inst->generic_name) continue;
        if (strcmp(inst->generic_name, "HashMap") != 0) continue;
        if (!inst->concrete_name || !inst->type_arg_names || inst->type_arg_count != 2) continue;

        const char *key = inst->type_arg_names[0];
        const char *val = inst->type_arg_names[1];
        if (!key || !val) continue;

        if (!(strcmp(key, "int") == 0 || strcmp(key, "string") == 0)) continue;
        if (!(strcmp(val, "int") == 0 || strcmp(val, "string") == 0)) continue;

        const char *struct_name = inst->concrete_name;
        const char *suffix = struct_name;
        if (strncmp(struct_name, "HashMap_", 8) == 0) suffix = struct_name + 8;

        const char *key_param_type = (strcmp(key, "string") == 0) ? "const char*" : "int64_t";
        const char *key_store_type = (strcmp(key, "string") == 0) ? "char*" : "int64_t";
        const char *val_param_type = (strcmp(val, "string") == 0) ? "const char*" : "int64_t";
        const char *val_store_type = (strcmp(val, "string") == 0) ? "char*" : "int64_t";
        const char *val_ret_type = (strcmp(val, "string") == 0) ? "const char*" : "int64_t";

        const char *keys_elem = (strcmp(key, "string") == 0) ? "ELEM_STRING" : "ELEM_INT";
        const char *values_elem = (strcmp(val, "string") == 0) ? "ELEM_STRING" : "ELEM_INT";
        const char *keys_push = (strcmp(key, "string") == 0) ? "string" : "int";
        const char *values_push = (strcmp(val, "string") == 0) ? "string" : "int";

        const char *hash_fn = (strcmp(key, "string") == 0) ? "nl_hashmap_hash_string" : "nl_hashmap_hash_int";
        const char *eq_fn = (strcmp(key, "string") == 0) ? "nl_hashmap_key_eq_string" : NULL;

        emitted_any = true;

        sb_appendf(sb, "typedef struct %s_Entry {\n", struct_name);
        sb_append(sb, "    uint8_t state; /* 0=empty, 1=filled, 2=tombstone */\n");
        sb_appendf(sb, "    %s key;\n", key_store_type);
        sb_appendf(sb, "    %s value;\n", val_store_type);
        sb_appendf(sb, "} %s_Entry;\n\n", struct_name);

        sb_appendf(sb, "struct %s {\n", struct_name);
        sb_append(sb, "    int64_t capacity;\n");
        sb_append(sb, "    int64_t size;\n");
        sb_append(sb, "    int64_t tombstones;\n");
        sb_appendf(sb, "    %s_Entry *entries;\n", struct_name);
        sb_append(sb, "};\n\n");

        sb_appendf(sb, "static %s* nl_hashmap_%s_alloc(int64_t cap) {\n", struct_name, suffix);
        sb_appendf(sb, "    %s *hm = (%s*)malloc(sizeof(%s));\n", struct_name, struct_name, struct_name);
        sb_append(sb, "    if (!hm) return NULL;\n");
        sb_append(sb, "    hm->capacity = cap;\n");
        sb_append(sb, "    hm->size = 0;\n");
        sb_append(sb, "    hm->tombstones = 0;\n");
        sb_appendf(sb, "    hm->entries = (%s_Entry*)calloc((size_t)cap, sizeof(%s_Entry));\n", struct_name, struct_name);
        sb_append(sb, "    if (!hm->entries) { free(hm); return NULL; }\n");
        sb_append(sb, "    return hm;\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static int64_t nl_hashmap_%s_find_slot(%s *hm, %s key, bool *out_found) {\n", suffix, struct_name, key_param_type);
        sb_append(sb, "    if (out_found) *out_found = false;\n");
        sb_append(sb, "    if (!hm || hm->capacity <= 0) return -1;\n");
        sb_appendf(sb, "    uint64_t h = %s(key);\n", hash_fn);
        sb_append(sb, "    int64_t mask = hm->capacity - 1;\n");
        sb_append(sb, "    int64_t idx = (int64_t)(h & (uint64_t)mask);\n");
        sb_append(sb, "    int64_t first_tomb = -1;\n");
        sb_append(sb, "    for (int64_t probe = 0; probe < hm->capacity; probe++) {\n");
        sb_appendf(sb, "        %s_Entry *e = &hm->entries[idx];\n", struct_name);
        sb_append(sb, "        if (e->state == 0) {\n");
        sb_append(sb, "            if (first_tomb != -1) idx = first_tomb;\n");
        sb_append(sb, "            return idx;\n");
        sb_append(sb, "        }\n");
        sb_append(sb, "        if (e->state == 2) {\n");
        sb_append(sb, "            if (first_tomb == -1) first_tomb = idx;\n");
        sb_append(sb, "        } else {\n");
        if (strcmp(key, "string") == 0) {
            sb_appendf(sb, "            if (%s(e->key, key)) { if (out_found) *out_found = true; return idx; }\n", eq_fn);
        } else {
            sb_append(sb, "            if (e->key == key) { if (out_found) *out_found = true; return idx; }\n");
        }
        sb_append(sb, "        }\n");
        sb_append(sb, "        idx = (idx + 1) & mask;\n");
        sb_append(sb, "    }\n");
        sb_append(sb, "    return first_tomb;\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static void nl_hashmap_%s_rehash(%s *hm, int64_t new_cap) {\n", suffix, struct_name);
        sb_append(sb, "    if (!hm) return;\n");
        sb_append(sb, "    if (new_cap < 8) new_cap = 8;\n");
        sb_append(sb, "    /* Ensure power-of-two capacity */\n");
        sb_append(sb, "    int64_t cap = 1;\n");
        sb_append(sb, "    while (cap < new_cap) cap <<= 1;\n");
        sb_appendf(sb, "    %s_Entry *old_entries = hm->entries;\n", struct_name);
        sb_append(sb, "    int64_t old_cap = hm->capacity;\n");
        sb_appendf(sb, "    hm->entries = (%s_Entry*)calloc((size_t)cap, sizeof(%s_Entry));\n", struct_name, struct_name);
        sb_append(sb, "    if (!hm->entries) { hm->entries = old_entries; return; }\n");
        sb_append(sb, "    hm->capacity = cap;\n");
        sb_append(sb, "    hm->size = 0;\n");
        sb_append(sb, "    hm->tombstones = 0;\n");
        sb_append(sb, "    for (int64_t i2 = 0; i2 < old_cap; i2++) {\n");
        sb_appendf(sb, "        %s_Entry *e = &old_entries[i2];\n", struct_name);
        sb_append(sb, "        if (e->state != 1) continue;\n");
        sb_append(sb, "        bool found = false;\n");
        sb_appendf(sb, "        int64_t idx = nl_hashmap_%s_find_slot(hm, e->key, &found);\n", suffix);
        sb_append(sb, "        if (idx >= 0) { hm->entries[idx] = *e; hm->entries[idx].state = 1; hm->size++; }\n");
        sb_append(sb, "    }\n");
        sb_append(sb, "    free(old_entries);\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static %s* nl_hashmap_%s_new(void) {\n", struct_name, suffix);
        sb_appendf(sb, "    return nl_hashmap_%s_alloc(16);\n", suffix);
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static void nl_hashmap_%s_put(%s *hm, %s key, %s value) {\n", suffix, struct_name, key_param_type, val_param_type);
        sb_append(sb, "    if (!hm) return;\n");
        sb_append(sb, "    if ((hm->size + hm->tombstones) * 10 >= hm->capacity * 7) {\n");
        sb_appendf(sb, "        nl_hashmap_%s_rehash(hm, hm->capacity * 2);\n", suffix);
        sb_append(sb, "    }\n");
        sb_append(sb, "    bool found = false;\n");
        sb_appendf(sb, "    int64_t idx = nl_hashmap_%s_find_slot(hm, key, &found);\n", suffix);
        sb_append(sb, "    if (idx < 0) return;\n");
        sb_appendf(sb, "    %s_Entry *e = &hm->entries[idx];\n", struct_name);
        sb_append(sb, "    if (found) {\n");
        if (strcmp(val, "string") == 0) {
            sb_append(sb, "        if (e->value) free(e->value);\n");
            sb_append(sb, "        e->value = value ? strdup(value) : strdup(\"\");\n");
        } else {
            sb_append(sb, "        e->value = value;\n");
        }
        sb_append(sb, "        return;\n");
        sb_append(sb, "    }\n");
        sb_append(sb, "    if (e->state == 2) { hm->tombstones--; }\n");
        sb_append(sb, "    e->state = 1;\n");
        if (strcmp(key, "string") == 0) {
            sb_append(sb, "    e->key = key ? strdup(key) : strdup(\"\");\n");
        } else {
            sb_append(sb, "    e->key = key;\n");
        }
        if (strcmp(val, "string") == 0) {
            sb_append(sb, "    e->value = value ? strdup(value) : strdup(\"\");\n");
        } else {
            sb_append(sb, "    e->value = value;\n");
        }
        sb_append(sb, "    hm->size++;\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static bool nl_hashmap_%s_has(%s *hm, %s key) {\n", suffix, struct_name, key_param_type);
        sb_append(sb, "    if (!hm) return false;\n");
        sb_append(sb, "    bool found = false;\n");
        sb_appendf(sb, "    int64_t idx = nl_hashmap_%s_find_slot(hm, key, &found);\n", suffix);
        sb_append(sb, "    (void)idx;\n");
        sb_append(sb, "    return found;\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static %s nl_hashmap_%s_get(%s *hm, %s key) {\n", val_ret_type, suffix, struct_name, key_param_type);
        sb_append(sb, "    if (!hm) ");
        if (strcmp(val, "string") == 0) sb_append(sb, "return \"\";\n"); else sb_append(sb, "return 0;\n");
        sb_append(sb, "    bool found = false;\n");
        sb_appendf(sb, "    int64_t idx = nl_hashmap_%s_find_slot(hm, key, &found);\n", suffix);
        sb_append(sb, "    if (!found || idx < 0) ");
        if (strcmp(val, "string") == 0) sb_append(sb, "return \"\";\n"); else sb_append(sb, "return 0;\n");
        sb_appendf(sb, "    %s_Entry *e = &hm->entries[idx];\n", struct_name);
        sb_append(sb, "    return e->value ? e->value : ");
        if (strcmp(val, "string") == 0) sb_append(sb, "\"\";\n"); else sb_append(sb, "0;\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static void nl_hashmap_%s_remove(%s *hm, %s key) {\n", suffix, struct_name, key_param_type);
        sb_append(sb, "    if (!hm) return;\n");
        sb_append(sb, "    bool found = false;\n");
        sb_appendf(sb, "    int64_t idx = nl_hashmap_%s_find_slot(hm, key, &found);\n", suffix);
        sb_append(sb, "    if (!found || idx < 0) return;\n");
        sb_appendf(sb, "    %s_Entry *e = &hm->entries[idx];\n", struct_name);
        if (strcmp(key, "string") == 0) {
            sb_append(sb, "    if (e->key) free(e->key);\n");
        }
        if (strcmp(val, "string") == 0) {
            sb_append(sb, "    if (e->value) free(e->value);\n");
        }
        sb_append(sb, "    e->state = 2;\n");
        sb_append(sb, "    hm->size--;\n");
        sb_append(sb, "    hm->tombstones++;\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static int64_t nl_hashmap_%s_length(%s *hm) {\n", suffix, struct_name);
        sb_append(sb, "    return hm ? hm->size : 0;\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static void nl_hashmap_%s_clear(%s *hm) {\n", suffix, struct_name);
        sb_append(sb, "    if (!hm) return;\n");
        sb_append(sb, "    for (int64_t i2 = 0; i2 < hm->capacity; i2++) {\n");
        sb_appendf(sb, "        %s_Entry *e = &hm->entries[i2];\n", struct_name);
        sb_append(sb, "        if (e->state != 1) { e->state = 0; continue; }\n");
        if (strcmp(key, "string") == 0) {
            sb_append(sb, "        if (e->key) free(e->key);\n");
        }
        if (strcmp(val, "string") == 0) {
            sb_append(sb, "        if (e->value) free(e->value);\n");
        }
        sb_append(sb, "        e->state = 0;\n");
        sb_append(sb, "    }\n");
        sb_append(sb, "    hm->size = 0;\n");
        sb_append(sb, "    hm->tombstones = 0;\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static void nl_hashmap_%s_free(%s *hm) {\n", suffix, struct_name);
        sb_append(sb, "    if (!hm) return;\n");
        sb_appendf(sb, "    nl_hashmap_%s_clear(hm);\n", suffix);
        sb_append(sb, "    free(hm->entries);\n");
        sb_append(sb, "    free(hm);\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static DynArray* nl_hashmap_%s_keys(%s *hm) {\n", suffix, struct_name);
        sb_appendf(sb, "    DynArray* out = dyn_array_new(%s);\n", keys_elem);
        sb_append(sb, "    if (!hm) return out;\n");
        sb_append(sb, "    for (int64_t i2 = 0; i2 < hm->capacity; i2++) {\n");
        sb_appendf(sb, "        %s_Entry *e = &hm->entries[i2];\n", struct_name);
        sb_append(sb, "        if (e->state != 1) continue;\n");
        sb_appendf(sb, "        dyn_array_push_%s(out, e->key);\n", keys_push);
        sb_append(sb, "    }\n");
        sb_append(sb, "    return out;\n");
        sb_append(sb, "}\n\n");

        sb_appendf(sb, "static DynArray* nl_hashmap_%s_values(%s *hm) {\n", suffix, struct_name);
        sb_appendf(sb, "    DynArray* out = dyn_array_new(%s);\n", values_elem);
        sb_append(sb, "    if (!hm) return out;\n");
        sb_append(sb, "    for (int64_t i2 = 0; i2 < hm->capacity; i2++) {\n");
        sb_appendf(sb, "        %s_Entry *e = &hm->entries[i2];\n", struct_name);
        sb_append(sb, "        if (e->state != 1) continue;\n");
        sb_appendf(sb, "        dyn_array_push_%s(out, e->value);\n", values_push);
        sb_append(sb, "    }\n");
        sb_append(sb, "    return out;\n");
        sb_append(sb, "}\n\n");
    }

    if (!emitted_any) {
        sb_append(sb, "/* (no HashMap instantiations) */\n\n");
    }

    sb_append(sb, "/* ========== End HashMap Runtime (Generated) ========== */\n\n");
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
                                          inst->type_arg_count, inst->type_info)) {
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
    name = native_opaque_projection(name);
    for (int i = 0; i < count; i++) {
        if (items[i].name && strcmp(items[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

/* I use the same concrete spelling for a record field and its layout edge. */
static char *native_record_field_name(Environment *env, StructDef *definition, int field) {
    TypeInfo *info = definition->field_type_info ? definition->field_type_info[field] : NULL;
    if (info && info->generic_name && info->type_param_count > 0 &&
        env_get_union(env, info->generic_name))
        return typeinfo_to_generic_arg_name(info);
    return definition->field_type_names && definition->field_type_names[field]
        ? strdup(definition->field_type_names[field]) : NULL;
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
    sb_appendf(sb, native_derived_forwarded(prefixed_name_dup) ? "struct %s {\n" : "typedef struct %s {\n", prefixed_name_dup);
    for (int j = 0; j < sdef->field_count; j++) {
        sb_append(sb, "    ");

        TypeInfo *info = sdef->field_type_info ? sdef->field_type_info[j] : NULL;
        if (info && native_derived_type_name(info)) {
            emit_native_type_info(env, sb, info);
        } else if (info && info->generic_name && info->type_param_count > 0 &&
            env_get_union(env, info->generic_name)) {
            char *field_name = native_record_field_name(env, sdef, j);
            if (!field_name) { fprintf(stderr, "I cannot allocate a native record field type\n"); exit(1); }
            sb_append(sb, get_prefixed_type_name(field_name));
            free(field_name);
        /* Opaque types are represented as TYPE_STRUCT with a registered opaque type name */
        } else if (sdef->field_types[j] == TYPE_STRUCT && sdef->field_type_names && sdef->field_type_names[j] &&
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
    if (native_derived_forwarded(prefixed_name_dup)) sb_append(sb, "};\n");
    else sb_appendf(sb, "} %s;\n", prefixed_name_dup);
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

                TypeInfo *complete = udef->variant_field_type_info && udef->variant_field_type_info[j]
                    ? udef->variant_field_type_info[j][k] : NULL;
                if (complete && native_derived_type_name(complete)) {
                    emit_native_type_info(env, sb, complete);
                } else if (ft == TYPE_STRUCT && udef->variant_field_type_names && udef->variant_field_type_names[j] &&
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

    sb_appendf(sb, native_derived_forwarded(prefixed_union) ? "struct %s {\n" : "typedef struct %s {\n", prefixed_union);
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
    if (native_derived_forwarded(prefixed_union)) sb_append(sb, "};\n\n");
    else sb_appendf(sb, "} %s;\n\n", prefixed_union);

    free((void*)prefixed_union);
}

static void emit_native_type_info(Environment *env, StringBuilder *sb, TypeInfo *info) {
    if (info->base_type == TYPE_LIST_GENERIC) {
        char *name = typeinfo_to_generic_arg_name(info);
        if (!name) { fprintf(stderr, "I cannot allocate a native list payload type\n"); exit(1); }
        sb_appendf(sb, "%s*", name);
        free(name);
        return;
    }
    const char *derived = native_derived_type_name(info);
    if (derived && (info->base_type == TYPE_TUPLE || info->base_type == TYPE_FUNCTION)) {
        sb_append(sb, derived); return;
    }
    if ((info->base_type == TYPE_STRUCT || info->base_type == TYPE_UNION || info->base_type == TYPE_ENUM) && info->generic_name) {
        if (env_get_opaque_type(env, info->generic_name)) { sb_append(sb, "void*"); return; }
        char *name = typeinfo_to_generic_arg_name(info);
        if (!name) { fprintf(stderr, "I cannot allocate a native payload type\n"); exit(1); }
        sb_append(sb, get_prefixed_type_name(name));
        free(name);
    } else sb_append(sb, type_to_c(info->base_type));
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

            TypeInfo *payload = inst->type_info
                ? resolve_union_payload_type_info(udef, j, k, inst->type_info) : NULL;
            if (payload) {
                emit_native_type_info(env, sb, payload);
                free_payload_type_info(payload);
                sb_appendf(sb, " %s;\n", udef->variant_field_names[j][k]);
                continue;
            }

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

    sb_appendf(sb, native_derived_forwarded(prefixed_union) ? "struct %s {\n" : "typedef struct %s {\n", prefixed_union);
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
    if (native_derived_forwarded(prefixed_union)) sb_append(sb, "};\n\n");
    else sb_appendf(sb, "} %s;\n\n", prefixed_union);

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
                                          inst->type_arg_count, inst->type_info)) {
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
                if (sdef->field_types[f] == TYPE_STRUCT || sdef->field_types[f] == TYPE_UNION || sdef->field_types[f] == TYPE_ENUM || sdef->field_types[f] == TYPE_GENERIC) {
                    char *field_name = native_record_field_name(env, sdef, f);
                    if (!field_name) continue;
                    int dep = find_composite_type_item(items, count, field_name);
                    free(field_name);
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
                    TypeInfo *payload = inst->type_info
                        ? resolve_union_payload_type_info(udef, v, f, inst->type_info) : NULL;
                    if (payload) {
                        if ((payload->base_type == TYPE_STRUCT || payload->base_type == TYPE_UNION || payload->base_type == TYPE_ENUM) && payload->generic_name &&
                            !env_get_opaque_type(env, payload->generic_name)) {
                            char *name = typeinfo_to_generic_arg_name(payload);
                            int dep = name ? find_composite_type_item(items, count, name) : -1;
                            free(name);
                            if (dep >= 0 && dep != i && !edges[(size_t)dep * (size_t)count + (size_t)i]) {
                                edges[(size_t)dep * (size_t)count + (size_t)i] = true;
                                indegree[i]++;
                            }
                        }
                        free_payload_type_info(payload);
                        continue;
                    }
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

/* Generate compile-time struct metadata reflection functions */
static void generate_struct_metadata(Environment *env, StringBuilder *sb) {
    if (!env || !sb) return;
    
    sb_append(sb, "/* ========== Auto-Generated Struct Metadata ========== */\n\n");
    
    for (int i = 0; i < env->struct_count; i++) {
        StructDef *sdef = &env->structs[i];
        if (!sdef || !sdef->name) continue;
        if (sdef->is_extern) continue;  /* Skip extern structs */
        
        const char *struct_name = sdef->name;
        int field_count = sdef->field_count;
        
        /* Function: __reflect_<StructName>_field_count() -> int */
        sb_appendf(sb, "inline int64_t ___reflect_%s_field_count(void) {\n", struct_name);
        sb_appendf(sb, "    return %d;\n", field_count);
        sb_append(sb, "}\n\n");
        
        /* Function: __reflect_<StructName>_field_name(index) -> string */
        sb_appendf(sb, "inline const char* ___reflect_%s_field_name(int64_t index) {\n", struct_name);
        for (int j = 0; j < field_count; j++) {
            if (j == 0) {
                sb_appendf(sb, "    if (index == %d) { return \"%s\"; }\n", 
                          j, sdef->field_names[j]);
            } else {
                sb_appendf(sb, "    else if (index == %d) { return \"%s\"; }\n", 
                          j, sdef->field_names[j]);
            }
        }
        sb_append(sb, "    return \"\";\n");
        sb_append(sb, "}\n\n");
        
        /* Function: __reflect_<StructName>_field_type(index) -> string */
        sb_appendf(sb, "inline const char* ___reflect_%s_field_type(int64_t index) {\n", struct_name);
        for (int j = 0; j < field_count; j++) {
            const char *type_str = NULL;
            switch (sdef->field_types[j]) {
                case TYPE_INT: type_str = "int"; break;
                case TYPE_FLOAT: type_str = "float"; break;
                case TYPE_STRING: type_str = "string"; break;
                case TYPE_BOOL: type_str = "bool"; break;
                case TYPE_VOID: type_str = "void"; break;
                case TYPE_STRUCT:
                    type_str = sdef->field_type_names[j] ? sdef->field_type_names[j] : "struct";
                    break;
                case TYPE_ARRAY: {
                    /* Construct "array<T>" type string */
                    static char array_type_buf[256];
                    const char *elem_type = "unknown";
                    if (sdef->field_element_types && sdef->field_element_types[j] == TYPE_INT) {
                        elem_type = "int";
                    } else if (sdef->field_element_types && sdef->field_element_types[j] == TYPE_STRING) {
                        elem_type = "string";
                    } else if (sdef->field_element_types && sdef->field_element_types[j] == TYPE_STRUCT) {
                        elem_type = sdef->field_type_names[j] ? sdef->field_type_names[j] : "struct";
                    }
                    snprintf(array_type_buf, sizeof(array_type_buf), "array<%s>", elem_type);
                    type_str = array_type_buf;
                    break;
                }
                default: type_str = "unknown"; break;
            }
            
            if (j == 0) {
                sb_appendf(sb, "    if (index == %d) { return \"%s\"; }\n", j, type_str);
            } else {
                sb_appendf(sb, "    else if (index == %d) { return \"%s\"; }\n", j, type_str);
            }
        }
        sb_append(sb, "    return \"\";\n");
        sb_append(sb, "}\n\n");
        
        /* Function: __reflect_<StructName>_has_field(name) -> bool */
        sb_appendf(sb, "inline bool ___reflect_%s_has_field(const char* name) {\n", struct_name);
        for (int j = 0; j < field_count; j++) {
            if (j == 0) {
                sb_appendf(sb, "    if (strcmp(name, \"%s\") == 0) { return 1; }\n", 
                          sdef->field_names[j]);
            } else {
                sb_appendf(sb, "    else if (strcmp(name, \"%s\") == 0) { return 1; }\n", 
                          sdef->field_names[j]);
            }
        }
        sb_append(sb, "    return 0;\n");
        sb_append(sb, "}\n\n");
        
        /* Function: __reflect_<StructName>_field_type_by_name(name) -> string */
        sb_appendf(sb, "inline const char* ___reflect_%s_field_type_by_name(const char* name) {\n", struct_name);
        for (int j = 0; j < field_count; j++) {
            const char *type_str = NULL;
            switch (sdef->field_types[j]) {
                case TYPE_INT: type_str = "int"; break;
                case TYPE_FLOAT: type_str = "float"; break;
                case TYPE_STRING: type_str = "string"; break;
                case TYPE_BOOL: type_str = "bool"; break;
                case TYPE_VOID: type_str = "void"; break;
                case TYPE_STRUCT:
                    type_str = sdef->field_type_names[j] ? sdef->field_type_names[j] : "struct";
                    break;
                case TYPE_ARRAY: {
                    static char array_type_buf2[256];
                    const char *elem_type = "unknown";
                    if (sdef->field_element_types && sdef->field_element_types[j] == TYPE_INT) {
                        elem_type = "int";
                    } else if (sdef->field_element_types && sdef->field_element_types[j] == TYPE_STRING) {
                        elem_type = "string";
                    } else if (sdef->field_element_types && sdef->field_element_types[j] == TYPE_STRUCT) {
                        elem_type = sdef->field_type_names[j] ? sdef->field_type_names[j] : "struct";
                    }
                    snprintf(array_type_buf2, sizeof(array_type_buf2), "array<%s>", elem_type);
                    type_str = array_type_buf2;
                    break;
                }
                default: type_str = "unknown"; break;
            }
            
            if (j == 0) {
                sb_appendf(sb, "    if (strcmp(name, \"%s\") == 0) { return \"%s\"; }\n", 
                          sdef->field_names[j], type_str);
            } else {
                sb_appendf(sb, "    else if (strcmp(name, \"%s\") == 0) { return \"%s\"; }\n", 
                          sdef->field_names[j], type_str);
            }
        }
        sb_append(sb, "    return \"\";\n");
        sb_append(sb, "}\n\n");
    }
    
    sb_append(sb, "/* ========== End Struct Metadata ========== */\n\n");
}

/* Generate compile-time module metadata introspection functions */
static void generate_module_metadata(Environment *env, StringBuilder *sb) {
    if (!env || !sb) return;
    
    sb_append(sb, "/* ========== Auto-Generated Module Metadata ========== */\n\n");
    
    #ifdef DEBUG_MODULE_INTROSPECTION
    fprintf(stderr, "DEBUG: Generating metadata for %d modules\n", env->module_count);
    #endif
    
    for (int i = 0; i < env->module_count; i++) {
        ModuleInfo *mod = &env->modules[i];
        if (!mod || !mod->name) continue;
        
        const char *module_name = mod->name;
        const char *module_symbol = module_symbol_suffix(module_name);
        
        #ifdef DEBUG_MODULE_INTROSPECTION
        fprintf(stderr, "DEBUG: Generating functions for module '%s' (unsafe=%d, has_ffi=%d)\n",
                module_name, mod->is_unsafe, mod->has_ffi);
        #endif
        
        /* Function: ___module_info_<NAME>() -> struct with module metadata */
        sb_append(sb, "/* Module metadata entry. */\n");
        
        /* For now, generate simple metadata functions */
        /* These can be called from NanoLang to introspect modules at compile-time */
        
        /* Function: ___module_is_unsafe_<NAME>() -> bool */
        sb_appendf(sb, "static inline bool ___module_is_unsafe_%s(void) {\n", module_symbol);
        sb_appendf(sb, "    return %s;\n", mod->is_unsafe ? "1" : "0");
        sb_append(sb, "}\n\n");
        
        /* Function: ___module_has_ffi_<NAME>() -> bool */
        sb_appendf(sb, "static inline bool ___module_has_ffi_%s(void) {\n", module_symbol);
        sb_appendf(sb, "    return %s;\n", mod->has_ffi ? "1" : "0");
        sb_append(sb, "}\n\n");
        
        /* Function: ___module_name_<NAME>() -> string */
        sb_appendf(sb, "static inline const char* ___module_name_%s(void) {\n", module_symbol);
        sb_appendf(sb, "    return %s;\n", module_c_literal(module_name));
        sb_append(sb, "}\n\n");
        
        /* Function: ___module_path_<NAME>() -> string */
        sb_appendf(sb, "static inline const char* ___module_path_%s(void) {\n", module_symbol);
        sb_appendf(sb, "    return %s;\n", module_c_literal(mod->path));
        sb_append(sb, "}\n\n");

        /* Function: ___module_function_count_<NAME>() -> int */
        sb_appendf(sb, "int64_t ___module_function_count_%s(void) {\n", module_symbol);
        sb_appendf(sb, "    return %d;\n", mod->function_count);
        sb_append(sb, "}\n\n");

        /* Function: ___module_function_name_<NAME>(idx: int) -> string */
        sb_appendf(sb, "const char* ___module_function_name_%s(int64_t idx) {\n", module_symbol);
        if (mod->function_count > 0 && mod->exported_functions) {
            for (int j = 0; j < mod->function_count; j++) {
                const char *fname = mod->exported_functions[j] ? mod->exported_functions[j] : "";
                if (j == 0) {
                    sb_appendf(sb, "    if (idx == %d) { return \"%s\"; }\n", j, fname);
                } else {
                    sb_appendf(sb, "    else if (idx == %d) { return \"%s\"; }\n", j, fname);
                }
            }
            sb_append(sb, "    else { return \"\"; }\n");
        } else {
            sb_append(sb, "    (void)idx;\n");
            sb_append(sb, "    return \"\";\n");
        }
        sb_append(sb, "}\n\n");

        /* Function: ___module_struct_count_<NAME>() -> int */
        sb_appendf(sb, "int64_t ___module_struct_count_%s(void) {\n", module_symbol);
        sb_appendf(sb, "    return %d;\n", mod->struct_count);
        sb_append(sb, "}\n\n");

        /* Function: ___module_struct_name_<NAME>(idx: int) -> string */
        sb_appendf(sb, "const char* ___module_struct_name_%s(int64_t idx) {\n", module_symbol);
        if (mod->struct_count > 0 && mod->exported_structs) {
            for (int j = 0; j < mod->struct_count; j++) {
                const char *sname = mod->exported_structs[j] ? mod->exported_structs[j] : "";
                if (j == 0) {
                    sb_appendf(sb, "    if (idx == %d) { return \"%s\"; }\n", j, sname);
                } else {
                    sb_appendf(sb, "    else if (idx == %d) { return \"%s\"; }\n", j, sname);
                }
            }
            sb_append(sb, "    else { return \"\"; }\n");
        } else {
            sb_append(sb, "    (void)idx;\n");
            sb_append(sb, "    return \"\";\n");
        }
        sb_append(sb, "}\n\n");
    }
    
    sb_append(sb, "/* ========== End Module Metadata ========== */\n\n");
}

static void generate_to_string_helpers(Environment *env, StringBuilder *sb) {
    sb_append(sb, "/* ========== To-String Helpers ========== */\n\n");

    /* Forward declarations (structs/unions can reference each other in to_string) */
    sb_append(sb, "/* To-String forward declarations */\n");

    for (int i = 0; i < env->enum_count; i++) {
        EnumDef *edef = &env->enums[i];
        if (!edef || !edef->name) continue;
        if (edef->is_extern) continue;
        const char *prefixed_enum = get_prefixed_type_name(edef->name);
        sb_appendf(sb, "static const char* nl_to_string_%s(%s v);\n", edef->name, prefixed_enum);
    }

    for (int i = 0; i < env->struct_count; i++) {
        StructDef *sdef = &env->structs[i];
        if (!sdef || !sdef->name) continue;
        if (sdef->is_extern) continue;
        const char *prefixed_struct = get_prefixed_type_name(sdef->name);
        sb_appendf(sb, "static const char* nl_to_string_%s(%s v);\n", sdef->name, prefixed_struct);
    }

    for (int i = 0; i < env->union_count; i++) {
        UnionDef *udef = &env->unions[i];
        if (!udef || !udef->name) continue;
        if (udef->generic_param_count > 0) continue;
        if (udef->is_extern) continue;
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
                                          inst->type_arg_count, inst->type_info)) {
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
        if (edef->is_extern) continue;

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
        if (sdef->is_extern) continue;

        const char *prefixed_struct = get_prefixed_type_name(sdef->name);
        sb_appendf(sb, "static const char* nl_to_string_%s(%s v) {\n", sdef->name, prefixed_struct);
        sb_append(sb, "    nl_fmt_sb_t sb = nl_fmt_sb_new(256);\n");
        sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, \"%s { \");\n", sdef->original_name ? sdef->original_name : sdef->name);

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
                        StructDef *field_sdef = env_get_struct(env, sdef->field_type_names[j]);
                        UnionDef *field_udef = env_get_union(env, sdef->field_type_names[j]);
                        if ((field_sdef && field_sdef->is_extern) || (field_udef && field_udef->is_extern)) {
                            sb_append(sb, "    nl_fmt_sb_append_cstr(&sb, \"<extern>\");\n");
                        } else {
                            sb_appendf(sb, "    nl_fmt_sb_append_cstr(&sb, nl_to_string_%s(v.%s));\n",
                                       sdef->field_type_names[j], sdef->field_names[j]);
                        }
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
        if (udef->is_extern) continue;

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
                                StructDef *field_sdef = env_get_struct(env, udef->variant_field_type_names[j][k]);
                                UnionDef *field_udef = env_get_union(env, udef->variant_field_type_names[j][k]);
                                if ((field_sdef && field_sdef->is_extern) || (field_udef && field_udef->is_extern)) {
                                    sb_append(sb, "            nl_fmt_sb_append_cstr(&sb, \"<extern>\");\n");
                                } else {
                                    sb_appendf(sb, "            nl_fmt_sb_append_cstr(&sb, nl_to_string_%s(u.data.%s.%s));\n",
                                               udef->variant_field_type_names[j][k],
                                               udef->variant_names[j],
                                               udef->variant_field_names[j][k]);
                                }
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
                                          inst->type_arg_count, inst->type_info)) {
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

        /* I use the same target-file identity as the module loader. */
        char *canonical = realpath(resolved, NULL);
        if (canonical) {
            free((char *)resolved);
            resolved = canonical;
        }

        /* Extract module name from file path BEFORE freeing resolved */
        char module_name_from_path[256];
        const char *last_slash = strrchr(resolved, '/');
        const char *base_name = last_slash ? last_slash + 1 : resolved;
        snprintf(module_name_from_path, sizeof(module_name_from_path), "%s", base_name);
        char *dot = strrchr(module_name_from_path, '.');
        if (dot) *dot = '\0';

        ASTNode *module_ast = get_cached_module_ast(resolved);
        free((char*)resolved);  /* Cast away const for free() */
        if (!module_ast || module_ast->type != AST_PROGRAM) continue;

        /* Check if module has an explicit module declaration */
        const char *module_name = NULL;
        for (int j = 0; j < module_ast->as.program.count; j++) {
            ASTNode *mi = module_ast->as.program.items[j];
            if (mi && mi->type == AST_MODULE_DECL && mi->as.module_decl.name) {
                module_name = mi->as.module_decl.name;
                break;
            }
        }
        
        /* If no module declaration, use name extracted from file path */
        if (!module_name) {
            module_name = module_name_from_path;
        }

        /* Check if module has a main function (if not, all functions are implicitly public) */
        bool module_has_main = false;
        for (int j = 0; j < module_ast->as.program.count; j++) {
            ASTNode *check = module_ast->as.program.items[j];
            if (check && check->type == AST_FUNCTION && strcmp(check->as.function.name, "main") == 0) {
                module_has_main = true;
                break;
            }
        }
        
        for (int j = 0; j < module_ast->as.program.count; j++) {
            ASTNode *mi = module_ast->as.program.items[j];
            if (!mi || mi->type != AST_FUNCTION) continue;
            /* In modules (no main), all functions are implicitly public */
            if (!mi->as.function.is_pub && module_has_main) continue;
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

            /* If the generated C file includes the relevant system headers, don't redeclare system APIs. */
            if (mi->as.function.is_extern && g_module_headers_count > 0) {
                if (strncmp(c_name, "SDL_", 4) == 0 && module_headers_contain("SDL.h")) continue;
                if (strncmp(c_name, "TTF_", 4) == 0 && module_headers_contain("SDL_ttf.h")) continue;
                if (strncmp(c_name, "IMG_", 4) == 0 && module_headers_contain("SDL_image.h")) continue;
                if (strncmp(c_name, "Mix_", 4) == 0 && module_headers_contain("SDL_mixer.h")) continue;
                if (strncmp(c_name, "sqlite3_", 8) == 0 && module_headers_contain("sqlite3.h")) continue;
                if (strncmp(c_name, "curl_", 5) == 0 && module_headers_contain("curl")) continue;
                if (strncmp(c_name, "glfw", 4) == 0 && module_headers_contain("glfw")) continue;
                if (module_header_declares_nl_wrapper(c_name)) continue;
            }

            /* If the module's own header is included, don't redeclare its exported wrapper functions. */
            if (mi->as.function.is_extern && g_module_headers_count > 0 && module_name && strncmp(c_name, "nl_", 3) == 0) {
                char module_header_needle[300];
                snprintf(module_header_needle, sizeof(module_header_needle), "%s.h", module_name);
                if (module_headers_contain(module_header_needle)) continue;
                snprintf(module_header_needle, sizeof(module_header_needle), "%s_helpers.h", module_name);
                if (module_headers_contain(module_header_needle)) continue;
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
            } else if (mi->as.function.return_type == TYPE_HASHMAP) {
                if (mi->as.function.return_type_info &&
                    mi->as.function.return_type_info->generic_name &&
                    strcmp(mi->as.function.return_type_info->generic_name, "HashMap") == 0 &&
                    mi->as.function.return_type_info->type_param_count == 2) {
                    char monomorphized_name[256];
                    if (build_monomorphized_name_from_typeinfo(
                            monomorphized_name, sizeof(monomorphized_name),
                            mi->as.function.return_type_info->generic_name,
                            mi->as.function.return_type_info->type_params,
                            mi->as.function.return_type_info->type_param_count)) {
                        sb_appendf(sb, "%s*", monomorphized_name);
                    } else {
                        sb_append(sb, "void*");
                    }
                } else {
                    sb_append(sb, "void*");
                }
            } else if (mi->as.function.return_type == TYPE_FUNCTION) {
                /* Avoid emitting incorrect prototypes (not needed for current stdlib) */
                continue;
            } else {
                sb_append(sb, type_to_c(mi->as.function.return_type));
            }

            sb_appendf(sb, " %s(", c_name);
            if (mi->as.function.param_count == 0) sb_append(sb, "void");

            /* Parameters */
            for (int p = 0; p < mi->as.function.param_count; p++) {
                if (p > 0) sb_append(sb, ", ");
                Parameter *param = &mi->as.function.params[p];
                if ((param->type == TYPE_BORROW_SHARED || param->type == TYPE_BORROW_MUT)) {
                    sb_appendf(sb, "%s%s*", param->type == TYPE_BORROW_SHARED ? "const " : "", get_prefixed_type_name(param->struct_type_name));
                } else if (param->type == TYPE_STRUCT && param->struct_type_name) {
                    OpaqueTypeDef *opaque = env ? env_get_opaque_type(env, param->struct_type_name) : NULL;
                    if (opaque) {
                        sb_append(sb, "void*");
                    } else {
                        sb_append(sb, get_prefixed_type_name(param->struct_type_name));
                    }
                } else if (param->type == TYPE_UNION && param->type_info &&
                           param->type_info->generic_name && param->type_info->type_param_count > 0) {
                    char monomorphized_name[256];
                    if (!build_monomorphized_name_from_typeinfo(
                            monomorphized_name, sizeof(monomorphized_name),
                            param->type_info->generic_name, param->type_info->type_params,
                            param->type_info->type_param_count)) {
                        fprintf(stderr, "I cannot represent this imported generic union parameter\n");
                        exit(1);
                    }
                    sb_append(sb, get_prefixed_type_name(monomorphized_name));
                } else if (param->type == TYPE_UNION && param->struct_type_name) {
                    sb_append(sb, get_prefixed_type_name(param->struct_type_name));
                } else if (param->type == TYPE_LIST_GENERIC && param->struct_type_name) {
                    sb_appendf(sb, "List_%s*", param->struct_type_name);
                } else if (param->type == TYPE_HASHMAP) {
                    if (param->type_info && param->type_info->generic_name &&
                        strcmp(param->type_info->generic_name, "HashMap") == 0 &&
                        param->type_info->type_param_count == 2) {
                        char monomorphized_name[256];
                        if (build_monomorphized_name_from_typeinfo(
                                monomorphized_name, sizeof(monomorphized_name),
                                param->type_info->generic_name,
                                param->type_info->type_params,
                                param->type_info->type_param_count)) {
                            sb_appendf(sb, "%s*", monomorphized_name);
                        } else {
                            sb_append(sb, "void*");
                        }
                    } else {
                        sb_append(sb, "void*");
                    }
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
    /* Check if we're compiling a module (no main function) */
    Function *main_func = env_get_function(env, "main");
    bool is_module = (main_func == NULL);
    
    /* Forward declare functions from current program */
    sb_append(sb, "/* Forward declarations for program functions */\n");
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        /* async fn declarations wrap a normal function node — treat them identically */
        if (item->type == AST_ASYNC_FN) item = item->as.async_fn.function;
        if (item->type == AST_FUNCTION) {
            /* Skip extern functions - they're declared above */
            if (item->as.function.is_extern) {
                continue;
            }

            /* Skip generic functions — monomorphized forward decls emitted below */
            if (func_node_is_generic(item)) continue;

            /* Add static for private functions
             * BUT: When compiling modules, export all functions by default
             * (static functions can't be linked from other compilation units)
             */
            if (!item->as.function.is_pub && !is_module) {
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
            } else if (item->as.function.return_type == TYPE_HASHMAP) {
                if (item->as.function.return_type_info &&
                    item->as.function.return_type_info->generic_name &&
                    strcmp(item->as.function.return_type_info->generic_name, "HashMap") == 0 &&
                    item->as.function.return_type_info->type_param_count == 2) {
                    char monomorphized_name[256];
                    if (build_monomorphized_name_from_typeinfo(
                            monomorphized_name, sizeof(monomorphized_name),
                            item->as.function.return_type_info->generic_name,
                            item->as.function.return_type_info->type_params,
                            item->as.function.return_type_info->type_param_count)) {
                        sb_appendf(sb, "%s*", monomorphized_name);
                    } else {
                        sb_append(sb, "void*");
                    }
                } else {
                    sb_append(sb, "void*");
                }
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
            if (item->as.function.param_count == 0) sb_append(sb, "void");
            
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
                } else if (item->as.function.params[j].type == TYPE_HASHMAP) {
                    if (item->as.function.params[j].type_info &&
                        item->as.function.params[j].type_info->generic_name &&
                        strcmp(item->as.function.params[j].type_info->generic_name, "HashMap") == 0 &&
                        item->as.function.params[j].type_info->type_param_count == 2) {
                        char monomorphized_name[256];
                        if (build_monomorphized_name_from_typeinfo(
                                monomorphized_name, sizeof(monomorphized_name),
                                item->as.function.params[j].type_info->generic_name,
                                item->as.function.params[j].type_info->type_params,
                                item->as.function.params[j].type_info->type_param_count)) {
                            sb_appendf(sb, "%s* %s", monomorphized_name, item->as.function.params[j].name);
                        } else {
                            sb_appendf(sb, "void* %s", item->as.function.params[j].name);
                        }
                    } else {
                        sb_appendf(sb, "void* %s", item->as.function.params[j].name);
                    }
                } else if ((item->as.function.params[j].type == TYPE_BORROW_SHARED || item->as.function.params[j].type == TYPE_BORROW_MUT)) {
                    sb_appendf(sb, "%s%s* %s", item->as.function.params[j].type == TYPE_BORROW_SHARED ? "const " : "", get_prefixed_type_name(item->as.function.params[j].struct_type_name), item->as.function.params[j].name);
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
                } else if (item->as.function.params[j].type == TYPE_UNION &&
                           item->as.function.params[j].type_info &&
                           item->as.function.params[j].type_info->generic_name &&
                           item->as.function.params[j].type_info->type_param_count > 0) {
                    /* Generic union parameter: Result<int, string> -> Result_int_string */
                    char monomorphized_name[256];
                    if (!build_monomorphized_name_from_typeinfo(
                            monomorphized_name, sizeof(monomorphized_name),
                            item->as.function.params[j].type_info->generic_name,
                            item->as.function.params[j].type_info->type_params,
                            item->as.function.params[j].type_info->type_param_count)) {
                        sb_appendf(sb, "int64_t %s", item->as.function.params[j].name);
                    } else {
                        const char *prefixed_name = get_prefixed_type_name(monomorphized_name);
                        sb_appendf(sb, "%s %s", prefixed_name, item->as.function.params[j].name);
                    }
                } else if (item->as.function.params[j].type == TYPE_UNION && item->as.function.params[j].struct_type_name) {
                    /* Use prefixed union name */
                    const char *prefixed_name = get_prefixed_type_name(item->as.function.params[j].struct_type_name);
                    sb_appendf(sb, "%s %s", prefixed_name, item->as.function.params[j].name);
                } else if (item->as.function.params[j].type == TYPE_TUPLE && item->as.function.params[j].type_info) {
                    /* Tuple parameter: use typedef name */
                    const char *typedef_name = register_tuple_type(tuple_registry, item->as.function.params[j].type_info);
                    sb_appendf(sb, "%s %s", typedef_name, item->as.function.params[j].name);
                } else {
                    sb_appendf(sb, "%s %s",
                              type_to_c(item->as.function.params[j].type),
                              item->as.function.params[j].name);
                }
            }
            sb_append(sb, ");\n");
        }
    }

    /* Emit forward declarations for monomorphized generic function instances */
    for (int gi = 0; gi < env->generic_func_instance_count; gi++) {
        GenericFuncInstance *inst = &env->generic_func_instances[gi];
        /* Find original AST node */
        for (int j = 0; j < program->as.program.count; j++) {
            ASTNode *it = program->as.program.items[j];
            if (it->type == AST_ASYNC_FN) it = it->as.async_fn.function;
            if (it->type == AST_FUNCTION && strcmp(it->as.function.name, inst->orig_name) == 0) {
                emit_generic_forward_decl(sb, it, inst, env, is_module);
                break;
            }
        }
    }

    sb_append(sb, "\n");
}

/* Functions for stdlib runtime generation moved to stdlib_runtime.c */

/* I pop generated-function metadata after all emitted work owns its text. */
static void pop_native_function_metadata(Environment *env, int first) {
    for (int i = first; i < env->symbol_count; ++i) {
        free(env->symbols[i].name);
        free(env->symbols[i].struct_type_name);
        env->symbols[i].name = NULL;
        env->symbols[i].struct_type_name = NULL;
    }
    env->symbol_count = first;
}

/* I bind generic parameters from their declaration and resolved instance only. */
static void bind_native_generic_parameter(Environment *env, const Parameter *param,
                                          const GenericFuncInstance *inst) {
    Type type = param->type;
    const char *name = param->struct_type_name;
    TypeInfo *info = param->type_info;
    TypeInfo resolved = {0};
    bool own_info = false;
    if (type == TYPE_STRUCT && is_type_var(name)) {
        for (int k = 0; k < inst->binding_count; ++k) {
            if (strcmp(inst->var_names[k], name) != 0) continue;
            type = inst->bound_types[k];
            name = inst->bound_type_names[k];
            resolved.base_type = type;
            resolved.generic_name = (char *)name;
            info = &resolved;
            own_info = true;
            break;
        }
    } else if (!info && type == TYPE_FUNCTION && param->fn_sig) {
        resolved.base_type = TYPE_FUNCTION;
        resolved.fn_sig = param->fn_sig;
        info = &resolved;
        own_info = true;
    }
    if (own_info) {
        TypeInfo *copy = NULL;
        if (!copy_payload_type_info_checked(info, &copy) ||
            !env_own_checker_type_info(env, copy)) {
            free_payload_type_info(copy);
            fprintf(stderr, "I cannot retain a generic parameter annotation.\n");
            exit(1);
        }
        info = copy;
    }
    char *owned_name = name ? strdup(name) : NULL;
    if (name && !owned_name) {
        fprintf(stderr, "I cannot retain a generic parameter name.\n");
        exit(1);
    }
    env_define_var_with_type_info(env, param->name, type, param->element_type,
                                 info, true, create_void());
    env->symbols[env->symbol_count - 1].struct_type_name = owned_name;
}

/* Emit implementation for one generic function instance (must appear after transpile_statement macro) */
static void emit_generic_implementation(StringBuilder *sb, const ASTNode *orig,
                                          const GenericFuncInstance *inst,
                                          Environment *env,
                                          FunctionTypeRegistry *fn_registry,
                                          bool is_module) {
    if (!orig->as.function.is_pub && !is_module) sb_append(sb, "static ");

    /* Return type */
    Type rt = orig->as.function.return_type;
    const char *rs = orig->as.function.return_struct_type_name;
    const char *rname = resolve_generic_param_c(rt, rs, inst);
    if (rname) sb_append(sb, rname);
    else        sb_append(sb, type_to_c(rt));

    sb_appendf(sb, " nl_%s(", inst->mono_name);
    if (orig->as.function.param_count == 0) sb_append(sb, "void");

    /* Parameters */
    for (int j = 0; j < orig->as.function.param_count; j++) {
        if (j > 0) sb_append(sb, ", ");
        Type pt = orig->as.function.params[j].type;
        const char *ps = orig->as.function.params[j].struct_type_name;
        const char *pname = resolve_generic_param_c(pt, ps, inst);
        if (pname) sb_appendf(sb, "%s %s", pname, orig->as.function.params[j].name);
        else        sb_appendf(sb, "%s %s", type_to_c(pt), orig->as.function.params[j].name);
    }
    sb_append(sb, ") ");

    /* Add parameters with concrete types to env for body transpilation */
    int saved_sym_count = env->symbol_count;
    for (int j = 0; j < orig->as.function.param_count; j++) {
        bind_native_generic_parameter(env, &orig->as.function.params[j], inst);
    }

    transpile_statement(sb, orig->as.function.body, 0, env, fn_registry);
    sb_append(sb, "\n");
    pop_native_function_metadata(env, saved_sym_count);
}

/* Generate function implementations from program AST */
static void generate_function_implementations(StringBuilder *sb, ASTNode *program, Environment *env,
                                              FunctionTypeRegistry *fn_registry, TupleTypeRegistry *tuple_registry) {
    /* Check if we're compiling a module (no main function) */
    Function *main_func = env_get_function(env, "main");
    bool is_module = (main_func == NULL);
    
    /* Transpile all functions (skip shadow tests and extern functions) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        /* async fn declarations wrap a normal function node — treat them identically */
        if (item->type == AST_ASYNC_FN) item = item->as.async_fn.function;
        if (item->type == AST_FUNCTION) {
            /* Skip extern functions - they're declared only, no implementation */
            if (item->as.function.is_extern) {
                continue;
            }

            /* Skip generic functions — monomorphized implementations emitted below */
            if (func_node_is_generic(item)) continue;

            /* Add static for private functions
             * BUT: When compiling modules, export all functions by default
             * (static functions can't be linked from other compilation units)
             */
            if (!item->as.function.is_pub && !is_module) {
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
            } else if (item->as.function.return_type == TYPE_HASHMAP) {
                if (item->as.function.return_type_info &&
                    item->as.function.return_type_info->generic_name &&
                    strcmp(item->as.function.return_type_info->generic_name, "HashMap") == 0 &&
                    item->as.function.return_type_info->type_param_count == 2) {
                    char monomorphized_name[256];
                    if (build_monomorphized_name_from_typeinfo(
                            monomorphized_name, sizeof(monomorphized_name),
                            item->as.function.return_type_info->generic_name,
                            item->as.function.return_type_info->type_params,
                            item->as.function.return_type_info->type_param_count)) {
                        sb_appendf(sb, "%s*", monomorphized_name);
                    } else {
                        sb_append(sb, "void*");
                    }
                } else {
                    sb_append(sb, "void*");
                }
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
            if (item->as.function.param_count == 0) sb_append(sb, "void");
            
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
                } else if (item->as.function.params[j].type == TYPE_HASHMAP) {
                    if (item->as.function.params[j].type_info &&
                        item->as.function.params[j].type_info->generic_name &&
                        strcmp(item->as.function.params[j].type_info->generic_name, "HashMap") == 0 &&
                        item->as.function.params[j].type_info->type_param_count == 2) {
                        char monomorphized_name[256];
                        if (build_monomorphized_name_from_typeinfo(
                                monomorphized_name, sizeof(monomorphized_name),
                                item->as.function.params[j].type_info->generic_name,
                                item->as.function.params[j].type_info->type_params,
                                item->as.function.params[j].type_info->type_param_count)) {
                            sb_appendf(sb, "%s* %s", monomorphized_name, item->as.function.params[j].name);
                        } else {
                            sb_appendf(sb, "void* %s", item->as.function.params[j].name);
                        }
                    } else {
                        sb_appendf(sb, "void* %s", item->as.function.params[j].name);
                    }
                } else if ((item->as.function.params[j].type == TYPE_BORROW_SHARED || item->as.function.params[j].type == TYPE_BORROW_MUT)) {
                    sb_appendf(sb, "%s%s* %s", item->as.function.params[j].type == TYPE_BORROW_SHARED ? "const " : "", get_prefixed_type_name(item->as.function.params[j].struct_type_name), item->as.function.params[j].name);
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
                } else if (item->as.function.params[j].type == TYPE_UNION &&
                           item->as.function.params[j].type_info &&
                           item->as.function.params[j].type_info->generic_name &&
                           item->as.function.params[j].type_info->type_param_count > 0) {
                    /* Generic union parameter: Result<int, string> -> Result_int_string */
                    char monomorphized_name[256];
                    if (!build_monomorphized_name_from_typeinfo(
                            monomorphized_name, sizeof(monomorphized_name),
                            item->as.function.params[j].type_info->generic_name,
                            item->as.function.params[j].type_info->type_params,
                            item->as.function.params[j].type_info->type_param_count)) {
                        sb_appendf(sb, "int64_t %s", item->as.function.params[j].name);
                    } else {
                        const char *prefixed_name = get_prefixed_type_name(monomorphized_name);
                        sb_appendf(sb, "%s %s", prefixed_name, item->as.function.params[j].name);
                    }
                } else if (item->as.function.params[j].type == TYPE_UNION && item->as.function.params[j].struct_type_name) {
                    /* Use prefixed union name */
                    const char *prefixed_name = get_prefixed_type_name(item->as.function.params[j].struct_type_name);
                    sb_appendf(sb, "%s %s", prefixed_name, item->as.function.params[j].name);
                } else if (item->as.function.params[j].type == TYPE_TUPLE && item->as.function.params[j].type_info) {
                    /* Tuple parameter: use typedef name */
                    const char *typedef_name = register_tuple_type(tuple_registry, item->as.function.params[j].type_info);
                    sb_appendf(sb, "%s %s", typedef_name, item->as.function.params[j].name);
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
                             item->as.function.params[j].type_info,
                             false, dummy_val);
                Symbol *located_param = &env->symbols[env->symbol_count - 1];
                if (native_effect_program) located_param->def_file = g_source_file_for_line_directives;
                located_param->def_line = item->line;
                located_param->def_column = item->column;
                if (item->as.function.body) {
                    located_param->scope_end_line = item->as.function.body->scope_end_line;
                    located_param->scope_end_column = item->as.function.body->scope_end_column;
                }
                
                /* A parameter owns its declared nominal metadata. */
                free(located_param->struct_type_name);
                located_param->struct_type_name = NULL;
                if (item->as.function.params[j].struct_type_name) {
                    located_param->struct_type_name = strdup(item->as.function.params[j].struct_type_name);
                    if (!located_param->struct_type_name) {
                        fprintf(stderr, "Error: Out of memory duplicating parameter type name\n");
                        exit(1);
                    }
                }
            }

            /* Functions with open-record (row-polymorphic) parameters can't be
             * compiled to concrete C code — emit a stub and let the interpreter
             * handle them via shadow blocks. */
            bool has_open_record_param = false;
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (item->as.function.params[j].type == TYPE_OPEN_RECORD) {
                    has_open_record_param = true; break;
                }
            }
            if (has_open_record_param) {
                Type ret = item->as.function.return_type;
                if (ret == TYPE_VOID)        sb_append(sb, "{ /* row-poly: interpreter-only */ }\n");
                else if (ret == TYPE_STRING) sb_append(sb, "{ return NULL; /* row-poly */ }\n");
                else                         sb_append(sb, "{ return 0; /* row-poly */ }\n");
                pop_native_function_metadata(env, saved_symbol_count);
                sb_append(sb, "\n");
                g_current_function = NULL;
                continue;
            }

            /* Function body */
            g_current_function = item;  /* Set context for union construction in returns */
            if (env && env->profile) {
                /* Signal the block emitter to inject timing guard at start of this body */
                g_profile_mode = true;
                g_profile_func_name = c_func_name;
            }
            if (env && env->trace) {
                /* Signal the block emitter to inject a trace guard at start of this body */
                g_trace_mode = true;
                g_trace_func_name = c_func_name;
            }
            transpile_statement(sb, item->as.function.body, 0, env, fn_registry);
            g_profile_mode = false;   /* Clear after body is emitted */
            g_profile_func_name = NULL;
            g_trace_mode = false;
            g_trace_func_name = NULL;
            g_current_function = NULL;  /* Clear context */
            sb_append(sb, "\n");

            /* I preserve shared type facts and retire only owned symbol names. */
            pop_native_function_metadata(env, saved_symbol_count);
        }
    }

    /* Emit implementations for monomorphized generic function instances */
    for (int gi = 0; gi < env->generic_func_instance_count; gi++) {
        GenericFuncInstance *inst = &env->generic_func_instances[gi];
        for (int j = 0; j < program->as.program.count; j++) {
            ASTNode *it = program->as.program.items[j];
            if (it->type == AST_ASYNC_FN) it = it->as.async_fn.function;
            if (it->type == AST_FUNCTION && strcmp(it->as.function.name, inst->orig_name) == 0) {
                emit_generic_implementation(sb, it, inst, env, fn_registry, is_module);
                break;
            }
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

    sb_append(sb, "/* isatty() wrapper - avoids int64_t type clash with unistd.h */\n");
    sb_append(sb, "static inline int64_t nl_isatty(int64_t fd) {\n");
    sb_append(sb, "    return (int64_t)isatty((int)fd);\n");
    sb_append(sb, "}\n\n");

    sb_append(sb, "#include \"runtime/cseed_capture.h\"\n");
    sb_append(sb, "#include \"runtime/process_capture.h\"\n");
    sb_append(sb, "#ifndef NANOLANG_STD_PROCESS_H\n");
    sb_append(sb, "static DynArray* nl_os_process_run(const char* command);\n");
    sb_append(sb, "NANO_DECLARE_LOCAL_ARRAY_ABI(nl_os_process_run);\n");
    sb_append(sb, "static DynArray* nl_os_process_run(const char* command) {\n");
    sb_append(sb, "    return nl_process_run_capture(command);\n");
    sb_append(sb, "}\n");
    sb_append(sb, "#else\n");
    sb_append(sb, "static const NanoLocalArrayAbi *const nl_os_process_run__nano_local_array_abi = NULL;\n");
    sb_append(sb, "#endif /* NANOLANG_STD_PROCESS_H */\n\n");
}

/* Generate C main() wrapper that calls nanolang main().
 * Important: Only emit this if THIS compilation unit defines a main().
 * (If "main" only exists in an imported module, emitting a wrapper here causes
 * undeclared-call/linkage issues and breaks test files that intentionally omit main.)
 */
static void generate_main_wrapper(StringBuilder *sb, ASTNode *program, Environment *env) {
    bool has_local_main = false;
    if (program && program->type == AST_PROGRAM) {
        for (int i = 0; i < program->as.program.count; i++) {
            ASTNode *item = program->as.program.items[i];
            if (!item) continue;
            if (item->type == AST_ASYNC_FN) item = item->as.async_fn.function;
            if (item->type != AST_FUNCTION) continue;
            if (!item->as.function.name) continue;
            if (strcmp(item->as.function.name, "main") != 0) continue;
            if (item->as.function.is_extern) continue;
            has_local_main = true;
            break;
        }
    }
    Function *main_func = has_local_main ? env_get_function(env, "main") : NULL;
    if (has_local_main && (!main_func || main_func->is_extern)) {
        /* If the program defines main(), it must resolve to a real function. */
        return;
    }
    
    sb_append(sb, "\n/* C main() entry point - calls nanolang main */\n");
    sb_append(sb, "/* Global argc/argv for CLI runtime support */\n");
    sb_append(sb, "int g_argc = 0;\n");
    sb_append(sb, "char **g_argv = NULL;\n\n");
    
    /* If profiling is enabled, use fork/exec wrapper */
    if (env && env->profile_gprof && has_local_main) {
        const char *c_main_name = get_c_func_name_with_module("main", main_func->module_name, main_func->is_extern);
        sb_append(sb, "int main(int argc, char **argv) {\n");
        sb_append(sb, "    g_argc = argc;\n");
        sb_append(sb, "    g_argv = argv;\n");
        sb_append(sb, "    setvbuf(stdout, NULL, _IOLBF, 0);\n");
        sb_appendf(sb, "    return _nl_run_with_profiling(argc, argv, %s);\n", c_main_name);
        sb_append(sb, "}\n");
    } else {
        /* Normal main without gprof profiling */
        sb_append(sb, "int main(int argc, char **argv) {\n");
        sb_append(sb, "    g_argc = argc;\n");
        sb_append(sb, "    g_argv = argv;\n");
        /* Line-buffer stdout so println output appears immediately even when piped */
        sb_append(sb, "    setvbuf(stdout, NULL, _IOLBF, 0);\n");
        if (env && (env->profile || env->trace)) {
            /* Resolve every diagnostics hook exactly once, at process startup. */
            sb_append(sb, "    _nl_diag_init();\n");
        }
        if (env && env->profile) {
            /* Register hotspot report to print at exit */
            sb_append(sb, "    atexit(_nl_prof_report);\n");
            /* Register flamegraph report if --profile-runtime was used */
            if (env->profile_runtime) {
                sb_append(sb, "    atexit(_nl_prof_flamegraph_report);\n");
            }
        }
        if (has_local_main) {
            const char *c_main_name = get_c_func_name_with_module("main", main_func->module_name, main_func->is_extern);
            sb_appendf(sb, "    return (int)%s();\n", c_main_name);
        } else {
            sb_append(sb, "    return 0;\n");
        }
        sb_append(sb, "}\n");
    }
}

static bool is_c_constant_initializer(ASTNode *expr) {
    if (!expr) return false;
    switch (expr->type) {
        case AST_NUMBER:
        case AST_FLOAT:
        case AST_BOOL:
        case AST_STRING:
            return true;
        default:
            return false;
    }
}

/* Generate top-level globals (constants + mutable globals).
 * For non-constant initializers, emit a small runtime initializer.
 */
static void generate_toplevel_globals(StringBuilder *sb, ASTNode *program, Environment *env,
                                      FunctionTypeRegistry *fn_registry, TupleTypeRegistry *tuple_registry) {
    sb_append(sb, "/* Top-level globals */\n");

    ASTNode **runtime_inits = NULL;
    int runtime_init_count = 0;
    int runtime_init_cap = 0;

    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type != AST_LET) continue;

        /* Skip constants that come from C headers - they're already defined in the headers */
        Symbol *sym = env_get_var(env, item->as.let.name);
        if (sym && sym->from_c_header) {
            continue;
        }

        bool is_const_init = is_c_constant_initializer(item->as.let.value);

        if (!item->as.let.is_mut && is_const_init) {
            /* Emit true constants as C constants */
            if (item->as.let.var_type == TYPE_STRING) {
                sb_append(sb, "static const char * const");
            } else {
                sb_append(sb, "static const ");
            }
            if (item->as.let.var_type == TYPE_HASHMAP &&
                item->as.let.type_info &&
                item->as.let.type_info->generic_name &&
                strcmp(item->as.let.type_info->generic_name, "HashMap") == 0 &&
                item->as.let.type_info->type_param_count == 2) {
                char monomorphized_name[256];
                if (build_monomorphized_name_from_typeinfo(
                        monomorphized_name, sizeof(monomorphized_name),
                        item->as.let.type_info->generic_name,
                        item->as.let.type_info->type_params,
                        item->as.let.type_info->type_param_count)) {
                    sb_appendf(sb, "%s*", monomorphized_name);
                } else {
                    sb_append(sb, "void*");
                }
            } else if (item->as.let.var_type != TYPE_STRING) {
                sb_append(sb, type_to_c(item->as.let.var_type));
            }
            sb_appendf(sb, " %s = ", item->as.let.name);
            if (item->as.let.var_type == TYPE_U8) sb_append(sb, "(uint8_t)(");
            transpile_expression(sb, item->as.let.value, env);
            if (item->as.let.var_type == TYPE_U8) sb_append(sb, ")");
            sb_append(sb, ";\n");
            continue;
        }

        /* Emit as a normal global (mutable or runtime-initialized constant) */
        sb_append(sb, "static ");
        if (item->as.let.var_type == TYPE_HASHMAP &&
            item->as.let.type_info &&
            item->as.let.type_info->generic_name &&
            strcmp(item->as.let.type_info->generic_name, "HashMap") == 0 &&
            item->as.let.type_info->type_param_count == 2) {
            char monomorphized_name[256];
            if (build_monomorphized_name_from_typeinfo(
                    monomorphized_name, sizeof(monomorphized_name),
                    item->as.let.type_info->generic_name,
                    item->as.let.type_info->type_params,
                    item->as.let.type_info->type_param_count)) {
                sb_appendf(sb, "%s*", monomorphized_name);
            } else {
                sb_append(sb, "void*");
            }
        } else if ((item->as.let.var_type == TYPE_STRUCT || item->as.let.var_type == TYPE_UNION ||
                    item->as.let.var_type == TYPE_ENUM) && item->as.let.type_info) {
            emit_native_type_info(env, sb, item->as.let.type_info);
        } else if (item->as.let.var_type == TYPE_STRUCT && item->as.let.type_name) {
            sb_append(sb, get_prefixed_type_name(item->as.let.type_name));
        } else if (item->as.let.var_type == TYPE_TUPLE && item->as.let.type_info) {
            sb_append(sb, register_tuple_type(tuple_registry, item->as.let.type_info));
        } else if (item->as.let.var_type == TYPE_FUNCTION && item->as.let.fn_sig) {
            sb_append(sb, register_function_signature(fn_registry, item->as.let.fn_sig));
        } else {
            sb_append(sb, type_to_c(item->as.let.var_type));
        }
        sb_appendf(sb, " %s", item->as.let.name);
        if (is_const_init) {
            sb_append(sb, " = ");
            if (item->as.let.var_type == TYPE_U8) sb_append(sb, "(uint8_t)(");
            transpile_expression(sb, item->as.let.value, env);
            if (item->as.let.var_type == TYPE_U8) sb_append(sb, ")");
        }
        sb_append(sb, ";\n");

        if (!is_const_init) {
            if (runtime_init_count >= runtime_init_cap) {
                int new_cap = runtime_init_cap == 0 ? 8 : runtime_init_cap * 2;
                ASTNode **new_arr = realloc(runtime_inits, sizeof(ASTNode*) * (size_t)new_cap);
                if (!new_arr) {
                    fprintf(stderr, "Error: Out of memory collecting top-level initializers\n");
                    exit(1);
                }
                runtime_inits = new_arr;
                runtime_init_cap = new_cap;
            }
            runtime_inits[runtime_init_count++] = item;
        }
    }

    sb_append(sb, "\n");

    if (runtime_init_count > 0) {
        sb_append(sb, "/* Top-level runtime initialization */\n");
        sb_append(sb, "static bool nl_toplevel_initialized = false;\n");
        sb_append(sb, "static void nl_init_toplevel(void) {\n");
        sb_append(sb, "    if (nl_toplevel_initialized) return;\n");
        sb_append(sb, "    nl_toplevel_initialized = true;\n");
        for (int i = 0; i < runtime_init_count; i++) {
            ASTNode *item = runtime_inits[i];
            sb_appendf(sb, "    %s = ", item->as.let.name);
            if (item->as.let.var_type == TYPE_U8) sb_append(sb, "(uint8_t)(");
            transpile_expression(sb, item->as.let.value, env);
            if (item->as.let.var_type == TYPE_U8) sb_append(sb, ")");
            sb_append(sb, ";\n");
        }
        sb_append(sb, "}\n");

        sb_append(sb, "#if defined(__GNUC__) || defined(__clang__)\n");
        sb_append(sb, "__attribute__((constructor))\n");
        sb_append(sb, "#endif\n");
        sb_append(sb, "static void nl_init_toplevel_ctor(void) { nl_init_toplevel(); }\n\n");
    }

    free(runtime_inits);
}

/* Collect module headers from all import statements */
static void collect_module_headers_from_imports(ASTNode *program, const char *source_file) {
    clear_module_headers();
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_IMPORT) {
            /* Resolve module path and collect headers.
             * Pass source_file so resolve_module_path can walk up from the
             * source file's directory to find the project root.  Without
             * this, the resolver falls back to CWD-relative probing which
             * fails when the compiler runs with CWD=/tmp and no module
             * headers are emitted, causing -Wimplicit-function-declaration
             * for every SDL/UI helper called in the generated C. */
            const char *module_path = resolve_module_path(item->as.import_stmt.module_path, source_file);
            if (module_path) {
                collect_headers_from_module(module_path);
                free((char*)module_path);  /* Cast away const for free() */
            }
        }
    }
}

/* Generate typedef declarations for function and tuple types */
static void generate_type_typedefs(StringBuilder *sb, FunctionTypeRegistry *fn_registry, 
                                     TupleTypeRegistry *tuple_registry, Environment *env) {
    /* Generate function type typedefs */
    if (fn_registry->count > 0) {
        sb_append(sb, "/* Function Type Typedefs */\n");
        for (int i = 0; i < fn_registry->count; i++) {
            TypeInfo info = {.base_type = TYPE_FUNCTION, .fn_sig = fn_registry->signatures[i]};
            if (native_derived_emitted(&info)) continue;
            generate_function_typedef(sb, fn_registry->signatures[i],
                                    fn_registry->typedef_names[i], env);
        }
        sb_append(sb, "\n");
    }
    
    /* Generate tuple type typedefs */
    if (tuple_registry->count > 0) {
        sb_appendf(sb, "/* Tuple Type Typedefs (found %d types) */\n", tuple_registry->count);
        for (int i = 0; i < tuple_registry->count; i++) {
            if (native_derived_emitted(tuple_registry->tuples[i])) continue;
            generate_tuple_typedef(sb, tuple_registry->tuples[i],
                                 tuple_registry->typedef_names[i], env);
        }
        sb_append(sb, "\n");
    }
}

/* Collect function signatures and tuple types from program AST */
static void collect_function_and_tuple_types(ASTNode *program, FunctionTypeRegistry *fn_registry,
                                               TupleTypeRegistry *tuple_registry) {
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        /* async fn declarations wrap a normal function node — treat them identically */
        if (item->type == AST_ASYNC_FN) item = item->as.async_fn.function;

        if (item->type == AST_LET) {
            collect_fn_sigs(item, fn_registry);
            if (item->as.let.var_type == TYPE_TUPLE && item->as.let.type_info)
                register_tuple_type(tuple_registry, item->as.let.type_info);
            collect_tuple_types_from_stmt(item, tuple_registry);
        }

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

            /* Check parameter types for tuple types */
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (item->as.function.params[j].type == TYPE_TUPLE &&
                    item->as.function.params[j].type_info) {
                    register_tuple_type(tuple_registry, item->as.function.params[j].type_info);
                }
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

        bool module_has_main = false;
        for (int j = 0; j < module_ast->as.program.count; j++) {
            ASTNode *declaration = module_ast->as.program.items[j];
            if (declaration && declaration->type == AST_FUNCTION &&
                strcmp(declaration->as.function.name, "main") == 0)
                module_has_main = true;
        }

        for (int j = 0; j < module_ast->as.program.count; j++) {
            ASTNode *mi = module_ast->as.program.items[j];
            if (!mi || mi->type != AST_FUNCTION) continue;
            if (!mi->as.function.is_pub && module_has_main) continue;

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
static void generate_module_extern_declarations(StringBuilder *sb, ASTNode *program, Environment *env, FunctionTypeRegistry *fn_registry) {
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

                /* std module fs functions: forward-declared inline in the
                 * generated C preamble by generate_c_headers() — no #include
                 * needed and no -I path dependency.  Skip them here so we do
                 * not emit a second extern declaration that would conflict with
                 * the already-correct signature in the preamble. */
                bool is_fs_h_function = (
                    strcmp(func->name, "fs_mkdir_p")       == 0 ||
                    strcmp(func->name, "path_relpath")      == 0 ||
                    strcmp(func->name, "file_copy")         == 0 ||
                    strcmp(func->name, "dir_copy")          == 0 ||
                    strcmp(func->name, "nl_os_process_run") == 0
                );

                if (is_system_function || is_module_with_header || is_fs_h_function) {
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
            } else if (func->return_type == TYPE_FUNCTION && func->return_fn_sig) {
                sb_append(sb, register_function_signature(fn_registry, func->return_fn_sig));
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
                } else if ((func->params[j].type == TYPE_BORROW_SHARED || func->params[j].type == TYPE_BORROW_MUT)) {
                    sb_appendf(sb, "%s%s*", func->params[j].type == TYPE_BORROW_SHARED ? "const " : "", get_prefixed_type_name(func->params[j].struct_type_name));
                } else if (func->params[j].type == TYPE_STRUCT && func->params[j].struct_type_name) {
                    /* Check if this is an opaque type */
                    OpaqueTypeDef *opaque = env_get_opaque_type(env, func->params[j].struct_type_name);
                    if (opaque) {
                        sb_append(sb, "void*");
                    } else {
                        const char *prefixed = get_prefixed_type_name(func->params[j].struct_type_name);
                        sb_append(sb, prefixed);
                    }
                } else if (func->params[j].type == TYPE_FUNCTION && func->params[j].fn_sig) {
                    sb_append(sb, register_function_signature(fn_registry, func->params[j].fn_sig));
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

static void generate_extern_declarations(StringBuilder *sb, ASTNode *program, Environment *env, FunctionTypeRegistry *fn_registry) {
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
            strcmp(func_name, "getenv") == 0 || strcmp(func_name, "setenv") == 0 || \
            strcmp(func_name, "unsetenv") == 0 || \
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
        /* If we have the relevant system headers in the generated C file, don't redeclare system APIs. */ \
        if (g_module_headers_count > 0) { \
            if (strncmp(func_name, "SDL_", 4) == 0 && module_headers_contain("SDL.h")) break; \
            if (strncmp(func_name, "TTF_", 4) == 0 && module_headers_contain("SDL_ttf.h")) break; \
            if (strncmp(func_name, "IMG_", 4) == 0 && module_headers_contain("SDL_image.h")) break; \
            if (strncmp(func_name, "Mix_", 4) == 0 && module_headers_contain("SDL_mixer.h")) break; \
            if (strncmp(func_name, "sqlite3_", 8) == 0 && module_headers_contain("sqlite3.h")) break; \
            if (strncmp(func_name, "curl_", 5) == 0 && module_headers_contain("curl")) break; \
            if (strncmp(func_name, "glfw", 4) == 0 && module_headers_contain("glfw")) break; \
            if (strncmp(func_name, "nl_queue_", 9) == 0 && module_headers_contain("dispatch.h")) break; \
            if (strncmp(func_name, "nl_group_", 9) == 0 && module_headers_contain("dispatch.h")) break; \
            if (module_header_declares_nl_wrapper(func_name)) break; \
            if (strncmp(func_name, "nl_", 3) == 0) { \
                Function *decl_func = env ? env_get_function(env, func_name) : NULL; \
                const char *decl_module_name = decl_func ? decl_func->module_name : NULL; \
                if (decl_module_name && decl_module_name[0] != '\0') { \
                    char module_header_needle[300]; \
                    snprintf(module_header_needle, sizeof(module_header_needle), "%s.h", decl_module_name); \
                    if (module_headers_contain(module_header_needle)) break; \
                    snprintf(module_header_needle, sizeof(module_header_needle), "%s_helpers.h", decl_module_name); \
                    if (module_headers_contain(module_header_needle)) break; \
                } \
            } \
        } \
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
        } else if ((_return_type) == TYPE_HASHMAP) { \
            if ((_return_type_info) && (_return_type_info)->generic_name && \
                strcmp((_return_type_info)->generic_name, "HashMap") == 0 && \
                (_return_type_info)->type_param_count == 2) { \
                char monomorphized_name[256]; \
                if (build_monomorphized_name_from_typeinfo( \
                        monomorphized_name, sizeof(monomorphized_name), \
                        (_return_type_info)->generic_name, \
                        (_return_type_info)->type_params, \
                        (_return_type_info)->type_param_count)) { \
                    sb_appendf(sb, "%s*", monomorphized_name); \
                } else { \
                    sb_append(sb, "void*"); \
                } \
            } else { \
                sb_append(sb, "void*"); \
            } \
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
        sb_appendf(sb, " %s(", module_helper_c_name(func_name)); \
        for (int j = 0; j < (_param_count); j++) { \
            if (j > 0) sb_append(sb, ", "); \
            const char *sdl_param_type = get_sdl_c_type(func_name, j, false); \
            if (sdl_param_type) { \
                sb_append(sb, sdl_param_type); \
            } else if (((_params)[j].type == TYPE_BORROW_SHARED || (_params)[j].type == TYPE_BORROW_MUT)) { \
                sb_appendf(sb, "%s%s*", (_params)[j].type == TYPE_BORROW_SHARED ? "const " : "", get_prefixed_type_name((_params)[j].struct_type_name)); \
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
            } else if ((_params)[j].type == TYPE_HASHMAP) { \
                if ((_params)[j].type_info && (_params)[j].type_info->generic_name && \
                    strcmp((_params)[j].type_info->generic_name, "HashMap") == 0 && \
                    (_params)[j].type_info->type_param_count == 2) { \
                    char monomorphized_name[256]; \
                    if (build_monomorphized_name_from_typeinfo( \
                            monomorphized_name, sizeof(monomorphized_name), \
                            (_params)[j].type_info->generic_name, \
                            (_params)[j].type_info->type_params, \
                            (_params)[j].type_info->type_param_count)) { \
                        sb_appendf(sb, "%s*", monomorphized_name); \
                    } else { \
                        sb_append(sb, "void*"); \
                    } \
                } else { \
                    sb_append(sb, "void*"); \
                } \
            } else if ((_params)[j].type == TYPE_FUNCTION) { \
                /* I preserve the declared callback ABI in module object builds. */ \
                sb_append(sb, register_function_signature(fn_registry, (_params)[j].fn_sig)); \
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

/* I dispatch synchronous effects through lexical frames and explicitly unwind
 * owned local guards before a handler returns from its installing function. */
static void generate_effect_dispatch(StringBuilder *sb, ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) return;
    native_effect_program = env->effect_count > 0;
    for (int i = 0; i < program->as.program.count; ++i)
        if (program->as.program.items[i]->type == AST_EFFECT_DECL) native_effect_program = true;
    if (!native_effect_program) return;
    sb_append(sb, "#include \"runtime/effect_runtime.h\"\n");
    for (int i = 0; i < env->effect_count; i++) {
        EffectDef *effect = &env->effects[i];
        const char *eff = effect->name;
        if (!eff) continue;
        for (int j = 0; j < effect->op_count; j++) {
            const char *op = effect->ops[j].name;
            int count = effect->ops[j].param_count;
            Type rtype = effect->ops[j].return_type;
            const char *crt = effect_c_type(rtype, effect->ops[j].return_type_name, env);
            sb_appendf(sb, "static %s nl_perform_%s_%s(", crt, eff, op);
            if (!count) sb_append(sb, "void");
            for (int k = 0; k < count; k++) {
                if (k) sb_append(sb, ", ");
                sb_appendf(sb, "%s _arg%d",
                    effect_c_type(effect->ops[j].params[k].type, effect->ops[j].params[k].struct_type_name, env), k);
            }
            sb_append(sb, ") { void *_args[] = {");
            for (int k = 0; k < count; k++) sb_appendf(sb, "%s&_arg%d", k ? "," : "", k);
            if (!count) sb_append(sb, "NULL");
            sb_append(sb, "}; ");
            if (rtype != TYPE_VOID) sb_appendf(sb, "%s _r = {0}; ", crt);
            sb_appendf(sb, "nl_effect_dispatch(\"%s.%s\", _args, %s); ", eff, op, rtype == TYPE_VOID ? "NULL" : "&_r");
            if (rtype != TYPE_VOID) sb_append(sb, "return _r; ");
            sb_append(sb, "}\n");
        }
    }
    sb_append(sb, "\n");
}

/* Transpile program to C */
#include "transpiler_opaque_declarations.inc"

static char *transpile_to_c_impl(ASTNode *program, Environment *env, const char *input_file) {
    if (ast_has_service_declaration(program)) { fprintf(stderr, "I have not resolved File service declarations for this consumer.\n"); return NULL; }
    if (!program || program->type != AST_PROGRAM) {
        return NULL;
    }
    
    if (!env) {
        fprintf(stderr, "Error: Environment is NULL in transpile_to_c\n");
        return NULL;
    }

    /* Set source file for #line directive emission */
    g_source_file_for_line_directives = input_file;

    /* Clear and collect headers from imported modules */
    collect_module_headers_from_imports(program, input_file);

    StringBuilder *sb = sb_create();

    /* POSIX feature macro for strdup, strnlen, etc. */
    sb_append(sb, "#define _GNU_SOURCE 1\n#define _DARWIN_C_SOURCE 1\n#define _POSIX_C_SOURCE 200809L\n\n");

    /* Generate headers */
    generate_c_headers(sb);

    /* Suppress unused warnings for runtime library and generated code */
    /* These pragmas work on both Clang and GCC */
    sb_append(sb, "#pragma GCC diagnostic push\n");
    sb_append(sb, "#pragma GCC diagnostic ignored \"-Wunused-function\"\n");
    sb_append(sb, "#pragma GCC diagnostic ignored \"-Wunused-variable\"\n");
    sb_append(sb, "#pragma GCC diagnostic ignored \"-Wunused-parameter\"\n");
    sb_append(sb, "#pragma GCC diagnostic ignored \"-Wunused-const-variable\"\n\n");

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

    /* Timing utilities */
    generate_timing_utilities(sb);

    /* Console I/O utilities (for REPL, interactive programs) */
    generate_console_io_utilities(sb);

    /* Cross-platform profiling system (only when -pg flag is used) */
    if (env && env->profile_gprof) {
        generate_profiling_system(sb, env->profile_output_path);
    }

    /* Shared diagnostics hook mechanism: one mechanism drives generated-C
     * tracing (--trace) and profiling (--profile). Emit it whenever either
     * hook is requested so both resolve from the same startup-time state. */
    if (env && (env->profile || env->trace)) {
        generate_diagnostics_runtime(sb, env->profile, env->trace);
    }

    /* Instrumented profiling system (only when --profile flag is used) */
    if (env && env->profile) {
        generate_instrumented_profiling_system(sb);
        /* Flamegraph output extension (--profile-runtime) */
        if (env->profile_runtime) {
            generate_flamegraph_profiling_system(sb, env->profile_flamegraph_path);
        }
    }

    /* Math and utility built-in functions */
    generate_math_utility_builtins(sb);

    /* Coroutine runtime builtins (coro_spawn, scheduler_run, etc.) */
    generate_coroutine_builtins(sb);

    /* ========== Function Type Typedefs ========== */
    /* Collect all function signatures and tuple types used in the program */
    FunctionTypeRegistry *fn_registry = create_fn_type_registry();
    TupleTypeRegistry *tuple_registry = create_tuple_type_registry(env);
    g_tuple_registry = tuple_registry;  /* Set global registry for expression transpilation */

    collect_module_function_types(program, fn_registry, input_file);
    collect_function_and_tuple_types(program, fn_registry, tuple_registry);
    /* I register transitive foreign signatures before their declarations. */
    for (int i = 0; i < env->function_count; ++i) {
        Function *function = &env->functions[i];
        if (!function->is_extern) continue;
        if (function->return_type == TYPE_FUNCTION && function->return_fn_sig)
            register_function_signature(fn_registry, function->return_fn_sig);
        for (int j = 0; j < function->param_count; ++j) {
            Parameter *parameter = &function->params[j];
            if (parameter->type == TYPE_FUNCTION && parameter->fn_sig)
                register_function_signature(fn_registry, parameter->fn_sig);
        }
    }


    NativeDerivedGraph *derived_graph = native_derived_prepare(env, fn_registry, tuple_registry);

    /* Generate enum typedefs first (before structs, since structs may use enums) */
    generate_enum_definitions(env, sb);

    /* Forward declare List types BEFORE structs */
    generate_list_specializations(env, sb);

    /* Forward declare HashMap types BEFORE structs */
    generate_hashmap_specializations(env, sb);

    /* Generate struct + union definitions in dependency-safe order */
    if (derived_graph) native_derived_emit(derived_graph, sb);
    else generate_struct_and_union_definitions_ordered(env, sb);

    /* Generate compile-time struct metadata reflection functions */
    generate_struct_metadata(env, sb);

    /* Generate compile-time module metadata introspection functions */
    if (env->emit_module_metadata) {
        generate_module_metadata(env, sb);
    }

    /* Generate List implementations and includes */
    generate_list_implementations(env, sb);

    /* Generate HashMap implementations */
    generate_hashmap_implementations(env, sb);

    /* Generate to_string helpers for user-defined types */
    generate_to_string_helpers(env, sb);
    
    /* Generate typedef declarations */
    generate_type_typedefs(sb, fn_registry, tuple_registry, env);

    /* Generate extern function declarations */
    generate_extern_declarations(sb, program, env, fn_registry);

    /* Generate typed entry points for synchronous effect operations. */
    generate_effect_dispatch(sb, program, env);

    /* Also generate extern declarations for extern functions from imported modules */
    generate_module_extern_declarations(sb, program, env, fn_registry);

    /* Forward declare imported module functions */
    generate_module_function_declarations(sb, program, env, input_file, fn_registry);
    
    /* I declare callable signatures before runtime global initializers use them. */
    generate_program_function_declarations(sb, program, env, fn_registry, tuple_registry);

    /* Emit top-level globals in their original initialization order. */
    generate_toplevel_globals(sb, program, env, fn_registry, tuple_registry);

    /* Generate function implementations */
    effect_helpers = sb_create(); effect_serial = 0;
    StringBuilder *implementations = sb_create();
    generate_function_implementations(implementations, program, env, fn_registry, tuple_registry);
    sb_append(sb, effect_helpers->buffer);
    sb_append(sb, implementations->buffer);
    free(effect_helpers->buffer); free(effect_helpers); effect_helpers = NULL;
    free(implementations->buffer); free(implementations);

    /* Add C main() wrapper for standalone executables (skip for module objects) */
    if (env && env->emit_c_main) {
        generate_main_wrapper(sb, program, env);
    }

    /* Re-enable warnings after all generated code */
    sb_append(sb, "\n#pragma GCC diagnostic pop\n");

    /* Cleanup */
    free_fn_type_registry(fn_registry);
    free_tuple_type_registry(tuple_registry);
    native_derived_graph_free(derived_graph); native_derived_graph = NULL;
    g_tuple_registry = NULL;  /* Clear global registry */
    clear_module_headers();  /* Clear collected headers */

    char *result = sb->buffer;
    free(sb);
    return result;
}

/* I retain no borrowed declaration pointer and restore context on every exit. */
char *transpile_to_c(ASTNode *program, Environment *env, const char *input_file) {
    NativeOpaqueNames names = {.prefix = env ? env_opaque_symbol_prefix(env) : 0};
    NativeOpaqueNames *previous_names = native_opaque_names;
    NativeDerivedGraph *previous_graph = native_derived_graph;
    TupleTypeRegistry *previous_tuples = g_tuple_registry;
    native_derived_graph = NULL;
    native_opaque_names = &names;
    uint32_t previous = native_declared_letters;
    native_declared_letters = 0;
    for (int i = 0; env && i < env->union_count; ++i) {
        const char *name = env->unions[i].name;
        if (name && name[0] >= 'A' && name[0] <= 'Z' && name[1] == '\0')
            native_declared_letters |= UINT32_C(1) << (name[0] - 'A');
    }
    for (int i = 0; env && i < env->enum_count; ++i) {
        const char *name = env->enums[i].name;
        if (!env->enums[i].is_extern && name && name[0] >= 'A' && name[0] <= 'Z' && name[1] == '\0')
            native_declared_letters |= UINT32_C(1) << (name[0] - 'A');
    }
    char *result = transpile_to_c_impl(program, env, input_file);
    native_declared_letters = previous;
    native_opaque_names = previous_names;
    native_derived_graph = previous_graph; g_tuple_registry = previous_tuples;
    native_opaque_names_free(&names);
    return result;
}
