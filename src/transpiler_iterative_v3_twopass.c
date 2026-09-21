/*
 * ITERATIVE TRANSPILER - Two-Pass Architecture
 * 
 * Pass 1: Traverse AST and build ordered list of work items
 * Pass 2: Process work items and generate output
 * 
 * This cleanly separates "what to output" from "outputting it"
 * No LIFO/FIFO issues - items are processed in the order they're created.
 */

#include "nanolang.h"
#include "module_builder.h"
#include "stdlib_runtime.h"
#include "utf8.h"
#include "checked_loop_binding.h"
#include <stdarg.h>
#include <string.h>

/* Forward declarations for types defined in transpiler.c (only when not included from transpiler.c) */
#ifndef TRANSPILER_INTERNAL_TYPES_DEFINED
#define TRANSPILER_INTERNAL_TYPES_DEFINED
typedef struct {
    char **typedef_names;
    void **signatures;
    int count;
    int capacity;
} FunctionTypeRegistry;

typedef struct {
    void **tuples;
    char **typedef_names;
    int count;
    int capacity;
} TupleTypeRegistry;
#endif /* TRANSPILER_INTERNAL_TYPES_DEFINED */

/* ============================================================================
 * TRANSPILER CONTEXT - Track current function for type resolution
 * (Declared in transpiler.c, visible here via include)
 * ============================================================================ */

extern ASTNode *g_current_function;  /* Current function being transpiled */
extern const char *g_source_file_for_line_directives; /* Source file for #line directives */
extern bool g_profile_mode;          /* --profile: inject timing guard in next function body */
extern const char *g_profile_func_name; /* function name for profiling guard */
extern bool g_trace_mode;            /* --trace: inject trace guard in next function body */
extern const char *g_trace_func_name; /* function name for tracing guard */

/* =========================================================================
 * GENERIC TYPE NAME HELPERS
 * ========================================================================= */

static bool build_monomorphized_name_from_typeinfo_iter(char *dest, size_t dest_size, TypeInfo *info) {
    if (!dest || !dest_size || !info || !info->generic_name || info->type_param_count <= 0) return false;
    char *name = typeinfo_to_generic_arg_name(info);
    if (!name) return false;
    int written = snprintf(dest, dest_size, "%s", native_opaque_projection(name));
    free(name);
    return written >= 0 && (size_t)written < dest_size;
}

static const char *hashmap_suffix_from_typeinfo(TypeInfo *hm_info, char *buf, size_t buf_size) {
    if (!hm_info || !buf || buf_size == 0) return NULL;
    if (!hm_info->generic_name || strcmp(hm_info->generic_name, "HashMap") != 0) return NULL;
    if (hm_info->type_param_count != 2) return NULL;

    if (!build_monomorphized_name_from_typeinfo_iter(buf, buf_size, hm_info)) return NULL;
    if (strncmp(buf, "HashMap_", 8) == 0) return buf + 8;
    return buf;
}

static const char *hashmap_suffix_from_expr(ASTNode *hm_expr, Environment *env, char *buf, size_t buf_size) {
    if (!hm_expr || !env) return NULL;
    if (hm_expr->type == AST_FIELD_ACCESS)
        return hashmap_suffix_from_typeinfo(hm_expr->as.field_access.resolved_type_info, buf, buf_size);
    if (hm_expr->type == AST_CALL) {
        if (hm_expr->as.call.checked_signature)
            return hashmap_suffix_from_typeinfo(hm_expr->as.call.checked_signature->return_type_info, buf, buf_size);
        if (!hm_expr->as.call.func_expr && hm_expr->as.call.name) {
            Symbol *symbol = env_get_var_visible_at(env, hm_expr->as.call.name, hm_expr->line, hm_expr->column);
            if (symbol && symbol->type == TYPE_FUNCTION && symbol->type_info && symbol->type_info->fn_sig)
                return hashmap_suffix_from_typeinfo(symbol->type_info->fn_sig->return_type_info, buf, buf_size);
            Function *function = env_get_function(env, hm_expr->as.call.name);
            if (function) return hashmap_suffix_from_typeinfo(function->return_type_info, buf, buf_size);
        }
    }
    if (hm_expr->type == AST_IDENTIFIER) {
        Symbol *sym = env_get_var_visible_at(env, hm_expr->as.identifier, hm_expr->line, hm_expr->column);
        if (sym && sym->type_info) {
            return hashmap_suffix_from_typeinfo(sym->type_info, buf, buf_size);
        }
    }
    return NULL;
}

__attribute__((unused))
static const char *match_union_c_name(ASTNode *match, Environment *env, char *buf, size_t buf_size) {
    if (!match || !env) return NULL;
    const char *base = match->as.match_expr.union_type_name;
    if (!base) return NULL;

    ASTNode *scrutinee = match->as.match_expr.expr;
    if (scrutinee && scrutinee->type == AST_IDENTIFIER) {
        Symbol *sym = env_get_var_visible_at(env, scrutinee->as.identifier, match->line, match->column);
        if (sym && sym->type == TYPE_UNION && sym->type_info && sym->type_info->generic_name &&
            sym->type_info->type_param_count > 0 && strcmp(sym->type_info->generic_name, base) == 0) {
            if (build_monomorphized_name_from_typeinfo_iter(buf, buf_size, sym->type_info)) {
                return buf;
            }
        }
    }

    return base;
}

static bool match_uses_checked_int_domain(ASTNode *match) {
    if (!match || !match->as.match_expr.scrutinee_type_checked) {
        fprintf(stderr,
                "I require a checked match scrutinee domain before native lowering at line %d.\n",
                match ? match->line : 0);
        exit(1);
    }

    if (match->as.match_expr.checked_scrutinee_type == TYPE_INT) return true;
    if (match->as.match_expr.checked_scrutinee_type == TYPE_UNION) return false;

    fprintf(stderr,
            "I require an int or known union match scrutinee before native lowering at line %d.\n",
            match->line);
    exit(1);
}

static const char *checked_match_union_name(ASTNode *match) {
    const char *name = match ? match->as.match_expr.union_type_name : NULL;
    if (name && name[0] != '\0') return native_opaque_projection(name);

    fprintf(stderr,
            "I lost the checked union identity before native lowering at line %d.\n",
            match ? match->line : 0);
    exit(1);
}

/* ============================================================================
 * WORK ITEM TYPES - Describe what output to generate
 * ============================================================================ */

typedef enum {
    WORK_LITERAL,      /* Output a literal string */
    WORK_FORMATTED,    /* Output a formatted string */
    WORK_INDENT,       /* Output indentation */
    WORK_GC_RELEASE,   /* Generate gc_release() call for a variable */
} WorkItemType;

typedef struct {
    WorkItemType type;
    union {
        char *literal;           /* For WORK_LITERAL */
        char *formatted;         /* For WORK_FORMATTED (pre-formatted) */
        int indent_level;        /* For WORK_INDENT */
        char *var_name;          /* For WORK_GC_RELEASE */
    } data;
} WorkItem;

/* ============================================================================
 * WORK LIST - Ordered list of work items (NOT a stack!)
 * ============================================================================ */

typedef struct UnderscoreAlias {
    struct UnderscoreAlias *next;
    char name[64];
} UnderscoreAlias;

typedef struct {
    WorkItem *items;
    int capacity;
    int count;
    FunctionTypeRegistry *fn_registry;
    UnderscoreAlias *underscore_aliases;
    const char *underscore_name;
} WorkList;

static WorkList *worklist_create(int initial_capacity) {
    WorkList *list = malloc(sizeof(WorkList));
    if (!list) {
        fprintf(stderr, "Error: Out of memory allocating WorkList\n");
        exit(1);
    }
    list->capacity = initial_capacity;
    list->count = 0;
    list->fn_registry = NULL;
    list->underscore_aliases = NULL;
    list->underscore_name = NULL;
    list->items = malloc(sizeof(WorkItem) * initial_capacity);
    if (!list->items) {
        fprintf(stderr, "Error: Out of memory allocating WorkList items\n");
        free(list);
        exit(1);
    }
    return list;
}

static void worklist_free(WorkList *list) {
    if (!list) return;
    for (int i = 0; i < list->count; i++) {
        if (list->items[i].type == WORK_LITERAL && list->items[i].data.literal) {
            free(list->items[i].data.literal);
        }
        if (list->items[i].type == WORK_FORMATTED && list->items[i].data.formatted) {
            free(list->items[i].data.formatted);
        }
        if (list->items[i].type == WORK_GC_RELEASE && list->items[i].data.var_name) {
            free(list->items[i].data.var_name);
        }
    }
    free(list->items);
    while (list->underscore_aliases) {
        UnderscoreAlias *next = list->underscore_aliases->next;
        free(list->underscore_aliases);
        list->underscore_aliases = next;
    }
    free(list);
}

/* I change native storage names, never source names or diagnostic locations. */
static const char *native_local_name(WorkList *list, const char *name) {
    return name && !strcmp(name, "_") && list->underscore_name
        ? list->underscore_name : name;
}

static const char *new_underscore_name(WorkList *list, Environment *env) {
    static unsigned serial;
    UnderscoreAlias *alias = malloc(sizeof *alias);
    if (!alias) { fprintf(stderr, "I cannot allocate a native binding name.\n"); exit(1); }
    do {
        snprintf(alias->name, sizeof alias->name, "__nl_underscore_%u", serial++);
    } while (env_get_var(env, alias->name));
    alias->next = list->underscore_aliases;
    list->underscore_aliases = alias;
    return alias->name;
}

static void worklist_grow(WorkList *list) {
    if ((size_t)list->capacity > SIZE_MAX / 2 / sizeof(WorkItem)) {
        fprintf(stderr, "Error: WorkList capacity overflow\n");
        exit(1);
    }
    int new_capacity = list->capacity * 2;
    WorkItem *new_items = realloc(list->items, sizeof(WorkItem) * new_capacity);
    if (!new_items) {
        fprintf(stderr, "Error: Out of memory in WorkList\n");
        exit(1);
    }
    list->items = new_items;
    list->capacity = new_capacity;
}

static void worklist_append(WorkList *list, WorkItem item) {
    if (list->count >= list->capacity) {
        worklist_grow(list);
    }
    list->items[list->count++] = item;
}

/* ============================================================================
 * FORWARD DECLARATIONS
 * ============================================================================ */
static void emit_indent_item(WorkList *list, int level);
static void emit_literal(WorkList *list, const char *str);
static void emit_formatted(WorkList *list, const char *fmt, ...);
static void build_expr(WorkList *list, ASTNode *expr, Environment *env);

/* ============================================================================
 * SCOPE TRACKING - Track GC-managed variables for automatic cleanup
 * ============================================================================ */

typedef struct {
    char *name;
    Type type;
    bool needs_gc_release;  /* True for strings, opaques, hashmaps */
} ScopeVar;

typedef struct {
    ScopeVar *vars;
    const char *underscore_name;
    int capacity;
    int count;
} Scope;

typedef struct {
    Scope *scopes;
    int capacity;
    int count;
} ScopeStack;

static ScopeStack *scope_stack_create() {
    ScopeStack *stack = malloc(sizeof(ScopeStack));
    if (!stack) {
        fprintf(stderr, "Error: Out of memory allocating ScopeStack\n");
        exit(1);
    }
    stack->capacity = 32;
    stack->count = 0;
    stack->scopes = malloc(sizeof(Scope) * stack->capacity);
    if (!stack->scopes) {
        fprintf(stderr, "Error: Out of memory allocating ScopeStack scopes\n");
        free(stack);
        exit(1);
    }
    return stack;
}

static void scope_stack_free(ScopeStack *stack) {
    if (!stack) return;
    for (int i = 0; i < stack->count; i++) {
        for (int j = 0; j < stack->scopes[i].count; j++) {
            free(stack->scopes[i].vars[j].name);
        }
        free(stack->scopes[i].vars);
    }
    free(stack->scopes);
    free(stack);
}

static void scope_stack_push(ScopeStack *stack, WorkList *list) {
    if (stack->count >= stack->capacity) {
        stack->capacity *= 2;
        stack->scopes = realloc(stack->scopes, sizeof(Scope) * stack->capacity);
        if (!stack->scopes) {
            fprintf(stderr, "Error: Out of memory growing ScopeStack\n");
            exit(1);
        }
    }
    Scope *scope = &stack->scopes[stack->count++];
    scope->underscore_name = list->underscore_name;
    scope->capacity = 16;
    scope->count = 0;
    scope->vars = malloc(sizeof(ScopeVar) * scope->capacity);
    if (!scope->vars) {
        fprintf(stderr, "Error: Out of memory allocating Scope vars\n");
        exit(1);
    }
}

static void scope_add_var(ScopeStack *stack, const char *name, Type type, const char *type_name, Environment *env) {
    if (stack->count == 0) return;  /* No scope active */

    /* Check if this type needs gc_release
     * NOTE: We do NOT release TYPE_STRING here because:
     * 1. String literals (const char*) should never be released
     * 2. GC-allocated strings are managed by the GC's ref counting
     * 3. Releasing strings at scope end would break returns and cause double-frees
     *
     * For opaque types: gc_release() safely handles both:
     * - Wrapped (owned) refs: Released normally
     * - Borrowed refs: gc_release() checks gc_is_managed() and returns early
     */
    bool needs_cleanup = false;
    if (type == TYPE_OPAQUE || type == TYPE_HASHMAP) {
        needs_cleanup = true;
    }
    /* Opaque types are represented as TYPE_STRUCT - check if this struct is opaque */
    else if (type == TYPE_STRUCT && type_name && env) {
        OpaqueTypeDef *opaque = env_get_opaque_type(env, type_name);
        if (opaque) {
            needs_cleanup = true;
        }
    }

    if (!needs_cleanup) return;  /* Don't track variables that don't need cleanup */

    Scope *scope = &stack->scopes[stack->count - 1];
    if (scope->count >= scope->capacity) {
        scope->capacity *= 2;
        scope->vars = realloc(scope->vars, sizeof(ScopeVar) * scope->capacity);
        if (!scope->vars) {
            fprintf(stderr, "Error: Out of memory growing Scope vars\n");
            exit(1);
        }
    }

    ScopeVar *var = &scope->vars[scope->count++];
    var->name = strdup(name);
    if (!var->name) {
        fprintf(stderr, "Error: Out of memory duplicating variable name\n");
        exit(1);
    }
    var->type = type;
    var->needs_gc_release = true;
}

static bool native_effect_program;

static void scope_emit_cleanup(ScopeStack *stack, WorkList *list, int indent) {
    if (stack->count == 0) return;

    Scope *scope = &stack->scopes[stack->count - 1];
    /* Emit cleanup in reverse order (LIFO) */
    for (int i = scope->count - 1; i >= 0; i--) {
        if (scope->vars[i].needs_gc_release) {
            emit_indent_item(list, indent);
            if (!native_effect_program) emit_formatted(list, "gc_release(%s);\n", scope->vars[i].name);
        } else if (native_effect_program) {
            emit_formatted(list, "_nl_gc_%s.owned = false;\n", scope->vars[i].name);
        }
    }
}

static void scope_stack_pop(ScopeStack *stack, WorkList *list) {
    if (stack->count == 0) return;

    Scope *scope = &stack->scopes[--stack->count];
    list->underscore_name = scope->underscore_name;
    for (int i = 0; i < scope->count; i++) {
        free(scope->vars[i].name);
    }
    free(scope->vars);
}

static void build_stmt(WorkList *list, ScopeStack *scopes, ASTNode *stmt, int indent,
                       Environment *env, FunctionTypeRegistry *fn_registry);

static void build_loop_body(WorkList *list, ScopeStack *scopes, ASTNode *body,
                            ASTNode *loop, int indent, Environment *env,
                            FunctionTypeRegistry *registry) {
    const char *outer = list->underscore_name;
    if (!reestablish_checked_loop_binding(env, loop)) {
        fprintf(stderr, "I require checked loop binding metadata at line %d.\n", loop->line);
        exit(1);
    }
    if (!strcmp(loop->as.for_stmt.var_name, "_")) list->underscore_name = NULL;
    build_stmt(list, scopes, body, indent, env, registry);
    list->underscore_name = outer;
}

static void build_scoped_statements(WorkList *list, ScopeStack *scopes, ASTNode *body,
                                    int indent, Environment *env,
                                    FunctionTypeRegistry *fn_registry) {
    scope_stack_push(scopes, list);
    if (body && body->type == AST_BLOCK) {
        for (int i = 0; i < body->as.block.count; i++)
            build_stmt(list, scopes, body->as.block.statements[i], indent, env, fn_registry);
    } else if (body) {
        emit_indent_item(list, indent);
        build_expr(list, body, env);
        emit_literal(list, ";\n");
    }
    scope_emit_cleanup(scopes, list, indent);
    scope_stack_pop(scopes, list);
}

/* ============================================================================
 * WORK ITEM BUILDERS - Append work items to list
 * ============================================================================ */

static void emit_literal(WorkList *list, const char *str) {
    WorkItem item;
    item.type = WORK_LITERAL;
    item.data.literal = strdup(str);
    if (!item.data.literal) {
        fprintf(stderr, "Error: Out of memory duplicating literal string\n");
        exit(1);
    }
    worklist_append(list, item);
}

static void emit_formatted(WorkList *list, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list measured;
    va_copy(measured, args);
    int length = vsnprintf(NULL, 0, fmt, measured);
    va_end(measured);
    char *buffer = length >= 0 ? malloc((size_t)length + 1) : NULL;
    if (!buffer) {
        va_end(args);
        fprintf(stderr, "I could not allocate formatted native output.\n");
        exit(1);
    }
    int written = vsnprintf(buffer, (size_t)length + 1, fmt, args);
    va_end(args);
    if (written != length) {
        free(buffer);
        fprintf(stderr, "I could not format complete native output.\n");
        exit(1);
    }
    
    WorkItem item;
    item.type = WORK_FORMATTED;
    item.data.formatted = buffer;
    worklist_append(list, item);
}

static bool foreign_function_has_array(const Function *fn) {
    if (!fn || !fn->is_extern) return false;
    if (fn->return_type == TYPE_ARRAY) return true;
    for (int i = 0; i < fn->param_count; ++i)
        if (fn->params[i].type == TYPE_ARRAY) return true;
    return false;
}

static void emit_foreign_reference(WorkList *list, const char *name, const Function *fn) {
    if (foreign_function_has_array(fn)) {
        emit_formatted(list,
            "(nano_require_native_array_abi((void*)%s, \"%s__nano_array_abi\", NANO_DYN_ARRAY_ABI_VERSION, \"%s\"), %s)",
            name, name, name, name);
    } else {
        emit_literal(list, name);
    }
}

static void emit_indent_item(WorkList *list, int level) {
    WorkItem item;
    item.type = WORK_INDENT;
    item.data.indent_level = level;
    worklist_append(list, item);
}

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/* Function name mapping now uses the unified builtin registry */
#include "builtins_registry.h"

/* My scalar lists are registered by register_builtin_functions, separately
 * from the registry cache. Source definitions have bodies; source externs have
 * is_extern. Neither shares this complete registration shape. */
static bool native_scalar_list_builtin(const char *name, Environment *env) {
    if (!name || (strncmp(name, "list_int_", 9) && strncmp(name, "list_string_", 12))) return false;
    Function *function = env_get_function(env, name);
    return function && !function->is_extern && !function->body &&
        !function->params && !function->shadow_test && !function->module_name && !function->alias_of;
}

static const char *map_function_name(const char *name, Environment *env) {
    const char *helper_name = module_helper_c_name(name);
    if (helper_name != name) return helper_name;
    /* Handle qualified names: module::func or nested::module::func */
    const char *double_colon = strstr(name, "::");
    if (double_colon) {
        /* Qualified name - convert to mangled form: module::func -> module__func */
        static _Thread_local char mangled[512];
        size_t i = 0, j = 0;
        while (name[i] && j < sizeof(mangled) - 1) {
            if (name[i] == ':' && name[i+1] == ':') {
                mangled[j++] = '_';
                mangled[j++] = '_';
                i += 2;
            } else {
                mangled[j++] = name[i++];
            }
        }
        mangled[j] = '\0';
        return mangled;
    }
    
    /* Handle legacy module-qualified names with dot notation */
    const char *dot = strchr(name, '.');
    if (dot) {
        /* Try module-qualified lookup first to preserve module scoping */
        Function *qualified_func = env_get_function(env, name);
        if (qualified_func) {
            const char *resolved_name = qualified_func->alias_of ? qualified_func->alias_of : qualified_func->name;
            extern const char *get_c_func_name_with_module(const char *nano_name, const char *module_name, bool is_extern);
            return get_c_func_name_with_module(resolved_name, qualified_func->module_name, qualified_func->is_extern);
        }
        /* Fallback: treat as unqualified for builtins/mappings */
        name = dot + 1;
    }
    
    /* I retain the selected declaration instead of its registry spelling. */
    if (env_native_array_operation(name)) {
        Function *selected = env_get_function(env, name);
        if (selected && (selected->body || selected->is_extern || selected->source_file || selected->alias_of)) {
            extern const char *get_c_func_name_with_module(const char *, const char *, bool);
            return get_c_func_name_with_module(selected->alias_of ? selected->alias_of : selected->name,
                                               selected->module_name, selected->is_extern);
        }
    }

    /* I preserve an actual list declaration before consulting builtin spelling. */
    if (!strncmp(name, "list_", 5) || !strncmp(name, "List_", 5)) {
        if (native_scalar_list_builtin(name, env)) return name;
        Function *selected = env_get_function(env, name);
        if (selected && !env_function_is_builtin(selected) &&
            !env_generated_list_element(env, selected).ordinal) {
            return get_c_func_name_with_module(selected->alias_of ? selected->alias_of : selected->name,
                                               selected->module_name, selected->is_extern);
        }
    }

    /* Check unified builtin registry */
    const char *c_name = builtin_c_name(name);
    if (c_name) {
        return c_name;
    }
    
    /* User-defined function? Look up to get module context */
    Function *func = env_get_function(env, name);
    
    if (func) {
        /* Use get_c_func_name_with_module for namespace mangling */
        const char *resolved_name = func->alias_of ? func->alias_of : func->name;
        extern const char *get_c_func_name_with_module(const char *nano_name, const char *module_name, bool is_extern);
        return get_c_func_name_with_module(resolved_name, func->module_name, func->is_extern);
    }
    
    return name;
}

static const TypeInfo *array_expr_type_info(ASTNode *expr, Environment *env) {
    if (!expr) return NULL;
    const TypeInfo *checked = checked_expression_type_info(expr, env);
    if (checked && checked->base_type == TYPE_ARRAY) return checked;
    if (expr->type == AST_CALL && expr->as.call.name &&
        strcmp(expr->as.call.name, "array_push") == 0 &&
        !env_array_push_is_builtin(env, expr->line, expr->column)) {
        check_expression(expr, env);
        if (expr->as.call.checked_signature)
            return expr->as.call.checked_signature->return_type_info;
        Function *selected = env_get_function(env, expr->as.call.name);
        return selected ? selected->return_type_info : NULL;
    }
    if (expr->type == AST_FIELD_ACCESS) {
        check_expression(expr, env);
        if (expr->as.field_access.resolved_type_info) return expr->as.field_access.resolved_type_info;
    }
    if (expr->type == AST_IDENTIFIER) {
        Symbol *sym = env_get_var_visible_at(env, expr->as.identifier, expr->line, expr->column);
        return sym && sym->type == TYPE_ARRAY ? sym->type_info : NULL;
    }
    if (expr->type == AST_CALL && expr->as.call.name &&
        strcmp(expr->as.call.name, "at") == 0 &&
        expr->as.call.arg_count == 2) {
        const TypeInfo *array = array_expr_type_info(expr->as.call.args[0], env);
        return array && array->base_type == TYPE_ARRAY ? array->element_type : NULL;
    }
    return NULL;
}

static Type resolved_array_element(Type element, const char *name, Environment *env) {
    return element == TYPE_STRUCT && name && env_get_enum(env, name) ? TYPE_ENUM : element;
}

static Type infer_array_element_type(ASTNode *array_expr, Environment *env) {
    if (!array_expr) return TYPE_UNKNOWN;
    if (array_expr->type == AST_CALL && !array_expr->as.call.func_expr &&
        array_expr->as.call.name && !strcmp(array_expr->as.call.name, "array_push") &&
        array_expr->as.call.arg_count == 2 &&
        env_array_push_is_builtin(env, array_expr->line, array_expr->column))
        return infer_array_element_type(array_expr->as.call.args[0], env);

    const TypeInfo *info = array_expr_type_info(array_expr, env);
    if (info && info->base_type == TYPE_ARRAY && info->element_type) {
        return resolved_array_element(info->element_type->base_type, info->element_type->generic_name, env);
    }

    if (array_expr->type == AST_MODULE_QUALIFIED_CALL) {
        const char *alias = array_expr->as.module_qualified_call.module_alias;
        const char *name = array_expr->as.module_qualified_call.function_name;
        size_t size = strlen(alias) + strlen(name) + 2;
        char *qualified = malloc(size);
        if (!qualified) return TYPE_UNKNOWN;
        snprintf(qualified, size, "%s.%s", alias, name);
        Function *producer = env_get_function(env, qualified);
        free(qualified);
        if (producer && producer->return_type == TYPE_ARRAY)
            return resolved_array_element(producer->return_element_type, producer->return_struct_type_name, env);
    }

    if (array_expr->type == AST_CALL && array_expr->as.call.name &&
        !array_expr->as.call.func_expr) {
        if ((strcmp(array_expr->as.call.name, "at") == 0 ||
             strcmp(array_expr->as.call.name, "array_get") == 0) &&
            array_expr->as.call.arg_count == 2) {
            ASTNode *container = array_expr->as.call.args[0];
            if (container && container->type == AST_ARRAY_LITERAL &&
                container->as.array_literal.element_count > 0) {
                ASTNode *first = container->as.array_literal.elements[0];
                if (first && check_expression(first, env) == TYPE_ARRAY) {
                    return infer_array_element_type(first, env);
                }
            }
        }
        Function *producer = env_get_function(env, array_expr->as.call.name);
        if (producer && producer->return_type == TYPE_ARRAY) {
            return resolved_array_element(producer->return_element_type, producer->return_struct_type_name, env);
        }
    }

    if (array_expr->type == AST_IDENTIFIER) {
        Symbol *sym = env_get_var_visible_at(env, array_expr->as.identifier, array_expr->line, array_expr->column);
        if (sym && sym->type == TYPE_ARRAY && sym->element_type != TYPE_UNKNOWN) {
            return resolved_array_element(sym->element_type, sym->struct_type_name, env);
        }
    }

    if (array_expr->type == AST_ARRAY_LITERAL) {
        if (array_expr->as.array_literal.element_type != TYPE_UNKNOWN) {
            return array_expr->as.array_literal.element_type;
        }
        if (array_expr->as.array_literal.element_count > 0) {
            return check_expression(array_expr->as.array_literal.elements[0], env);
        }
    }

    if (array_expr->type == AST_FIELD_ACCESS) {
        const char *struct_name = get_struct_type_name(array_expr->as.field_access.object, env);
        if (struct_name) {
            StructDef *sdef = env_get_struct(env, struct_name);
            if (sdef && sdef->field_element_types) {
                const char *field_name = array_expr->as.field_access.field_name;
                for (int i = 0; i < sdef->field_count; i++) {
                    if (strcmp(sdef->field_names[i], field_name) == 0) {
                        if (sdef->field_types[i] == TYPE_ARRAY && sdef->field_element_types[i] != TYPE_UNKNOWN) {
                            return resolved_array_element(sdef->field_element_types[i], sdef->field_type_names ? sdef->field_type_names[i] : NULL, env);
                        }
                        break;
                    }
                }
            }
        }
    }

    return TYPE_UNKNOWN;
}

/* ============================================================================
 * HELPER: Serialize expression AST to human-readable string (for error messages)
 * I build owned diagnostic text without fixed-size source fragments.
 * ============================================================================ */

/* I escape source spelling for inclusion in a generated C diagnostic literal. */
static void append_diagnostic_string(StringBuilder *sb, const char *text) {
    sb_append(sb, "\\\"");
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        switch (*p) {
            case '\\': sb_append(sb, "\\\\"); break;
            case '"': sb_append(sb, "\\\""); break;
            case '\n': sb_append(sb, "\\n"); break;
            case '\r': sb_append(sb, "\\r"); break;
            case '\t': sb_append(sb, "\\t"); break;
            default:
                if (*p < 32 || *p == 127) sb_appendf(sb, "\\%03o", *p);
                else {
                    char byte[2] = {(char)*p, '\0'};
                    sb_append(sb, byte);
                }
                break;
        }
    }
    sb_append(sb, "\\\"");
}

static void expr_to_string_impl(ASTNode *expr, StringBuilder *sb) {
    if (!expr) return;
    
    switch (expr->type) {
        case AST_NUMBER:
            sb_appendf(sb, "%lld", expr->as.number);
            break;
        case AST_FLOAT:
            sb_appendf(sb, "%g", expr->as.float_val);
            break;
        case AST_STRING:
            append_diagnostic_string(sb, expr->as.string_val ? expr->as.string_val : "");
            break;
        case AST_BOOL:
            sb_append(sb, expr->as.bool_val ? "true" : "false");
            break;
        case AST_IDENTIFIER:
            sb_append(sb, expr->as.identifier ? expr->as.identifier : "?");
            break;
        case AST_PREFIX_OP: {
            sb_append(sb, "(");
            /* Map token type to operator string */
            const char *op = "?";
            switch (expr->as.prefix_op.op) {
                case TOKEN_PLUS: op = "+"; break;
                case TOKEN_MINUS: op = "-"; break;
                case TOKEN_STAR: op = "*"; break;
                case TOKEN_SLASH: op = "/"; break;
                case TOKEN_PERCENT: op = "%"; break;
                case TOKEN_EQ: op = "=="; break;
                case TOKEN_NE: op = "!="; break;
                case TOKEN_LT: op = "<"; break;
                case TOKEN_LE: op = "<="; break;
                case TOKEN_GT: op = ">"; break;
                case TOKEN_GE: op = ">="; break;
                case TOKEN_AND: op = "and"; break;
                case TOKEN_OR: op = "or"; break;
                case TOKEN_NOT: op = "not"; break;
                default: break;
            }
            sb_append(sb, op);
            for (int i = 0; i < expr->as.prefix_op.arg_count; i++) {
                sb_append(sb, " ");
                expr_to_string_impl(expr->as.prefix_op.args[i], sb);
            }
            sb_append(sb, ")");
            break;
        }
        case AST_CALL:
            sb_append(sb, "(");
            sb_append(sb, expr->as.call.name ? expr->as.call.name : "?");
            for (int i = 0; i < expr->as.call.arg_count; i++) {
                sb_append(sb, " ");
                expr_to_string_impl(expr->as.call.args[i], sb);
            }
            sb_append(sb, ")");
            break;
        default:
            sb_append(sb, "...");
            break;
    }
}

/* I return owned storage; each emission site releases it after copying. */
static char *expr_to_string(ASTNode *expr) {
    StringBuilder *sb = sb_create();
    expr_to_string_impl(expr, sb);
    char *text = sb->buffer;
    free(sb);
    return text;
}

/* ============================================================================
 * HELPER: Try to evaluate condition at compile time (for contract elision)
 * Returns: 1 = always true, 0 = always false, -1 = cannot determine
 * ============================================================================ */

static int try_eval_bool_const(ASTNode *expr) {
    if (!expr) return -1;
    
    switch (expr->type) {
        case AST_BOOL:
            return expr->as.bool_val ? 1 : 0;
            
        case AST_NUMBER:
            /* Numbers are truthy if non-zero */
            return expr->as.number != 0 ? 1 : 0;
            
        case AST_PREFIX_OP: {
            /* Handle simple comparison operators with constant operands */
            if (expr->as.prefix_op.arg_count == 2) {
                ASTNode *left = expr->as.prefix_op.args[0];
                ASTNode *right = expr->as.prefix_op.args[1];
                
                /* Both must be constant numbers */
                if (left->type != AST_NUMBER || right->type != AST_NUMBER)
                    return -1;
                    
                long long l = left->as.number;
                long long r = right->as.number;
                
                switch (expr->as.prefix_op.op) {
                    case TOKEN_EQ: return (l == r) ? 1 : 0;
                    case TOKEN_NE: return (l != r) ? 1 : 0;
                    case TOKEN_LT: return (l < r) ? 1 : 0;
                    case TOKEN_LE: return (l <= r) ? 1 : 0;
                    case TOKEN_GT: return (l > r) ? 1 : 0;
                    case TOKEN_GE: return (l >= r) ? 1 : 0;
                    default: break;
                }
            }
            
            /* Handle 'not' with constant operand */
            if (expr->as.prefix_op.op == TOKEN_NOT && expr->as.prefix_op.arg_count == 1) {
                int inner = try_eval_bool_const(expr->as.prefix_op.args[0]);
                if (inner >= 0) return inner ? 0 : 1;
            }
            
            /* Handle 'and' / 'or' with constant operands */
            if (expr->as.prefix_op.arg_count == 2) {
                int left_val = try_eval_bool_const(expr->as.prefix_op.args[0]);
                int right_val = try_eval_bool_const(expr->as.prefix_op.args[1]);
                
                if (expr->as.prefix_op.op == TOKEN_AND) {
                    if (left_val == 0 || right_val == 0) return 0;  /* false and x = false */
                    if (left_val == 1 && right_val == 1) return 1;  /* true and true = true */
                }
                if (expr->as.prefix_op.op == TOKEN_OR) {
                    if (left_val == 1 || right_val == 1) return 1;  /* true or x = true */
                    if (left_val == 0 && right_val == 0) return 0;  /* false or false = false */
                }
            }
            break;
        }
        
        default:
            break;
    }
    
    return -1;  /* Cannot determine */
}

/* ============================================================================
 * PASS 1: BUILD WORK ITEMS (Expression Transpiler)
 * Traverses AST and appends work items in correct output order
 * ============================================================================ */

/* Forward declarations */
static void build_expr(WorkList *list, ASTNode *expr, Environment *env);

/* I bind arguments in source order before entering an ordinary C call. */
static unsigned next_ordered_call_id(Environment *env, int arg_count) {
    static _Thread_local unsigned next_call_id;
    unsigned call_id;
    bool available;
    do {
        call_id = next_call_id++;
        char callee_name[64];
        snprintf(callee_name, sizeof(callee_name), "__nl_callee_%u", call_id);
        available = !env || !env_get_var(env, callee_name);
        for (int i = 0; i < arg_count; ++i) {
            char name[64];
            snprintf(name, sizeof(name), "__nl_arg_%u_%d", call_id, i);
            if (env && env_get_var(env, name)) available = false;
        }
    } while (!available);
    return call_id;
}
static unsigned build_ordered_call_args(WorkList *list, ASTNode **args,
                                        int arg_count, Environment *env,
                                        const char *callee_value) {
    unsigned call_id = next_ordered_call_id(env, arg_count);
    if (callee_value) {
        emit_formatted(list, "__auto_type __nl_callee_%u = %s; ", call_id, callee_value);
    }
    for (int i = 0; i < arg_count; ++i) {
        emit_formatted(list, "__auto_type __nl_arg_%u_%d = ", call_id, i);
        build_expr(list, args[i], env);
        emit_literal(list, "; ");
    }
    return call_id;
}

/* I preserve the source-level zero spelling for an opaque null after call
 * arguments have been snapshotted into typed C temporaries.  An integer
 * temporary is not a C null-pointer constant, so the final call boundary must
 * restore the declared pointer type explicitly. */
static bool call_parameter_is_opaque(const Function *function, int index,
                                     Environment *env) {
    if (!function || !function->params || index < 0 ||
        index >= function->param_count) return false;
    const Parameter *parameter = &function->params[index];
    if (parameter->type == TYPE_OPAQUE) return true;
    return parameter->type == TYPE_STRUCT && parameter->struct_type_name && env &&
           env_get_opaque_type(env, parameter->struct_type_name) != NULL;
}

static unsigned next_nested_array_id(Environment *env) {
    static _Thread_local unsigned next_id;
    unsigned id;
    bool available;
    do {
        id = next_id++;
        char name[64];
        snprintf(name, sizeof(name), "__nl_nested_array_%u", id);
        available = !env || !env_get_var(env, name);
    } while (!available);
    return id;
}

static void build_stmt(WorkList *list, ScopeStack *scopes, ASTNode *stmt, int indent, Environment *env,
                       FunctionTypeRegistry *fn_registry);

/* I restore a checked payload after emitted outer locals enter the environment. */
static void restore_native_match_binding(Environment *env, ASTNode *match, int arm,
                                         const char *owner) {
    const char *name = match->as.match_expr.pattern_bindings[arm];
    const char *variant = match->as.match_expr.pattern_variants[arm];
    if (!owner || !name || !*name || !strcmp(name, "_") ||
        !strcmp(variant, "_") || !strncmp(variant, "INT:", 4) ||
        !strncmp(variant, "OR:", 3)) return;
    size_t size = strlen(owner) + strlen(variant) + 2;
    char *nominal = malloc(size);
    if (!nominal) {
        fprintf(stderr, "I cannot allocate match binding metadata\n");
        exit(1);
    }
    snprintf(nominal, size, "%s.%s", owner, variant);
    Symbol checked;
    bool found = false;
    for (int i = env->symbol_count - 1; i >= 0; --i) {
        Symbol *symbol = &env->symbols[i];
        if (symbol->name && !strcmp(symbol->name, name) &&
            symbol->struct_type_name && !strcmp(symbol->struct_type_name, nominal) &&
            symbol->def_line == match->line && symbol->def_column == match->column &&
            (!symbol->def_file || !env->current_file ||
             !strcmp(symbol->def_file, env->current_file))) {
            checked = *symbol;
            found = true;
            break;
        }
    }
    if (!found) {
        free(nominal);
        fprintf(stderr, "I require the checked nominal match binding\n");
        exit(1);
    }
    env_define_var_with_type_info(env, name, checked.type, checked.element_type,
                                 checked.type_info, checked.is_mut, create_void());
    Symbol *binding = &env->symbols[env->symbol_count - 1];
    free(binding->struct_type_name);
    binding->struct_type_name = nominal;
    binding->nominal_owner = checked.nominal_owner;
    binding->callable_owner = checked.callable_owner;
    binding->inferred_nominal = checked.inferred_nominal;
    binding->checker_nominal_view = checked.checker_nominal_view;
    binding->def_line = checked.def_line;
    binding->def_column = checked.def_column;
    binding->def_file = checked.def_file;
    binding->is_resource = checked.is_resource;
    binding->scope_end_line = match->as.match_expr.arm_bodies[arm]->scope_end_line;
    binding->scope_end_column = match->as.match_expr.arm_bodies[arm]->scope_end_column;
}

static void build_match_arm_value(WorkList *list, ASTNode *body, Environment *env) {
    if (!body) return;
    if (body->type != AST_BLOCK) {
        emit_literal(list, "_out = ");
        build_expr(list, body, env);
        emit_literal(list, "; ");
        return;
    }
    ScopeStack *scopes = scope_stack_create();
    scope_stack_push(scopes, list);
    for (int i = 0; i < body->as.block.count; i++) {
        ASTNode *stmt = body->as.block.statements[i];
        if (i == body->as.block.count - 1 && ast_is_value_expression(stmt->type)) {
            emit_literal(list, "_out = ");
            build_expr(list, stmt, env);
            emit_literal(list, "; ");
            /* I transfer a directly yielded local's owned reference to _out. */
            if (stmt->type == AST_IDENTIFIER) {
                Scope *scope = &scopes->scopes[scopes->count - 1];
                for (int j = 0; j < scope->count; j++) {
                    if (strcmp(scope->vars[j].name, native_local_name(list, stmt->as.identifier)) == 0)
                        scope->vars[j].needs_gc_release = false;
                }
            }
        } else {
            build_stmt(list, scopes, stmt, 0, env, list->fn_registry);
        }
    }
    scope_emit_cleanup(scopes, list, 0);
    scope_stack_pop(scopes, list);
    scope_stack_free(scopes);
}

static bool is_generic_list_runtime_fn(const char *name, Environment *env) {
    if (!name || !env || (strncmp(name, "list_", 5) && strncmp(name, "List_", 5)))
        return false;
    Function *selected = env_get_function(env, name);
    if (selected && !env_generated_list_element(env, selected).ordinal) return false;
    static const char *operations[] = {"new", "push", "get", "set", "insert", "remove",
        "pop", "length", "capacity", "is_empty", "clear", "free"};
    for (int i = 0; i < env->generic_instance_count; ++i) {
        const char *element = native_list_element(env, &env->generic_instances[i]);
        if (!element) continue;
        size_t n = strlen(element);
        if (strncmp(name + 5, element, n) || name[5 + n] != '_') continue;
        for (size_t j = 0; j < sizeof(operations) / sizeof(operations[0]); ++j)
            if (!strcmp(name + 6 + n, operations[j])) return true;
    }
    return false;
}


/* I lift synchronous handlers into ordinary C functions. Captures point at
 * the installing lexical frame; handlers cannot escape that frame. */
static const char *effect_c_type(Type type, const char *name, Environment *env) {
    if (name && (type == TYPE_STRUCT || type == TYPE_UNION || type == TYPE_ENUM)) {
        if (env_get_opaque_type(env, name)) return "void*";
        return get_prefixed_type_name(name);
    }
    return type_to_c(type);
}
static StringBuilder *effect_helpers;
static int effect_serial;
static ASTNode *effect_guarded_foreign_call;
static bool in_effect_handler;
static Type effect_lexical_return;
static const char *effect_lexical_name;
static Environment *effect_environment;
static Symbol **effect_captures;
static int effect_capture_count;
static int *effect_capture_slots;
static const char *effect_capture_name(const char *name, int line, int column) {
    static char result[512];
    for (int i = 0; i < effect_capture_count; ++i) {
        if (strcmp(effect_captures[i]->name, name) == 0) {
            Symbol *visible = env_get_var_visible_at(effect_environment, name, line, column);
            if (visible && (visible->def_line != effect_captures[i]->def_line || visible->def_column != effect_captures[i]->def_column)) return name;
            snprintf(result, sizeof(result), "(*(%s *)_frame->captures[%d])",
                     effect_c_type(effect_captures[i]->type, effect_captures[i]->struct_type_name, effect_environment), effect_capture_slots[i]);
            return result;
        }
    }
    return name;
}
static Type effect_body_type(ASTNode *body, Environment *env) {
    if (!body) return TYPE_VOID;
    if (body->type == AST_BLOCK) {
        if (!body->as.block.count) return TYPE_VOID;
        body = body->as.block.statements[body->as.block.count - 1];
    }
    return ast_is_value_expression(body->type) ? check_expression(body, env) : TYPE_VOID;
}
static void build_effect_handle(WorkList *, ASTNode *, Environment *);

/* I evaluate a map receiver before its key/value expressions, exactly once.
 * Fresh names keep nested operations and source bindings independent. */
static void build_ordered_hashmap_call(WorkList *list, ASTNode *call, Environment *env,
                                       const char *suffix, const char *operation) {
    static unsigned next_id;
    unsigned id;
    bool available;
    do {
        id = next_id++;
        available = true;
        for (int i = 0; i < call->as.call.arg_count; ++i) {
            char name[64];
            snprintf(name, sizeof(name), "__nl_map_%u_arg_%d", id, i);
            if (env_get_var(env, name)) available = false;
        }
    } while (!available);
    emit_literal(list, "({ ");
    for (int i = 0; i < call->as.call.arg_count; ++i) {
        emit_formatted(list, "__auto_type __nl_map_%u_arg_%d = ", id, i);
        build_expr(list, call->as.call.args[i], env);
        emit_literal(list, "; ");
    }
    emit_formatted(list, "nl_hashmap_%s_%s(", suffix, operation);
    for (int i = 0; i < call->as.call.arg_count; ++i) {
        if (i) emit_literal(list, ", ");
        emit_formatted(list, "__nl_map_%u_arg_%d", id, i);
    }
    emit_literal(list, "); })");
}

#include "transpiler_opaque_arrays.inc"

static void build_expr(WorkList *list, ASTNode *expr, Environment *env) {
    if (!expr) return;

    if (native_effect_program && expr != effect_guarded_foreign_call &&
        (expr->type == AST_CALL || expr->type == AST_MODULE_QUALIFIED_CALL)) {
        char qualified[1024];
        const char *name = expr->type == AST_CALL ? expr->as.call.name : NULL;
        if (expr->type == AST_MODULE_QUALIFIED_CALL) {
            snprintf(qualified, sizeof(qualified), "%s.%s", expr->as.module_qualified_call.module_alias, expr->as.module_qualified_call.function_name);
            name = qualified;
        }
        Function *foreign = name ? env_get_function(env, name) : NULL;
        if (foreign && foreign->is_extern) {
            int count = expr->type == AST_CALL ? expr->as.call.arg_count : expr->as.module_qualified_call.arg_count;
            ASTNode **args = expr->type == AST_CALL ? expr->as.call.args : expr->as.module_qualified_call.args;
            ASTNode **temporaries = calloc((size_t)count + 1, sizeof(*temporaries));
            int serial = effect_serial++;
            emit_literal(list, "({ ");
            for (int i = 0; i < count; ++i) {
                emit_formatted(list, "__auto_type _foreign_%d_%d = ", serial, i);
                build_expr(list, args[i], env);
                emit_literal(list, "; ");
                temporaries[i] = calloc(1, sizeof(ASTNode));
                temporaries[i]->type = AST_IDENTIFIER;
                char temp_name[80]; snprintf(temp_name, sizeof(temp_name), "_foreign_%d_%d", serial, i);
                temporaries[i]->as.identifier = strdup(temp_name);
            }
            ASTNode guarded = *expr;
            if (expr->type == AST_CALL) guarded.as.call.args = temporaries;
            else guarded.as.module_qualified_call.args = temporaries;
            emit_literal(list, "++nl_effect_foreign_depth; ");
            if (foreign->return_type != TYPE_VOID) emit_literal(list, "__auto_type _foreign_result = ");
            ASTNode *saved = effect_guarded_foreign_call;
            effect_guarded_foreign_call = &guarded;
            build_expr(list, &guarded, env);
            effect_guarded_foreign_call = saved;
            emit_literal(list, "; --nl_effect_foreign_depth; ");
            emit_literal(list, foreign->return_type == TYPE_VOID ? "(void)0; })" : "_foreign_result; })");
            for (int i = 0; i < count; ++i) { free(temporaries[i]->as.identifier); free(temporaries[i]); }
            free(temporaries);
            return;
        }
    }
    switch (expr->type) {
        case AST_NUMBER:
            if (expr->as.number == INT64_MIN) {
                emit_literal(list, "INT64_MIN");
            } else {
                emit_formatted(list, "%lldLL", expr->as.number);
            }
            break;
            
        case AST_FLOAT:
            if (expr->as.float_val == (double)(int64_t)expr->as.float_val) {
                emit_formatted(list, "%.1f", expr->as.float_val);
            } else {
                emit_formatted(list, "%g", expr->as.float_val);
            }
            break;
            
        case AST_STRING:
            /* I lower f-strings in the lexer; ordinary braces remain literal. */
            emit_formatted(list, "\"%s\"", expr->as.string_val);
            break;
            
        case AST_BOOL:
            emit_literal(list, expr->as.bool_val ? "true" : "false");
            break;
            
        case AST_IDENTIFIER: {
            if (in_effect_handler) {
                const char *captured = effect_capture_name(expr->as.identifier, expr->line, expr->column);
                if (captured != expr->as.identifier) { emit_literal(list, captured); break; }
            }
            /* Check for constant inlining */
            Symbol *sym = env_get_var_visible_at(env, expr->as.identifier, expr->line, expr->column);
            /* I prefer the latest global value over its older declaration
             * placeholder, but never over a visible local shadow. */
            if (!sym || sym->is_global) {
                for (int i = env->symbol_count - 1; i >= 0; --i) {
                    Symbol *global = &env->symbols[i];
                    if (global->is_global && global->name &&
                        strcmp(global->name, expr->as.identifier) == 0) {
                        sym = global;
                        break;
                    }
                }
            }
            if (sym && (sym->type == TYPE_BORROW_SHARED || sym->type == TYPE_BORROW_MUT)) {
                emit_formatted(list, "(*%s)", native_local_name(list, expr->as.identifier));
                break;
            }
            if (sym && sym->type == TYPE_VOID) {
                emit_literal(list, "((void)0)");
                break;
            }
            if (sym && !sym->is_mut) {
                if (sym->value.type == VAL_INT) {
                    if (sym->value.as.int_val == INT64_MIN) {
                        emit_literal(list, "INT64_MIN");
                    } else {
                        emit_formatted(list, "%lldLL", (long long)sym->value.as.int_val);
                    }
                    return;
                } else if (sym->value.type == VAL_FLOAT) {
                    emit_formatted(list, "%g", sym->value.as.float_val);
                    return;
                } else if (sym->value.type == VAL_BOOL) {
                    emit_literal(list, sym->value.as.bool_val ? "true" : "false");
                    return;
                }
            }

            /* If a local/global variable with this name is in scope, emit it directly.
             * A local variable shadows any same-named function from an imported module.
             * Only fall through to the function-name check when no variable is in scope. */
            if (sym) {
                emit_literal(list, native_local_name(list, expr->as.identifier));
                break;
            }

            /* Check if it's a function identifier */
            Function *func_def = env_get_function(env, expr->as.identifier);
            if (func_def && !func_def->is_extern && func_def->body != NULL) {
                emit_literal(list, map_function_name(expr->as.identifier, env));
            } else if (func_def && func_def->is_extern) {
                emit_foreign_reference(list, map_function_name(expr->as.identifier, env), func_def);
            } else {
                emit_foreign_reference(list, expr->as.identifier, func_def);
            }
            break;
        }
        
        case AST_QUALIFIED_NAME: {
            /* Qualified name: std::io::read -> std__io__read */
            static _Thread_local char mangled[512];
            size_t j = 0;
            
            /* Join all parts with __ separator */
            for (int i = 0; i < expr->as.qualified_name.part_count; i++) {
                if (i > 0) {
                    mangled[j++] = '_';
                    mangled[j++] = '_';
                }
                const char *part = expr->as.qualified_name.name_parts[i];
                size_t k = 0;
                while (part[k] && j < sizeof(mangled) - 3) {
                    mangled[j++] = part[k++];
                }
            }
            mangled[j] = '\0';
            
            emit_literal(list, mangled);
            break;
        }
        
        case AST_PREFIX_OP: {
            TokenType op = expr->as.prefix_op.op;
            int arg_count = expr->as.prefix_op.arg_count;
            
            if (arg_count == 2) {
                /* I snapshot exact scalar operands in source order, once. */
                if ((op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR || op == TOKEN_SLASH) &&
                    check_expression(expr->as.prefix_op.args[0], env) == TYPE_FLOAT &&
                    check_expression(expr->as.prefix_op.args[1], env) == TYPE_FLOAT) {
                    char left[80], right[80];
                    unsigned suffix = 0;
                    do {
                        snprintf(left, sizeof left, "nano_rt_f64_left_%u", suffix);
                        snprintf(right, sizeof right, "nano_rt_f64_right_%u", suffix++);
                    } while (env_get_var(env, left) || env_get_var(env, right) ||
                             env_get_function(env, left) || env_get_function(env, right));
                    emit_formatted(list, "({ double %s = ", left);
                    build_expr(list, expr->as.prefix_op.args[0], env);
                    emit_formatted(list, "; double %s = ", right);
                    build_expr(list, expr->as.prefix_op.args[1], env);
                    emit_formatted(list, "; nano_rt_f64_%s(%s, %s); })",
                                   op == TOKEN_PLUS ? "add" : op == TOKEN_MINUS ? "sub" :
                                   op == TOKEN_STAR ? "mul" : "div", left, right);
                    break;
                }
                /* Binary operator */
                if (op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR || op == TOKEN_SLASH || op == TOKEN_PERCENT) {
                    Type t1 = check_expression(expr->as.prefix_op.args[0], env);
                    Type t2 = check_expression(expr->as.prefix_op.args[1], env);
                    if (t1 == TYPE_ARRAY || t2 == TYPE_ARRAY) {
                        bool left_is_array = (t1 == TYPE_ARRAY);
                        bool right_is_array = (t2 == TYPE_ARRAY);

                        ASTNode *left_expr = expr->as.prefix_op.args[0];
                        ASTNode *right_expr = expr->as.prefix_op.args[1];

                        const char *fn = NULL;
                        const char *fn_scalar = NULL;
                        const char *fn_rscalar = NULL;
                        switch (op) {
                            case TOKEN_PLUS:
                                fn = "nl_array_add";
                                if (t2 == TYPE_FLOAT || t1 == TYPE_FLOAT) {
                                    fn_scalar = "nl_array_add_scalar_float";
                                    fn_rscalar = "nl_array_radd_scalar_float";
                                } else if (t2 == TYPE_STRING || t1 == TYPE_STRING) {
                                    fn_scalar = "nl_array_add_scalar_string";
                                    fn_rscalar = "nl_array_radd_scalar_string";
                                } else {
                                    fn_scalar = "nl_array_add_scalar_int";
                                    fn_rscalar = "nl_array_radd_scalar_int";
                                }
                                break;
                            case TOKEN_MINUS:
                                fn = "nl_array_sub";
                                if (t2 == TYPE_FLOAT || t1 == TYPE_FLOAT) {
                                    fn_scalar = "nl_array_sub_scalar_float";
                                    fn_rscalar = "nl_array_rsub_scalar_float";
                                } else {
                                    fn_scalar = "nl_array_sub_scalar_int";
                                    fn_rscalar = "nl_array_rsub_scalar_int";
                                }
                                break;
                            case TOKEN_STAR:
                                fn = "nl_array_mul";
                                if (t2 == TYPE_FLOAT || t1 == TYPE_FLOAT) {
                                    fn_scalar = "nl_array_mul_scalar_float";
                                    fn_rscalar = "nl_array_rmul_scalar_float";
                                } else {
                                    fn_scalar = "nl_array_mul_scalar_int";
                                    fn_rscalar = "nl_array_rmul_scalar_int";
                                }
                                break;
                            case TOKEN_SLASH:
                                fn = "nl_array_div";
                                if (t2 == TYPE_FLOAT || t1 == TYPE_FLOAT) {
                                    fn_scalar = "nl_array_div_scalar_float";
                                    fn_rscalar = "nl_array_rdiv_scalar_float";
                                } else {
                                    fn_scalar = "nl_array_div_scalar_int";
                                    fn_rscalar = "nl_array_rdiv_scalar_int";
                                }
                                break;
                            case TOKEN_PERCENT:
                                fn = "nl_array_mod";
                                fn_scalar = "nl_array_mod_scalar_int";
                                fn_rscalar = "nl_array_rmod_scalar_int";
                                break;
                            default:
                                break;
                        }

                        if (left_is_array && right_is_array) {
                            Type elem = infer_array_element_type(left_expr, env);
                            Type elem2 = infer_array_element_type(right_expr, env);

                            if (elem != TYPE_UNKNOWN && elem2 != TYPE_UNKNOWN && elem == elem2 &&
                                (elem == TYPE_INT || elem == TYPE_U8 || elem == TYPE_FLOAT || elem == TYPE_STRING || elem == TYPE_ARRAY)) {
                                const char *elem_enum = (elem == TYPE_INT) ? "ELEM_INT" :
                                                        (elem == TYPE_U8) ? "ELEM_U8" :
                                                        (elem == TYPE_FLOAT) ? "ELEM_FLOAT" :
                                                        (elem == TYPE_STRING) ? "ELEM_STRING" :
                                                        "ELEM_ARRAY";
                                const char *get_suffix = (elem == TYPE_INT) ? "int" :
                                                         (elem == TYPE_U8) ? "u8" :
                                                         (elem == TYPE_FLOAT) ? "float" :
                                                         (elem == TYPE_STRING) ? "string" :
                                                         "array";
                                const char *push_suffix = get_suffix;
                                const char *c_type = (elem == TYPE_INT) ? "int64_t" :
                                                     (elem == TYPE_U8) ? "uint8_t" :
                                                     (elem == TYPE_FLOAT) ? "double" :
                                                     (elem == TYPE_STRING) ? "const char*" :
                                                     "DynArray*";

                                emit_literal(list, "({ DynArray* _a = ");
                                build_expr(list, left_expr, env);
                                emit_literal(list, "; DynArray* _b = ");
                                build_expr(list, right_expr, env);
                                emit_formatted(list, "; assert(dyn_array_length(_a) == dyn_array_length(_b)); DynArray* _out = dyn_array_new(%s); int64_t _len = dyn_array_length(_a); ", elem_enum);
                                emit_formatted(list, "for (int64_t _i = 0; _i < _len; _i++) { %s _x = dyn_array_get_%s(_a, _i); %s _y = dyn_array_get_%s(_b, _i); ", c_type, get_suffix, c_type, get_suffix);
                                if (elem == TYPE_STRING) {
                                    if (op == TOKEN_PLUS) {
                                        emit_literal(list, "dyn_array_push_string(_out, nl_str_concat(_x, _y)); ");
                                    } else {
                                        emit_literal(list, "assert(false && \"string arrays only support +\"); ");
                                    }
                                } else if (elem == TYPE_ARRAY) {
                                    emit_formatted(list, "dyn_array_push_%s(_out, %s(_x, _y)); ", push_suffix, fn);
                                } else {
                                    const char *op_str = (op == TOKEN_PLUS) ? "+" :
                                                        (op == TOKEN_MINUS) ? "-" :
                                                        (op == TOKEN_STAR) ? "*" :
                                                        (op == TOKEN_SLASH) ? "/" :
                                                        "%";
                                    if (elem == TYPE_FLOAT && op != TOKEN_PERCENT) {
                                        const char *helper = op == TOKEN_PLUS ? "add" : op == TOKEN_MINUS ? "sub" : op == TOKEN_STAR ? "mul" : "div";
                                        emit_formatted(list, "dyn_array_push_float(_out, nano_rt_f64_%s(_x, _y)); ", helper);
                                    } else emit_formatted(list, "dyn_array_push_%s(_out, _x %s _y); ", push_suffix, op_str);
                                }
                                emit_literal(list, "} _out; })");
                            } else {
                                emit_literal(list, "({ DynArray* _a = ");
                                build_expr(list, left_expr, env);
                                emit_literal(list, "; DynArray* _b = ");
                                build_expr(list, right_expr, env);
                                emit_formatted(list, "; %s(_a, _b); })", fn ? fn : "nl_array_add");
                            }
                            break;
                        }

                        /* array-scalar broadcast */
                        ASTNode *arr_expr = left_is_array ? left_expr : right_expr;
                        ASTNode *scalar_expr = left_is_array ? right_expr : left_expr;
                        Type elem = infer_array_element_type(arr_expr, env);

                        if (elem == TYPE_INT || elem == TYPE_U8 || elem == TYPE_FLOAT || elem == TYPE_STRING) {
                            const char *elem_enum = (elem == TYPE_INT) ? "ELEM_INT" : (elem == TYPE_U8) ? "ELEM_U8" : (elem == TYPE_FLOAT) ? "ELEM_FLOAT" : "ELEM_STRING";
                            const char *get_suffix = (elem == TYPE_INT) ? "int" : (elem == TYPE_U8) ? "u8" : (elem == TYPE_FLOAT) ? "float" : "string";
                            const char *push_suffix = get_suffix;
                            const char *c_type = (elem == TYPE_INT) ? "int64_t" : (elem == TYPE_U8) ? "uint8_t" : (elem == TYPE_FLOAT) ? "double" : "const char*";
                            const char *op_str = (op == TOKEN_PLUS) ? "+" :
                                                (op == TOKEN_MINUS) ? "-" :
                                                (op == TOKEN_STAR) ? "*" :
                                                (op == TOKEN_SLASH) ? "/" :
                                                "%";

                            if (left_is_array) {
                                emit_literal(list, "({ DynArray* _a = ");
                                build_expr(list, arr_expr, env);
                                emit_formatted(list, "; %s _s = ", c_type);
                                build_expr(list, scalar_expr, env);
                            } else {
                                emit_formatted(list, "({ %s _s = ", c_type);
                                build_expr(list, scalar_expr, env);
                                emit_literal(list, "; DynArray* _a = ");
                                build_expr(list, arr_expr, env);
                            }
                            emit_formatted(list, "; DynArray* _out = dyn_array_new(%s); int64_t _len = dyn_array_length(_a); ", elem_enum);
                            emit_formatted(list, "for (int64_t _i = 0; _i < _len; _i++) { %s _x = dyn_array_get_%s(_a, _i); ", c_type, get_suffix);
                            if (elem == TYPE_STRING) {
                                if (op == TOKEN_PLUS) {
                                    if (left_is_array) {
                                        emit_literal(list, "dyn_array_push_string(_out, nl_str_concat(_x, _s)); ");
                                    } else {
                                        emit_literal(list, "dyn_array_push_string(_out, nl_str_concat(_s, _x)); ");
                                    }
                                } else {
                                    emit_literal(list, "assert(false && \"string arrays only support +\"); ");
                                }
                            } else if (elem == TYPE_FLOAT && op != TOKEN_PERCENT) {
                                const char *helper = op == TOKEN_PLUS ? "add" : op == TOKEN_MINUS ? "sub" : op == TOKEN_STAR ? "mul" : "div";
                                emit_formatted(list, "dyn_array_push_float(_out, nano_rt_f64_%s(%s, %s)); ",
                                               helper, left_is_array ? "_x" : "_s", left_is_array ? "_s" : "_x");
                            } else {
                                if (left_is_array) {
                                    emit_formatted(list, "dyn_array_push_%s(_out, _x %s _s); ", push_suffix, op_str);
                                } else {
                                    emit_formatted(list, "dyn_array_push_%s(_out, _s %s _x); ", push_suffix, op_str);
                                }
                            }
                            emit_literal(list, "} _out; })");
                        } else {
                            /* Fallback: use runtime helper (asserts element type at runtime) */
                            if (left_is_array) {
                                emit_literal(list, "({ DynArray* _a = ");
                                build_expr(list, arr_expr, env);
                                emit_formatted(list, "; %s(_a, ", fn_scalar ? fn_scalar : "nl_array_add_scalar_int");
                                build_expr(list, scalar_expr, env);
                                emit_literal(list, "); })");
                            } else {
                                emit_literal(list, "({ __auto_type _s = ");
                                build_expr(list, scalar_expr, env);
                                emit_literal(list, "; DynArray* _a = ");
                                build_expr(list, arr_expr, env);
                                emit_formatted(list, "; %s(_s, _a); })", fn_rscalar ? fn_rscalar : "nl_array_radd_scalar_int");
                            }
                        }
                        break;
                    }
                }

                /* Check for string operations */
                Type op_t1 = check_expression(expr->as.prefix_op.args[0], env);
                Type op_t2 = check_expression(expr->as.prefix_op.args[1], env);
                
                bool is_string_comp = false;
                if ((op == TOKEN_EQ || op == TOKEN_NE) && op_t1 == TYPE_STRING && op_t2 == TYPE_STRING) {
                    is_string_comp = true;
                }
                
                bool is_string_concat = false;
                if (op == TOKEN_PLUS && op_t1 == TYPE_STRING && op_t2 == TYPE_STRING) {
                    is_string_concat = true;
                }
                
                if (is_string_comp) {
                    /* String comparison: strcmp(a, b) == 0 */
                    emit_literal(list, "(strcmp(");
                    build_expr(list, expr->as.prefix_op.args[0], env);
                    emit_literal(list, ", ");
                    build_expr(list, expr->as.prefix_op.args[1], env);
                    if (op == TOKEN_EQ) {
                        emit_literal(list, ") == 0)");
                    } else {
                        emit_literal(list, ") != 0)");
                    }
                } else if (is_string_concat) {
                    /* String concatenation: nl_str_concat(a, b) */
                    emit_literal(list, "nl_str_concat(");
                    build_expr(list, expr->as.prefix_op.args[0], env);
                    emit_literal(list, ", ");
                    build_expr(list, expr->as.prefix_op.args[1], env);
                    emit_literal(list, ")");
                } else if ((op == TOKEN_SLASH || op == TOKEN_PERCENT)
                           && check_expression(expr->as.prefix_op.args[0], env) == TYPE_INT
                           && check_expression(expr->as.prefix_op.args[1], env) == TYPE_INT) {
                    /* Integer / and % must match the language's total semantics
                     * (div-by-zero -> 0; INT64_MIN / -1 and INT64_MIN % -1
                     * defined, not signed-overflow UB/SIGFPE, matching the Coq
                     * model and the NanoISA VM). Emit a runtime helper instead
                     * of a raw C divide. Float division stays a plain `/`. */
                    emit_literal(list, op == TOKEN_SLASH ? "nano_rt_idiv(" : "nano_rt_imod(");
                    build_expr(list, expr->as.prefix_op.args[0], env);
                    emit_literal(list, ", ");
                    build_expr(list, expr->as.prefix_op.args[1], env);
                    emit_literal(list, ")");
                } else {
                    /* Regular binary operator */
                    bool needs_parens = (op == TOKEN_PLUS || op == TOKEN_MINUS ||
                                       op == TOKEN_STAR || op == TOKEN_SLASH || op == TOKEN_PERCENT ||
                                       op == TOKEN_AND || op == TOKEN_OR);

                    if (needs_parens) emit_literal(list, "(");
                    build_expr(list, expr->as.prefix_op.args[0], env);

                    const char *op_str = NULL;
                    switch (op) {
                        case TOKEN_PLUS: op_str = " + "; break;
                        case TOKEN_MINUS: op_str = " - "; break;
                        case TOKEN_STAR: op_str = " * "; break;
                        case TOKEN_SLASH: op_str = " / "; break;
                        case TOKEN_PERCENT: op_str = " % "; break;
                        case TOKEN_EQ: op_str = " == "; break;
                        case TOKEN_NE: op_str = " != "; break;
                        case TOKEN_LT: op_str = " < "; break;
                        case TOKEN_LE: op_str = " <= "; break;
                        case TOKEN_GT: op_str = " > "; break;
                        case TOKEN_GE: op_str = " >= "; break;
                        case TOKEN_AND: op_str = " && "; break;
                        case TOKEN_OR: op_str = " || "; break;
                        default: op_str = " OP "; break;
                    }
                    emit_literal(list, op_str);
                    build_expr(list, expr->as.prefix_op.args[1], env);
                    if (needs_parens) emit_literal(list, ")");
                }
            } else if (arg_count == 1) {
                /* Unary operator */
                if (op == TOKEN_NOT) {
                    ASTNode *arg = expr->as.prefix_op.args[0];
                    bool arg_is_binary_op = (arg && arg->type == AST_PREFIX_OP && arg->as.prefix_op.arg_count == 2);
                    
                    emit_literal(list, "(!");
                    /* Add extra parentheses around binary operations to fix precedence */
                    if (arg_is_binary_op) {
                        emit_literal(list, "(");
                    }
                    build_expr(list, arg, env);
                    if (arg_is_binary_op) {
                        emit_literal(list, ")");
                    }
                    emit_literal(list, ")");
                } else if (op == TOKEN_MINUS) {
                    Type t = check_expression(expr->as.prefix_op.args[0], env);
                    ASTNode *arg = expr->as.prefix_op.args[0];
                    if (t == TYPE_INT && arg && arg->type == AST_NUMBER &&
                        arg->as.number == INT64_MIN) {
                        /* The parser stores the magnitude of -2^63 in an int64_t.
                         * Spell the result with the standard macro so C never has
                         * to represent the out-of-range positive literal first. */
                        emit_literal(list, "INT64_MIN");
                    } else if (t == TYPE_ARRAY) {
                        ASTNode *arr_expr = expr->as.prefix_op.args[0];
                        Type elem = infer_array_element_type(arr_expr, env);
                        if (elem == TYPE_INT || elem == TYPE_FLOAT) {
                            const char *elem_enum = (elem == TYPE_INT) ? "ELEM_INT" : "ELEM_FLOAT";
                            const char *get_suffix = (elem == TYPE_INT) ? "int" : "float";
                            const char *push_suffix = get_suffix;
                            const char *c_type = (elem == TYPE_INT) ? "int64_t" : "double";
                            emit_literal(list, "({ DynArray* _a = ");
                            build_expr(list, arr_expr, env);
                            emit_formatted(list, "; DynArray* _out = dyn_array_new(%s); int64_t _len = dyn_array_length(_a); ", elem_enum);
                            emit_formatted(list, "for (int64_t _i = 0; _i < _len; _i++) { %s _x = dyn_array_get_%s(_a, _i); dyn_array_push_%s(_out, -_x); } _out; })", c_type, get_suffix, push_suffix);
                        } else {
                            emit_literal(list, "({ assert(false && \"unary minus requires array<int> or array<float>\"); (DynArray*)0; })");
                        }
                    } else {
                        emit_literal(list, "(-(");
                        build_expr(list, expr->as.prefix_op.args[0], env);
                        emit_literal(list, "))");
                    }
                } else if (op == TOKEN_QUESTION) {
                    /* ? try-propagate: desugars to:
                     * ({ nl_X _nl_try_r = inner; if (_nl_try_r.tag == nl_X_TAG_Err) return _nl_try_r; _nl_try_r.data.Ok.val; })
                     */
                    ASTNode *inner = expr->as.prefix_op.args[0];
                    const char *union_name_raw = get_struct_type_name(inner, env);
                    if (!union_name_raw) union_name_raw = "Result";
                    char union_name[256];
                    strncpy(union_name, union_name_raw, sizeof(union_name) - 1);
                    union_name[sizeof(union_name) - 1] = '\0';

                    const char *err_variant = "Err";
                    const char *ok_variant = "Ok";
                    const char *ok_field = "val";

                    UnionDef *udef = env_get_union(env, union_name);
                    if (udef) {
                        for (int vi = 0; vi < udef->variant_count; vi++) {
                            if (strcmp(udef->variant_names[vi], "Err") == 0) {
                                err_variant = udef->variant_names[vi];
                            } else if (strcmp(udef->variant_names[vi], "Ok") == 0) {
                                ok_variant = udef->variant_names[vi];
                                if (udef->variant_field_counts[vi] > 0) {
                                    ok_field = udef->variant_field_names[vi][0];
                                }
                            }
                        }
                    }

                    /* Copy prefixed names into local buffers before subsequent calls overwrite them */
                    char c_type_buf[256];
                    snprintf(c_type_buf, sizeof(c_type_buf), "%s", get_prefixed_type_name(union_name));
                    char err_tag_buf[512];
                    snprintf(err_tag_buf, sizeof(err_tag_buf), "%s", get_prefixed_tag_name(union_name, err_variant));

                    emit_literal(list, "({ ");
                    emit_formatted(list, "%s _nl_try_r = ", c_type_buf);
                    build_expr(list, inner, env);
                    emit_formatted(list, "; if (_nl_try_r.tag == %s) return _nl_try_r; ", err_tag_buf);
                    emit_formatted(list, "_nl_try_r.data.%s.%s; })", ok_variant, ok_field);
                }
            }
            break;
        }
        
        case AST_CALL: {
            if (expr->as.call.borrow_mode) {
                emit_literal(list, "(&(");
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, "))");
                break;
            }
            /* Map function name */
            const char *func_name = expr->as.call.name;
            
            /* Check for NULL function name (happens with function pointers: ((get_operation choice) a b)) */
            if (!func_name) {
                /* This is a function pointer call - use func_expr */
                if (expr->as.call.func_expr) {
                    build_expr(list, expr->as.call.func_expr, env);
                    emit_literal(list, "(");
                    for (int i = 0; i < expr->as.call.arg_count; i++) {
                        if (i > 0) emit_literal(list, ", ");
                        build_expr(list, expr->as.call.args[i], env);
                    }
                    emit_literal(list, ")");
                }
                break;
            }
            
            if (env_native_array_operation(func_name) &&
                !env_native_array_is_builtin(env, func_name, expr->line, expr->column)) goto native_array_declared_call;
            if (native_opaque_array_call(list, expr, env)) break;

            /* Special handling for println - needs type dispatch */
            if (strcmp(func_name, "println") == 0 && expr->as.call.arg_count == 1) {
                Type arg_type = check_expression(expr->as.call.args[0], env);
                if (arg_type == TYPE_STRING) {
                    emit_literal(list, "nl_println_string(");
                } else if (arg_type == TYPE_FLOAT) {
                    emit_literal(list, "nl_println_float(");
                } else if (arg_type == TYPE_BOOL) {
                    emit_literal(list, "nl_println_bool(");
                } else if (arg_type == TYPE_INT || arg_type == TYPE_U8) {
                    emit_literal(list, "nl_println_int(");
                } else {
                    emit_literal(list, "nl_println_string(");
                    if (arg_type == TYPE_ARRAY) {
                        emit_literal(list, "nl_to_string_array(");
                        build_expr(list, expr->as.call.args[0], env);
                        emit_literal(list, ")");
                    } else if (arg_type == TYPE_STRUCT || arg_type == TYPE_UNION) {
                        const char *type_name = get_struct_type_name(expr->as.call.args[0], env);
                        if (type_name) {
                            emit_formatted(list, "nl_to_string_%s(", type_name);
                            build_expr(list, expr->as.call.args[0], env);
                            emit_literal(list, ")");
                        } else {
                            emit_literal(list, "\"<struct>\"");
                        }
                    } else if (arg_type == TYPE_ENUM) {
                        emit_literal(list, "nl_to_string_int(");
                        build_expr(list, expr->as.call.args[0], env);
                        emit_literal(list, ")");
                    } else {
                        emit_literal(list, "\"<unknown>\"");
                    }
                    emit_literal(list, ")");
                    break;
                }
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, ")");
            }
            /* Special handling for print - needs type dispatch */
            else if (strcmp(func_name, "print") == 0 && expr->as.call.arg_count == 1) {
                Type arg_type = check_expression(expr->as.call.args[0], env);
                if (arg_type == TYPE_STRING) {
                    emit_literal(list, "nl_print_string(");
                } else if (arg_type == TYPE_FLOAT) {
                    emit_literal(list, "nl_print_float(");
                } else if (arg_type == TYPE_BOOL) {
                    emit_literal(list, "nl_print_bool(");
                } else if (arg_type == TYPE_INT || arg_type == TYPE_U8) {
                    emit_literal(list, "nl_print_int(");
                } else {
                    emit_literal(list, "nl_print_string(");
                    if (arg_type == TYPE_ARRAY) {
                        emit_literal(list, "nl_to_string_array(");
                        build_expr(list, expr->as.call.args[0], env);
                        emit_literal(list, ")");
                    } else if (arg_type == TYPE_STRUCT || arg_type == TYPE_UNION) {
                        const char *type_name = get_struct_type_name(expr->as.call.args[0], env);
                        if (type_name) {
                            emit_formatted(list, "nl_to_string_%s(", type_name);
                            build_expr(list, expr->as.call.args[0], env);
                            emit_literal(list, ")");
                        } else {
                            emit_literal(list, "\"<struct>\"");
                        }
                    } else if (arg_type == TYPE_ENUM) {
                        emit_literal(list, "nl_to_string_int(");
                        build_expr(list, expr->as.call.args[0], env);
                        emit_literal(list, ")");
                    } else {
                        emit_literal(list, "\"<unknown>\"");
                    }
                    emit_literal(list, ")");
                    break;
                }
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, ")");
            }

            /* Special handling for to_string/cast_string */
            else if ((strcmp(func_name, "to_string") == 0 || strcmp(func_name, "cast_string") == 0) &&
                     expr->as.call.arg_count == 1) {
                ASTNode *arg = expr->as.call.args[0];
                Type arg_type = check_expression(arg, env);

                if (arg_type == TYPE_STRING) {
                    emit_literal(list, "nl_to_string_string(");
                    build_expr(list, arg, env);
                    emit_literal(list, ")");
                } else if (arg_type == TYPE_INT || arg_type == TYPE_U8) {
                    emit_literal(list, "nl_to_string_int(");
                    build_expr(list, arg, env);
                    emit_literal(list, ")");
                } else if (arg_type == TYPE_FLOAT) {
                    emit_literal(list, "nl_to_string_float(");
                    build_expr(list, arg, env);
                    emit_literal(list, ")");
                } else if (arg_type == TYPE_BOOL) {
                    emit_literal(list, "nl_to_string_bool(");
                    build_expr(list, arg, env);
                    emit_literal(list, ")");
                } else if (arg_type == TYPE_ARRAY) {
                    emit_literal(list, "nl_to_string_array(");
                    build_expr(list, arg, env);
                    emit_literal(list, ")");
                } else if (arg_type == TYPE_STRUCT || arg_type == TYPE_UNION) {
                    const char *type_name = get_struct_type_name(arg, env);
                    if (type_name) {
                        emit_formatted(list, "nl_to_string_%s(", type_name);
                        build_expr(list, arg, env);
                        emit_literal(list, ")");
                    } else {
                        emit_literal(list, "\"<struct>\"");
                    }
                } else if (arg_type == TYPE_ENUM) {
                    /* Best-effort: print as int when enum name isn't known */
                    emit_literal(list, "nl_to_string_int(");
                    build_expr(list, arg, env);
                    emit_literal(list, ")");
                } else {
                    emit_literal(list, "\"<unknown>\"");
                }
            }

            /* format(template, arg1, arg2, ...) -> string interpolation */
            else if (strcmp(func_name, "format") == 0 && expr->as.call.arg_count >= 1) {
                int n_args = expr->as.call.arg_count - 1;
                emit_literal(list, "nl_format(");
                build_expr(list, expr->as.call.args[0], env);
                emit_formatted(list, ", %d", n_args);
                for (int i = 1; i < expr->as.call.arg_count; i++) {
                    emit_literal(list, ", ");
                    ASTNode *farg = expr->as.call.args[i];
                    Type farg_type = check_expression(farg, env);
                    if (farg_type == TYPE_STRING) {
                        build_expr(list, farg, env);
                    } else if (farg_type == TYPE_INT || farg_type == TYPE_U8) {
                        emit_literal(list, "int_to_string(");
                        build_expr(list, farg, env);
                        emit_literal(list, ")");
                    } else if (farg_type == TYPE_FLOAT) {
                        emit_literal(list, "float_to_string(");
                        build_expr(list, farg, env);
                        emit_literal(list, ")");
                    } else if (farg_type == TYPE_BOOL) {
                        emit_literal(list, "nl_to_string_bool(");
                        build_expr(list, farg, env);
                        emit_literal(list, ")");
                    } else {
                        /* Fallback: cast to string */
                        emit_literal(list, "nl_to_string_int((int64_t)(");
                        build_expr(list, farg, env);
                        emit_literal(list, "))");
                    }
                }
                emit_literal(list, ")");
            }

            /* Result<T,E> helpers (intrinsics; generic-function stopgap) */
            else if (strcmp(func_name, "result_is_ok") == 0 && expr->as.call.arg_count == 1) {
                emit_literal(list, "({ __auto_type _r = ");
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, "; (_r.tag == 0); })");
            }
            else if (strcmp(func_name, "result_is_err") == 0 && expr->as.call.arg_count == 1) {
                emit_literal(list, "({ __auto_type _r = ");
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, "; (_r.tag != 0); })");
            }
            else if (strcmp(func_name, "result_unwrap") == 0 && expr->as.call.arg_count == 1) {
                emit_literal(list, "({ __auto_type _r = ");
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, "; if (_r.tag != 0) { fprintf(stderr, \"panic: unwrap Err\\n\"); exit(1); } _r.data.Ok.value; })");
            }
            else if (strcmp(func_name, "result_unwrap_err") == 0 && expr->as.call.arg_count == 1) {
                emit_literal(list, "({ __auto_type _r = ");
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, "; if (_r.tag == 0) { fprintf(stderr, \"panic: unwrap_err Ok\\n\"); exit(1); } _r.data.Err.error; })");
            }
            else if (strcmp(func_name, "result_unwrap_or") == 0 && expr->as.call.arg_count == 2) {
                emit_literal(list, "({ __auto_type _r = ");
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, "; __auto_type _d = ");
                build_expr(list, expr->as.call.args[1], env);
                emit_literal(list, "; (_r.tag == 0) ? _r.data.Ok.value : _d; })");
            }
            else if (strcmp(func_name, "result_map") == 0 && expr->as.call.arg_count == 2) {
                emit_literal(list, "({ __auto_type _r = ");
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, "; __auto_type _f = ");
                build_expr(list, expr->as.call.args[1], env);
                emit_literal(list, "; if (_r.tag == 0) { _r.data.Ok.value = _f(_r.data.Ok.value); } _r; })");
            }
            else if (strcmp(func_name, "result_and_then") == 0 && expr->as.call.arg_count == 2) {
                emit_literal(list, "({ __auto_type _r = ");
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, "; __auto_type _f = ");
                build_expr(list, expr->as.call.args[1], env);
                emit_literal(list, "; (_r.tag == 0) ? _f(_r.data.Ok.value) : _r; })");
            }

            /* HashMap<K,V> core built-ins (only if no user-defined function exists) */
            else if (env_get_function(env, func_name) == NULL && strcmp(func_name, "map_new") == 0 && expr->as.call.arg_count == 0) {
                const char *mono = expr->as.call.return_struct_type_name;
                if (!mono) {
                    emit_literal(list, "({ assert(false && \"map_new requires HashMap<K,V> type context\"); (void*)0; })");
                } else {
                    const char *suffix = mono;
                    if (strncmp(mono, "HashMap_", 8) == 0) suffix = mono + 8;
                    emit_formatted(list, "nl_hashmap_%s_new()", suffix);
                }
            }
            else if (env_get_function(env, func_name) == NULL &&
                     (strcmp(func_name, "map_put") == 0 || strcmp(func_name, "map_set") == 0) &&
                     expr->as.call.arg_count == 3) {
                char buf[256];
                const char *suffix = hashmap_suffix_from_expr(expr->as.call.args[0], env, buf, sizeof(buf));
                if (!suffix) {
                    emit_literal(list, "({ assert(false && \"cannot infer HashMap<K,V> for map_put\"); (void)0; })");
                } else {
                    build_ordered_hashmap_call(list, expr, env, suffix, "put");
                }
            }
            else if (env_get_function(env, func_name) == NULL && strcmp(func_name, "map_get") == 0 && expr->as.call.arg_count == 2) {
                char buf[256];
                const char *suffix = hashmap_suffix_from_expr(expr->as.call.args[0], env, buf, sizeof(buf));
                if (!suffix) {
                    emit_literal(list, "({ assert(false && \"cannot infer HashMap<K,V> for map_get\"); 0; })");
                } else {
                    build_ordered_hashmap_call(list, expr, env, suffix, "get");
                }
            }
            else if (env_get_function(env, func_name) == NULL && strcmp(func_name, "map_has") == 0 && expr->as.call.arg_count == 2) {
                char buf[256];
                const char *suffix = hashmap_suffix_from_expr(expr->as.call.args[0], env, buf, sizeof(buf));
                if (!suffix) {
                    emit_literal(list, "({ assert(false && \"cannot infer HashMap<K,V> for map_has\"); false; })");
                } else {
                    build_ordered_hashmap_call(list, expr, env, suffix, "has");
                }
            }
            else if (env_get_function(env, func_name) == NULL && strcmp(func_name, "map_remove") == 0 && expr->as.call.arg_count == 2) {
                char buf[256];
                const char *suffix = hashmap_suffix_from_expr(expr->as.call.args[0], env, buf, sizeof(buf));
                if (!suffix) {
                    emit_literal(list, "({ assert(false && \"cannot infer HashMap<K,V> for map_remove\"); (void)0; })");
                } else {
                    build_ordered_hashmap_call(list, expr, env, suffix, "remove");
                }
            }
            else if (env_get_function(env, func_name) == NULL &&
                     (strcmp(func_name, "map_length") == 0 || strcmp(func_name, "map_size") == 0) &&
                     expr->as.call.arg_count == 1) {
                char buf[256];
                const char *suffix = hashmap_suffix_from_expr(expr->as.call.args[0], env, buf, sizeof(buf));
                if (!suffix) {
                    emit_literal(list, "({ assert(false && \"cannot infer HashMap<K,V> for map_length\"); 0; })");
                } else {
                    build_ordered_hashmap_call(list, expr, env, suffix, "length");
                }
            }
            else if (env_get_function(env, func_name) == NULL &&
                     (strcmp(func_name, "map_clear") == 0 || strcmp(func_name, "map_free") == 0) &&
                     expr->as.call.arg_count == 1) {
                char buf[256];
                const char *suffix = hashmap_suffix_from_expr(expr->as.call.args[0], env, buf, sizeof(buf));
                if (!suffix) {
                    emit_literal(list, "({ assert(false && \"cannot infer HashMap<K,V> for map_clear\"); (void)0; })");
                } else {
                    const char *op = (strcmp(func_name, "map_free") == 0) ? "free" : "clear";
                    build_ordered_hashmap_call(list, expr, env, suffix, op);
                }
            }
            else if (env_get_function(env, func_name) == NULL &&
                     (strcmp(func_name, "map_keys") == 0 || strcmp(func_name, "map_values") == 0) &&
                     expr->as.call.arg_count == 1) {
                char buf[256];
                const char *suffix = hashmap_suffix_from_expr(expr->as.call.args[0], env, buf, sizeof(buf));
                if (!suffix) {
                    emit_literal(list, "({ assert(false && \"cannot infer HashMap<K,V> for map_keys\"); (DynArray*)0; })");
                } else {
                    const char *op = (strcmp(func_name, "map_values") == 0) ? "values" : "keys";
                    build_ordered_hashmap_call(list, expr, env, suffix, op);
                }
            }

            /* I evaluate array search inputs once, in source order. C does not
             * order arguments to an ordinary call, even when my helper is pure. */
            else if ((strcmp(func_name, "array_index_of") == 0 ||
                      strcmp(func_name, "array_contains") == 0) &&
                     expr->as.call.arg_count == 2) {
                emit_literal(list, "({ __auto_type __nl_search_array = ");
                build_expr(list, expr->as.call.args[0], env);
                emit_literal(list, "; __auto_type __nl_search_needle = ");
                build_expr(list, expr->as.call.args[1], env);
                emit_formatted(list, "; nl_%s(__nl_search_array, __nl_search_needle); })", func_name);
            }

            /* Special handling for filter() - compiled lowering */
            else if (strcmp(func_name, "filter") == 0 && expr->as.call.arg_count == 2) {
                /* filter(array<T>, fn(T)->bool) -> array<T>
                 * Arrays are DynArray* in compiled mode.
                 */
                ASTNode *array_arg = expr->as.call.args[0];
                ASTNode *fn_arg = expr->as.call.args[1];

                Type elem_type = TYPE_INT;  /* default */
                const char *struct_name = NULL;

                if (array_arg && array_arg->type == AST_IDENTIFIER) {
                    Symbol *sym = env_get_var_visible_at(env, array_arg->as.identifier, array_arg->line, array_arg->column);
                    if (sym && sym->element_type != TYPE_UNKNOWN) {
                        elem_type = sym->element_type;
                        if (elem_type == TYPE_STRUCT && sym->struct_type_name) {
                            struct_name = sym->struct_type_name;
                        }
                    }
                } else if (array_arg && array_arg->type == AST_ARRAY_LITERAL &&
                           array_arg->as.array_literal.element_type != TYPE_UNKNOWN) {
                    elem_type = array_arg->as.array_literal.element_type;
                    if (elem_type == TYPE_STRUCT && array_arg->as.array_literal.element_count > 0) {
                        ASTNode *first = array_arg->as.array_literal.elements[0];
                        if (first && first->type == AST_STRUCT_LITERAL) {
                            struct_name = first->as.struct_literal.struct_name;
                        }
                    }
                }

                bool empty_literal = array_arg && array_arg->type == AST_ARRAY_LITERAL &&
                                     array_arg->as.array_literal.element_count == 0;
                if (empty_literal) {
                    Type input = filter_predicate_element_type(fn_arg, env);
                    if (input == TYPE_INT || input == TYPE_FLOAT ||
                        input == TYPE_BOOL || input == TYPE_STRING) elem_type = input;
                }

                const char *elem_enum = "ELEM_INT";
                const char *type_suffix = "int";
                const char *c_type = "int64_t";

                if (elem_type == TYPE_FLOAT) {
                    elem_enum = "ELEM_FLOAT";
                    type_suffix = "float";
                    c_type = "double";
                } else if (elem_type == TYPE_STRING) {
                    elem_enum = "ELEM_STRING";
                    type_suffix = "string";
                    c_type = "const char*";
                } else if (elem_type == TYPE_BOOL) {
                    elem_enum = "ELEM_BOOL";
                    type_suffix = "bool";
                    c_type = "bool";
                } else if (elem_type == TYPE_ARRAY) {
                    elem_enum = "ELEM_ARRAY";
                    type_suffix = "array";
                    c_type = "DynArray*";
                } else if (elem_type == TYPE_STRUCT) {
                    elem_enum = "ELEM_STRUCT";
                }

                emit_literal(list, "({ DynArray* _arr = ");
                if (empty_literal) emit_formatted(list, "dyn_array_new(%s)", elem_enum);
                else build_expr(list, array_arg, env);
                emit_literal(list, "; __auto_type _pred = ");
                build_expr(list, fn_arg, env);
                emit_formatted(list, "; DynArray* _out = dyn_array_new(%s); ", elem_enum);
                emit_literal(list, "int64_t _len = dyn_array_length(_arr); ");
                emit_literal(list, "for (int64_t _i = 0; _i < _len; _i++) { ");

                if (elem_type == TYPE_STRUCT && struct_name) {
                    emit_formatted(list,
                                   "nl_%s _elem = *((nl_%s*)dyn_array_get_struct(_arr, _i)); ",
                                   struct_name, struct_name);
                    emit_literal(list, "if (_pred(_elem)) { ");
                    emit_formatted(list, "dyn_array_push_struct(_out, &_elem, sizeof(nl_%s)); ", struct_name);
                    emit_literal(list, "} ");
                } else if (elem_type == TYPE_STRUCT && !struct_name) {
                    emit_literal(list, "int64_t _elem = dyn_array_get_int(_arr, _i); ");
                    emit_literal(list, "if (_pred(_elem)) { dyn_array_push_int(_out, _elem); } ");
                } else {
                    emit_formatted(list, "%s _elem = dyn_array_get_%s(_arr, _i); ", c_type, type_suffix);
                    emit_literal(list, "if (_pred(_elem)) { ");
                    emit_formatted(list, "dyn_array_push_%s(_out, _elem); ", type_suffix);
                    emit_literal(list, "} ");
                }

                emit_literal(list, "} _out; })");
            }

            /* Special handling for map() - compiled lowering */
            else if (strcmp(func_name, "map") == 0 && expr->as.call.arg_count == 2) {
                /* map(array<T>, fn(T)->U) -> array<U>
                 * Arrays are DynArray* in compiled mode.
                 */
                ASTNode *array_arg = expr->as.call.args[0];
                ASTNode *fn_arg = expr->as.call.args[1];

                Type elem_type = TYPE_INT;  /* default */
                const char *struct_name = NULL;

                if (array_arg && array_arg->type == AST_IDENTIFIER) {
                    Symbol *sym = env_get_var_visible_at(env, array_arg->as.identifier, array_arg->line, array_arg->column);
                    if (sym && sym->element_type != TYPE_UNKNOWN) {
                        elem_type = sym->element_type;
                        if (elem_type == TYPE_STRUCT && sym->struct_type_name) {
                            struct_name = sym->struct_type_name;
                        }
                    }
                } else if (array_arg && array_arg->type == AST_ARRAY_LITERAL &&
                           array_arg->as.array_literal.element_type != TYPE_UNKNOWN) {
                    elem_type = array_arg->as.array_literal.element_type;
                    if (elem_type == TYPE_STRUCT && array_arg->as.array_literal.element_count > 0) {
                        ASTNode *first = array_arg->as.array_literal.elements[0];
                        if (first && first->type == AST_STRUCT_LITERAL) {
                            struct_name = first->as.struct_literal.struct_name;
                        }
                    }
                }

                const char *elem_enum = "ELEM_INT";
                const char *type_suffix = "int";
                const char *c_type = "int64_t";

                if (elem_type == TYPE_FLOAT) {
                    elem_enum = "ELEM_FLOAT";
                    type_suffix = "float";
                    c_type = "double";
                } else if (elem_type == TYPE_STRING) {
                    elem_enum = "ELEM_STRING";
                    type_suffix = "string";
                    c_type = "const char*";
                } else if (elem_type == TYPE_BOOL) {
                    elem_enum = "ELEM_BOOL";
                    type_suffix = "bool";
                    c_type = "bool";
                } else if (elem_type == TYPE_ARRAY) {
                    elem_enum = "ELEM_ARRAY";
                    type_suffix = "array";
                    c_type = "DynArray*";
                } else if (elem_type == TYPE_STRUCT) {
                    elem_enum = "ELEM_STRUCT";
                }

                emit_literal(list, "({ DynArray* _arr = ");
                build_expr(list, array_arg, env);
                emit_literal(list, "; __auto_type _f = ");
                build_expr(list, fn_arg, env);
                Type result_type = map_transform_result_type(fn_arg, env);
                const char *result_suffix = type_suffix;
                if (result_type == TYPE_INT) { elem_enum = "ELEM_INT"; result_suffix = "int"; }
                else if (result_type == TYPE_FLOAT) { elem_enum = "ELEM_FLOAT"; result_suffix = "float"; }
                else if (result_type == TYPE_BOOL) { elem_enum = "ELEM_BOOL"; result_suffix = "bool"; }
                else if (result_type == TYPE_STRING) { elem_enum = "ELEM_STRING"; result_suffix = "string"; }
                else if (result_type == TYPE_ARRAY) { elem_enum = "ELEM_ARRAY"; result_suffix = "array"; }
                emit_formatted(list, "; DynArray* _out = dyn_array_new(%s); ", elem_enum);
                emit_literal(list, "int64_t _len = dyn_array_length(_arr); ");
                emit_literal(list, "for (int64_t _i = 0; _i < _len; _i++) { ");

                if (elem_type == TYPE_STRUCT && struct_name) {
                    emit_formatted(list,
                                   "nl_%s _elem = *((nl_%s*)dyn_array_get_struct(_arr, _i)); ",
                                   struct_name, struct_name);
                    emit_formatted(list,
                                   "nl_%s _mapped = _f(_elem); dyn_array_push_struct(_out, &_mapped, sizeof(nl_%s)); ",
                                   struct_name, struct_name);
                } else if (elem_type == TYPE_STRUCT && !struct_name) {
                    /* Best effort fallback - treat as int to avoid generating invalid C types */
                    emit_literal(list, "int64_t _elem = dyn_array_get_int(_arr, _i); ");
                    emit_literal(list, "int64_t _mapped = _f(_elem); dyn_array_push_int(_out, _mapped); ");
                } else {
                    emit_formatted(list, "%s _elem = dyn_array_get_%s(_arr, _i); ", c_type, type_suffix);
                    emit_formatted(list, "__auto_type _mapped = _f(_elem); dyn_array_push_%s(_out, _mapped); ", result_suffix);
                }

                emit_literal(list, "} _out; })");
            }

            /* Special handling for reduce() - compiled lowering */
            else if (strcmp(func_name, "reduce") == 0 && expr->as.call.arg_count == 3) {
                /* reduce(array<T>, acc, fn(acc, T)->acc) -> acc */
                ASTNode *array_arg = expr->as.call.args[0];
                ASTNode *initial_arg = expr->as.call.args[1];
                ASTNode *fn_arg = expr->as.call.args[2];

                Type elem_type = TYPE_INT;  /* default */
                const char *elem_struct_name = NULL;

                if (array_arg && array_arg->type == AST_IDENTIFIER) {
                    Symbol *sym = env_get_var_visible_at(env, array_arg->as.identifier, array_arg->line, array_arg->column);
                    if (sym && sym->element_type != TYPE_UNKNOWN) {
                        elem_type = sym->element_type;
                        if (elem_type == TYPE_STRUCT && sym->struct_type_name) {
                            elem_struct_name = sym->struct_type_name;
                        }
                    }
                } else if (array_arg && array_arg->type == AST_ARRAY_LITERAL &&
                           array_arg->as.array_literal.element_type != TYPE_UNKNOWN) {
                    elem_type = array_arg->as.array_literal.element_type;
                    if (elem_type == TYPE_STRUCT && array_arg->as.array_literal.element_count > 0) {
                        ASTNode *first = array_arg->as.array_literal.elements[0];
                        if (first && first->type == AST_STRUCT_LITERAL) {
                            elem_struct_name = first->as.struct_literal.struct_name;
                        }
                    }
                }

                Type acc_type = check_expression(initial_arg, env);
                const char *acc_c_type = "int64_t";
                const char *acc_struct_name = NULL;

                if (acc_type == TYPE_FLOAT) {
                    acc_c_type = "double";
                } else if (acc_type == TYPE_BOOL) {
                    acc_c_type = "bool";
                } else if (acc_type == TYPE_STRING) {
                    acc_c_type = "const char*";
                } else if (acc_type == TYPE_ARRAY) {
                    acc_c_type = "DynArray*";
                } else if (acc_type == TYPE_STRUCT) {
                    /* Try to recover struct name */
                    if (initial_arg && initial_arg->type == AST_STRUCT_LITERAL) {
                        acc_struct_name = initial_arg->as.struct_literal.struct_name;
                    } else if (initial_arg && initial_arg->type == AST_IDENTIFIER) {
                        Symbol *sym = env_get_var_visible_at(env, initial_arg->as.identifier, initial_arg->line, initial_arg->column);
                        if (sym && sym->struct_type_name) {
                            acc_struct_name = sym->struct_type_name;
                        }
                    }
                    if (acc_struct_name) {
                        static _Thread_local char buf[256];
                        snprintf(buf, sizeof(buf), "nl_%s", acc_struct_name);
                        acc_c_type = buf;
                    }
                }

                const char *elem_suffix = "int";
                const char *elem_c_type = "int64_t";
                if (elem_type == TYPE_FLOAT) {
                    elem_suffix = "float";
                    elem_c_type = "double";
                } else if (elem_type == TYPE_STRING) {
                    elem_suffix = "string";
                    elem_c_type = "const char*";
                } else if (elem_type == TYPE_BOOL) {
                    elem_suffix = "bool";
                    elem_c_type = "bool";
                } else if (elem_type == TYPE_ARRAY) {
                    elem_suffix = "array";
                    elem_c_type = "DynArray*";
                }

                emit_literal(list, "({ DynArray* _arr = ");
                build_expr(list, array_arg, env);
                emit_literal(list, "; __auto_type _f = ");
                build_expr(list, fn_arg, env);
                emit_literal(list, "; ");

                emit_formatted(list, "%s _acc = ", acc_c_type);
                build_expr(list, initial_arg, env);
                emit_literal(list, "; ");

                emit_literal(list, "int64_t _len = dyn_array_length(_arr); ");
                emit_literal(list, "for (int64_t _i = 0; _i < _len; _i++) { ");

                if (elem_type == TYPE_STRUCT && elem_struct_name) {
                    emit_formatted(list,
                                   "nl_%s _elem = *((nl_%s*)dyn_array_get_struct(_arr, _i)); ",
                                   elem_struct_name, elem_struct_name);
                    emit_literal(list, "_acc = _f(_acc, _elem); ");
                } else if (elem_type == TYPE_STRUCT && !elem_struct_name) {
                    emit_literal(list, "int64_t _elem = dyn_array_get_int(_arr, _i); ");
                    emit_literal(list, "_acc = _f(_acc, _elem); ");
                } else {
                    emit_formatted(list, "%s _elem = dyn_array_get_%s(_arr, _i); ", elem_c_type, elem_suffix);
                    emit_literal(list, "_acc = _f(_acc, _elem); ");
                }

                emit_literal(list, "} _acc; })");
            }

            /* Special handling for at() and array_set() - needs type-specific functions */
            else if ((strcmp(func_name, "at") == 0 || strcmp(func_name, "array_get") == 0 ||
                      strcmp(func_name, "array_set") == 0) &&
                     expr->as.call.arg_count >= 2) {
                /* Detect element type from array argument (first arg) */
                const char *struct_name = NULL;
                ASTNode *array_arg = expr->as.call.args[0];

                Type elem_type = infer_array_element_type(array_arg, env);
                if (elem_type == TYPE_UNKNOWN) {
                    elem_type = TYPE_INT;  /* Fallback */
                }

                /* If this is an array of structs, try to recover the element struct name */
                if (elem_type == TYPE_STRUCT) {
                    if (array_arg->type == AST_CALL && array_arg->as.call.name &&
                        !array_arg->as.call.func_expr) {
                        Function *producer = env_get_function(env, array_arg->as.call.name);
                        if (producer && producer->return_type == TYPE_ARRAY) {
                            struct_name = producer->return_struct_type_name;
                        }
                    }
                    if (array_arg->type == AST_IDENTIFIER) {
                        Symbol *sym = env_get_var_visible_at(env, array_arg->as.identifier, array_arg->line, array_arg->column);
                        if (sym && sym->struct_type_name) {
                            struct_name = sym->struct_type_name;
                        }
                    } else if (array_arg->type == AST_FIELD_ACCESS) {
                        const char *container_struct = get_struct_type_name(array_arg->as.field_access.object, env);
                        if (container_struct) {
                            StructDef *sdef = env_get_struct(env, container_struct);
                            if (sdef && sdef->field_type_names && sdef->field_element_types) {
                                const char *field_name = array_arg->as.field_access.field_name;
                                for (int i = 0; i < sdef->field_count; i++) {
                                    if (strcmp(sdef->field_names[i], field_name) == 0) {
                                        if (sdef->field_types[i] == TYPE_ARRAY &&
                                            sdef->field_element_types[i] == TYPE_STRUCT &&
                                            sdef->field_type_names[i]) {
                                            struct_name = sdef->field_type_names[i];
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (elem_type == TYPE_STRUCT && (strcmp(func_name, "at") == 0 || strcmp(func_name, "array_get") == 0))
                    struct_name = get_struct_type_name(expr, env);

                /* For structs, use dyn_array_get/set_struct with casts */
                if (elem_type == TYPE_STRUCT && struct_name) {
                    if ((strcmp(func_name, "at") == 0 || strcmp(func_name, "array_get") == 0)) {
                        /* Generate: *((nl_StructName*)dyn_array_get_struct(arr, idx)) */
                        emit_formatted(list, "(*((nl_%s*)dyn_array_get_struct(", struct_name);
                        build_expr(list, expr->as.call.args[0], env);  /* array */
                        emit_literal(list, ", ");
                        build_expr(list, expr->as.call.args[1], env);  /* index */
                        emit_literal(list, ")))");
                    } else {
                        /* Generate: dyn_array_set_struct(arr, idx, &value, sizeof(nl_StructName)) */
                        emit_literal(list, "dyn_array_set_struct(");
                        build_expr(list, expr->as.call.args[0], env);  /* array */
                        emit_literal(list, ", ");
                        build_expr(list, expr->as.call.args[1], env);  /* index */
                        emit_literal(list, ", &(");
                        build_expr(list, expr->as.call.args[2], env);  /* value */
                        emit_formatted(list, "), sizeof(nl_%s))", struct_name);
                    }
                } else {
                    /* Map element type to suffix for primitive types */
                    const char *type_suffix = "int";
                    if (elem_type == TYPE_U8) {
                        type_suffix = "u8";
                    }
                    if (elem_type == TYPE_FLOAT) {
                        type_suffix = "float";
                    } else if (elem_type == TYPE_STRING) {
                        type_suffix = "string";
                    } else if (elem_type == TYPE_BOOL) {
                        type_suffix = "bool";
                    } else if (elem_type == TYPE_ARRAY) {
                        type_suffix = "array";  /* For nested arrays */
                    }
                    
                    /* Generate type-specific function name */
                    char func_buf[64];
                    if ((strcmp(func_name, "at") == 0 || strcmp(func_name, "array_get") == 0)) {
                        snprintf(func_buf, sizeof(func_buf), "nl_array_at_%s", type_suffix);
                    } else {
                        snprintf(func_buf, sizeof(func_buf), "nl_array_set_%s", type_suffix);
                    }

                    ASTNode *value_arg = expr->as.call.arg_count >= 3
                        ? expr->as.call.args[2] : NULL;
                    bool empty_nested_replacement =
                        strcmp(func_name, "array_set") == 0 && elem_type == TYPE_ARRAY &&
                        value_arg && value_arg->type == AST_ARRAY_LITERAL &&
                        value_arg->as.array_literal.element_count == 0 &&
                        value_arg->as.array_literal.element_type == TYPE_UNKNOWN;
                    if (empty_nested_replacement) {
                        /* The C AST retains only one declared field-array level.
                         * For [] replacing an existing nested element, preserve
                         * that element's runtime tag and evaluate source operands
                         * once, in their written order. */
                        emit_literal(list, "({ ");
                        unsigned call_id = build_ordered_call_args(
                            list, expr->as.call.args, 2, env, NULL);
                        emit_formatted(list,
                            "nl_array_set_array(__nl_arg_%u_0, __nl_arg_%u_1, "
                            "dyn_array_new(dyn_array_get_elem_type("
                            "nl_array_at_array(__nl_arg_%u_0, __nl_arg_%u_1)))); })",
                            call_id, call_id, call_id, call_id);
                    } else {
                        emit_literal(list, func_buf);
                        emit_literal(list, "(");
                        for (int i = 0; i < expr->as.call.arg_count; i++) {
                            if (i > 0) emit_literal(list, ", ");
                            build_expr(list, expr->as.call.args[i], env);
                        }
                        emit_literal(list, ")");
                    }
                }
            }
            /* Special handling for array_new() - creates new dynamic array with size and initial value */
            else if (strcmp(func_name, "array_new") == 0 && expr->as.call.arg_count == 2) {
                /* array_new(size, default_value) -> DynArray* */
                ASTNode *value_arg = expr->as.call.args[1];
                Type elem_type = check_expression(value_arg, env);

                /* Map element type to ElementType enum */
                const char *elem_type_str = "ELEM_INT";
                if (elem_type == TYPE_U8) {
                    elem_type_str = "ELEM_U8";
                }
                if (elem_type == TYPE_FLOAT) {
                    elem_type_str = "ELEM_FLOAT";
                } else if (elem_type == TYPE_STRING) {
                    elem_type_str = "ELEM_STRING";
                } else if (elem_type == TYPE_BOOL) {
                    elem_type_str = "ELEM_BOOL";
                } else if (elem_type == TYPE_ARRAY) {
                    elem_type_str = "ELEM_ARRAY";
                } else if (elem_type == TYPE_STRUCT) {
                    elem_type_str = "ELEM_STRUCT";
                }
                
                /* I capture both operands once, before checking the count. */
                emit_literal(list, "({ ");
                unsigned call_id = build_ordered_call_args(list, expr->as.call.args, 2, env, NULL);
                emit_formatted(list, "if (__nl_arg_%u_0 < 0) { fputs(\"I require a non-negative array count\\n\", stderr); abort(); } ", call_id);
                emit_formatted(list, "DynArray* _arr = dyn_array_new(%s); ", elem_type_str);
                emit_formatted(list, "for (int64_t _i = 0; _i < __nl_arg_%u_0; _i++) { ", call_id);

                if (elem_type == TYPE_STRUCT) {
                    emit_formatted(list, "dyn_array_push_struct(_arr, &__nl_arg_%u_1, sizeof(__nl_arg_%u_1)); ", call_id, call_id);
                } else {
                    /* For primitives: dyn_array_push_<type>(_arr, value) */
                    const char *type_suffix = "int";
                    if (elem_type == TYPE_U8) {
                        type_suffix = "u8";
                    }
                    if (elem_type == TYPE_FLOAT) {
                        type_suffix = "float";
                    } else if (elem_type == TYPE_STRING) {
                        type_suffix = "string";
                    } else if (elem_type == TYPE_BOOL) {
                        type_suffix = "bool";
                    } else if (elem_type == TYPE_ARRAY) {
                        type_suffix = "array";
                    }
                    
                    emit_formatted(list, "dyn_array_push_%s(_arr, __nl_arg_%u_1); ", type_suffix, call_id);
                }
                
                emit_literal(list, "} _arr; })");
            }
            /* Special handling for array_push() and array_pop() - dynamic array operations */
            else if (strcmp(func_name, "array_push") == 0 &&
                     env_array_push_is_builtin(env, expr->line, expr->column) &&
                     expr->as.call.arg_count == 2) {
                Type elem_type = infer_array_element_type(expr->as.call.args[0], env);
                if (elem_type == TYPE_UNKNOWN)
                    elem_type = check_expression(expr->as.call.args[1], env);
                /* I capture the receiver then the value. A returned record needs
                 * addressable storage, and either operand may have effects. */
                emit_literal(list, "({ ");
                unsigned call_id = build_ordered_call_args(list, expr->as.call.args, 2, env, NULL);
                if (elem_type == TYPE_STRUCT) {
                    emit_formatted(list,
                        "dyn_array_push_struct(__nl_arg_%u_0, &__nl_arg_%u_1, sizeof(__nl_arg_%u_1)); })",
                        call_id, call_id, call_id);
                } else {
                    const char *suffix = "int";
                    if (elem_type == TYPE_U8) suffix = "u8";
                    else if (elem_type == TYPE_FLOAT) suffix = "float";
                    else if (elem_type == TYPE_STRING) suffix = "string";
                    else if (elem_type == TYPE_BOOL) suffix = "bool";
                    else if (elem_type == TYPE_ARRAY) suffix = "array";
                    emit_formatted(list, "dyn_array_push_%s(__nl_arg_%u_0, __nl_arg_%u_1); })",
                                   suffix, call_id, call_id);
                }
            }
            else if (strcmp(func_name, "array_pop") == 0 && expr->as.call.arg_count == 1) {
                /* Detect element type from array argument */
                Type elem_type = TYPE_INT;  /* Default to int */
                const char *struct_name = NULL;
                
                ASTNode *array_arg = expr->as.call.args[0];
                if (array_arg->type == AST_IDENTIFIER) {
                    const char *array_name = array_arg->as.identifier;
                    Symbol *sym = env_get_var_visible_at(env, array_name, array_arg->line, array_arg->column);
                    if (sym && sym->element_type != TYPE_UNKNOWN) {
                        elem_type = sym->element_type;
                        if (elem_type == TYPE_STRUCT && sym->struct_type_name) {
                            struct_name = sym->struct_type_name;
                        }
                    }
                }
                
                /* For structs, use dyn_array_pop_struct */
                if (elem_type == TYPE_STRUCT && struct_name) {
                    /* Generate: ({ bool _s; nl_StructName _v; dyn_array_pop_struct(arr, &_v, sizeof(nl_StructName), &_s); _v; }) */
                    emit_formatted(list, "({ bool _s; nl_%s _v; dyn_array_pop_struct(", struct_name);
                    build_expr(list, expr->as.call.args[0], env);  /* array */
                    emit_formatted(list, ", &_v, sizeof(nl_%s), &_s); _v; })", struct_name);
                } else {
                    /* Map element type to suffix for primitive types */
                    const char *type_suffix = "int";
                    if (elem_type == TYPE_U8) {
                        type_suffix = "u8";
                    }
                    if (elem_type == TYPE_FLOAT) {
                        type_suffix = "float";
                    } else if (elem_type == TYPE_STRING) {
                        type_suffix = "string";
                    } else if (elem_type == TYPE_BOOL) {
                        type_suffix = "bool";
                    } else if (elem_type == TYPE_ARRAY) {
                        type_suffix = "array";  /* For nested arrays */
                    }
                    
                    /* Generate wrapper that handles bool success parameter */
                    char func_buf[128];
                    snprintf(func_buf, sizeof(func_buf), 
                             "({ bool _s; %s _v = dyn_array_pop_%s(", 
                             (elem_type == TYPE_U8 ? "uint8_t" :
                              elem_type == TYPE_FLOAT ? "double" : 
                              elem_type == TYPE_STRING ? "const char*" :
                              elem_type == TYPE_BOOL ? "bool" :
                              elem_type == TYPE_ARRAY ? "DynArray*" : "int64_t"),
                             type_suffix);
                    emit_literal(list, func_buf);
                    build_expr(list, expr->as.call.args[0], env);  /* array */
                    emit_literal(list, ", &_s); _v; })");
                }
            }
            else {
native_array_declared_call: ;
                /* Regular function call */
                const char *mapped_name = func_name;
                /* Use monomorphized name for generic function calls */
                if (expr->as.call.concrete_func_name) {
                    static _Thread_local char generic_buf[512];
                    snprintf(generic_buf, sizeof(generic_buf), "nl_%s", expr->as.call.concrete_func_name);
                    mapped_name = generic_buf;
                } else if (is_generic_list_runtime_fn(func_name, env)) {
                    static _Thread_local char buf[512];
                    snprintf(buf, sizeof(buf), "nl_list_%s", func_name + 5);
                    mapped_name = buf;
                } else {
                    mapped_name = map_function_name(mapped_name, env);
                }

                /* ARC: Check if function returns opaque type requiring manual free
                 * Be very defensive - only apply to extern functions we can safely lookup */
                bool needs_wrapping = false;
                bool needs_unwrap_check = false;
                Function *func_info = NULL;

                /* Only attempt lookup if we have a valid function name and environment
                 * Skip for built-in/special functions that might not be in env */
                if (func_name && env &&
                    strcmp(func_name, "println") != 0 &&
                    strcmp(func_name, "print") != 0 &&
                    !is_generic_list_runtime_fn(func_name, env)) {

                    func_info = env_get_function(env, func_name);

                    /* Only wrap extern functions with explicit cleanup that aren't borrowed refs */
                    if (func_info && func_info->is_extern && func_info->requires_manual_free && !func_info->returns_borrowed) {
                        if (func_info->cleanup_function && func_info->cleanup_function[0] != '\0') {
                            needs_wrapping = true;
                            needs_unwrap_check = true;
                        }
                    }
                }

                Symbol *callee_symbol = env_get_var_visible_at(env, func_name, expr->line, expr->column);
                bool capture_callee = callee_symbol && callee_symbol->type == TYPE_FUNCTION;
                if (capture_callee) {
                    mapped_name = native_local_name(list, effect_capture_name(func_name, expr->line, expr->column));
                    func_info = NULL;
                    needs_wrapping = false;
                    needs_unwrap_check = false;
                }
                /* I retain the callee name before recursive lowering reuses its buffer. */
                char *call_name = strdup(mapped_name);
                if (!call_name) {
                    fprintf(stderr, "I could not retain the native call name.\n");
                    exit(1);
                }
                emit_literal(list, "({ ");
                unsigned call_id = build_ordered_call_args(list, expr->as.call.args,
                                                           expr->as.call.arg_count, env,
                                                           capture_callee ? call_name : NULL);

                bool scalar_list = !capture_callee && native_scalar_list_builtin(func_name, env);
                if (!capture_callee && (scalar_list || is_generic_list_runtime_fn(func_name, env))) {
                    static const char *operations[] = {"with_capacity", "is_empty", "new", "get", "set",
                        "insert", "remove", "push", "pop", "length", "capacity", "clear", "free"};
                    const char *operation = NULL;
                    size_t name_length = strlen(func_name);
                    for (size_t j = 0; j < sizeof(operations) / sizeof(operations[0]); ++j) {
                        size_t n = strlen(operations[j]);
                        if (name_length > n && func_name[name_length - n - 1] == '_' &&
                            !strcmp(func_name + name_length - n, operations[j])) {
                            operation = operations[j]; break;
                        }
                    }
                    if (operation && !strcmp(operation, "with_capacity") && expr->as.call.arg_count == 1) {
                        emit_formatted(list, "__nl_arg_%u_0 = nl_native_list_capacity(__nl_arg_%u_0); ", call_id, call_id);
                    } else if (operation && strcmp(operation, "new") && strcmp(operation, "free") &&
                               expr->as.call.arg_count > 0) {
                        emit_formatted(list, "if (!__nl_arg_%u_0) nl_record_list_fail(); ", call_id);
                        if ((!strcmp(operation, "get") || !strcmp(operation, "set") ||
                             !strcmp(operation, "insert") || !strcmp(operation, "remove")) &&
                            expr->as.call.arg_count > 1) {
                            size_t element_length = name_length - strlen(operation) - 6;
                            emit_formatted(list, "__nl_arg_%u_1 = nl_native_list_index(__nl_arg_%u_1, %slist_%.*s_length(__nl_arg_%u_0), %s); ",
                                call_id, call_id, scalar_list ? "" : "nl_", (int)element_length, func_name + 5,
                                call_id, !strcmp(operation, "insert") ? "true" : "false");
                        }
                    }
                }

                /* If wrapping needed, emit gc_wrap_external( */
                if (needs_wrapping) {
                    emit_literal(list, "gc_wrap_external(");
                }

                if (capture_callee) emit_formatted(list, "__nl_callee_%u", call_id);
                else emit_foreign_reference(list, call_name, func_info);
                free(call_name);
                emit_literal(list, "(");

                /* Emit arguments - unwrap if opaque type */
                for (int i = 0; i < expr->as.call.arg_count; i++) {
                    if (i > 0) emit_literal(list, ", ");

                    bool opaque_parameter =
                        call_parameter_is_opaque(func_info, i, env);

                    /* ARC: Check if parameter is opaque type that needs unwrapping */
                    bool needs_unwrap = false;
                    if (needs_unwrap_check && func_info && i < func_info->param_count && env) {
                        Type param_type = func_info->params[i].type;
                        /* Check if it's an opaque type */
                        if (param_type == TYPE_STRUCT && func_info->params[i].struct_type_name) {
                            OpaqueTypeDef *opaque = env_get_opaque_type(env, func_info->params[i].struct_type_name);
                            if (opaque) {
                                needs_unwrap = true;
                            }
                        } else if (param_type == TYPE_OPAQUE) {
                            needs_unwrap = true;
                        }
                    }

                    if (needs_unwrap) {
                        emit_literal(list, "gc_unwrap(");
                        if (opaque_parameter) emit_literal(list, "(void*)");
                        emit_formatted(list, "__nl_arg_%u_%d", call_id, i);
                        emit_literal(list, ")");
                    } else {
                        if (opaque_parameter) emit_literal(list, "(void*)");
                        emit_formatted(list, "__nl_arg_%u_%d", call_id, i);
                    }
                }

                emit_literal(list, ")");

                /* If wrapping needed, close gc_wrap_external with finalizer */
                if (needs_wrapping) {
                    emit_literal(list, ", ");
                    emit_literal(list, func_info->cleanup_function);
                    emit_literal(list, ")");
                }
                emit_literal(list, "; })");
            }
            break;
        }
        
        case AST_MODULE_QUALIFIED_CALL: {
            /* Module-qualified function call: (Module.function args...) */
            const char *module_alias = expr->as.module_qualified_call.module_alias;
            const char *function_name = expr->as.module_qualified_call.function_name;
            
            /* Generate qualified function name: "Module.function" */
            char *qualified_name = malloc(strlen(module_alias) + strlen(function_name) + 2);
            sprintf(qualified_name, "%s.%s", module_alias, function_name);
            
            /* Map to C function name */
            char *c_name = strdup(map_function_name(qualified_name, env));
            if (!c_name) {
                fprintf(stderr, "I could not retain the qualified native call name.\n");
                exit(1);
            }
            emit_literal(list, "({ ");
            unsigned call_id = build_ordered_call_args(list, expr->as.module_qualified_call.args,
                                                       expr->as.module_qualified_call.arg_count, env, NULL);
            Function *qualified_function = env_get_function(env, qualified_name);
            emit_foreign_reference(list, c_name, qualified_function);
            emit_literal(list, "(");
            
            /* Emit arguments */
            for (int i = 0; i < expr->as.module_qualified_call.arg_count; i++) {
                if (i > 0) emit_literal(list, ", ");
                if (call_parameter_is_opaque(qualified_function, i, env))
                    emit_literal(list, "(void*)");
                emit_formatted(list, "__nl_arg_%u_%d", call_id, i);
            }
            
            emit_literal(list, "); })");
            free(c_name);
            free(qualified_name);
            break;
        }
        
        case AST_IF: {
            /* Ternary operator */
            emit_literal(list, "(");
            build_expr(list, expr->as.if_stmt.condition, env);
            emit_literal(list, " ? ");
            build_expr(list, expr->as.if_stmt.then_branch, env);
            emit_literal(list, " : ");
            build_expr(list, expr->as.if_stmt.else_branch, env);
            emit_literal(list, ")");
            break;
        }

        case AST_COND: {
            /* Transpile cond expression to nested ternary operators */
            emit_literal(list, "(");
            for (int i = 0; i < expr->as.cond_expr.clause_count; i++) {
                if (i > 0) {
                    emit_literal(list, " : (");
                }
                build_expr(list, expr->as.cond_expr.conditions[i], env);
                emit_literal(list, " ? ");
                build_expr(list, expr->as.cond_expr.values[i], env);
            }
            emit_literal(list, " : ");
            build_expr(list, expr->as.cond_expr.else_value, env);
            for (int i = 0; i < expr->as.cond_expr.clause_count; i++) {
                emit_literal(list, ")");
            }
            break;
        }
        
        case AST_TUPLE_LITERAL: {
            int element_count = expr->as.tuple_literal.element_count;
            
            if (element_count == 0) {
                emit_literal(list, "(struct {int _dummy;}){0}");
                break;
            }
            
            const TypeInfo *complete = checked_expression_type_info(expr, env);
            const char *typedef_name = NULL;
            if (!complete || !type_info_tuple_valid(complete) || !g_tuple_registry)
                native_opaque_name_failure();
            for (int i = 0; i < g_tuple_registry->count; ++i)
                if (type_infos_equal(complete, g_tuple_registry->tuples[i])) {
                    typedef_name = g_tuple_registry->typedef_names[i]; break;
                }
            if (!typedef_name) native_opaque_name_failure();
            /* I snapshot each child once before assembling the exact tuple. */
            unsigned id = next_ordered_call_id(env, element_count);
            emit_literal(list, "({ ");
            for (int i = 0; i < element_count; ++i) {
                emit_formatted(list, "__auto_type __nl_arg_%u_%d = ", id, i);
                build_expr(list, expr->as.tuple_literal.elements[i], env);
                emit_literal(list, "; ");
            }
            emit_formatted(list, "(%s){", typedef_name);
            for (int i = 0; i < element_count; ++i) {
                if (i) emit_literal(list, ", ");
                emit_formatted(list, "._%d = __nl_arg_%u_%d", i, id, i);
            }
            emit_literal(list, "}; })");
            break;
        }
        
        case AST_FIELD_ACCESS: {
            /* Check if this is an enum variant (Enum.Variant) */
            if (expr->as.field_access.object->type == AST_IDENTIFIER) {
                const char *object_name = expr->as.field_access.object->as.identifier;
                if (env_get_enum(env, object_name)) {
                    /* It's an enum - generate prefixed variant name */
                    const char *prefixed = get_prefixed_variant_name(object_name, expr->as.field_access.field_name);
                    emit_literal(list, prefixed);
                    break;
                }
            }
            
            /* Regular field access */
            build_expr(list, expr->as.field_access.object, env);
            emit_literal(list, ".");
            emit_literal(list, expr->as.field_access.field_name);
            break;
        }
            
        case AST_TUPLE_INDEX:
            build_expr(list, expr->as.tuple_index.tuple, env);
            emit_formatted(list, "._%d", expr->as.tuple_index.index);
            break;

        case AST_TRY_OP: {
            /* Desugar expr? to:
             *   ({ nl_UNION _nl_try_r = EXPR;
             *      if (_nl_try_r.tag == nl_UNION_TAG_Err) return _nl_try_r;
             *      _nl_try_r.data.Ok.FIELD; })
             */
            const char *union_name = native_opaque_projection(expr->as.try_op.union_type_name);
            const char *ok_field  = expr->as.try_op.ok_field_name;
            if (!union_name) union_name = "Result";
            if (!ok_field)  ok_field  = "val";
            const char *prefixed = get_prefixed_type_name(union_name);
            emit_literal(list, "({ ");
            emit_formatted(list, "%s _nl_try_r = ", prefixed);
            build_expr(list, expr->as.try_op.operand, env);
            emit_formatted(list, "; if (_nl_try_r.tag == nl_%s_TAG_Err) return _nl_try_r;", union_name);
            emit_formatted(list, " _nl_try_r.data.Ok.%s; })", ok_field);
            break;
        }
            
        case AST_STRUCT_LITERAL: {
            /* Struct literal: StructName { field1: val1, field2: val2 } */
            const char *struct_name = expr->as.struct_literal.struct_name;
            int field_count = expr->as.struct_literal.field_count;

            /* Union variant construction is currently parsed as a struct literal with a dotted name:
             *   UnionName.Variant { ... }
             * If UnionName is a known union, emit a tagged union literal instead of a struct literal.
             */
            if (struct_name) {
                const char *dot = strchr(struct_name, '.');
                if (dot) {
                    char union_name_buf[256];
                    size_t union_len = (size_t)(dot - struct_name);
                    if (union_len > 0 && union_len < sizeof(union_name_buf)) {
                        memcpy(union_name_buf, struct_name, union_len);
                        union_name_buf[union_len] = '\0';
                        const char *variant_name = dot + 1;

                        if (env_get_union(env, union_name_buf)) {
                            const char *prefixed_union;
                            char monomorphized_name[256];
                            bool is_generic = false;

                            /* Generic unions: if we're returning Result<int, string>, emit nl_Result_int_string */
                            if (g_current_function &&
                                g_current_function->as.function.return_type == TYPE_UNION &&
                                g_current_function->as.function.return_type_info &&
                                g_current_function->as.function.return_type_info->generic_name &&
                                g_current_function->as.function.return_type_info->type_param_count > 0 &&
                                strcmp(union_name_buf, g_current_function->as.function.return_type_info->generic_name) == 0) {
                                if (!build_monomorphized_name_from_typeinfo_iter(monomorphized_name, sizeof(monomorphized_name),
                                                                                g_current_function->as.function.return_type_info)) {
                                    snprintf(monomorphized_name, sizeof(monomorphized_name), "%s", union_name_buf);
                                }
                                prefixed_union = get_prefixed_type_name(monomorphized_name);
                                is_generic = true;
                            } else {
                                prefixed_union = get_prefixed_type_name(union_name_buf);
                            }

                            int variant_idx = env_get_union_variant_index(env, union_name_buf, variant_name);
                            if (variant_idx < 0) {
                                fprintf(stderr, "Error: Unknown variant '%s' in union '%s'\n", variant_name, union_name_buf);
                                emit_literal(list, "/* ERROR: unknown variant */");
                                break;
                            }

                            if (is_generic) {
                                emit_formatted(list, "(%s){ .tag = nl_%s_TAG_%s", prefixed_union, monomorphized_name, variant_name);
                            } else {
                                emit_formatted(list, "(%s){ .tag = nl_%s_TAG_%s", prefixed_union, union_name_buf, variant_name);
                            }

                            if (field_count > 0) {
                                emit_formatted(list, ", .data.%s = {", variant_name);
                                for (int i = 0; i < field_count; i++) {
                                    if (i > 0) emit_literal(list, ", ");
                                    emit_formatted(list, ".%s = ", expr->as.struct_literal.field_names[i]);
                                    build_expr(list, expr->as.struct_literal.field_values[i], env);
                                }
                                emit_literal(list, "}");
                            }

                            emit_literal(list, "}");
                            break;
                        }
                    }
                }
            }

            /* Look up struct definition to propagate types to empty array fields */
            StructDef *sdef = env_get_struct(env, struct_name);
            
            /* Propagate element types to empty array literals in struct fields */
            if (sdef) {
                for (int i = 0; i < field_count; i++) {
                    ASTNode *field_value = expr->as.struct_literal.field_values[i];
                    const char *field_name = expr->as.struct_literal.field_names[i];
                    
                    /* Check if this field value is an empty array literal */
                    if (field_value && 
                        field_value->type == AST_ARRAY_LITERAL &&
                        field_value->as.array_literal.element_count == 0 &&
                        field_value->as.array_literal.element_type == TYPE_UNKNOWN) {
                        
                        /* Find matching field in struct definition */
                        for (int j = 0; j < sdef->field_count; j++) {
                            if (strcmp(field_name, sdef->field_names[j]) == 0) {
                                /* Found matching field - check if it's an array type */
                                if (sdef->field_types[j] == TYPE_ARRAY) {
                                    /* Propagate the element type from struct definition */
                                    field_value->as.array_literal.element_type = sdef->field_element_types[j];
                                }
                                break;
                            }
                        }
                    }
                }
            }
            
            ASTNode *spread = expr->as.struct_literal.spread_source;
            char spread_name[64];
            if (spread && sdef) {
                unsigned index = 0;
                do {
                    snprintf(spread_name, sizeof(spread_name), "__nano_spread_%u", index++);
                } while (env_get_var(env, spread_name) || env_get_function(env, spread_name));
                emit_formatted(list, "({ __auto_type %s = ", spread_name);
                build_expr(list, spread, env);
                emit_formatted(list, "; (void)%s; ", spread_name);
            }
            emit_formatted(list, "(%s){", get_prefixed_type_name(struct_name));
            for (int i = 0; i < field_count; i++) {
                if (i > 0) emit_literal(list, ", ");
                emit_formatted(list, ".%s = ", expr->as.struct_literal.field_names[i]);
                build_expr(list, expr->as.struct_literal.field_values[i], env);
            }
            if (spread && sdef) {
                int emitted = field_count;
                for (int i = 0; i < sdef->field_count; i++) {
                    bool overridden = false;
                    for (int j = 0; j < field_count; j++)
                        if (!strcmp(sdef->field_names[i], expr->as.struct_literal.field_names[j]))
                            overridden = true;
                    if (overridden) continue;
                    if (emitted++) emit_literal(list, ", ");
                    emit_formatted(list, ".%s = %s.%s", sdef->field_names[i], spread_name, sdef->field_names[i]);
                }
            }
            emit_literal(list, "}");
            if (spread && sdef) emit_literal(list, "; })");
            break;
        }
        
        case AST_UNION_CONSTRUCT: {
            /* Union construction: UnionName.Variant { field1: val1, field2: val2 } */
            const char *union_name = expr->as.union_construct.union_name;
            const char *variant_name = expr->as.union_construct.variant_name;
            
            /* Check if we're in a function returning a generic union */
            const char *prefixed_union;
            char monomorphized_name[256];
            bool is_generic = false;
            
            TypeInfo *context = expr->as.union_construct.type_info;
            if (!context && g_current_function &&
                g_current_function->as.function.return_type == TYPE_UNION)
                context = g_current_function->as.function.return_type_info;
            if (context && context->generic_name && context->type_param_count > 0 &&
                strcmp(union_name, context->generic_name) == 0) {
                if (!build_monomorphized_name_from_typeinfo_iter(monomorphized_name, sizeof(monomorphized_name), context)) {
                    fprintf(stderr, "I cannot represent this concrete union constructor\n");
                    exit(1);
                }
                prefixed_union = get_prefixed_type_name(monomorphized_name);
                is_generic = true;
            } else prefixed_union = get_prefixed_type_name(union_name);

            /* Get variant index */
            int variant_idx = env_get_union_variant_index(env, union_name, variant_name);
            if (variant_idx < 0) {
                fprintf(stderr, "Error: Unknown variant '%s' in union '%s'\n", variant_name, union_name);
                emit_literal(list, "/* ERROR: unknown variant */");
                break;
            }
            
            /* Generate union construction: (UnionName){ .tag = TAG, .data.variant = {...} } */
            if (is_generic) {
                /* For generic unions, use monomorphized tag name */
                emit_formatted(list, "(%s){ .tag = nl_%s_TAG_%s", 
                              prefixed_union, monomorphized_name, variant_name);
            } else {
                /* For non-generic unions, use base tag name */
                emit_formatted(list, "(%s){ .tag = nl_%s_TAG_%s", 
                              prefixed_union, union_name, variant_name);
            }
            
            if (expr->as.union_construct.field_count > 0) {
                emit_formatted(list, ", .data.%s = {", variant_name);
                for (int i = 0; i < expr->as.union_construct.field_count; i++) {
                    if (i > 0) emit_literal(list, ", ");
                    emit_formatted(list, ".%s = ", expr->as.union_construct.field_names[i]);
                    build_expr(list, expr->as.union_construct.field_values[i], env);
                }
                emit_literal(list, "}");
            }
            emit_literal(list, "}");
            break;
        }
        
        case AST_ARRAY_LITERAL: {
            if (native_opaque_array_literal(list, expr, env)) break;
            /* Array literal: [1, 2, 3] - Use dynarray_literal_* helper functions */
            int count = expr->as.array_literal.element_count;
            
            if (count == 0) {
                /* Empty array - need to determine type */
                Type elem_type = expr->as.array_literal.element_type;
                
                if (elem_type == TYPE_UNKNOWN) {
                    /* Default to int for empty arrays without type info */
                    elem_type = TYPE_INT;
                }
                
                /* Generate call to appropriate constructor */
                if (elem_type == TYPE_INT || elem_type == TYPE_ENUM) {
                    emit_literal(list, "dynarray_literal_int(0)");
                } else if (elem_type == TYPE_U8) {
                    emit_literal(list, "dynarray_literal_u8(0)");
                } else if (elem_type == TYPE_FLOAT) {
                    emit_literal(list, "dynarray_literal_float(0)");
                } else if (elem_type == TYPE_ARRAY) {
                    /* Nested array */
                    emit_literal(list, "dyn_array_new(ELEM_ARRAY)");
                } else if (elem_type == TYPE_STRING) {
                    emit_literal(list, "dyn_array_new(ELEM_STRING)");
                } else if (elem_type == TYPE_BOOL) {
                    emit_literal(list, "dyn_array_new(ELEM_BOOL)");
                } else if (elem_type == TYPE_STRUCT) {
                    /* Array of structs - size will be set on first push */
                    emit_literal(list, "dyn_array_new(ELEM_STRUCT)");
                } else {
                    /* Fallback for other types */
                    emit_literal(list, "dyn_array_new(ELEM_INT)");
                }
            } else {
                /* Non-empty array - prefer stored element type (possibly set from type context) */
                Type elem_type = expr->as.array_literal.element_type;
                if (elem_type == TYPE_UNKNOWN) {
                    elem_type = check_expression(expr->as.array_literal.elements[0], env);
                }
                
                /* Generate call to appropriate helper function */
                if (elem_type == TYPE_INT || elem_type == TYPE_ENUM) {
                    emit_formatted(list, "dynarray_literal_int(%d", count);
                    for (int i = 0; i < count; i++) {
                        emit_literal(list, ", (int64_t)(");
                        build_expr(list, expr->as.array_literal.elements[i], env);
                        emit_literal(list, ")");
                    }
                    emit_literal(list, ")");
                } else if (elem_type == TYPE_U8) {
                    emit_formatted(list, "dynarray_literal_u8(%d", count);
                    for (int i = 0; i < count; i++) {
                        emit_literal(list, ", ");
                        build_expr(list, expr->as.array_literal.elements[i], env);
                    }
                    emit_literal(list, ")");
                } else if (elem_type == TYPE_FLOAT) {
                    emit_formatted(list, "dynarray_literal_float(%d", count);
                    for (int i = 0; i < count; i++) {
                        emit_literal(list, ", ");
                        build_expr(list, expr->as.array_literal.elements[i], env);
                    }
                    emit_literal(list, ")");
                } else if (elem_type == TYPE_STRING) {
                    emit_formatted(list, "dynarray_literal_string(%d", count);
                    for (int i = 0; i < count; i++) {
                        emit_literal(list, ", ");
                        build_expr(list, expr->as.array_literal.elements[i], env);
                    }
                    emit_literal(list, ")");
                } else if (elem_type == TYPE_BOOL) {
                    emit_formatted(list, "dynarray_literal_bool(%d", count);
                    for (int i = 0; i < count; i++) {
                        emit_literal(list, ", ");
                        build_expr(list, expr->as.array_literal.elements[i], env);
                    }
                    emit_literal(list, ")");
                } else if (elem_type == TYPE_STRUCT) {
                    /* I snapshot each record in source order before copying its
                     * value into a dynamic record array. */
                    unsigned array_id = next_nested_array_id(env);
                    emit_literal(list, "({ ");
                    unsigned values = build_ordered_call_args(list,
                        expr->as.array_literal.elements, count, env, NULL);
                    emit_formatted(list,
                        "DynArray* __nl_nested_array_%u = dyn_array_new(ELEM_STRUCT); ", array_id);
                    for (int i = 0; i < count; i++) {
                        emit_formatted(list,
                            "dyn_array_push_struct(__nl_nested_array_%u, &__nl_arg_%u_%d, sizeof(__nl_arg_%u_%d)); ",
                            array_id, values, i, values, i);
                    }
                    emit_formatted(list, "__nl_nested_array_%u; })", array_id);
                } else if (elem_type == TYPE_ARRAY) {
                    /* Nested literals contain DynArray pointers, so their outer
                     * value must remain a DynArray rather than decay from a C
                     * compound array to DynArray**. Sequential pushes preserve
                     * the language's left-to-right child evaluation. */
                    unsigned array_id = next_nested_array_id(env);
                    emit_formatted(list,
                        "({ DynArray* __nl_nested_array_%u = dyn_array_new(ELEM_ARRAY); ",
                        array_id);
                    for (int i = 0; i < count; i++) {
                        emit_formatted(list, "dyn_array_push_array(__nl_nested_array_%u, ",
                                       array_id);
                        build_expr(list, expr->as.array_literal.elements[i], env);
                        emit_literal(list, "); ");
                    }
                    emit_formatted(list, "__nl_nested_array_%u; })", array_id);
                } else {
                    /* For other types, fallback to old behavior */
                    const char *c_type = type_to_c(elem_type);
                    emit_formatted(list, "(%s[]){", c_type);
                    for (int i = 0; i < count; i++) {
                        if (i > 0) emit_literal(list, ", ");
                        build_expr(list, expr->as.array_literal.elements[i], env);
                    }
                    emit_literal(list, "}");
                }
            }
            break;
        }
        
        case AST_MATCH: {
            /* Match expression: match opt { Some(s) => s.value, None(n) => 0 } */

            /* The checker decides the exact match domain. In particular, a
             * wildcard-only int match has no INT: arm from which to infer it. */
            bool is_int_match_expr = match_uses_checked_int_domain(expr);

            /* Detect if any arm has a guard — if so, use if-else chain instead of switch */
            int has_any_guard_expr = 0;
            if (expr->as.match_expr.guard_exprs) {
                for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                    if (expr->as.match_expr.guard_exprs[i]) {
                        has_any_guard_expr = 1;
                        break;
                    }
                }
            }

            const char *union_c_name = is_int_match_expr
                ? "" : checked_match_union_name(expr);

            /* Look up union definition to check variant field counts.
             * For generic unions, union_c_name is monomorphized; use the base name for lookup. */
            const char *base_union_name = union_c_name;
            char base_buf[256];
            UnionDef *udef = env_get_union(env, base_union_name);
            if (!udef) {
                const char *us = strchr(union_c_name, '_');
                if (us) {
                    size_t n = (size_t)(us - union_c_name);
                    if (n < sizeof(base_buf)) {
                        memcpy(base_buf, union_c_name, n);
                        base_buf[n] = '\0';
                        base_union_name = base_buf;
                        udef = env_get_union(env, base_union_name);
                    }
                }
            }

            /* I lower the checked match value, not the enclosing return type. */
            Type result_type = expr->as.match_expr.result_type;
            const char *out_type = type_to_c(result_type);
            if (result_type == TYPE_VOID || result_type == TYPE_UNKNOWN) out_type = "int64_t";
            char out_type_buf[256];
            if (result_type == TYPE_STRUCT || result_type == TYPE_UNION || result_type == TYPE_ENUM) {
                const char *name = get_struct_type_name(expr, env);
                if (name) {
                    snprintf(out_type_buf, sizeof(out_type_buf), "%s", get_prefixed_type_name(name));
                    out_type = out_type_buf;
                }
            }

            const char *prefixed_union = get_prefixed_type_name(union_c_name);

            if (has_any_guard_expr) {
                emit_literal(list, "({ ");
                if (is_int_match_expr) {
                    emit_literal(list, "int64_t _m = ");
                    build_expr(list, expr->as.match_expr.expr, env);
                    emit_literal(list, "; ");
                } else {
                    emit_literal(list, prefixed_union);
                    emit_literal(list, " _m = ");
                    build_expr(list, expr->as.match_expr.expr, env);
                    emit_literal(list, "; ");
                }
                emit_literal(list, out_type);
                emit_literal(list, " _out = {0}; int _matched = 0; ");

                for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                    const char *variant_name = expr->as.match_expr.pattern_variants[i];
                    const char *binding_name = expr->as.match_expr.pattern_bindings[i];
                    ASTNode *arm_body = expr->as.match_expr.arm_bodies[i];
                    restore_native_match_binding(env, expr, i, udef ? udef->name : NULL);
                    ASTNode *guard = expr->as.match_expr.guard_exprs ? expr->as.match_expr.guard_exprs[i] : NULL;

                    if (strcmp(variant_name, "_") == 0) {
                        emit_literal(list, "if (!_matched) { ");
                    } else if (strncmp(variant_name, "INT:", 4) == 0) {
                        emit_literal(list, "if (!_matched && _m == ");
                        emit_literal(list, variant_name + 4);
                        emit_literal(list, ") { ");
                    } else if (strncmp(variant_name, "OR:", 3) == 0) {
                        /* Or-pattern expr (guard mode): emit _m.tag == A || _m.tag == B */
                        char or_copy_g[512]; strncpy(or_copy_g, variant_name + 3, sizeof(or_copy_g)-1); or_copy_g[sizeof(or_copy_g)-1]='\0';
                        char *alts_g[64]; int n_alts_g = 0;
                        char *tok_og = strtok(or_copy_g, ":"); while (tok_og && n_alts_g < 64) { alts_g[n_alts_g++] = tok_og; tok_og = strtok(NULL, ":"); }
                        emit_literal(list, "if (!_matched && (");
                        for (int oi = 0; oi < n_alts_g; oi++) {
                            if (oi > 0) emit_literal(list, " || ");
                            emit_literal(list, "_m.tag == nl_"); emit_literal(list, union_c_name);
                            emit_literal(list, "_TAG_"); emit_literal(list, alts_g[oi]);
                        }
                        emit_literal(list, ")) { ");
                    } else {
                        emit_literal(list, "if (!_matched && _m.tag == nl_");
                        emit_literal(list, union_c_name);
                        emit_literal(list, "_TAG_");
                        emit_literal(list, variant_name);
                        emit_literal(list, ") { ");

                        /* Declare binding */
                        int variant_field_count = 0;
                        if (udef) {
                            for (int v = 0; v < udef->variant_count; v++) {
                                if (strcmp(udef->variant_names[v], variant_name) == 0) {
                                    variant_field_count = udef->variant_field_counts[v];
                                    break;
                                }
                            }
                        }
                        if (variant_field_count > 0 && binding_name && strcmp(binding_name, "_") != 0) {
                            emit_literal(list, "nl_");
                            emit_literal(list, union_c_name);
                            emit_literal(list, "_");
                            emit_literal(list, variant_name);
                            emit_literal(list, " ");
                            emit_literal(list, binding_name);
                            emit_literal(list, " = _m.data.");
                            emit_literal(list, variant_name);
                            emit_literal(list, "; ");
                        } else {
                            emit_literal(list, "(void)_m.data.");
                            emit_literal(list, variant_name);
                            emit_literal(list, "; ");
                        }
                    }

                    if (guard) {
                        emit_literal(list, "if (");
                        build_expr(list, guard, env);
                        emit_literal(list, ") { _matched = 1; ");
                    } else {
                        emit_literal(list, "_matched = 1; ");
                    }

                    /* Emit arm body */
                        build_match_arm_value(list, arm_body, env);

                    if (guard) {
                        emit_literal(list, "} ");  /* close if (guard) */
                    }
                    emit_literal(list, "} ");  /* close if (!_matched && ...) */
                }

                emit_literal(list, "_out; })");
            } else {
                /* No guards: use original switch-based approach */
                emit_literal(list, "({ ");
                if (is_int_match_expr) {
                    emit_literal(list, "int64_t _m = ");
                    build_expr(list, expr->as.match_expr.expr, env);
                    emit_literal(list, "; ");
                    emit_literal(list, out_type);
                    emit_literal(list, " _out = {0}; switch (_m) { ");
                } else {
                    emit_literal(list, prefixed_union);
                    emit_literal(list, " _m = ");
                    build_expr(list, expr->as.match_expr.expr, env);
                    emit_literal(list, "; ");
                    emit_literal(list, out_type);
                    emit_literal(list, " _out = {0}; switch (_m.tag) { ");
                }

                /* Generate each match arm */
                for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                    const char *variant_name = expr->as.match_expr.pattern_variants[i];
                    const char *binding_name = expr->as.match_expr.pattern_bindings[i];
                    ASTNode *arm_body = expr->as.match_expr.arm_bodies[i];
                    restore_native_match_binding(env, expr, i, udef ? udef->name : NULL);

                    if (strcmp(variant_name, "_") == 0) {
                        /* Wildcard arm: _ => expr  emits default: */
                        emit_literal(list, "default: { ");

                        build_match_arm_value(list, arm_body, env);

                        emit_literal(list, "break; } ");
                    } else if (strncmp(variant_name, "INT:", 4) == 0) {
                        /* Integer literal arm: case N: { */
                        emit_literal(list, "case ");
                        emit_literal(list, variant_name + 4);  /* skip "INT:" prefix */
                        emit_literal(list, ": { ");

                        build_match_arm_value(list, arm_body, env);

                        emit_literal(list, "break; } ");
                    } else if (strncmp(variant_name, "OR:", 3) == 0) {
                        /* Or-pattern expr (no guard): emit fall-through cases */
                        char or_copy[512]; strncpy(or_copy, variant_name + 3, sizeof(or_copy)-1); or_copy[sizeof(or_copy)-1]='\0';
                        char *alts_expr[64]; int n_alts_expr = 0;
                        char *tok_oe = strtok(or_copy, ":"); while (tok_oe && n_alts_expr < 64) { alts_expr[n_alts_expr++] = tok_oe; tok_oe = strtok(NULL, ":"); }
                        for (int oi = 0; oi < n_alts_expr - 1; oi++) {
                            emit_literal(list, "case nl_"); emit_literal(list, union_c_name);
                            emit_literal(list, "_TAG_"); emit_literal(list, alts_expr[oi]); emit_literal(list, ": ");
                        }
                        emit_literal(list, "case nl_"); emit_literal(list, union_c_name);
                        emit_literal(list, "_TAG_"); emit_literal(list, alts_expr[n_alts_expr-1]); emit_literal(list, ": { ");
                        build_match_arm_value(list, arm_body, env);
                        emit_literal(list, "break; } ");
                    } else {
                        /* case nl_UnionName_TAG_Variant: { */
                        emit_literal(list, "case nl_");
                        emit_literal(list, union_c_name);
                        emit_literal(list, "_TAG_");
                        emit_literal(list, variant_name);
                        emit_literal(list, ": { ");

                        /* Find variant index to check field count */
                        int variant_field_count = 0;
                        if (udef) {
                            for (int v = 0; v < udef->variant_count; v++) {
                                if (strcmp(udef->variant_names[v], variant_name) == 0) {
                                    variant_field_count = udef->variant_field_counts[v];
                                    break;
                                }
                            }
                        }

                        /* Declare binding only if variant has fields */
                        if (variant_field_count > 0 && binding_name && strcmp(binding_name, "_") != 0) {
                            emit_literal(list, "nl_");
                            emit_literal(list, union_c_name);
                            emit_literal(list, "_");
                            emit_literal(list, variant_name);
                            emit_literal(list, " ");
                            emit_literal(list, binding_name);
                            emit_literal(list, " = _m.data.");
                            emit_literal(list, variant_name);
                            emit_literal(list, "; ");
                        } else {
                            emit_literal(list, "(void)_m.data.");
                            emit_literal(list, variant_name);
                            emit_literal(list, "; ");
                        }

                        /* Handle arm body */
                        build_match_arm_value(list, arm_body, env);

                        emit_literal(list, "break; } ");
                    }
                }

                /* Close switch and compound expression */
                emit_literal(list, "} _out; })");
            }
            break;
        }
        case AST_HANDLE_EXPR: {
            build_effect_handle(list, expr, env);
            break;
        }

        case AST_EFFECT_DECL:
            /* Effect declarations are type-level only — no C code emitted */
            break;


        case AST_AWAIT:
            /* In synchronous evaluation mode, await is transparent — emit the inner expr */
            build_expr(list, expr->as.await_expr.expr, env);
            break;

        case AST_EFFECT_HANDLER: {
            ASTNode normalized = *expr;
            normalized.type = AST_HANDLE_EXPR;
            normalized.as.handle_expr.body = expr->as.effect_handler.body;
            normalized.as.handle_expr.effect_name = expr->as.effect_handler.effect_name;
            normalized.as.handle_expr.handler_count = expr->as.effect_handler.handler_count;
            normalized.as.handle_expr.handler_op_names = expr->as.effect_handler.handler_op_names;
            normalized.as.handle_expr.handler_bodies = expr->as.effect_handler.handler_bodies;
            int count = expr->as.effect_handler.handler_count;
            normalized.as.handle_expr.handler_param_counts = calloc((size_t)count, sizeof(int));
            normalized.as.handle_expr.handler_param_names = calloc((size_t)count, sizeof(char **));
            for (int i = 0; i < count; ++i) {
                normalized.as.handle_expr.handler_param_counts[i] = expr->as.effect_handler.handler_param_names[i] ? 1 : 0;
                normalized.as.handle_expr.handler_param_names[i] = &expr->as.effect_handler.handler_param_names[i];
            }
            build_effect_handle(list, &normalized, env);
            free(normalized.as.handle_expr.handler_param_counts);
            free(normalized.as.handle_expr.handler_param_names);
            break;
        }

        case AST_EFFECT_OP: {
            EffectOp *op = effect_get_op(env_get_effect(env, expr->as.effect_op.effect_name), expr->as.effect_op.op_name);
            emit_literal(list, "({ ");
            for (int i = 0; i < expr->as.effect_op.arg_count; ++i) {
                emit_formatted(list, "%s _effect_arg%d = ", effect_c_type(op->params[i].type, op->params[i].struct_type_name, env), i);
                build_expr(list, expr->as.effect_op.args[i], env);
                emit_literal(list, "; ");
            }
            emit_formatted(list, "nl_perform_%s_%s(", expr->as.effect_op.effect_name, expr->as.effect_op.op_name);
            for (int i = 0; i < expr->as.effect_op.arg_count; ++i)
                emit_formatted(list, "%s_effect_arg%d", i ? ", " : "", i);
            emit_literal(list, "); })");
            break;
        }

        default:
            emit_formatted(list, "/* unsupported expr type %d */", expr->type);
            break;
    }
}

/* ============================================================================
 * PASS 1: BUILD WORK ITEMS (Statement Transpiler)
 * ============================================================================ */

/* Loop exits invalidate my explicit vectorization promise. */
static bool loop_body_has_exit(const ASTNode *node) {
    if (!node) return false;
    switch (node->type) {
        case AST_RETURN: case AST_BREAK: case AST_CONTINUE: return true;
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                if (loop_body_has_exit(node->as.block.statements[i])) return true;
            return false;
        case AST_UNSAFE_BLOCK:
            for (int i = 0; i < node->as.unsafe_block.count; i++)
                if (loop_body_has_exit(node->as.unsafe_block.statements[i])) return true;
            return false;
        case AST_IF:
            return loop_body_has_exit(node->as.if_stmt.then_branch)
                || loop_body_has_exit(node->as.if_stmt.else_branch);
        case AST_WHILE: return loop_body_has_exit(node->as.while_stmt.body);
        case AST_FOR: return loop_body_has_exit(node->as.for_stmt.body);
        case AST_MATCH:
            for (int i = 0; i < node->as.match_expr.arm_count; i++)
                if (loop_body_has_exit(node->as.match_expr.arm_bodies[i])) return true;
            return false;
        case AST_COND:
            for (int i = 0; i < node->as.cond_expr.clause_count; i++)
                if (loop_body_has_exit(node->as.cond_expr.values[i])) return true;
            return loop_body_has_exit(node->as.cond_expr.else_value);
        default: return false;
    }
}

static void build_stmt(WorkList *list, ScopeStack *scopes, ASTNode *stmt, int indent, Environment *env,
                       FunctionTypeRegistry *fn_registry) {
    if (!stmt) return;

    switch (stmt->type) {
        case AST_BLOCK:
            emit_indent_item(list, indent);
            emit_literal(list, "{\n");

            /* --profile: inject timing guard at start of function body blocks (indent==0) */
            if (indent == 0 && g_profile_mode && g_profile_func_name) {
                emit_formatted(list,
                    "    __attribute__((cleanup(_nl_prof_guard_exit))) _NlProfGuard _nl_prof_g ="
                    " {_nl_prof_get_entry(\"%s\"), {0, 0}};\n"
                    "    if (_nl_prof_g.entry) clock_gettime(CLOCK_MONOTONIC, &_nl_prof_g.start);\n",
                    g_profile_func_name);
                g_profile_mode = false;  /* consumed -- don't inject into nested blocks */
            }

            /* --trace: inject a function-call trace guard at the start of function
             * body blocks (indent==0). The guard shares the diagnostics hook, so a
             * disabled trace only reads a cached int and does no other work. */
            if (indent == 0 && g_trace_mode && g_trace_func_name) {
                emit_formatted(list,
                    "    __attribute__((cleanup(_nl_trace_guard_exit))) _NlTraceGuard _nl_trace_g ="
                    " {_nl_diag_trace ? \"%s\" : (const char *)0};\n"
                    "    if (_nl_trace_g.name) _nl_trace_enter(_nl_trace_g.name);\n",
                    g_trace_func_name);
                g_trace_mode = false;  /* consumed -- don't inject into nested blocks */
            }

            /* Push new scope for this block */
            scope_stack_push(scopes, list);

            for (int i = 0; i < stmt->as.block.count; i++) {
                ASTNode *child = stmt->as.block.statements[i];
                if (g_source_file_for_line_directives && child && child->line > 0) {
                    emit_formatted(list, "#line %d %s\n",
                                   child->line, module_c_literal(g_source_file_for_line_directives));
                }
                build_stmt(list, scopes, child, indent + 1, env, fn_registry);
            }

            /* Emit cleanup for variables in this scope */
            scope_emit_cleanup(scopes, list, indent + 1);

            /* Pop scope */
            scope_stack_pop(scopes, list);

            emit_indent_item(list, indent);
            emit_literal(list, "}\n");
            break;

        case AST_EFFECT_DECL:
            /* Effect declarations are type-level only — no C code emitted */
            break;

        case AST_HANDLE_EXPR:
            /* Handle expression used as statement: evaluate body */
            emit_indent_item(list, indent);
            build_expr(list, stmt, env);
            emit_literal(list, ";\n");
            break;

        case AST_UNSAFE_BLOCK:
            /* Unsafe blocks transpile to regular C blocks */
            emit_indent_item(list, indent);
            emit_literal(list, "/* unsafe */ {\n");
            scope_stack_push(scopes, list);
            for (int i = 0; i < stmt->as.unsafe_block.count; i++) {
                build_stmt(list, scopes, stmt->as.unsafe_block.statements[i], indent + 1, env, fn_registry);
            }
            scope_emit_cleanup(scopes, list, indent + 1);
            scope_stack_pop(scopes, list);
            emit_indent_item(list, indent);
            emit_literal(list, "}\n");
            break;


        case AST_PAR_BLOCK: {
            int *order = stmt->as.par_block.is_flow ? passive_binding_order(stmt) : NULL;
            if (stmt->as.par_block.is_flow && !order) {
                fprintf(stderr, "I cannot establish a valid flow execution order.\n");
                exit(1);
            }
            /* I export bindings at the enclosing scope in the stable serial order. */
            for (int i = 0; i < stmt->as.par_block.count; i++)
                build_stmt(list, scopes, stmt->as.par_block.bindings[order ? order[i] : i], indent, env, fn_registry);
            free(order);
            break;
        }

        case AST_PAR_LET:
            /* par-let: emit bindings as sequential C declarations, then the body expression.
             * Parallelism is the runtime's responsibility; this is the safe sequential fallback. */
            emit_indent_item(list, indent);
            emit_literal(list, "/* par-let begin (independent bindings follow) */\n");
            for (int i = 0; i < stmt->as.par_let.count; i++) {
                emit_indent_item(list, indent);
                const char *binding_name = !strcmp(stmt->as.par_let.names[i], "_")
                    ? new_underscore_name(list, env) : stmt->as.par_let.names[i];
                emit_literal(list, "__auto_type ");
                emit_literal(list, binding_name);
                emit_literal(list, " = ");
                build_expr(list, stmt->as.par_let.values[i], env);
                emit_literal(list, "; /* par-let binding */\n");
                if (!strcmp(stmt->as.par_let.names[i], "_")) list->underscore_name = binding_name;
            }
            emit_indent_item(list, indent);
            emit_literal(list, "/* par-let end */\n");
            emit_indent_item(list, indent);
            build_expr(list, stmt->as.par_let.body, env);
            emit_literal(list, ";\n");
            break;
        case AST_MATCH: {
            /* Use the checked domain rather than reconstructing it from arms.
             * This preserves wildcard-only integer matches. */
            bool is_int_match_stmt = match_uses_checked_int_domain(stmt);

            /* Detect if any arm has a guard — if so, use if-else chain instead of switch */
            int has_any_guard_stmt = 0;
            if (stmt->as.match_expr.guard_exprs) {
                for (int _gi = 0; _gi < stmt->as.match_expr.arm_count; _gi++) {
                    if (stmt->as.match_expr.guard_exprs[_gi]) {
                        has_any_guard_stmt = 1;
                        break;
                    }
                }
            }

            const char *union_c_name = is_int_match_stmt
                ? "" : checked_match_union_name(stmt);

            /* For generic unions, typechecker stores the monomorphized name (e.g. Result_int_string).
             * We still need the base name (e.g. Result) to look up the union definition. */
            const char *base_union_name = union_c_name;
            char base_buf[256];
            UnionDef *udef = env_get_union(env, base_union_name);
            if (!udef) {
                const char *us = strchr(union_c_name, '_');
                if (us) {
                    size_t n = (size_t)(us - union_c_name);
                    if (n < sizeof(base_buf)) {
                        memcpy(base_buf, union_c_name, n);
                        base_buf[n] = '\0';
                        base_union_name = base_buf;
                        udef = env_get_union(env, base_union_name);
                    }
                }
            }

            const char *prefixed_union = get_prefixed_type_name(union_c_name);

            emit_indent_item(list, indent);
            emit_literal(list, "{\n");

            emit_indent_item(list, indent + 1);
            if (is_int_match_stmt) {
                emit_literal(list, "int64_t _m = ");
                build_expr(list, stmt->as.match_expr.expr, env);
                emit_literal(list, ";\n");
            } else {
                emit_literal(list, prefixed_union);
                emit_literal(list, " _m = ");
                build_expr(list, stmt->as.match_expr.expr, env);
                emit_literal(list, ";\n");
            }

            if (has_any_guard_stmt) {
                /* Guard mode: emit sequential if (!_matched && ...) blocks */
                emit_indent_item(list, indent + 1);
                emit_literal(list, "int _matched = 0;\n");

                for (int i = 0; i < stmt->as.match_expr.arm_count; i++) {
                    const char *variant_name = stmt->as.match_expr.pattern_variants[i];
                    const char *binding_name = stmt->as.match_expr.pattern_bindings[i];
                    ASTNode *arm_body = stmt->as.match_expr.arm_bodies[i];
                    restore_native_match_binding(env, stmt, i, udef ? udef->name : NULL);
                    ASTNode *guard = stmt->as.match_expr.guard_exprs ? stmt->as.match_expr.guard_exprs[i] : NULL;

                    emit_indent_item(list, indent + 1);
                    if (strcmp(variant_name, "_") == 0) {
                        emit_literal(list, "if (!_matched) {\n");
                    } else if (strncmp(variant_name, "INT:", 4) == 0) {
                        emit_literal(list, "if (!_matched && _m == ");
                        emit_literal(list, variant_name + 4);
                        emit_literal(list, ") {\n");
                    } else if (strncmp(variant_name, "OR:", 3) == 0) {
                        /* Or-pattern stmt (guard mode): emit _m.tag == A || _m.tag == B */
                        char or_copy_sg[512]; strncpy(or_copy_sg, variant_name + 3, sizeof(or_copy_sg)-1); or_copy_sg[sizeof(or_copy_sg)-1]='\0';
                        char *alts_sg[64]; int n_alts_sg = 0;
                        char *tok_osg = strtok(or_copy_sg, ":"); while (tok_osg && n_alts_sg < 64) { alts_sg[n_alts_sg++] = tok_osg; tok_osg = strtok(NULL, ":"); }
                        emit_literal(list, "if (!_matched && (");
                        for (int oi = 0; oi < n_alts_sg; oi++) {
                            if (oi > 0) emit_literal(list, " || ");
                            emit_literal(list, "_m.tag == nl_"); emit_literal(list, union_c_name);
                            emit_literal(list, "_TAG_"); emit_literal(list, alts_sg[oi]);
                        }
                        emit_literal(list, ")) {\n");
                    } else {
                        emit_literal(list, "if (!_matched && _m.tag == nl_");
                        emit_literal(list, union_c_name);
                        emit_literal(list, "_TAG_");
                        emit_literal(list, variant_name);
                        emit_literal(list, ") {\n");

                        /* Declare binding */
                        int variant_field_count = 0;
                        if (udef) {
                            for (int v = 0; v < udef->variant_count; v++) {
                                if (strcmp(udef->variant_names[v], variant_name) == 0) {
                                    variant_field_count = udef->variant_field_counts[v];
                                    break;
                                }
                            }
                        }
                        if (variant_field_count > 0 && binding_name && strcmp(binding_name, "_") != 0) {
                            emit_indent_item(list, indent + 2);
                            emit_literal(list, "nl_");
                            emit_literal(list, union_c_name);
                            emit_literal(list, "_");
                            emit_literal(list, variant_name);
                            emit_literal(list, " ");
                            emit_literal(list, binding_name);
                            emit_literal(list, " = _m.data.");
                            emit_literal(list, variant_name);
                            emit_literal(list, ";\n");
                        } else {
                            emit_indent_item(list, indent + 2);
                            emit_literal(list, "(void)_m.data.");
                            emit_literal(list, variant_name);
                            emit_literal(list, ";\n");
                        }
                    }

                    if (guard) {
                        emit_indent_item(list, indent + 2);
                        emit_literal(list, "if (");
                        build_expr(list, guard, env);
                        emit_literal(list, ") {\n");
                        emit_indent_item(list, indent + 3);
                        emit_literal(list, "_matched = 1;\n");
                        build_scoped_statements(list, scopes, arm_body, indent + 3, env, fn_registry);
                        emit_indent_item(list, indent + 2);
                        emit_literal(list, "}\n");
                    } else {
                        emit_indent_item(list, indent + 2);
                        emit_literal(list, "_matched = 1;\n");
                        build_scoped_statements(list, scopes, arm_body, indent + 2, env, fn_registry);
                    }

                    emit_indent_item(list, indent + 1);
                    emit_literal(list, "}\n");
                }
            } else {
                /* No guards: use original switch-based approach */
                emit_indent_item(list, indent + 1);
                if (is_int_match_stmt) {
                    emit_literal(list, "switch (_m) {\n");
                } else {
                    emit_literal(list, "switch (_m.tag) {\n");
                }

                int has_wildcard_arm = 0;
                for (int i = 0; i < stmt->as.match_expr.arm_count; i++) {
                    if (strcmp(stmt->as.match_expr.pattern_variants[i], "_") == 0) {
                        has_wildcard_arm = 1;
                        break;
                    }
                }

                for (int i = 0; i < stmt->as.match_expr.arm_count; i++) {
                    const char *variant_name = stmt->as.match_expr.pattern_variants[i];
                    const char *binding_name = stmt->as.match_expr.pattern_bindings[i];
                    ASTNode *arm_body = stmt->as.match_expr.arm_bodies[i];
                    restore_native_match_binding(env, stmt, i, udef ? udef->name : NULL);

                    emit_indent_item(list, indent + 2);
                    if (strcmp(variant_name, "_") == 0) {
                        /* Wildcard arm: _ => { body }  emits default: */
                        emit_literal(list, "default: {\n");

                        build_scoped_statements(list, scopes, arm_body, indent + 3, env, fn_registry);

                        emit_indent_item(list, indent + 3);
                        emit_literal(list, "break;\n");
                        emit_indent_item(list, indent + 2);
                        emit_literal(list, "}\n");
                    } else if (strncmp(variant_name, "INT:", 4) == 0) {
                        /* Integer literal arm: case N: { body } */
                        emit_literal(list, "case ");
                        emit_literal(list, variant_name + 4);  /* skip "INT:" prefix */
                        emit_literal(list, ": {\n");

                        build_scoped_statements(list, scopes, arm_body, indent + 3, env, fn_registry);

                        emit_indent_item(list, indent + 3);
                        emit_literal(list, "break;\n");
                        emit_indent_item(list, indent + 2);
                        emit_literal(list, "}\n");
                    } else if (strncmp(variant_name, "OR:", 3) == 0) {
                        /* Or-pattern stmt (no guard): fall-through cases */
                        char or_copy_s[512]; strncpy(or_copy_s, variant_name + 3, sizeof(or_copy_s)-1); or_copy_s[sizeof(or_copy_s)-1]='\0';
                        char *alts_s[64]; int n_alts_s = 0;
                        char *tok_os = strtok(or_copy_s, ":"); while (tok_os && n_alts_s < 64) { alts_s[n_alts_s++] = tok_os; tok_os = strtok(NULL, ":"); }
                        for (int oi = 0; oi < n_alts_s - 1; oi++) {
                            emit_literal(list, "case nl_"); emit_literal(list, union_c_name);
                            emit_literal(list, "_TAG_"); emit_literal(list, alts_s[oi]); emit_literal(list, ":\n");
                            emit_indent_item(list, indent + 2);
                        }
                        emit_literal(list, "case nl_"); emit_literal(list, union_c_name);
                        emit_literal(list, "_TAG_"); emit_literal(list, alts_s[n_alts_s-1]); emit_literal(list, ": {\n");
                        build_scoped_statements(list, scopes, arm_body, indent + 3, env, fn_registry);
                        emit_indent_item(list, indent + 3); emit_literal(list, "break;\n");
                        emit_indent_item(list, indent + 2); emit_literal(list, "}\n");
                    } else {
                        emit_literal(list, "case nl_");
                        emit_literal(list, union_c_name);
                        emit_literal(list, "_TAG_");
                        emit_literal(list, variant_name);
                        emit_literal(list, ": {\n");

                        int variant_field_count = 0;
                        if (udef) {
                            for (int v = 0; v < udef->variant_count; v++) {
                                if (strcmp(udef->variant_names[v], variant_name) == 0) {
                                    variant_field_count = udef->variant_field_counts[v];
                                    break;
                                }
                            }
                        }

                        if (variant_field_count > 0) {
                            if (binding_name && strcmp(binding_name, "_") != 0) {
                                emit_indent_item(list, indent + 3);
                                emit_literal(list, "nl_");
                                emit_literal(list, union_c_name);
                                emit_literal(list, "_");
                                emit_literal(list, variant_name);
                                emit_literal(list, " ");
                                emit_literal(list, binding_name);
                                emit_literal(list, " = _m.data.");
                                emit_literal(list, variant_name);
                                emit_literal(list, ";\n");
                            } else {
                                emit_indent_item(list, indent + 3);
                                emit_literal(list, "(void)_m.data.");
                                emit_literal(list, variant_name);
                                emit_literal(list, ";\n");
                            }
                        } else {
                            emit_indent_item(list, indent + 3);
                            emit_literal(list, "(void)_m.data.");
                            emit_literal(list, variant_name);
                            emit_literal(list, ";\n");
                        }

                        build_scoped_statements(list, scopes, arm_body, indent + 3, env, fn_registry);

                        emit_indent_item(list, indent + 3);
                        emit_literal(list, "break;\n");
                        emit_indent_item(list, indent + 2);
                        emit_literal(list, "}\n");
                    }
                }

                /* Add default: __builtin_unreachable() only when no wildcard arm.
                 * This tells the C compiler all variants are covered and avoids
                 * "control reaches end of non-void function" warnings. */
                if (!has_wildcard_arm) {
                    emit_indent_item(list, indent + 2);
                    emit_literal(list, "default: __builtin_unreachable();\n");
                }
                emit_indent_item(list, indent + 1);
                emit_literal(list, "}\n");
            }
            emit_indent_item(list, indent);
            emit_literal(list, "}\n");
            break;
        }
            
        case AST_RETURN:
            if (native_effect_program && !in_effect_handler && stmt->as.return_stmt.value && stmt->as.return_stmt.value->type == AST_IDENTIFIER) {
                const char *name = native_local_name(list, stmt->as.return_stmt.value->as.identifier);
                for (int si = scopes->count - 1; si >= 0; --si) {
                    Scope *scope = &scopes->scopes[si];
                    for (int vi = scope->count - 1; vi >= 0; --vi) {
                        if (!strcmp(scope->vars[vi].name, name)) {
                            emit_formatted(list, "_nl_gc_%s.owned = false;\n", name);
                            si = -1; break;
                        }
                    }
                }
            }
            if (in_effect_handler) {
                if (stmt->as.return_stmt.value) {
                    emit_formatted(list, "*(%s *)_frame->lexical_result = ", effect_c_type(effect_lexical_return, effect_lexical_name, env));
                    build_expr(list, stmt->as.return_stmt.value, env);
                    emit_literal(list, "; ");
                }
                if (effect_lexical_return == TYPE_OPAQUE || effect_lexical_return == TYPE_HASHMAP ||
                    (effect_lexical_return == TYPE_STRUCT && effect_lexical_name && env_get_opaque_type(env, effect_lexical_name)))
                    emit_literal(list, "nl_effect_preserve_return(_frame->lexical_result); ");
                emit_literal(list, "nl_effect_escape(_frame);\n");
                break;
            }
            emit_indent_item(list, indent);
            emit_literal(list, "return");
            if (stmt->as.return_stmt.value) {
                emit_literal(list, " ");
                build_expr(list, stmt->as.return_stmt.value, env);
            }
            emit_literal(list, ";\n");
            break;

        case AST_BREAK:
            emit_indent_item(list, indent);
            emit_literal(list, "break;\n");
            break;

        case AST_CONTINUE:
            emit_indent_item(list, indent);
            emit_literal(list, "continue;\n");
            break;
            
        case AST_LET: {
            const char *binding_name = !strcmp(stmt->as.let.name, "_")
                ? new_underscore_name(list, env) : stmt->as.let.name;
            emit_indent_item(list, indent);
            
            /* A checked empty pattern over a name has no fields or runtime work. */
            if (stmt->as.let.is_destructure && !stmt->as.let.destructure_count &&
                stmt->as.let.value && stmt->as.let.value->type == AST_IDENTIFIER) {
                emit_literal(list, "(void)0;\n");
            }
            /* I infer the stored variant typedef after checking its complete identity. */
            else if (stmt->as.let.is_destructure) {
                emit_formatted(list, "__auto_type %s = ", binding_name);
                build_expr(list, stmt->as.let.value, env);
                emit_literal(list, ";\n");
            }
            /* I retain evaluation, but a void binding has no C storage. */
            else if (stmt->as.let.var_type == TYPE_VOID) {
                emit_literal(list, "(void)(");
                if (stmt->as.let.value) build_expr(list, stmt->as.let.value, env);
                else emit_literal(list, "0");
                emit_literal(list, ");\n");
            }
            /* Handle tuple types - use __auto_type to infer from RHS */
            else if (stmt->as.let.var_type == TYPE_TUPLE) {
                emit_formatted(list, "__auto_type %s = ", binding_name);
                build_expr(list, stmt->as.let.value, env);
                emit_literal(list, ";\n");
            }
            /* Handle function types - use typedef from registry */
            else if (stmt->as.let.var_type == TYPE_FUNCTION && stmt->as.let.fn_sig && fn_registry) {
                const char *typedef_name = register_function_signature(fn_registry, stmt->as.let.fn_sig);
                emit_formatted(list, "%s %s", typedef_name, binding_name);
                
                if (stmt->as.let.value) {
                    emit_literal(list, " = ");
                    build_expr(list, stmt->as.let.value, env);
                }
                emit_literal(list, ";\n");
            }
            /* Handle List<T> (generic lists) */
            else if (stmt->as.let.var_type == TYPE_LIST_GENERIC && stmt->as.let.type_name) {
                emit_formatted(list, "List_%s* %s", stmt->as.let.type_name, binding_name);
                if (stmt->as.let.value) {
                    emit_literal(list, " = ");
                    build_expr(list, stmt->as.let.value, env);
                }
                emit_literal(list, ";\n");
            }
            /* Handle HashMap<K,V> */
            else if (stmt->as.let.var_type == TYPE_HASHMAP && stmt->as.let.type_info &&
                     stmt->as.let.type_info->generic_name && stmt->as.let.type_info->type_param_count == 2) {
                char monomorphized_name[256];
                if (!build_monomorphized_name_from_typeinfo_iter(monomorphized_name, sizeof(monomorphized_name),
                                                                stmt->as.let.type_info)) {
                    snprintf(monomorphized_name, sizeof(monomorphized_name), "HashMap");
                }

                emit_formatted(list, "%s* %s", monomorphized_name, binding_name);
                if (stmt->as.let.value) {
                    emit_literal(list, " = ");
                    build_expr(list, stmt->as.let.value, env);
                }
                emit_literal(list, ";\n");
            }
            /* Handle struct/enum/union types that need prefixing */
            else if (stmt->as.let.var_type == TYPE_STRUCT ||
                     stmt->as.let.var_type == TYPE_ENUM ||
                     stmt->as.let.var_type == TYPE_UNION) {
                /* Check if this is a generic union instantiation */
                if (stmt->as.let.var_type == TYPE_UNION && stmt->as.let.type_info && 
                    stmt->as.let.type_info->generic_name && stmt->as.let.type_info->type_param_count > 0) {
                    /* Build monomorphized name: Result<int, string> -> Result_int_string */
                    char monomorphized_name[256];
                    if (!build_monomorphized_name_from_typeinfo_iter(monomorphized_name, sizeof(monomorphized_name),
                                                                    stmt->as.let.type_info)) {
                        snprintf(monomorphized_name, sizeof(monomorphized_name), "%s", stmt->as.let.type_info->generic_name);
                    }
                    
                    const char *prefixed = get_prefixed_type_name(monomorphized_name);
                    emit_formatted(list, "%s %s", prefixed, binding_name);
                    
                    if (stmt->as.let.value) {
                        emit_literal(list, " = ");
                        
                        /* Special handling for union construction with generic unions */
                        if (stmt->as.let.value->type == AST_UNION_CONSTRUCT) {
                            ASTNode *uc = stmt->as.let.value;
                            const char *variant_name = uc->as.union_construct.variant_name;
                            int variant_idx = env_get_union_variant_index(env, 
                                uc->as.union_construct.union_name, variant_name);
                            (void)variant_idx;  /* Retrieved for validation but not used in codegen */
                            
                            /* Generate union construction with monomorphized type name */
                            emit_formatted(list, "(%s){ .tag = nl_%s_TAG_%s", 
                                          prefixed, monomorphized_name, variant_name);
                            
                            if (uc->as.union_construct.field_count > 0) {
                                emit_formatted(list, ", .data.%s = {", variant_name);
                                for (int i = 0; i < uc->as.union_construct.field_count; i++) {
                                    if (i > 0) emit_literal(list, ", ");
                                    emit_formatted(list, ".%s = ", uc->as.union_construct.field_names[i]);
                                    build_expr(list, uc->as.union_construct.field_values[i], env);
                                }
                                emit_literal(list, "}");
                            }
                            emit_literal(list, "}");
                        } else if (stmt->as.let.value->type == AST_STRUCT_LITERAL) {
                            /* UnionName.Variant { ... } is parsed as a dotted struct literal; handle it here so
                             * generic unions use the monomorphized C type/tag.
                             */
                            ASTNode *sl = stmt->as.let.value;
                            const char *sl_name = sl->as.struct_literal.struct_name;
                            const char *dot = sl_name ? strchr(sl_name, '.') : NULL;
                            if (dot) {
                                char union_name_buf[256];
                                size_t union_len = (size_t)(dot - sl_name);
                                if (union_len > 0 && union_len < sizeof(union_name_buf)) {
                                    memcpy(union_name_buf, sl_name, union_len);
                                    union_name_buf[union_len] = '\0';
                                    const char *variant_name = dot + 1;

                                    if (stmt->as.let.type_info && stmt->as.let.type_info->generic_name &&
                                        strcmp(union_name_buf, stmt->as.let.type_info->generic_name) == 0 &&
                                        env_get_union(env, union_name_buf)) {

                                        int variant_idx = env_get_union_variant_index(env, union_name_buf, variant_name);
                                        (void)variant_idx;

                                        emit_formatted(list, "(%s){ .tag = nl_%s_TAG_%s", 
                                                      prefixed, monomorphized_name, variant_name);

                                        int field_count = sl->as.struct_literal.field_count;
                                        if (field_count > 0) {
                                            emit_formatted(list, ", .data.%s = {", variant_name);
                                            for (int i = 0; i < field_count; i++) {
                                                if (i > 0) emit_literal(list, ", ");
                                                emit_formatted(list, ".%s = ", sl->as.struct_literal.field_names[i]);
                                                build_expr(list, sl->as.struct_literal.field_values[i], env);
                                            }
                                            emit_literal(list, "}");
                                        }

                                        emit_literal(list, "}");
                                    } else {
                                        build_expr(list, stmt->as.let.value, env);
                                    }
                                } else {
                                    build_expr(list, stmt->as.let.value, env);
                                }
                            } else {
                                build_expr(list, stmt->as.let.value, env);
                            }
                        } else {
                            build_expr(list, stmt->as.let.value, env);
                        }
                    }
                    emit_literal(list, ";\n");
                }
                else {
                    /* Check if it's an enum */
                    bool is_enum = false;
                    const char *type_name = stmt->as.let.type_name;
                    
                    if (!type_name && stmt->as.let.value) {
                        /* Try to infer from value */
                        if (stmt->as.let.value->type == AST_FIELD_ACCESS &&
                            stmt->as.let.value->as.field_access.object->type == AST_IDENTIFIER) {
                            type_name = stmt->as.let.value->as.field_access.object->as.identifier;
                            if (env_get_enum(env, type_name)) {
                                is_enum = true;
                            }
                        }
                    }
                    
                    if (is_enum || (type_name && env_get_enum(env, type_name))) {
                        /* Enum type */
                        const char *prefixed = get_prefixed_type_name(type_name);
                        emit_formatted(list, "%s %s = ", prefixed, binding_name);
                        build_expr(list, stmt->as.let.value, env);
                        emit_literal(list, ";\n");
                    } else {
                        /* Check if this is an opaque type */
                        if (!type_name && stmt->as.let.value) {
                            type_name = get_struct_type_name(stmt->as.let.value, env);
                        }
                        
                        OpaqueTypeDef *opaque = NULL;
                        if (type_name) {
                            opaque = env_get_opaque_type(env, type_name);
                        }
                        
                        if (opaque) {
                            /* Opaque types are stored as void* */
                            emit_formatted(list, "void* %s", binding_name);
                        } else if (type_name) {
                            /* Regular struct type */
                            const char *prefixed = get_prefixed_type_name(type_name);
                            emit_formatted(list, "%s %s", prefixed, binding_name);
                        } else {
                            emit_formatted(list, "void* %s", binding_name);
                        }
                        
                        if (stmt->as.let.value) {
                            emit_literal(list, " = ");
                            build_expr(list, stmt->as.let.value, env);
                        }
                        emit_literal(list, ";\n");
                    }
                }
            }
            else {
                /* Regular types (int, float, string, bool, etc.) */
                const char *c_type = type_to_c(stmt->as.let.var_type);
                emit_formatted(list, "%s %s", c_type, binding_name);

                if (stmt->as.let.value) {
                    /* Propagate element type from let to empty array literals */
                    if (stmt->as.let.var_type == TYPE_ARRAY &&
                        stmt->as.let.value->type == AST_ARRAY_LITERAL &&
                        stmt->as.let.value->as.array_literal.element_count == 0 &&
                        stmt->as.let.value->as.array_literal.element_type == TYPE_UNKNOWN &&
                        stmt->as.let.element_type != TYPE_UNKNOWN) {
                        stmt->as.let.value->as.array_literal.element_type = stmt->as.let.element_type;
                    }
                    emit_literal(list, " = ");
                    build_expr(list, stmt->as.let.value, env);
                }
                emit_literal(list, ";\n");
            }
            
            /* The initializer above sees the previous lexical binding. */
            if (!strcmp(stmt->as.let.name, "_")) list->underscore_name = binding_name;

            /* Register in environment */
            Symbol *checked_binding = NULL;
            for (int i = env->symbol_count - 1; i >= 0; --i) {
                Symbol *candidate = &env->symbols[i];
                if (candidate->name && strcmp(candidate->name, stmt->as.let.name) == 0 &&
                    candidate->def_line == stmt->line && candidate->def_column == stmt->column &&
                    (!candidate->def_file || !env->current_file ||
                     strcmp(candidate->def_file, env->current_file) == 0)) {
                    checked_binding = candidate;
                    break;
                }
            }
            /* Copy borrowed checker metadata before definition can move symbols. */
            bool have_checked = checked_binding != NULL;
            Symbol checked = have_checked ? *checked_binding : (Symbol){0};
            int scope_end_line = checked_binding ? checked_binding->scope_end_line : 0;
            int scope_end_column = checked_binding ? checked_binding->scope_end_column : 0;
            const char *nominal = stmt->as.let.type_name ? stmt->as.let.type_name :
                (checked_binding ? checked_binding->struct_type_name : NULL);
            char *owned_nominal = nominal ? strdup(nominal) : NULL;
            if (nominal && !owned_nominal) {
                fprintf(stderr, "I cannot allocate declaration type metadata.\n");
                exit(1);
            }
            env_define_var_with_type_info(env, stmt->as.let.name, stmt->as.let.var_type,
                                         stmt->as.let.element_type,
                                         stmt->as.let.type_info,
                                         stmt->as.let.is_mut, create_void());
            Symbol *emitted_binding = &env->symbols[env->symbol_count - 1];
            emitted_binding->def_line = stmt->line;
            emitted_binding->def_column = stmt->column;
            if (native_effect_program) emitted_binding->def_file = g_source_file_for_line_directives;
            if (have_checked) {
                emitted_binding->nominal_owner = checked.nominal_owner;
                emitted_binding->callable_owner = checked.callable_owner;
                emitted_binding->inferred_nominal = checked.inferred_nominal;
                emitted_binding->checker_nominal_view = checked.checker_nominal_view;
            }
            emitted_binding->scope_end_line = scope_end_line;
            emitted_binding->scope_end_column = scope_end_column;
            free(emitted_binding->struct_type_name);
            emitted_binding->struct_type_name = owned_nominal;

            /* Track variable for GC cleanup if needed */
            scope_add_var(scopes, binding_name, stmt->as.let.var_type, stmt->as.let.type_name, env);
            if (native_effect_program && scopes->count > 0) {
                Scope *scope = &scopes->scopes[scopes->count - 1];
                if (scope->count > 0 && !strcmp(scope->vars[scope->count - 1].name, binding_name)) {
                    emit_formatted(list, "NlEffectGC _nl_gc_%s __attribute__((cleanup(nl_effect_gc_pop))) = {nl_effect_gc_top, (void **)&%s, true, true}; nl_effect_gc_top = &_nl_gc_%s;\n", binding_name, binding_name, binding_name);
                }
            }

            break;
        }
        
        case AST_SET:
            if (stmt->as.set.field_name) {
                emit_indent_item(list, indent);
                emit_literal(list, native_local_name(list, stmt->as.set.name));
                emit_literal(list, "->");
                emit_literal(list, stmt->as.set.field_name);
                emit_literal(list, " = ");
                build_expr(list, stmt->as.set.value, env);
                emit_literal(list, ";\n");
                break;
            }
            if (env_get_var_visible_at(env, stmt->as.set.name, stmt->line, stmt->column) &&
                env_get_var_visible_at(env, stmt->as.set.name, stmt->line, stmt->column)->type == TYPE_VOID) {
                emit_indent_item(list, indent);
                emit_literal(list, "(void)(");
                build_expr(list, stmt->as.set.value, env);
                emit_literal(list, ");\n");
                break;
            }
            /* Detect self-assignment (set x x) and skip generating code */
            if (stmt->as.set.value &&
                stmt->as.set.value->type == AST_IDENTIFIER &&
                strcmp(stmt->as.set.value->as.identifier, stmt->as.set.name) == 0) {
                /* Self-assignment detected: skip code generation to avoid -Wself-assign warning */
                emit_indent_item(list, indent);
                emit_literal(list, "/* self-assignment elided: ");
                emit_literal(list, native_local_name(list, stmt->as.set.name));
                emit_literal(list, " = ");
                emit_literal(list, native_local_name(list, stmt->as.set.name));
                emit_literal(list, " */\n");
                break;
            }
            
            emit_indent_item(list, indent);
            emit_literal(list, native_local_name(list, in_effect_handler ? effect_capture_name(stmt->as.set.name, stmt->line, stmt->column) : stmt->as.set.name));
            emit_literal(list, " = ");
            
            /* Special handling for empty array literals: infer type from target variable */
            if (stmt->as.set.value &&
                stmt->as.set.value->type == AST_ARRAY_LITERAL &&
                stmt->as.set.value->as.array_literal.element_count == 0 &&
                stmt->as.set.value->as.array_literal.element_type == TYPE_UNKNOWN) {
                
                /* Look up the target variable's element type */
                Symbol *sym = env_get_var_visible_at(env, stmt->as.set.name, stmt->line, stmt->column);
                if (sym && sym->element_type != TYPE_UNKNOWN) {
                    /* Propagate the element type to the empty array literal */
                    stmt->as.set.value->as.array_literal.element_type = sym->element_type;
                }
            }
            
            build_expr(list, stmt->as.set.value, env);
            emit_literal(list, ";\n");
            break;
            
        case AST_WHILE:
            emit_indent_item(list, indent);
            emit_literal(list, "while (");
            build_expr(list, stmt->as.while_stmt.condition, env);
            emit_literal(list, ") ");
            build_stmt(list, scopes, stmt->as.while_stmt.body, indent, env, fn_registry);
            break;
            
        case AST_FOR: {
            /* I evaluate both bounds once, in order, before binding the loop variable. */
            const char *var = stmt->as.for_stmt.var_name;
            ASTNode *range = stmt->as.for_stmt.range_expr;
            if (range && range->type == AST_CALL &&
                range->as.call.name && strcmp(range->as.call.name, "range") == 0 &&
                range->as.call.arg_count == 2) {
                emit_indent_item(list, indent);
                emit_literal(list, "{\n");
                emit_indent_item(list, indent + 1);
                unsigned bounds_id = build_ordered_call_args(list, range->as.call.args, 2, env, NULL);
                emit_literal(list, "\n");
                emit_indent_item(list, indent + 1);
                emit_formatted(list, "for (int64_t %s = __nl_arg_%u_0; %s < __nl_arg_%u_1; %s++) ",
                               var, bounds_id, var, bounds_id, var);
                build_loop_body(list, scopes, stmt->as.for_stmt.body, stmt, indent + 1, env, fn_registry);
                emit_indent_item(list, indent);
                emit_literal(list, "}\n");
            } else {
                /* Check if range is a dyn_array (array<T>) variable */
                bool is_dyn_array = false;
                Type dyn_elem_type = TYPE_INT;

                if (range && range->type == AST_IDENTIFIER) {
                    Symbol *arr_sym = env_get_var_visible_at(env, range->as.identifier, range->line, range->column);
                    if (arr_sym && arr_sym->type == TYPE_ARRAY) {
                        is_dyn_array = true;
                        dyn_elem_type = arr_sym->element_type;
                    }
                }

                if (is_dyn_array) {
                    /* for x in array_var =>
                     * { DynArray* __nl_arr = array_var;
                     *   int64_t __nl_len = dyn_array_length(__nl_arr);
                     *   for (int64_t __nl_idx = 0; __nl_idx < __nl_len; __nl_idx++) {
                     *     <elem_type> var = dyn_array_get_<T>(__nl_arr, __nl_idx);
                     *     body_stmts; } }
                     */
                    const char *get_fn;
                    const char *c_elem_type;
                    if (dyn_elem_type == TYPE_FLOAT) {
                        get_fn = "dyn_array_get_float";
                        c_elem_type = "double";
                    } else if (dyn_elem_type == TYPE_STRING) {
                        get_fn = "dyn_array_get_string";
                        c_elem_type = "const char*";
                    } else if (dyn_elem_type == TYPE_BOOL) {
                        get_fn = "dyn_array_get_bool";
                        c_elem_type = "int64_t";
                    } else {
                        get_fn = "dyn_array_get_int";
                        c_elem_type = "int64_t";
                    }
                    emit_indent_item(list, indent);
                    emit_literal(list, "{\n");
                    emit_indent_item(list, indent + 1);
                    emit_literal(list, "DynArray* __nl_arr = ");
                    build_expr(list, range, env);
                    emit_literal(list, ";\n");
                    emit_indent_item(list, indent + 1);
                    emit_literal(list, "int64_t __nl_len = dyn_array_length(__nl_arr);\n");
                    /* Vectorization hints for numeric element types */
                    if ((dyn_elem_type == TYPE_INT || dyn_elem_type == TYPE_FLOAT) &&
                        !loop_body_has_exit(stmt->as.for_stmt.body)) {
                        emit_indent_item(list, indent + 1);
                        emit_literal(list, "#if defined(__GNUC__) && !defined(__clang__)\n");
                        emit_indent_item(list, indent + 1);
                        emit_literal(list, "#pragma GCC ivdep\n");
                        emit_indent_item(list, indent + 1);
                        emit_literal(list, "#endif\n");
                        emit_indent_item(list, indent + 1);
                        emit_literal(list, "#ifdef __clang__\n");
                        emit_indent_item(list, indent + 1);
                        emit_literal(list, "#pragma clang loop vectorize(enable) interleave(enable)\n");
                        emit_indent_item(list, indent + 1);
                        emit_literal(list, "#endif\n");
                    }
                    emit_indent_item(list, indent + 1);
                    emit_literal(list, "for (int64_t __nl_idx = 0; __nl_idx < __nl_len; __nl_idx++) {\n");
                    emit_indent_item(list, indent + 2);
                    const TypeInfo *array_info = checked_expression_type_info(range, env);
                    const TypeInfo *element_info = array_info && array_info->base_type == TYPE_ARRAY ? array_info->element_type : NULL;
                    if (type_info_exact_array_element(element_info) && native_array_struct_value(element_info)) {
                        char *complete = native_array_c_type(element_info, env);
                        native_array_load(list, element_info, complete, "__nl_arr", "__nl_idx", var);
                        emit_literal(list, "\n"); free(complete);
                    } else emit_formatted(list, "%s %s = %s(__nl_arr, __nl_idx);\n",
                                   c_elem_type, var, get_fn);
                    /* Emit body statements */
                    ASTNode *dyn_body = stmt->as.for_stmt.body;
                    scope_stack_push(scopes, list);
                    if (!reestablish_checked_loop_binding(env, stmt)) {
                        fprintf(stderr, "I require checked loop binding metadata at line %d.\n", stmt->line);
                        exit(1);
                    }
                    if (!strcmp(var, "_")) list->underscore_name = NULL;
                    if (dyn_body && dyn_body->type == AST_BLOCK) {
                        for (int bi = 0; bi < dyn_body->as.block.count; bi++) {
                            build_stmt(list, scopes, dyn_body->as.block.statements[bi], indent + 2, env, fn_registry);
                        }
                    }
                    scope_stack_pop(scopes, list);
                    emit_indent_item(list, indent + 1);
                    emit_literal(list, "}\n");
                    emit_indent_item(list, indent);
                    emit_literal(list, "}\n");
                } else {

                /* Check if range is a List variable */
                bool is_list = false;
                Type list_elem_type = TYPE_INT;
                const char *list_struct_name = NULL;

                if (range && range->type == AST_IDENTIFIER) {
                    Symbol *list_sym = env_get_var_visible_at(env, range->as.identifier, range->line, range->column);
                    if (list_sym) {
                        if (list_sym->type == TYPE_LIST_INT || list_sym->type == TYPE_LIST_TOKEN) {
                            is_list = true;
                            list_elem_type = TYPE_INT;
                        } else if (list_sym->type == TYPE_LIST_STRING) {
                            is_list = true;
                            list_elem_type = TYPE_STRING;
                        } else if (list_sym->type == TYPE_LIST_GENERIC) {
                            is_list = true;
                            list_elem_type = list_sym->element_type;
                            if (list_elem_type == TYPE_STRUCT && list_sym->struct_type_name) {
                                list_struct_name = list_sym->struct_type_name;
                            }
                        }
                    }
                }

                if (is_list) {
                    /* for x in list =>
                     * { <ListType>* __nl_lst = list; int64_t __nl_len = <list>_length(__nl_lst);
                     *   for (int64_t __nl_idx = 0; __nl_idx < __nl_len; __nl_idx++) {
                     *     <elem_type> var = <list>_get(__nl_lst, __nl_idx);
                     *     body_stmts; } }
                     */
                    /* Determine list C type and get/length function names */
                    const char *list_c_type = "List_int*";
                    const char *list_len_fn = "list_int_length";
                    char elem_decl_buf[256];

                    if (list_elem_type == TYPE_STRING) {
                        list_c_type = "List_string*";
                        list_len_fn = "list_string_length";
                        snprintf(elem_decl_buf, sizeof(elem_decl_buf),
                                 "const char* %s = list_string_get(__nl_lst, __nl_idx);\n", var);
                    } else if (list_elem_type == TYPE_STRUCT && list_struct_name) {
                        /* Generic struct list: List_TypeName* */
                        static char lc_buf[64]; static char ll_buf[64];
                        snprintf(lc_buf, sizeof(lc_buf), "List_%s*", list_struct_name);
                        snprintf(ll_buf, sizeof(ll_buf), "List_%s_length", list_struct_name);
                        list_c_type = lc_buf; list_len_fn = ll_buf;
                        snprintf(elem_decl_buf, sizeof(elem_decl_buf),
                                 "nl_%s %s = *List_%s_get(__nl_lst, __nl_idx);\n",
                                 list_struct_name, var, list_struct_name);
                    } else {
                        /* INT (default) */
                        snprintf(elem_decl_buf, sizeof(elem_decl_buf),
                                 "int64_t %s = list_int_get(__nl_lst, __nl_idx);\n", var);
                    }

                    emit_indent_item(list, indent);
                    emit_literal(list, "{\n");
                    emit_indent_item(list, indent + 1);
                    emit_formatted(list, "%s __nl_lst = ", list_c_type);
                    build_expr(list, range, env);
                    emit_literal(list, ";\n");
                    emit_indent_item(list, indent + 1);
                    emit_formatted(list, "int64_t __nl_len = %s(__nl_lst);\n", list_len_fn);
                    emit_indent_item(list, indent + 1);
                    emit_literal(list, "for (int64_t __nl_idx = 0; __nl_idx < __nl_len; __nl_idx++) {\n");

                    emit_indent_item(list, indent + 2);
                    emit_literal(list, elem_decl_buf);

                    /* Emit body statements */
                    ASTNode *body = stmt->as.for_stmt.body;
                    scope_stack_push(scopes, list);
                    if (!reestablish_checked_loop_binding(env, stmt)) {
                        fprintf(stderr, "I require checked loop binding metadata at line %d.\n", stmt->line);
                        exit(1);
                    }
                    if (!strcmp(var, "_")) list->underscore_name = NULL;
                    if (body && body->type == AST_BLOCK) {
                        for (int bi = 0; bi < body->as.block.count; bi++) {
                            build_stmt(list, scopes, body->as.block.statements[bi], indent + 2, env, fn_registry);
                        }
                    }
                    scope_stack_pop(scopes, list);

                    emit_indent_item(list, indent + 1);
                    emit_literal(list, "}\n");
                    emit_indent_item(list, indent);
                    emit_literal(list, "}\n");
                } else {
                    /* Fallback */
                    emit_indent_item(list, indent);
                    emit_literal(list, "/* unsupported for-in pattern */;\n");
                }
                } /* end else (!is_dyn_array) */
            }
            break;
        }
            
        case AST_IF:
            emit_indent_item(list, indent);
            emit_literal(list, "if (");
            build_expr(list, stmt->as.if_stmt.condition, env);
            emit_literal(list, ") ");
            build_stmt(list, scopes, stmt->as.if_stmt.then_branch, indent, env, fn_registry);
            if (stmt->as.if_stmt.else_branch) {
                emit_indent_item(list, indent);
                emit_literal(list, "else ");
                build_stmt(list, scopes, stmt->as.if_stmt.else_branch, indent, env, fn_registry);
            }
            break;

        case AST_COND: {
            /* Transpile cond to nested if/else */
            for (int i = 0; i < stmt->as.cond_expr.clause_count; i++) {
                emit_indent_item(list, indent);
                if (i == 0) {
                    emit_literal(list, "if (");
                } else {
                    emit_literal(list, "else if (");
                }
                build_expr(list, stmt->as.cond_expr.conditions[i], env);
                emit_literal(list, ") {\n");
                
                /* Emit the value as a statement (usually a return) */
                emit_indent_item(list, indent + 1);
                build_expr(list, stmt->as.cond_expr.values[i], env);
                emit_literal(list, ";\n");
                
                emit_indent_item(list, indent);
                emit_literal(list, "} ");
            }
            
            /* Emit else clause */
            emit_literal(list, "else {\n");
            emit_indent_item(list, indent + 1);
            build_expr(list, stmt->as.cond_expr.else_value, env);
            emit_literal(list, ";\n");
            emit_indent_item(list, indent);
            emit_literal(list, "}\n");
            break;
        }
            
        case AST_PRINT: {
            emit_indent_item(list, indent);

            /* Get the type of the expression */
            Type expr_type = check_expression(stmt->as.print.expr, env);

            /* Map type to print/println function (with nl_ prefix) */
            const char *print_func;
            if (stmt->as.print.is_println) {
                if (expr_type == TYPE_STRING) print_func = "nl_println_string";
                else if (expr_type == TYPE_FLOAT) print_func = "nl_println_float";
                else if (expr_type == TYPE_BOOL) print_func = "nl_println_bool";
                else print_func = "nl_println_int";
            } else {
                if (expr_type == TYPE_STRING) print_func = "nl_print_string";
                else if (expr_type == TYPE_FLOAT) print_func = "nl_print_float";
                else if (expr_type == TYPE_BOOL) print_func = "nl_print_bool";
                else print_func = "nl_print_int";
            }

            emit_literal(list, print_func);
            emit_literal(list, "(");
            build_expr(list, stmt->as.print.expr, env);
            emit_literal(list, ");\n");
            break;
        }
            
        case AST_ASSERT: {
            /* Try to evaluate condition at compile time */
            int const_result = try_eval_bool_const(stmt->as.assert.condition);
            
            if (const_result == 1) {
                /* Condition is provably true - elide the check (emit comment) */
                emit_indent_item(list, indent);
                emit_formatted(list, "/* contract at line %d: always true, elided */\n", stmt->line);
                break;
            }
            
            if (const_result == 0) {
                /* Condition is provably false - emit warning and unconditional failure */
                fprintf(stderr, "Warning at line %d: contract condition is always false\n", stmt->line);
                emit_indent_item(list, indent);
                char *cond_str = expr_to_string(stmt->as.assert.condition);
                emit_formatted(list, "{ fputs(\"Contract violation at line %d: %s (always false)\\n\", stderr); exit(1); }\n",
                              stmt->line, cond_str);
                free(cond_str);
                break;
            }
            
            /* Generate runtime assertion check with informative error message */
            emit_indent_item(list, indent);
            emit_formatted(list, "if (!(");
            build_expr(list, stmt->as.assert.condition, env);
            emit_literal(list, ")) { ");
            
            /* Generate descriptive error message showing the condition */
            char *cond_str = expr_to_string(stmt->as.assert.condition);
            emit_formatted(list, "fputs(\"Contract violation at line %d: %s\\n\", stderr); ",
                          stmt->line, cond_str);
            free(cond_str);
            
            emit_literal(list, "exit(1); }\n");
            break;
        }

        case AST_EFFECT_HANDLER:
        case AST_EFFECT_OP:
            /* Delegate to expression path. */
            emit_indent_item(list, indent);
            build_expr(list, stmt, env);
            emit_literal(list, ";\n");
            break;

        default:
            /* Expression statement */
            emit_indent_item(list, indent);
            build_expr(list, stmt, env);
            emit_literal(list, ";\n");
            break;
    }
}

/* ============================================================================
 * PASS 2: PROCESS WORK ITEMS AND GENERATE OUTPUT
 * Simply walks through the list and outputs each item
 * ============================================================================ */

static void process_worklist(StringBuilder *sb, WorkList *list) {
    for (int i = 0; i < list->count; i++) {
        WorkItem *item = &list->items[i];
        
        switch (item->type) {
            case WORK_LITERAL:
                sb_append(sb, item->data.literal);
                break;

            case WORK_FORMATTED:
                sb_append(sb, item->data.formatted);
                break;

            case WORK_INDENT:
                emit_indent(sb, item->data.indent_level);
                break;

            case WORK_GC_RELEASE:
                /* Generate gc_release() call for variable */
                sb_appendf(sb, "gc_release(%s);", item->data.var_name);
                break;
        }
    }
}

/* ============================================================================
 * PUBLIC API - Two-Pass Transpiler
 * ============================================================================ */

void transpile_expression_iterative(StringBuilder *sb, ASTNode *expr, Environment *env) {
    if (!expr) return;
    
    /* Pass 1: Build work items */
    WorkList *list = worklist_create(1000);
    build_expr(list, expr, env);
    
    /* Pass 2: Process and output */
    process_worklist(sb, list);
    
    /* Cleanup */
    worklist_free(list);
}

void transpile_statement_iterative(StringBuilder *sb, ASTNode *stmt, int indent,
                                  Environment *env, FunctionTypeRegistry *fn_registry) {
    if (!stmt) return;

    /* Pass 1: Build work items with scope tracking */
    WorkList *list = worklist_create(5000);
    list->fn_registry = fn_registry;
    ScopeStack *scopes = scope_stack_create();

    /* Function body is already a scope, but we track variables in it */
    scope_stack_push(scopes, list);
    build_stmt(list, scopes, stmt, indent, env, fn_registry);
    scope_stack_pop(scopes, list);

    /* Pass 2: Process and output */
    process_worklist(sb, list);

    /* Cleanup */
    worklist_free(list);
    scope_stack_free(scopes);
}

static void build_effect_handle(WorkList *list, ASTNode *expr, Environment *env) {
    bool saved_handler = in_effect_handler;
    Type saved_return = effect_lexical_return;
    const char *saved_return_name = effect_lexical_name;
    Symbol **saved_captures = effect_captures;
    int saved_count = effect_capture_count;
    int *saved_slots = effect_capture_slots;
    int id = effect_serial++;
    Type result = effect_body_type(expr->as.handle_expr.body, env);
    ASTNode *result_expr = expr->as.handle_expr.body;
    if (result_expr->type == AST_BLOCK && result_expr->as.block.count)
        result_expr = result_expr->as.block.statements[result_expr->as.block.count - 1];
    const char *result_name = get_struct_type_name(result_expr, env);
    if (result_expr->type == AST_EFFECT_OP) {
        EffectOp *result_op = effect_get_op(env_get_effect(env, result_expr->as.effect_op.effect_name), result_expr->as.effect_op.op_name);
        if (result_op) result_name = result_op->return_type_name;
    }
    Type lexical = g_current_function->as.function.return_type;
    Symbol **captures = calloc((size_t)env->symbol_count, sizeof(*captures));
    int count = 0;
    for (int i = 0; i < env->symbol_count; ++i) {
        Symbol *sym = &env->symbols[i];
        if (!sym->def_file || (g_source_file_for_line_directives && strcmp(sym->def_file, g_source_file_for_line_directives))) continue;
        if (!sym->is_global && sym->def_line > 0 && sym->type != TYPE_VOID &&
            env_get_var_visible_at(env, sym->name, expr->line, expr->column) == sym)
            { captures[count] = malloc(sizeof(*sym)); *captures[count++] = *sym; }
    }
    for (int h = 0; h <= expr->as.handle_expr.handler_count; ++h) {
        StringBuilder *helper_source = sb_create();
        bool body_helper = h == expr->as.handle_expr.handler_count;
        EffectOp body_op = {.param_count = 0, .return_type = result, .return_type_name = (char *)result_name};
        const char *name = body_helper ? NULL : expr->as.handle_expr.handler_op_names[h];
        EffectOp *op = body_helper ? &body_op : effect_get_op(env_get_effect(env, expr->as.handle_expr.effect_name), name);
        ASTNode *handler_body = body_helper ? expr->as.handle_expr.body : expr->as.handle_expr.handler_bodies[h];
        sb_appendf(helper_source, "static void nl_handler_%d_%d(NlEffectFrame *_frame, void **_args, void *_result) {\n", id, h);
        for (int p = 0; p < op->param_count; ++p)
            sb_appendf(helper_source, "%s %s = *(%s *)_args[%d];\n", effect_c_type(op->params[p].type, op->params[p].struct_type_name, env), expr->as.handle_expr.handler_param_names[h][p], effect_c_type(op->params[p].type, op->params[p].struct_type_name, env), p);
        if (op->return_type != TYPE_VOID) sb_appendf(helper_source, "%s _out = {0};\n", effect_c_type(op->return_type, op->return_type_name, env));
        WorkList *handler = worklist_create(100);
        handler->fn_registry = list->fn_registry;
        in_effect_handler = true; effect_lexical_return = lexical; effect_lexical_name = g_current_function->as.function.return_struct_type_name; effect_environment = env;
        effect_captures = captures; effect_capture_count = count;
        /* Operation parameters shadow lexical captures. */
        Symbol **filtered = calloc((size_t)count + 1, sizeof(*filtered));
        int *slots = calloc((size_t)count + 1, sizeof(*slots));
        int n = 0;
        for (int c = 0; c < count; ++c) {
            bool shadow = false;
            for (int p = 0; p < op->param_count; ++p)
                if (!strcmp(captures[c]->name, expr->as.handle_expr.handler_param_names[h][p])) shadow = true;
            if (!shadow) { slots[n] = c; filtered[n++] = captures[c]; }
        }
        /* I keep slot numbers stable even when a parameter shadows a capture. */
        effect_captures = filtered; effect_capture_count = n; effect_capture_slots = slots;
        int saved_symbols = env->symbol_count;
        for (int p = 0; p < op->param_count; ++p) {
            Value value = {0}; value.type = VAL_VOID;
            env_define_var_with_type_info(env, expr->as.handle_expr.handler_param_names[h][p], op->params[p].type, op->params[p].element_type, op->params[p].type_info, true, value);
            Symbol *parameter = &env->symbols[env->symbol_count - 1];
            parameter->struct_type_name = op->params[p].struct_type_name ? strdup(op->params[p].struct_type_name) : NULL;
            parameter->def_file = g_source_file_for_line_directives;
            parameter->def_line = handler_body->line;
            parameter->def_column = handler_body->column;
            parameter->scope_end_line = handler_body->scope_end_line;
            parameter->scope_end_column = handler_body->scope_end_column;
        }
        if (op->return_type != TYPE_VOID) build_match_arm_value(handler, handler_body, env);
        else { ScopeStack *scopes = scope_stack_create(); scope_stack_push(scopes, handler); build_stmt(handler, scopes, handler_body, 0, env, list->fn_registry); scope_stack_free(scopes); }
        env->symbol_count = saved_symbols;
        in_effect_handler = false; effect_captures = NULL; effect_capture_count = 0;
        process_worklist(helper_source, handler); worklist_free(handler); free(filtered); free(slots);
        if (op->return_type != TYPE_VOID) sb_appendf(helper_source, "*(%s *)_result = _out;\n", effect_c_type(op->return_type, op->return_type_name, env));
        sb_append(helper_source, "}\n");
        sb_append(effect_helpers, helper_source->buffer);
        free(helper_source->buffer); free(helper_source);
        in_effect_handler = saved_handler; effect_lexical_return = saved_return; effect_lexical_name = saved_return_name;
        effect_captures = saved_captures; effect_capture_count = saved_count; effect_capture_slots = saved_slots;
    }
    emit_literal(list, "({ ");
    emit_formatted(list, "void *_captures_%d[] = {", id);
    for (int c = 0; c < count; ++c) emit_formatted(list, "%s&%s", c ? ", " : "", saved_handler ? effect_capture_name(captures[c]->name, expr->line, expr->column) : captures[c]->name);
    if (!count) emit_literal(list, "NULL");
    emit_literal(list, "}; ");
    emit_formatted(list, "NlEffectEntry _entries_%d[] = {", id);
    for (int h = 0; h < expr->as.handle_expr.handler_count; ++h)
        emit_formatted(list, "%s{\"%s.%s\", nl_handler_%d_%d}", h ? ", " : "", expr->as.handle_expr.effect_name, expr->as.handle_expr.handler_op_names[h], id, h);
    emit_literal(list, "}; ");
    if (lexical != TYPE_VOID) emit_formatted(list, "%s _lexical_%d = {0}; ", effect_c_type(lexical, g_current_function->as.function.return_struct_type_name, env), id);
    emit_formatted(list, "NlEffectFrame _frame_%d __attribute__((cleanup(nl_effect_pop))) = {.previous=nl_effect_top, .entries=_entries_%d, .count=%d, .captures=_captures_%d, .cleanup_mark=nl_effect_gc_top, .foreign_depth=nl_effect_foreign_depth, .lexical_result=", id, id, expr->as.handle_expr.handler_count, id);
    if (lexical == TYPE_VOID) emit_literal(list, "NULL"); else emit_formatted(list, "&_lexical_%d", id);
    emit_literal(list, "}; ");
    if (result != TYPE_VOID) emit_formatted(list, "%s _out = {0}; ", effect_c_type(result, result_name, env));
    emit_formatted(list, "if (nl_effect_run(&_frame_%d, nl_handler_%d_%d, %s)) {", id, id, expr->as.handle_expr.handler_count, result == TYPE_VOID ? "NULL" : "&_out");
    if (saved_handler) {
        if (lexical != TYPE_VOID) emit_formatted(list, "*(%s *)_frame->lexical_result = _lexical_%d; ", effect_c_type(lexical, g_current_function->as.function.return_struct_type_name, env), id);
        emit_formatted(list, "nl_effect_pop(&_frame_%d); nl_effect_escape(_frame);", id);
    } else if (lexical != TYPE_VOID) emit_formatted(list, "%s _ret = _lexical_%d; return _ret;", effect_c_type(lexical, g_current_function->as.function.return_struct_type_name, env), id);
    else emit_literal(list, "return;");
    emit_literal(list, "} ");
    if (result != TYPE_VOID) emit_literal(list, "_out; "); else emit_literal(list, "(void)0; ");
    emit_literal(list, "})");
    for (int c = 0; c < count; ++c) free(captures[c]);
    free(captures);
}
