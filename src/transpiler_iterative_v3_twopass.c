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
#include <stdarg.h>
#include <string.h>

/* ============================================================================
 * TRANSPILER CONTEXT - Track current function for type resolution
 * (Declared in transpiler.c, visible here via include)
 * ============================================================================ */

extern ASTNode *g_current_function;  /* Current function being transpiled */

/* =========================================================================
 * GENERIC TYPE NAME HELPERS
 * ========================================================================= */

static bool typeinfo_to_monomorph_segment(TypeInfo *ti, char *out, size_t out_size) {
    if (!out || out_size == 0) return false;
    if (!ti) return snprintf(out, out_size, "unknown") < (int)out_size;

    switch (ti->base_type) {
        case TYPE_INT:
            return snprintf(out, out_size, "int") < (int)out_size;
        case TYPE_U8:
            return snprintf(out, out_size, "u8") < (int)out_size;
        case TYPE_STRING:
            return snprintf(out, out_size, "string") < (int)out_size;
        case TYPE_BOOL:
            return snprintf(out, out_size, "bool") < (int)out_size;
        case TYPE_FLOAT:
            return snprintf(out, out_size, "float") < (int)out_size;
        case TYPE_STRUCT:
        case TYPE_UNION:
        case TYPE_ENUM:
            if (ti->generic_name) return snprintf(out, out_size, "%s", ti->generic_name) < (int)out_size;
            return snprintf(out, out_size, "unknown") < (int)out_size;
        case TYPE_ARRAY: {
            char elem[128];
            if (!typeinfo_to_monomorph_segment(ti->element_type, elem, sizeof(elem))) {
                return snprintf(out, out_size, "array_unknown") < (int)out_size;
            }
            return snprintf(out, out_size, "array_%s", elem) < (int)out_size;
        }
        default:
            return snprintf(out, out_size, "unknown") < (int)out_size;
    }
}

static bool build_monomorphized_name_from_typeinfo_iter(char *dest, size_t dest_size, TypeInfo *info) {
    if (!dest || dest_size == 0) return false;
    if (!info || !info->generic_name || info->type_param_count <= 0) return false;

    int written = snprintf(dest, dest_size, "%s", info->generic_name);
    if (written < 0 || (size_t)written >= dest_size) return false;

    size_t pos = (size_t)written;
    for (int i = 0; i < info->type_param_count; i++) {
        char seg[128];
        if (!typeinfo_to_monomorph_segment(info->type_params[i], seg, sizeof(seg))) {
            return false;
        }

        written = snprintf(dest + pos, dest_size - pos, "_%s", seg);
        if (written < 0 || (size_t)written >= dest_size - pos) return false;
        pos += (size_t)written;
    }

    return true;
}

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

/* ============================================================================
 * WORK ITEM TYPES - Describe what output to generate
 * ============================================================================ */

typedef enum {
    WORK_LITERAL,      /* Output a literal string */
    WORK_FORMATTED,    /* Output a formatted string */
    WORK_INDENT,       /* Output indentation */
} WorkItemType;

typedef struct {
    WorkItemType type;
    union {
        char *literal;           /* For WORK_LITERAL */
        char *formatted;         /* For WORK_FORMATTED (pre-formatted) */
        int indent_level;        /* For WORK_INDENT */
    } data;
} WorkItem;

/* ============================================================================
 * WORK LIST - Ordered list of work items (NOT a stack!)
 * ============================================================================ */

typedef struct {
    WorkItem *items;
    int capacity;
    int count;
} WorkList;

static WorkList *worklist_create(int initial_capacity) {
    WorkList *list = malloc(sizeof(WorkList));
    if (!list) {
        fprintf(stderr, "Error: Out of memory allocating WorkList\n");
        exit(1);
    }
    list->capacity = initial_capacity;
    list->count = 0;
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
    }
    free(list->items);
    free(list);
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
    char buffer[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    
    WorkItem item;
    item.type = WORK_FORMATTED;
    item.data.formatted = strdup(buffer);
    if (!item.data.formatted) {
        fprintf(stderr, "Error: Out of memory duplicating formatted string\n");
        exit(1);
    }
    worklist_append(list, item);
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

/* Function name mapping table */
typedef struct {
    const char *nano_name;
    const char *c_name;
} FunctionMapping;

static const FunctionMapping function_map[] = {
    {"println", "println"},
    {"print", "print"},
    {"cast_int", "nl_cast_int"},
    {"cast_float", "nl_cast_float"},
    {"file_read", "nl_os_file_read"},
    {"file_read_bytes", "nl_os_file_read_bytes"},
    {"file_write", "nl_os_file_write"},
    {"file_append", "nl_os_file_append"},
    {"file_remove", "nl_os_file_remove"},
    {"file_rename", "nl_os_file_rename"},
    {"file_exists", "nl_os_file_exists"},
    {"file_size", "nl_os_file_size"},
    {"tmp_dir", "nl_os_tmp_dir"},
    {"mktemp", "nl_os_mktemp"},
    {"mktemp_dir", "nl_os_mktemp_dir"},
    {"dir_create", "nl_os_dir_create"},
    {"dir_remove", "nl_os_dir_remove"},
    {"dir_list", "nl_os_dir_list"},
    {"dir_exists", "nl_os_dir_exists"},
    {"getcwd", "nl_os_getcwd"},
    {"chdir", "nl_os_chdir"},
    {"fs_walkdir", "nl_os_walkdir"},
    {"path_isfile", "nl_os_path_isfile"},
    {"path_isdir", "nl_os_path_isdir"},
    {"path_join", "nl_os_path_join"},
    {"path_basename", "nl_os_path_basename"},
    {"path_dirname", "nl_os_path_dirname"},
    {"path_normalize", "nl_os_path_normalize"},
    {"system", "nl_os_system"},
    {"exit", "nl_os_exit"},
    {"getenv", "nl_os_getenv"},
    {"process_run", "nl_os_process_run"},
    {"abs", "nl_abs"},
    {"min", "nl_min"},
    {"max", "nl_max"},
    {"sqrt", "sqrt"},
    {"pow", "pow"},
    {"floor", "floor"},
    {"ceil", "ceil"},
    {"round", "round"},
    {"str_length", "strlen"},
    {"str_concat", "nl_str_concat"},
    {"str_substring", "nl_str_substring"},
    {"str_contains", "nl_str_contains"},
    {"str_equals", "nl_str_equals"},
    {"bytes_from_string", "nl_bytes_from_string"},
    {"string_from_bytes", "nl_string_from_bytes"},
    {"array_slice", "nl_array_slice"},
    {"char_at", "char_at"},                     /* Generated inline, no prefix */
    {"string_from_char", "string_from_char"},   /* Generated inline, no prefix */
    {"int_to_string", "int_to_string"},         /* Generated inline, no prefix */
    {"string_to_int", "string_to_int"},         /* Generated inline, no prefix */
    {"is_digit", "is_digit"},                   /* Generated inline, no prefix */
    {"is_alpha", "is_alpha"},                   /* Generated inline, no prefix */
    {"is_alnum", "is_alnum"},                   /* Generated inline, no prefix */
    {"is_whitespace", "is_whitespace"},         /* Generated inline, no prefix */
    {"is_upper", "is_upper"},                   /* Generated inline, no prefix */
    {"is_lower", "is_lower"},                   /* Generated inline, no prefix */
    {"digit_value", "digit_value"},             /* Generated inline, no prefix */
    {"char_to_lower", "char_to_lower"},         /* Generated inline, no prefix */
    {"char_to_upper", "char_to_upper"},         /* Generated inline, no prefix */
    {"array_length", "dyn_array_length"},
    {"at", "dyn_array_get"},
    {"array_set", "dyn_array_put"},
    {"array_get", "dyn_array_get"},
    {"array_remove_at", "dyn_array_remove_at"},
};

static const char *map_function_name(const char *name, Environment *env) {
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
        name = dot + 1;
    }
    
    /* Check mapping table */
    for (size_t i = 0; i < sizeof(function_map)/sizeof(function_map[0]); i++) {
        if (strcmp(name, function_map[i].nano_name) == 0) {
            return function_map[i].c_name;
        }
    }
    
    /* User-defined function? Look up to get module context */
    Function *func = env_get_function(env, name);
    
    if (func && !func->is_extern && func->body != NULL) {
        /* Use get_c_func_name_with_module for namespace mangling */
        extern const char *get_c_func_name_with_module(const char *nano_name, const char *module_name);
        return get_c_func_name_with_module(name, func->module_name);
    }
    
    return name;
}

static Type infer_array_element_type(ASTNode *array_expr, Environment *env) {
    if (!array_expr) return TYPE_UNKNOWN;

    if (array_expr->type == AST_IDENTIFIER) {
        Symbol *sym = env_get_var(env, array_expr->as.identifier);
        if (sym && sym->type == TYPE_ARRAY && sym->element_type != TYPE_UNKNOWN) {
            return sym->element_type;
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
                            return sdef->field_element_types[i];
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
 * PASS 1: BUILD WORK ITEMS (Expression Transpiler)
 * Traverses AST and appends work items in correct output order
 * ============================================================================ */

/* Forward declarations */
static void build_expr(WorkList *list, ASTNode *expr, Environment *env);
static void build_stmt(WorkList *list, ASTNode *stmt, int indent, Environment *env, 
                       FunctionTypeRegistry *fn_registry);

static bool is_generic_list_runtime_fn(const char *name) {
    if (!name) return false;
    if (strncmp(name, "list_", 5) != 0) return false;
    /* Built-in runtime lists (do NOT get nl_ prefix) */
    if (strncmp(name, "list_int_", 9) == 0) return false;
    if (strncmp(name, "list_string_", 12) == 0) return false;
    if (strncmp(name, "list_token_", 11) == 0) return false;
    return true;
}

static void build_expr(WorkList *list, ASTNode *expr, Environment *env) {
    if (!expr) return;
    
    switch (expr->type) {
        case AST_NUMBER:
            emit_formatted(list, "%lldLL", expr->as.number);
            break;
            
        case AST_FLOAT:
            if (expr->as.float_val == (double)(int64_t)expr->as.float_val) {
                emit_formatted(list, "%.1f", expr->as.float_val);
            } else {
                emit_formatted(list, "%g", expr->as.float_val);
            }
            break;
            
        case AST_STRING:
            emit_formatted(list, "\"%s\"", expr->as.string_val);
            break;
            
        case AST_BOOL:
            emit_literal(list, expr->as.bool_val ? "true" : "false");
            break;
            
        case AST_IDENTIFIER: {
            /* Check for constant inlining */
            Symbol *sym = env_get_var(env, expr->as.identifier);
            if (sym && !sym->is_mut) {
                if (sym->value.type == VAL_INT) {
                    emit_formatted(list, "%lldLL", (long long)sym->value.as.int_val);
                    return;
                } else if (sym->value.type == VAL_FLOAT) {
                    emit_formatted(list, "%g", sym->value.as.float_val);
                    return;
                } else if (sym->value.type == VAL_BOOL) {
                    emit_literal(list, sym->value.as.bool_val ? "true" : "false");
                    return;
                }
            }
            
            /* Check if it's a function identifier */
            Function *func_def = env_get_function(env, expr->as.identifier);
            if (func_def && !func_def->is_extern && func_def->body != NULL) {
                emit_formatted(list, "nl_%s", expr->as.identifier);
            } else {
                emit_literal(list, expr->as.identifier);
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
                                    emit_formatted(list, "dyn_array_push_%s(_out, _x %s _y); ", push_suffix, op_str);
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

                            emit_literal(list, "({ DynArray* _a = ");
                            build_expr(list, arr_expr, env);
                            emit_literal(list, "; ");
                            emit_formatted(list, "%s _s = ", c_type);
                            build_expr(list, scalar_expr, env);
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
                            emit_literal(list, "({ DynArray* _a = ");
                            build_expr(list, arr_expr, env);
                            emit_literal(list, "; ");
                            if (left_is_array) {
                                emit_formatted(list, "%s(_a, ", fn_scalar ? fn_scalar : "nl_array_add_scalar_int");
                                build_expr(list, scalar_expr, env);
                                emit_literal(list, "); })");
                            } else {
                                emit_formatted(list, "%s(", fn_rscalar ? fn_rscalar : "nl_array_radd_scalar_int");
                                build_expr(list, scalar_expr, env);
                                emit_literal(list, ", _a); })");
                            }
                        }
                        break;
                    }
                }

                bool is_string_comp = false;
                if (op == TOKEN_EQ || op == TOKEN_NE) {
                    Type t1 = check_expression(expr->as.prefix_op.args[0], env);
                    Type t2 = check_expression(expr->as.prefix_op.args[1], env);
                    if (t1 == TYPE_STRING && t2 == TYPE_STRING) {
                        is_string_comp = true;
                    }
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
                } else {
                    /* Regular binary operator */
                    bool needs_parens = (op == TOKEN_PLUS || op == TOKEN_MINUS || 
                                       op == TOKEN_STAR || op == TOKEN_SLASH || op == TOKEN_PERCENT);
                    
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
                    emit_literal(list, "(!");
                    build_expr(list, expr->as.prefix_op.args[0], env);
                    emit_literal(list, ")");
                } else if (op == TOKEN_MINUS) {
                    Type t = check_expression(expr->as.prefix_op.args[0], env);
                    if (t == TYPE_ARRAY) {
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
                        emit_literal(list, "(-");
                        build_expr(list, expr->as.prefix_op.args[0], env);
                        emit_literal(list, ")");
                    }
                }
            }
            break;
        }
        
        case AST_CALL: {
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
                    Symbol *sym = env_get_var(env, array_arg->as.identifier);
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
                /* map(array<T>, fn(T)->T) -> array<T>
                 * Arrays are DynArray* in compiled mode.
                 */
                ASTNode *array_arg = expr->as.call.args[0];
                ASTNode *fn_arg = expr->as.call.args[1];

                Type elem_type = TYPE_INT;  /* default */
                const char *struct_name = NULL;

                if (array_arg && array_arg->type == AST_IDENTIFIER) {
                    Symbol *sym = env_get_var(env, array_arg->as.identifier);
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
                    emit_formatted(list, "%s _mapped = _f(_elem); dyn_array_push_%s(_out, _mapped); ", c_type, type_suffix);
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
                    Symbol *sym = env_get_var(env, array_arg->as.identifier);
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
                        Symbol *sym = env_get_var(env, initial_arg->as.identifier);
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
            else if ((strcmp(func_name, "at") == 0 || strcmp(func_name, "array_set") == 0) && 
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
                    if (array_arg->type == AST_IDENTIFIER) {
                        Symbol *sym = env_get_var(env, array_arg->as.identifier);
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
                
                /* For structs, use dyn_array_get/set_struct with casts */
                if (elem_type == TYPE_STRUCT && struct_name) {
                    if (strcmp(func_name, "at") == 0) {
                        /* Generate: *((nl_StructName*)dyn_array_get_struct(arr, idx)) */
                        emit_formatted(list, "*((nl_%s*)dyn_array_get_struct(", struct_name);
                        build_expr(list, expr->as.call.args[0], env);  /* array */
                        emit_literal(list, ", ");
                        build_expr(list, expr->as.call.args[1], env);  /* index */
                        emit_literal(list, "))");
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
                    if (strcmp(func_name, "at") == 0) {
                        snprintf(func_buf, sizeof(func_buf), "nl_array_at_%s", type_suffix);
                    } else {
                        snprintf(func_buf, sizeof(func_buf), "nl_array_set_%s", type_suffix);
                    }
                    
                    emit_literal(list, func_buf);
                    emit_literal(list, "(");
                    for (int i = 0; i < expr->as.call.arg_count; i++) {
                        if (i > 0) emit_literal(list, ", ");
                        build_expr(list, expr->as.call.args[i], env);
                    }
                    emit_literal(list, ")");
                }
            }
            /* Special handling for array_new() - creates new dynamic array with size and initial value */
            else if (strcmp(func_name, "array_new") == 0 && expr->as.call.arg_count == 2) {
                /* array_new(size, default_value) -> DynArray* */
                ASTNode *size_arg = expr->as.call.args[0];
                ASTNode *value_arg = expr->as.call.args[1];
                
                /* Infer element type from default value */
                Type elem_type = check_expression(value_arg, env);
                const char *struct_name = NULL;
                
                /* For structs, get the struct name */
                if (elem_type == TYPE_STRUCT) {
                    if (value_arg->type == AST_IDENTIFIER) {
                        Symbol *value_sym = env_get_var(env, value_arg->as.identifier);
                        if (value_sym && value_sym->struct_type_name) {
                            struct_name = value_sym->struct_type_name;
                        }
                    } else if (value_arg->type == AST_STRUCT_LITERAL) {
                        struct_name = value_arg->as.struct_literal.struct_name;
                    }
                }
                
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
                
                /* Generate initialization code using a compound statement */
                /* ({ DynArray* _arr = dyn_array_new(ELEM_TYPE); for (...) { dyn_array_push_*(_arr, val); } _arr; }) */
                emit_formatted(list, "({ DynArray* _arr = dyn_array_new(%s); ", elem_type_str);
                emit_literal(list, "int64_t _size = ");
                build_expr(list, size_arg, env);
                emit_literal(list, "; for (int64_t _i = 0; _i < _size; _i++) { ");
                
                /* Generate appropriate push call based on type */
                if (elem_type == TYPE_STRUCT && struct_name) {
                    /* For structs: dyn_array_push_struct(_arr, &value, sizeof(nl_StructName)) */
                    emit_literal(list, "dyn_array_push_struct(_arr, &(");
                    build_expr(list, value_arg, env);
                    emit_formatted(list, "), sizeof(nl_%s)); ", struct_name);
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
                    
                    emit_formatted(list, "dyn_array_push_%s(_arr, ", type_suffix);
                    build_expr(list, value_arg, env);
                    emit_literal(list, "); ");
                }
                
                emit_literal(list, "} _arr; })");
            }
            /* Special handling for array_push() and array_pop() - dynamic array operations */
            else if (strcmp(func_name, "array_push") == 0 && expr->as.call.arg_count == 2) {
                /* Detect element type from array argument (first arg) or value argument (second arg) */
                Type elem_type = TYPE_INT;  /* Default to int */
                const char *struct_name = NULL;
                
                ASTNode *array_arg = expr->as.call.args[0];
                ASTNode *value_arg = expr->as.call.args[1];
                
                if (array_arg->type == AST_IDENTIFIER) {
                    /* Array is a variable - look up its element type */
                    const char *array_name = array_arg->as.identifier;
                    Symbol *sym = env_get_var(env, array_name);
                    if (sym && sym->element_type != TYPE_UNKNOWN) {
                        elem_type = sym->element_type;
                        /* For array<struct>, the struct name is stored in struct_type_name */
                        if (elem_type == TYPE_STRUCT && sym->struct_type_name) {
                            struct_name = sym->struct_type_name;
                        }
                    } else {
                        /* Try to infer from value type */
                        elem_type = check_expression(value_arg, env);
                    }
                } else {
                    /* Try to infer from value type */
                    elem_type = check_expression(value_arg, env);
                }
                
                /* If we still don't have struct name, try to infer from value argument */
                if (elem_type == TYPE_STRUCT && !struct_name) {
                    /* Check if value is a variable with struct type */
                    if (value_arg->type == AST_IDENTIFIER) {
                        Symbol *value_sym = env_get_var(env, value_arg->as.identifier);
                        if (value_sym && value_sym->type == TYPE_STRUCT && value_sym->struct_type_name) {
                            struct_name = value_sym->struct_type_name;
                        }
                    } else if (value_arg->type == AST_STRUCT_LITERAL) {
                        /* Struct literal has the name directly */
                        struct_name = value_arg->as.struct_literal.struct_name;
                    }
                }
                
                /* For structs, use dyn_array_push_struct with sizeof */
                if (elem_type == TYPE_STRUCT && struct_name) {
                    /* Generate: dyn_array_push_struct(arr, &value, sizeof(nl_StructName)) */
                    emit_literal(list, "dyn_array_push_struct(");
                    build_expr(list, expr->as.call.args[0], env);  /* array */
                    emit_literal(list, ", &(");
                    build_expr(list, expr->as.call.args[1], env);  /* value */
                    emit_formatted(list, "), sizeof(nl_%s))", struct_name);
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
                    
                    /* Generate: dyn_array_push_<type>(arr, value) */
                    char func_buf[64];
                    snprintf(func_buf, sizeof(func_buf), "dyn_array_push_%s", type_suffix);
                    
                    emit_literal(list, func_buf);
                    emit_literal(list, "(");
                    build_expr(list, expr->as.call.args[0], env);  /* array */
                    emit_literal(list, ", ");
                    build_expr(list, expr->as.call.args[1], env);  /* value */
                    emit_literal(list, ")");
                }
            }
            else if (strcmp(func_name, "array_pop") == 0 && expr->as.call.arg_count == 1) {
                /* Detect element type from array argument */
                Type elem_type = TYPE_INT;  /* Default to int */
                const char *struct_name = NULL;
                
                ASTNode *array_arg = expr->as.call.args[0];
                if (array_arg->type == AST_IDENTIFIER) {
                    const char *array_name = array_arg->as.identifier;
                    Symbol *sym = env_get_var(env, array_name);
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
                /* Regular function call */
                const char *mapped_name = func_name;
                if (is_generic_list_runtime_fn(func_name)) {
                    static _Thread_local char buf[512];
                    snprintf(buf, sizeof(buf), "nl_%s", func_name);
                    mapped_name = buf;
                }
                mapped_name = map_function_name(mapped_name, env);
                
                emit_literal(list, mapped_name);
                emit_literal(list, "(");
                for (int i = 0; i < expr->as.call.arg_count; i++) {
                    if (i > 0) emit_literal(list, ", ");
                    build_expr(list, expr->as.call.args[i], env);
                }
                emit_literal(list, ")");
            }
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
        
        case AST_TUPLE_LITERAL: {
            int element_count = expr->as.tuple_literal.element_count;
            
            if (element_count == 0) {
                emit_literal(list, "(struct {int _dummy;}){0}");
                break;
            }
            
            /* Try to find typedef from pre-collected registry */
            const char *typedef_name = NULL;
            extern TupleTypeRegistry *g_tuple_registry;
            
            if (g_tuple_registry && element_count > 0) {
                if (expr->as.tuple_literal.element_types) {
                    /* Element types are set - look up by exact match */
                    for (int i = 0; i < g_tuple_registry->count; i++) {
                        TypeInfo *registered = g_tuple_registry->tuples[i];
                        if (registered->tuple_element_count == element_count) {
                            bool match = true;
                            for (int j = 0; j < element_count; j++) {
                                if (registered->tuple_types[j] != expr->as.tuple_literal.element_types[j]) {
                                    match = false;
                                    break;
                                }
                            }
                            if (match) {
                                typedef_name = g_tuple_registry->typedef_names[i];
                                break;
                            }
                        }
                    }
                } else {
                    /* Element types not set - try to infer and match */
                    Type inferred_types[element_count];
                    for (int i = 0; i < element_count; i++) {
                        Type elem_type = TYPE_INT;
                        ASTNode *elem = expr->as.tuple_literal.elements[i];
                        if (elem) {
                            if (elem->type == AST_NUMBER) elem_type = TYPE_INT;
                            else if (elem->type == AST_STRING) elem_type = TYPE_STRING;
                            else if (elem->type == AST_BOOL) elem_type = TYPE_BOOL;
                            else if (elem->type == AST_FLOAT) elem_type = TYPE_FLOAT;
                            else if (elem->type == AST_IDENTIFIER) elem_type = TYPE_INT;
                        }
                        inferred_types[i] = elem_type;
                    }
                    
                    /* Look up by inferred types */
                    for (int i = 0; i < g_tuple_registry->count; i++) {
                        TypeInfo *registered = g_tuple_registry->tuples[i];
                        if (registered->tuple_element_count == element_count) {
                            bool match = true;
                            for (int j = 0; j < element_count; j++) {
                                if (registered->tuple_types[j] != inferred_types[j]) {
                                    match = false;
                                    break;
                                }
                            }
                            if (match) {
                                typedef_name = g_tuple_registry->typedef_names[i];
                                break;
                            }
                        }
                    }
                }
            }
            
            if (typedef_name) {
                /* Use typedef */
                emit_formatted(list, "(%s){", typedef_name);
            } else {
                /* Fall back to inline struct */
                emit_literal(list, "(struct { ");
                for (int i = 0; i < element_count; i++) {
                    Type elem_type = expr->as.tuple_literal.element_types ? 
                                   expr->as.tuple_literal.element_types[i] : TYPE_INT;
                    const char *c_type = type_to_c(elem_type);
                    emit_formatted(list, "%s _%d; ", c_type, i);
                }
                emit_literal(list, "}){");
            }
            
            /* Emit field initializers IN ORDER */
            for (int i = 0; i < element_count; i++) {
                if (i > 0) emit_literal(list, ", ");
                emit_formatted(list, "._%d = ", i);
                build_expr(list, expr->as.tuple_literal.elements[i], env);
            }
            emit_literal(list, "}");
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
            
        case AST_STRUCT_LITERAL: {
            /* Struct literal: StructName { field1: val1, field2: val2 } */
            const char *struct_name = expr->as.struct_literal.struct_name;
            const char *prefixed = get_prefixed_type_name(struct_name);
            int field_count = expr->as.struct_literal.field_count;
            
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
            
            emit_formatted(list, "(%s){", prefixed);
            for (int i = 0; i < field_count; i++) {
                if (i > 0) emit_literal(list, ", ");
                emit_formatted(list, ".%s = ", expr->as.struct_literal.field_names[i]);
                build_expr(list, expr->as.struct_literal.field_values[i], env);
            }
            emit_literal(list, "}");
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
            
            if (g_current_function && 
                g_current_function->as.function.return_type == TYPE_UNION &&
                g_current_function->as.function.return_type_info &&
                g_current_function->as.function.return_type_info->generic_name &&
                g_current_function->as.function.return_type_info->type_param_count > 0 &&
                strcmp(union_name, g_current_function->as.function.return_type_info->generic_name) == 0) {
                /* Build monomorphized name: Result<int, string> -> Result_int_string */
                if (!build_monomorphized_name_from_typeinfo_iter(monomorphized_name, sizeof(monomorphized_name),
                                                                g_current_function->as.function.return_type_info)) {
                    snprintf(monomorphized_name, sizeof(monomorphized_name), "%s", union_name);
                }
                
                prefixed_union = get_prefixed_type_name(monomorphized_name);
                is_generic = true;
            } else {
                /* Non-generic union or no function context - use base name */
                prefixed_union = get_prefixed_type_name(union_name);
            }
            
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
                if (elem_type == TYPE_INT) {
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
                if (elem_type == TYPE_INT) {
                    emit_formatted(list, "dynarray_literal_int(%d", count);
                    for (int i = 0; i < count; i++) {
                        emit_literal(list, ", ");
                        build_expr(list, expr->as.array_literal.elements[i], env);
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
            /* Generate: ({ UnionType _m = scrutinee; int64_t _out = 0; switch(_m.tag) { case TAG_V: { VariantType nl_binding = _m.data.V; _out = body; break; } } _out; }) */
            
            const char *union_c_name = expr->as.match_expr.union_type_name;
            if (!union_c_name) {
                emit_literal(list, "/* match: unknown union type */0");
                break;
            }
            
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
            
            /* Determine output type - check if we're in a function returning a struct */
            const char *out_type = "int64_t";  /* Default fallback */
            char out_type_buf[256] = "int64_t";  /* Buffer to save out_type */
            if (g_current_function && 
                g_current_function->as.function.return_type == TYPE_STRUCT &&
                g_current_function->as.function.return_struct_type_name) {
                const char *temp = get_prefixed_type_name(g_current_function->as.function.return_struct_type_name);
                strncpy(out_type_buf, temp, sizeof(out_type_buf) - 1);
                out_type_buf[sizeof(out_type_buf) - 1] = '\0';
                out_type = out_type_buf;
            }
            
            const char *prefixed_union = get_prefixed_type_name(union_c_name);
            
            /* Start compound expression */
            emit_literal(list, "({ ");
            emit_literal(list, prefixed_union);
            emit_literal(list, " _m = ");
            build_expr(list, expr->as.match_expr.expr, env);
            emit_literal(list, "; ");
            emit_literal(list, out_type);
            emit_literal(list, " _out = {0}; switch (_m.tag) { ");
            
            /* Generate each match arm */
            for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                const char *variant_name = expr->as.match_expr.pattern_variants[i];
                const char *binding_name = expr->as.match_expr.pattern_bindings[i];
                ASTNode *arm_body = expr->as.match_expr.arm_bodies[i];
                
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
                if (variant_field_count > 0) {
                    /* Binding name should NOT have nl_ prefix - it's referenced directly in code */
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
                    /* Empty variant - just suppress unused warning */
                    emit_literal(list, "(void)_m.data.");
                    emit_literal(list, variant_name);
                    emit_literal(list, "; ");
                }
                
                /* Handle arm body */
                if (arm_body) {
                    if (arm_body->type == AST_BLOCK) {
                        /* For block bodies, find return statement */
                        for (int j = 0; j < arm_body->as.block.count; j++) {
                            ASTNode *stmt = arm_body->as.block.statements[j];
                            if (stmt && stmt->type == AST_RETURN && stmt->as.return_stmt.value) {
                                emit_literal(list, "_out = ");
                                build_expr(list, stmt->as.return_stmt.value, env);
                                emit_literal(list, "; ");
                                break;
                            }
                        }
                    } else {
                        /* Expression body */
                        emit_literal(list, "_out = ");
                        build_expr(list, arm_body, env);
                        emit_literal(list, "; ");
                    }
                }
                
                emit_literal(list, "break; } ");
            }
            
            /* Close switch and compound expression */
            emit_literal(list, "} _out; })");
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

static void build_stmt(WorkList *list, ASTNode *stmt, int indent, Environment *env,
                       FunctionTypeRegistry *fn_registry) {
    if (!stmt) return;
    
    switch (stmt->type) {
        case AST_BLOCK:
            emit_indent_item(list, indent);
            emit_literal(list, "{\n");
            for (int i = 0; i < stmt->as.block.count; i++) {
                build_stmt(list, stmt->as.block.statements[i], indent + 1, env, fn_registry);
            }
            emit_indent_item(list, indent);
            emit_literal(list, "}\n");
            break;

        case AST_MATCH: {
            const char *union_c_name = stmt->as.match_expr.union_type_name;
            if (!union_c_name) {
                emit_indent_item(list, indent);
                emit_literal(list, "/* match: unknown union type */;\n");
                break;
            }

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
            emit_literal(list, prefixed_union);
            emit_literal(list, " _m = ");
            build_expr(list, stmt->as.match_expr.expr, env);
            emit_literal(list, ";\n");

            emit_indent_item(list, indent + 1);
            emit_literal(list, "switch (_m.tag) {\n");

            for (int i = 0; i < stmt->as.match_expr.arm_count; i++) {
                const char *variant_name = stmt->as.match_expr.pattern_variants[i];
                const char *binding_name = stmt->as.match_expr.pattern_bindings[i];
                ASTNode *arm_body = stmt->as.match_expr.arm_bodies[i];

                emit_indent_item(list, indent + 2);
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

                if (arm_body) {
                    if (arm_body->type == AST_BLOCK) {
                        for (int j = 0; j < arm_body->as.block.count; j++) {
                            build_stmt(list, arm_body->as.block.statements[j], indent + 3, env, fn_registry);
                        }
                    } else {
                        emit_indent_item(list, indent + 3);
                        build_expr(list, arm_body, env);
                        emit_literal(list, ";\n");
                    }
                }

                emit_indent_item(list, indent + 3);
                emit_literal(list, "break;\n");
                emit_indent_item(list, indent + 2);
                emit_literal(list, "}\n");
            }

            emit_indent_item(list, indent + 2);
            emit_literal(list, "default: break;\n");
            emit_indent_item(list, indent + 1);
            emit_literal(list, "}\n");
            emit_indent_item(list, indent);
            emit_literal(list, "}\n");
            break;
        }
            
        case AST_RETURN:
            emit_indent_item(list, indent);
            emit_literal(list, "return");
            if (stmt->as.return_stmt.value) {
                emit_literal(list, " ");
                build_expr(list, stmt->as.return_stmt.value, env);
            }
            emit_literal(list, ";\n");
            break;
            
        case AST_LET: {
            emit_indent_item(list, indent);
            
            /* Handle tuple types - use __auto_type to infer from RHS */
            if (stmt->as.let.var_type == TYPE_TUPLE) {
                emit_formatted(list, "__auto_type %s = ", stmt->as.let.name);
                build_expr(list, stmt->as.let.value, env);
                emit_literal(list, ";\n");
            }
            /* Handle function types - use typedef from registry */
            else if (stmt->as.let.var_type == TYPE_FUNCTION && stmt->as.let.fn_sig) {
                const char *typedef_name = register_function_signature(fn_registry, stmt->as.let.fn_sig);
                emit_formatted(list, "%s %s", typedef_name, stmt->as.let.name);
                
                if (stmt->as.let.value) {
                    emit_literal(list, " = ");
                    build_expr(list, stmt->as.let.value, env);
                }
                emit_literal(list, ";\n");
            }
            /* Handle List<T> (generic lists) */
            else if (stmt->as.let.var_type == TYPE_LIST_GENERIC && stmt->as.let.type_name) {
                emit_formatted(list, "List_%s* %s", stmt->as.let.type_name, stmt->as.let.name);
                if (stmt->as.let.value) {
                    emit_literal(list, " = ");
                    build_expr(list, stmt->as.let.value, env);
                }
                emit_literal(list, ";\n");
            }
            /* Handle struct/enum types that need prefixing */
            else if (stmt->as.let.var_type == TYPE_STRUCT || stmt->as.let.var_type == TYPE_UNION) {
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
                    emit_formatted(list, "%s %s", prefixed, stmt->as.let.name);
                    
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
                        emit_formatted(list, "%s %s = ", prefixed, stmt->as.let.name);
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
                            emit_formatted(list, "void* %s", stmt->as.let.name);
                        } else if (type_name) {
                            /* Regular struct type */
                            const char *prefixed = get_prefixed_type_name(type_name);
                            emit_formatted(list, "%s %s", prefixed, stmt->as.let.name);
                        } else {
                            emit_formatted(list, "void* %s", stmt->as.let.name);
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
                emit_formatted(list, "%s %s", c_type, stmt->as.let.name);
                
                if (stmt->as.let.value) {
                    emit_literal(list, " = ");
                    build_expr(list, stmt->as.let.value, env);
                }
                emit_literal(list, ";\n");
            }
            
            /* Register in environment */
            env_define_var_with_type_info(env, stmt->as.let.name, stmt->as.let.var_type,
                                         stmt->as.let.element_type, NULL, stmt->as.let.is_mut, create_void());
            
            /* For array<struct>, set struct_type_name so array_push can find it */
            if (stmt->as.let.var_type == TYPE_ARRAY && 
                stmt->as.let.element_type == TYPE_STRUCT &&
                stmt->as.let.type_name) {
                Symbol *sym = env_get_var(env, stmt->as.let.name);
                if (sym) {
                    sym->struct_type_name = strdup(stmt->as.let.type_name);
                    if (!sym->struct_type_name) {
                        fprintf(stderr, "Error: Out of memory duplicating struct type name\n");
                        exit(1);
                    }
                }
            }
            break;
        }
        
        case AST_SET:
            emit_indent_item(list, indent);
            emit_literal(list, stmt->as.set.name);
            emit_literal(list, " = ");
            
            /* Special handling for empty array literals: infer type from target variable */
            if (stmt->as.set.value &&
                stmt->as.set.value->type == AST_ARRAY_LITERAL &&
                stmt->as.set.value->as.array_literal.element_count == 0 &&
                stmt->as.set.value->as.array_literal.element_type == TYPE_UNKNOWN) {
                
                /* Look up the target variable's element type */
                Symbol *sym = env_get_var(env, stmt->as.set.name);
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
            build_stmt(list, stmt->as.while_stmt.body, indent, env, fn_registry);
            break;
            
        case AST_IF:
            emit_indent_item(list, indent);
            emit_literal(list, "if (");
            build_expr(list, stmt->as.if_stmt.condition, env);
            emit_literal(list, ") ");
            build_stmt(list, stmt->as.if_stmt.then_branch, indent, env, fn_registry);
            if (stmt->as.if_stmt.else_branch) {
                emit_indent_item(list, indent);
                emit_literal(list, "else ");
                build_stmt(list, stmt->as.if_stmt.else_branch, indent, env, fn_registry);
            }
            break;
            
        case AST_PRINT: {
            emit_indent_item(list, indent);
            
            /* Get the type of the expression */
            Type expr_type = check_expression(stmt->as.print.expr, env);
            
            /* Map type to print function (with nl_ prefix) */
            const char *print_func = "nl_print_int";
            if (expr_type == TYPE_STRING) {
                print_func = "nl_print_string";
            } else if (expr_type == TYPE_FLOAT) {
                print_func = "nl_print_float";
            } else if (expr_type == TYPE_BOOL) {
                print_func = "nl_print_bool";
            }
            
            emit_literal(list, print_func);
            emit_literal(list, "(");
            build_expr(list, stmt->as.print.expr, env);
            emit_literal(list, ");\n");
            break;
        }
            
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
    
    /* Pass 1: Build work items */
    WorkList *list = worklist_create(5000);
    build_stmt(list, stmt, indent, env, fn_registry);
    
    /* Pass 2: Process and output */
    process_worklist(sb, list);
    
    /* Cleanup */
    worklist_free(list);
}
