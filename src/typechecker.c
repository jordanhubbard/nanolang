#include "nanolang.h"
#include "effects.h"
#include "tracing.h"
#include "resource_tracking.h"
#include "colors.h"
#include "builtins_registry.h"
#include <ctype.h>

/* Returns true if name is a single uppercase letter — a generic type variable */
static bool is_type_variable_name(const char *name) {
    return name != NULL && name[0] != '\0' && name[1] == '\0' && isupper((unsigned char)name[0]);
}

/* Returns true if any parameter of func uses a type variable */
static bool func_is_generic(const Function *func) {
    if (!func->params) return false;
    for (int i = 0; i < func->param_count; i++) {
        if (func->params[i].type == TYPE_STRUCT && is_type_variable_name(func->params[i].struct_type_name))
            return true;
    }
    return false;
}

/* Build monomorphized name: "identity" + T->int => "identity_int" */
static void build_generic_mono_name(char *out, size_t out_size,
                                     const char *func_name,
                                     char **var_names, Type *bound_types, char **bound_type_names,
                                     int binding_count) {
    size_t pos = (size_t)snprintf(out, out_size, "%s", func_name);
    for (int i = 0; i < binding_count && pos < out_size - 1; i++) {
        const char *suffix = NULL;
        if (bound_types[i] == TYPE_STRUCT && bound_type_names[i]) {
            suffix = bound_type_names[i];
        } else {
            switch (bound_types[i]) {
                case TYPE_INT:    suffix = "int"; break;
                case TYPE_FLOAT:  suffix = "float"; break;
                case TYPE_BOOL:   suffix = "bool"; break;
                case TYPE_STRING: suffix = "string"; break;
                default:          suffix = "unknown"; break;
            }
        }
        pos += (size_t)snprintf(out + pos, out_size - pos, "_%s", suffix);
    }
    (void)var_names; /* used for identity-resolution; not needed in name build */
}

/* Use json_diagnostics without including json_diagnostics.h (DiagnosticSeverity conflict
 * with compiler_schema.h). Declare only what we need via extern. */
extern bool g_json_output_enabled;
extern void json_error(const char *code, const char *message, const char *file,
                       int line, int column, const char *suggestion);
extern void json_warning(const char *code, const char *message, const char *file,
                         int line, int column, const char *suggestion);

/* Global typecheck error count — forward-declared so all functions can access it
 * (definition appears later in this file after the builtin name table). */
static int g_typecheck_error_count;

static void emit_context_error(
    const char *title,
    int line,
    int column,
    int caret_len,
    const char *message,
    const char *hint
);

/* Type checking context */
typedef struct {
    Environment *env;
    Type current_function_return_type;
    Type current_function_return_element_type;
    const TypeInfo *current_function_return_info;
    const char *current_function_return_struct_name;  /* For struct return types */
    bool has_error;
    bool warnings_enabled;
    bool in_unsafe_block;   /* Track if we're inside an unsafe block */
    int loop_depth;         /* Track if we're inside a loop (for break/continue validation) */
} TypeChecker;

/* I retain the enclosing function context when checking expression blocks. */
static _Thread_local TypeChecker *active_statement_checker;

static char *typeinfo_to_monomorphized_generic_name(TypeInfo *info);

static char *typeinfo_to_monomorphized_generic_name(TypeInfo *info) {
    if (!info || !info->generic_name) return NULL;
    return typeinfo_to_generic_arg_name(info);
}

/* Helper: Recursively check if an AST expression references a given variable name.
 * Used by par-let dependency validation to detect cross-binding references. */
static bool ast_references_name(ASTNode *node, const char *name) {
    if (!node || !name) return false;
    switch (node->type) {
        case AST_IDENTIFIER:
            return strcmp(node->as.identifier, name) == 0;
        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; i++)
                if (ast_references_name(node->as.prefix_op.args[i], name)) return true;
            return false;
        case AST_CALL:
            for (int i = 0; i < node->as.call.arg_count; i++)
                if (ast_references_name(node->as.call.args[i], name)) return true;
            return false;
        case AST_IF:
            return ast_references_name(node->as.if_stmt.condition, name) ||
                   ast_references_name(node->as.if_stmt.then_branch, name) ||
                   ast_references_name(node->as.if_stmt.else_branch, name);
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++)
                if (ast_references_name(node->as.block.statements[i], name)) return true;
            return false;
        case AST_FIELD_ACCESS:
            return ast_references_name(node->as.field_access.object, name);
        case AST_ARRAY_LITERAL:
            for (int i = 0; i < node->as.array_literal.element_count; i++)
                if (ast_references_name(node->as.array_literal.elements[i], name)) return true;
            return false;
        case AST_LET:
            return ast_references_name(node->as.let.value, name);
        case AST_RETURN:
            return ast_references_name(node->as.return_stmt.value, name);
        default:
            return false;
    }
}

static bool par_closed_call(ASTNode *node, Environment *env);

/* I admit only closed scalar expressions in the passive frontend slice. */
static bool par_scalar_expression(ASTNode *node, Environment *env) {
    if (!node) return false;
    switch (node->type) {
        case AST_NUMBER: case AST_FLOAT: case AST_BOOL: case AST_STRING:
            return true;
        case AST_IDENTIFIER: {
            Symbol *symbol = env_get_var_visible_at(env, node->as.identifier, node->line, node->column);
            return symbol && !symbol->is_mut && !symbol->is_resource &&
                (symbol->type == TYPE_INT || symbol->type == TYPE_FLOAT ||
                 symbol->type == TYPE_BOOL || symbol->type == TYPE_STRING);
        }
        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; ++i)
                if (!par_scalar_expression(node->as.prefix_op.args[i], env)) return false;
            return true;
        case AST_CALL:
            if (!par_closed_call(node, env)) return false;
            for (int i = 0; i < node->as.call.arg_count; ++i)
                if (!par_scalar_expression(node->as.call.args[i], env)) return false;
            return true;
        case AST_MODULE_QUALIFIED_CALL:
            if (!par_closed_call(node, env)) return false;
            for (int i = 0; i < node->as.module_qualified_call.arg_count; ++i)
                if (!par_scalar_expression(node->as.module_qualified_call.args[i], env)) return false;
            return true;
        default:
            return false;
    }
}

/* Helper: Check if symbol was explicitly imported via selective import */
static bool is_symbol_imported(const char *symbol_name, const char *module_path, Environment *env) {
    if (!env->import_tracker) return true;  /* No tracking = allow all */

    /* If no imports from this module, not imported */
    bool has_import_from_module = false;

    for (int i = 0; i < env->import_tracker->import_count; i++) {
        SelectiveImport *imp = &env->import_tracker->imports[i];

        /* Check if this import is from the right module */
        if (imp->module_path && strcmp(imp->module_path, module_path) == 0) {
            has_import_from_module = true;

            /* Wildcard import - all symbols accessible */
            if (imp->is_wildcard) return true;

            /* Check if symbol in imported list */
            if (imp->imported_symbols) {
                for (int j = 0; j < imp->symbol_count; j++) {
                    if (strcmp(imp->imported_symbols[j], symbol_name) == 0) {
                        return true;
                    }
                }
            }
        }
    }

    /* If we have imports from this module but symbol not found, it wasn't imported */
    /* If we don't have any imports from this module, allow (legacy behavior) */
    return !has_import_from_module;
}

/* Helper: Check if a function is accessible from current module */
static bool is_function_accessible(Function *func, Environment *env, int line, int column) {
    if (!func) return false;
    
    /* If no module context, everything is accessible (legacy/global scope) */
    if (!env->current_module) return true;
    
    /* If function has no module, it's global (legacy) - accessible */
    if (!func->module_name) return true;
    
    /* If same module, always accessible */
    if (func->module_name && strcmp(func->module_name, env->current_module) == 0) {
        return true;
    }
    
    /* Different module - check visibility */
    if (!func->is_pub) {
        char message[512];
        snprintf(message, sizeof(message),
                 "I cannot call private function '%s' from module '%s'.",
                 func->name, func->module_name);
        emit_context_error("E009 PRIVATE ACCESS", line, column,
                           (int)safe_strlen(func->name), message,
                           "Call an exported function, or declare this function pub in its owning module.");
        return false;
    }

    /* Check if symbol was explicitly imported via selective import */
    if (!is_symbol_imported(func->name, func->module_name, env)) {
        char message[512];
        snprintf(message, sizeof(message),
                 "I cannot call function '%s' from module '%s' without importing it.",
                 func->name, func->module_name);
        emit_context_error("E009 IMPORT ACCESS", line, column,
                           (int)safe_strlen(func->name), message,
                           "Include this function in the selective import.");
        return false;
    }

    return true;
}

/* Helper: Check if a struct is accessible from current module
 * NOTE: Currently unused - reserved for future struct visibility enforcement
 */
#if 0  /* Disabled - not yet needed */
static bool is_struct_accessible(StructDef *sdef, Environment *env, int line, int column) {
    if (!sdef) return false;
    
    /* If no module context, everything is accessible */
    if (!env->current_module) return true;
    
    /* If struct has no module, it's global - accessible */
    if (!sdef->module_name) return true;
    
    /* If same module, always accessible */
    if (sdef->module_name && strcmp(sdef->module_name, env->current_module) == 0) {
        return true;
    }
    
    /* Different module - check visibility */
    if (!sdef->is_pub) {
        fprintf(stderr, "Error at line %d, column %d: Struct '%s' is private to module '%s'\n",
                line, column, sdef->name, sdef->module_name);
        fprintf(stderr, "  Note: Use 'pub struct %s { ... }' to make it accessible from other modules\n",
                sdef->name);
        fprintf(stderr, "  Hint: Private types are only accessible within their defining module\n");
        return false;
    }
    
    return true;
}
#endif  /* Disabled - not yet needed */

/* Check for unused variables in current scope and emit warnings */
static void check_unused_variables(TypeChecker *tc, int start_index) {
    if (!tc->warnings_enabled) return;
    
    for (int i = start_index; i < tc->env->symbol_count; i++) {
        Symbol *sym = &tc->env->symbols[i];
        if (!sym->is_used && sym->def_line > 0) {
            /* Skip loop variables (they start with underscores by convention) */
            if (sym->name[0] == '_') continue;
            
            fprintf(stderr, "Warning at line %d, column %d: Unused variable '%s'\n",
                    sym->def_line, sym->def_column, sym->name);
        }
    }
}

const char *get_struct_type_name(ASTNode *expr, Environment *env);
static FunctionSignature *function_result_signature(ASTNode *call, Environment *env);
static TypeInfo *infer_complete_tuple_literal(ASTNode *literal, Environment *env);

static TypeInfo *try_get_expr_type_info(ASTNode *expr, Environment *env) {
    if (!expr) return NULL;
    const TypeInfo *retained_array = env_array_expression_info(env, expr);
    if (retained_array) return (TypeInfo *)retained_array;
    if (expr->type == AST_ARRAY_LITERAL && expr->as.array_literal.element_count > 0) {
        ASTNode *first = expr->as.array_literal.elements[0];
        TypeInfo flat = {0};
        FunctionSignature *signature = NULL;
        const TypeInfo *element = try_get_expr_type_info(first, env);
        if (!element) {
            const char *name = get_struct_type_name(first, env);
            if (name && env_get_opaque_type(env, name)) {
                flat = (TypeInfo){.base_type = TYPE_STRUCT, .generic_name = (char *)name};
                element = &flat;
            } else {
                signature = checked_callable_signature_copy(first, env);
                if (signature) {
                    flat = (TypeInfo){.base_type = TYPE_FUNCTION, .fn_sig = signature};
                    element = &flat;
                }
            }
        }
        bool retained = element && type_info_exact_array_element(element);
        if (retained) {
            TypeInfo array = {.base_type = TYPE_ARRAY, .element_type = (TypeInfo *)element};
            if (!env_bind_array_expression(env, expr, &array)) env->opaque_resolution_failed = true;
        }
        free_function_signature(signature);
        if (retained) return (TypeInfo *)env_array_expression_info(env, expr);
    }
    if (expr->type == AST_UNION_CONSTRUCT) return expr->as.union_construct.type_info;
    if (expr->type == AST_TUPLE_LITERAL) {
        const TypeInfo *tuple = env_tuple_literal_info(env, expr);
        return tuple ? (TypeInfo *)tuple : infer_complete_tuple_literal(expr, env);
    }
    if (expr->type == AST_TUPLE_INDEX) {
        TypeInfo *tuple = try_get_expr_type_info(expr->as.tuple_index.tuple, env);
        int index = expr->as.tuple_index.index;
        if (tuple && tuple->base_type == TYPE_TUPLE && tuple->type_param_count &&
            type_info_tuple_valid(tuple) && index >= 0 && index < tuple->tuple_element_count)
            return tuple->type_params[index];
    }
    if (expr->type == AST_IDENTIFIER) {
        Symbol *sym = env_get_var_visible_at(env, expr->as.identifier, expr->line, expr->column);
        if (sym) return sym->type_info;
    }
    if (expr->type == AST_FIELD_ACCESS) {
        if (expr->as.field_access.resolved_type_info)
            return expr->as.field_access.resolved_type_info;
        const char *owner = get_struct_type_name(expr->as.field_access.object, env);
        if (owner) {
            for (int u = 0; u < env->union_count; ++u) {
                UnionDef *def = &env->unions[u];
                size_t length = strlen(def->name);
                if (strncmp(owner, def->name, length) || owner[length] != '.') continue;
                int arm = env_get_union_variant_index(env, def->name, owner + length + 1);
                if (arm < 0) continue;
                for (int field = 0; field < def->variant_field_counts[arm]; ++field) {
                    if (strcmp(def->variant_field_names[arm][field], expr->as.field_access.field_name)) continue;
                    TypeInfo *arguments = try_get_expr_type_info(expr->as.field_access.object, env);
                    expr->as.field_access.resolved_type_info =
                        resolve_union_payload_type_info(def, arm, field, arguments);
                    return expr->as.field_access.resolved_type_info;
                }
            }
        }
        StructDef *record = owner ? env_get_struct(env, owner) : NULL;
        if (record && record->field_type_info) {
            for (int i = 0; i < record->field_count; ++i)
                if (strcmp(record->field_names[i], expr->as.field_access.field_name) == 0)
                    return record->field_type_info[i];
        }
    }
    if (expr->type == AST_MODULE_QUALIFIED_CALL) {
        const char *owner = expr->as.module_qualified_call.module_alias;
        const char *name = expr->as.module_qualified_call.function_name;
        size_t a = strlen(owner), b = strlen(name);
        if (b > SIZE_MAX - 2 || a > SIZE_MAX - b - 2) { env->opaque_resolution_failed = true; return NULL; }
        char *qualified = malloc(a + b + 2);
        if (!qualified) { env->opaque_resolution_failed = true; return NULL; }
        memcpy(qualified, owner, a); qualified[a] = '.'; memcpy(qualified + a + 1, name, b + 1);
        Function *function = env_get_function(env, qualified); free(qualified);
        return function ? function->return_type_info : NULL;
    }
    if (expr->type == AST_CALL) {
        if (expr->as.call.checked_signature)
            return expr->as.call.checked_signature->return_type_info;
        if (expr->as.call.func_expr) {
            FunctionSignature *signature = function_result_signature(expr->as.call.func_expr, env);
            return signature ? signature->return_type_info : NULL;
        }
        if (expr->as.call.name) {
            Symbol *symbol = env_get_var_visible_at(env, expr->as.call.name, expr->line, expr->column);
            if (symbol && symbol->type == TYPE_FUNCTION && symbol->type_info &&
                symbol->type_info->base_type == TYPE_FUNCTION && symbol->type_info->fn_sig)
                return symbol->type_info->fn_sig->return_type_info;
        }
    }
    if (expr->type == AST_CALL && expr->as.call.name && !expr->as.call.func_expr &&
        env_native_array_is_builtin(env, expr->as.call.name, expr->line, expr->column)) {
        const char *name = expr->as.call.name;
        if (expr->as.call.arg_count >= 1 &&
            (!strcmp(name, "filter") || !strcmp(name, "array_slice") || !strcmp(name, "array_remove_at") ||
             (!strcmp(name, "array_push") && env_array_push_is_builtin(env, expr->line, expr->column))))
            return try_get_expr_type_info(expr->as.call.args[0], env);
        if (!strcmp(name, "array_pop") && expr->as.call.arg_count == 1) {
            TypeInfo *array = try_get_expr_type_info(expr->as.call.args[0], env);
            return array && array->base_type == TYPE_ARRAY ? array->element_type : NULL;
        }
        if (expr->as.call.arg_count == 2 && (!strcmp(name, "array_new") || !strcmp(name, "map"))) {
            FunctionSignature *signature = !strcmp(name, "map")
                ? checked_callable_signature_copy(expr->as.call.args[1], env) : NULL;
            TypeInfo flat = {0};
            const TypeInfo *element = NULL;
            if (signature) {
                flat = (TypeInfo){.base_type = signature->return_type, .generic_name = signature->return_struct_name,
                    .fn_sig = signature->return_fn_sig};
                if (signature->return_type_info) flat = *signature->return_type_info;
                if (!flat.fn_sig) flat.fn_sig = signature->return_fn_sig;
                element = &flat;
            } else if (!strcmp(name, "array_new")) {
                element = try_get_expr_type_info(expr->as.call.args[1], env);
                if (!element) {
                    const char *opaque = get_struct_type_name(expr->as.call.args[1], env);
                    if (opaque && env_get_opaque_type(env, opaque)) {
                        flat = (TypeInfo){.base_type = TYPE_STRUCT, .generic_name = (char *)opaque};
                        element = &flat;
                    }
                    else {
                        signature = checked_callable_signature_copy(expr->as.call.args[1], env);
                        if (signature) {
                            flat = (TypeInfo){.base_type = TYPE_FUNCTION, .fn_sig = signature};
                            element = &flat;
                        } else {
                            Type primitive = check_expression(expr->as.call.args[1], env);
                            if (primitive == TYPE_INT || primitive == TYPE_U8 || primitive == TYPE_FLOAT ||
                                primitive == TYPE_BOOL || primitive == TYPE_STRING) {
                                flat = (TypeInfo){.base_type = primitive}; element = &flat;
                            }
                        }
                    }
                }
            }
            TypeInfo array = {.base_type = TYPE_ARRAY, .element_type = (TypeInfo *)element};
            bool affected = element && (type_info_exact_array_element(element) || !strcmp(name, "array_new"));
            if (!affected && signature && !strcmp(name, "map")) affected = type_info_needs_array_context(try_get_expr_type_info(expr->as.call.args[0], env));
            bool retained = affected && env_bind_array_expression(env, expr, &array);
            if (affected && !retained) env->opaque_resolution_failed = true;
            free_function_signature(signature);
            if (retained) return (TypeInfo *)env_array_expression_info(env, expr);
        }
    }
    if (expr->type == AST_CALL && expr->as.call.name) {
        if (!expr->as.call.func_expr && expr->as.call.arg_count == 2 &&
            (strcmp(expr->as.call.name, "at") == 0 ||
             strcmp(expr->as.call.name, "array_get") == 0)) {
            TypeInfo *array = try_get_expr_type_info(expr->as.call.args[0], env);
            if (array && array->base_type == TYPE_ARRAY) return array->element_type;
        }
        Function *func = env_get_function(env, expr->as.call.name);
        if (func && func->return_type_info) return func->return_type_info;
    }
    return NULL;
}

/* I infer this owning snapshot from actual expression facts, never C names. */
static TypeInfo *infer_complete_tuple_literal(ASTNode *literal, Environment *env) {
    int count = literal->as.tuple_literal.element_count;
    if (count < 0 || (size_t)count > SIZE_MAX / sizeof(TypeInfo *)) return NULL;
    TypeInfo *tuple = calloc(1, sizeof *tuple);
    if (!tuple) { env->opaque_resolution_failed = true; return NULL; }
    tuple->base_type = TYPE_TUPLE;
    tuple->tuple_element_count = tuple->type_param_count = count;
    if (count) {
        tuple->type_params = calloc((size_t)count, sizeof *tuple->type_params);
        if (!tuple->type_params) goto allocation_failure;
    }
    for (int i = 0; i < count; ++i) {
        ASTNode *child = literal->as.tuple_literal.elements[i];
        const TypeInfo *info = try_get_expr_type_info(child, env);
        FunctionSignature *signature = NULL;
        TypeInfo flat = {0};
        if (!info) {
            flat.base_type = literal->as.tuple_literal.element_types
                ? literal->as.tuple_literal.element_types[i] : check_expression(child, env);
            const char *name = get_struct_type_name(child, env);
            flat.generic_name = (char *)name;
            if (flat.base_type == TYPE_FUNCTION) {
                signature = checked_callable_signature_copy(child, env);
                flat.fn_sig = signature;
            }
            info = &flat;
        }
        if (info->base_type == TYPE_UNKNOWN ||
            (info->base_type == TYPE_FUNCTION && !info->fn_sig)) {
            free_function_signature(signature); free_payload_type_info(tuple); return NULL;
        }
        tuple->type_params[i] = copy_complete_type_info_checked(info);
        free_function_signature(signature);
        if (!tuple->type_params[i]) goto allocation_failure;
    }
    if (!type_info_tuple_refresh(tuple) || !env_bind_tuple_literal(env, literal, tuple)) goto allocation_failure;
    free_payload_type_info(tuple);
    return (TypeInfo *)env_tuple_literal_info(env, literal);
allocation_failure:
    free_payload_type_info(tuple); env->opaque_resolution_failed = true; return NULL;
}

const TypeInfo *checked_expression_type_info(ASTNode *expr, Environment *env) {
    return try_get_expr_type_info(expr, env);
}
FunctionSignature *checked_callable_signature_copy(ASTNode *expr, Environment *env) {
    if (!expr) return NULL;
    TypeInfo *info = try_get_expr_type_info(expr, env);
    if (info && info->base_type == TYPE_FUNCTION && info->fn_sig) return copy_function_signature(info->fn_sig);
    if (expr->type == AST_IDENTIFIER) {
        Function *function = env_get_function(env, expr->as.identifier);
        if (function) return function_signature_from_function(function);
    }
    FunctionSignature *signature = function_result_signature(expr, env);
    return signature ? copy_function_signature(signature) : NULL;
}

static void check_concrete_union_arrays(Environment *env, const TypeInfo *expected,
                                        ASTNode *value, unsigned depth);
static bool indirect_argument_matches(ASTNode *argument, Environment *env,
                                      const TypeInfo *expected, Type fallback, int depth);
static bool check_opaque_array_callback(ASTNode *call, Environment *env, bool filter) {
    FunctionSignature *signature = checked_callable_signature_copy(call->as.call.args[1], env);
    TypeInfo flat = {0};
    const TypeInfo *parameter = NULL;
    if (signature && signature->param_count == 1) {
        flat = (TypeInfo){.base_type = signature->param_types[0],
            .generic_name = signature->param_struct_names ? signature->param_struct_names[0] : NULL};
        parameter = signature->param_type_info && signature->param_type_info[0]
            ? signature->param_type_info[0] : &flat;
    }
    const TypeInfo *array = try_get_expr_type_info(call->as.call.args[0], env);
    const TypeInfo *result = filter ? array : try_get_expr_type_info(call, env);
    /* The actual callback declaration supplies a literal input's element type.
     * I validate all source leaves before retaining the independent snapshot. */
    ASTNode *source = call->as.call.args[0];
    if (!array && parameter && source->type == AST_ARRAY_LITERAL &&
        (type_info_exact_array_element(parameter) || type_info_needs_array_context(result))) {
        TypeInfo context = {.base_type = TYPE_ARRAY, .element_type = (TypeInfo *)parameter};
        if (indirect_argument_matches(source, env, &context, TYPE_ARRAY, 0)) {
            check_concrete_union_arrays(env, &context, source, 0);
            if (!env_bind_array_expression(env, source, &context)) env->opaque_resolution_failed = true;
            array = try_get_expr_type_info(source, env);
        }
        if (filter) result = array;
    }
    if (!type_info_needs_array_context(array) && !type_info_needs_array_context(result) &&
        !type_info_exact_array_element(parameter)) {
        free_function_signature(signature); return true;
    }
    bool ok = signature && parameter && array && array->base_type == TYPE_ARRAY && array->element_type &&
        type_infos_equal(array->element_type, parameter) && (!filter || signature->return_type == TYPE_BOOL);
    free_function_signature(signature);
    if (!ok) emit_context_error("E001 TYPE MISMATCH", call->line, call->column, 1,
        "I require the same complete element annotation in this callback.",
        "Preserve the declaring module and every tuple, union and callable child.");
    return ok;
}

/* I compare opaque values by retained declaration identity, never ABI spelling.
 * A zero literal is the existing null spelling only for an opaque destination. */
static bool check_opaque_value(Environment *env, Type expected_type, const char *name,
                               ASTNode *value) {
    if (!value || (expected_type != TYPE_STRUCT && expected_type != TYPE_OPAQUE)) return true;
    OpaqueTypeDef *expected = name ? env_get_opaque_type(env, name) : NULL;
    const char *actual_name = get_struct_type_name(value, env);
    TypeInfo *actual_info = try_get_expr_type_info(value, env);
    if (!actual_name && actual_info)
        actual_name = actual_info->opaque_type_name ? actual_info->opaque_type_name : actual_info->generic_name;
    OpaqueTypeDef *actual = actual_name ? env_get_opaque_type(env, actual_name) : NULL;
    if (!expected && !actual) return expected_type != TYPE_OPAQUE;
    if (expected && value->type == AST_NUMBER && value->as.number == 0) return true;
    if (expected && actual && !strcmp(expected->identity, actual->identity)) return true;
    emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
        "I require the same opaque declaration identity at this value boundary.",
        "Use the declared opaque handle; matching C pointer spellings do not establish identity.");
    return false;
}

static Type type_from_typeinfo(TypeInfo *info, const char **out_struct_name) {
    if (out_struct_name) *out_struct_name = NULL;
    if (!info) return TYPE_UNKNOWN;

    if (info->base_type == TYPE_STRUCT && info->generic_name) {
        if (out_struct_name) *out_struct_name = info->generic_name;
        return TYPE_STRUCT;
    }
    return info->base_type;
}

/* Utility functions */
Type token_to_type(TokenType token) {
    switch (token) {
        case TOKEN_TYPE_INT: return TYPE_INT;
        case TOKEN_TYPE_FLOAT: return TYPE_FLOAT;
        case TOKEN_TYPE_BOOL: return TYPE_BOOL;
        case TOKEN_TYPE_STRING: return TYPE_STRING;
        case TOKEN_TYPE_BSTRING: return TYPE_BSTRING;
        case TOKEN_TYPE_VOID: return TYPE_VOID;
        default: return TYPE_UNKNOWN;
    }
}

const char *type_to_string(Type type) {
    switch (type) {
        case TYPE_INT: return "int";
        case TYPE_U8: return "u8";
        case TYPE_FLOAT: return "float";
        case TYPE_BOOL: return "bool";
        case TYPE_STRING: return "string";
        case TYPE_VOID: return "void";
        case TYPE_ARRAY: return "array";
        case TYPE_STRUCT: return "struct";
        case TYPE_ENUM: return "enum";
        case TYPE_UNION: return "union";
        case TYPE_FUNCTION: return "function";
        case TYPE_LIST_INT: return "list_int";
        case TYPE_LIST_STRING: return "list_string";
        case TYPE_HASHMAP: return "HashMap";
        case TYPE_OPEN_RECORD: return "open_record";
        case TYPE_UNKNOWN: return "unknown";
        default: return "unknown";
    }
}

/* Forward declarations */
static Type check_statement(TypeChecker *tc, ASTNode *node);

/* Recursion depth tracking to prevent stack overflow */
static int g_check_expr_depth = 0;
static int g_check_stmt_depth = 0;
#define MAX_CHECK_EXPR_DEPTH 2000
#define MAX_CHECK_STMT_DEPTH 2000

/* Check if an AST node contains calls to extern functions */
static bool contains_extern_calls(ASTNode *node, Environment *env) {
    if (!node) return false;
    
    switch (node->type) {
        case AST_CALL: {
            const char *func_name = node->as.call.name;
            Function *func = env_get_function(env, func_name);
            if (func && func->is_extern) {
                return true;
            }
            /* Check arguments recursively */
            for (int i = 0; i < node->as.call.arg_count; i++) {
                if (contains_extern_calls(node->as.call.args[i], env)) {
                    return true;
                }
            }
            return false;
        }
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++) {
                if (contains_extern_calls(node->as.block.statements[i], env)) {
                    return true;
                }
            }
            return false;
        case AST_IF:
            if (contains_extern_calls(node->as.if_stmt.condition, env)) return true;
            if (contains_extern_calls(node->as.if_stmt.then_branch, env)) return true;
            if (node->as.if_stmt.else_branch && contains_extern_calls(node->as.if_stmt.else_branch, env)) return true;
            return false;
        case AST_COND:
            for (int i = 0; i < node->as.cond_expr.clause_count; i++) {
                if (contains_extern_calls(node->as.cond_expr.conditions[i], env)) return true;
                if (contains_extern_calls(node->as.cond_expr.values[i], env)) return true;
            }
            if (contains_extern_calls(node->as.cond_expr.else_value, env)) return true;
            return false;
        case AST_WHILE:
            if (contains_extern_calls(node->as.while_stmt.condition, env)) return true;
            if (contains_extern_calls(node->as.while_stmt.body, env)) return true;
            return false;
        case AST_RETURN:
            if (node->as.return_stmt.value && contains_extern_calls(node->as.return_stmt.value, env)) return true;
            return false;
        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; i++) {
                if (contains_extern_calls(node->as.prefix_op.args[i], env)) return true;
            }
            return false;
        case AST_ARRAY_LITERAL:
            for (int i = 0; i < node->as.array_literal.element_count; i++) {
                if (contains_extern_calls(node->as.array_literal.elements[i], env)) return true;
            }
            return false;
        case AST_FIELD_ACCESS:
            return contains_extern_calls(node->as.field_access.object, env);
        case AST_LET:
            return contains_extern_calls(node->as.let.value, env);
        case AST_SET:
            return contains_extern_calls(node->as.set.value, env);
        default:
            return false;
    }
}

#include "typechecker_purity.c"
#include "typechecker_passive.c"

/* Check if types are compatible */
static Type type_from_typeinfo(TypeInfo *info, const char **out_struct_name);
static bool types_match(Type t1, Type t2) {
    if (t1 == t2) return true;

    /* TYPE_UNKNOWN is always compatible — polymorphic builtins (map/filter/reduce/fold)
     * and higher-order functions return TYPE_UNKNOWN; runtime enforces correctness. */
    if (t1 == TYPE_UNKNOWN || t2 == TYPE_UNKNOWN) return true;

    /* u8 is assignment-compatible with int (u8 is codegen'd as uint8_t). */
    if ((t1 == TYPE_U8 && t2 == TYPE_INT) ||
        (t1 == TYPE_INT && t2 == TYPE_U8)) {
        return true;
    }
    
    /* Generic lists match with int (list functions return int handles) */
    if ((t1 == TYPE_LIST_GENERIC && t2 == TYPE_INT) ||
        (t1 == TYPE_INT && t2 == TYPE_LIST_GENERIC)) {
        return true;
    }
    
    /* Function types match with int when checking function-typed parameters */
    /* This is a temporary workaround - function-typed parameters return TYPE_INT as placeholder */
    if ((t1 == TYPE_FUNCTION && t2 == TYPE_INT) ||
        (t1 == TYPE_INT && t2 == TYPE_FUNCTION)) {
        return true;  /* Allow for now - runtime will handle */
    }
    
    /* Enums match with int (enums are represented as integers in C) */
    if ((t1 == TYPE_ENUM && t2 == TYPE_INT) ||
        (t1 == TYPE_INT && t2 == TYPE_ENUM)) {
        return true;
    }

    /* Enums are also assignment-compatible with u8 (useful for byte-sized tags). */
    if ((t1 == TYPE_ENUM && t2 == TYPE_U8) ||
        (t1 == TYPE_U8 && t2 == TYPE_ENUM)) {
        return true;
    }

    /* Open records are compatible with any struct (row-polymorphism). */
    if (t1 == TYPE_OPEN_RECORD || t2 == TYPE_OPEN_RECORD) return true;

    return false;
}

/* I resolve nominal enum annotations before selecting scalar array operations. */
static Type resolved_array_element(Type element, const char *name, Environment *env) {
    return element == TYPE_STRUCT && name && env_get_enum(env, name) ? TYPE_ENUM : element;
}

/* I validate the inferred literal kind before annotation propagation changes it. */
static bool check_array_literal_annotation(TypeChecker *tc, ASTNode *literal, Type expected, const char *name) {
    expected = resolved_array_element(expected, name, tc->env);
    if (literal->as.array_literal.element_count > 0 &&
            !types_match(literal->as.array_literal.element_type, expected)) {
        char message[256];
        snprintf(message, sizeof message, "I expected array elements of type %s, but found %s.",
                 type_to_string(expected), type_to_string(literal->as.array_literal.element_type));
        emit_context_error("E001 TYPE MISMATCH", literal->line, literal->column, 1,
                           message, "Use elements matching the array annotation.");
        tc->has_error = true;
        return false;
    }
    literal->as.array_literal.element_type = expected;
    return true;
}

static bool hashmap_extract_kv(TypeInfo *hm_info, Type *out_key, Type *out_value) {
    if (out_key) *out_key = TYPE_UNKNOWN;
    if (out_value) *out_value = TYPE_UNKNOWN;
    if (!hm_info || !hm_info->generic_name) return false;
    if (strcmp(hm_info->generic_name, "HashMap") != 0) return false;
    if (hm_info->type_param_count != 2 || !hm_info->type_params) return false;

    if (out_key) *out_key = type_from_typeinfo(hm_info->type_params[0], NULL);
    if (out_value) *out_value = type_from_typeinfo(hm_info->type_params[1], NULL);
    return true;
}

/* I borrow nominal element identity from the array declaration. */
static const char *array_record_name(ASTNode *array, Environment *env) {
    if (!array) return NULL;
    if (array->type == AST_ARRAY_LITERAL && array->as.array_literal.element_count > 0)
        return get_struct_type_name(array->as.array_literal.elements[0], env);
    TypeInfo *info = try_get_expr_type_info(array, env);
    if (info && info->base_type == TYPE_ARRAY && info->element_type &&
        info->element_type->base_type == TYPE_STRUCT && info->element_type->generic_name)
        return info->element_type->generic_name;
    if (array->type == AST_IDENTIFIER) {
        Symbol *symbol = env_get_var_visible_at(env, array->as.identifier, array->line, array->column);
        return symbol && symbol->type == TYPE_ARRAY && symbol->element_type == TYPE_STRUCT
            ? symbol->struct_type_name : NULL;
    }
    if (array->type == AST_FIELD_ACCESS) {
        const char *owner = get_struct_type_name(array->as.field_access.object, env);
        StructDef *record = owner ? env_get_struct(env, owner) : NULL;
        if (!record) return NULL;
        for (int i = 0; i < record->field_count; i++)
            if (!strcmp(record->field_names[i], array->as.field_access.field_name) &&
                record->field_types[i] == TYPE_ARRAY && record->field_element_types &&
                record->field_element_types[i] == TYPE_STRUCT && record->field_type_names)
                return record->field_type_names[i];
    }
    if (array->type == AST_MODULE_QUALIFIED_CALL) {
        const char *alias = array->as.module_qualified_call.module_alias;
        const char *name = array->as.module_qualified_call.function_name;
        size_t length = strlen(alias) + strlen(name) + 2;
        char *qualified = malloc(length);
        if (!qualified) return NULL;
        snprintf(qualified, length, "%s.%s", alias, name);
        Function *function = env_get_function(env, qualified);
        free(qualified);
        if (function && function->return_type == TYPE_ARRAY && function->return_element_type == TYPE_STRUCT)
            return function->return_struct_type_name;
        return NULL;
    }
    if (array->type == AST_CALL && !array->as.call.func_expr && array->as.call.name) {
        const char *name = array->as.call.name;
        if (!env_native_array_is_builtin(env, name, array->line, array->column)) {
            const TypeInfo *info = try_get_expr_type_info(array, env);
            if (info && info->base_type == TYPE_ARRAY && info->element_type)
                return info->element_type->generic_name;
            Function *function = env_get_function(env, name);
            return function && function->return_type == TYPE_ARRAY ? function->return_struct_type_name : NULL;
        }
        bool builtin_push = !strcmp(name, "array_push") &&
            env_array_push_is_builtin(env, array->line, array->column);
        if (((builtin_push || !strcmp(name, "filter")) && array->as.call.arg_count == 2) ||
            (!strcmp(name, "array_slice") && array->as.call.arg_count == 3))
            return array_record_name(array->as.call.args[0], env);
        if (!strcmp(name, "array_new") && array->as.call.arg_count == 2)
            return get_struct_type_name(array->as.call.args[1], env);
        if (!strcmp(name, "map") && array->as.call.arg_count == 2 &&
            array->as.call.args[1]->type == AST_IDENTIFIER) {
            ASTNode *callback = array->as.call.args[1];
            if (!env_get_var_visible_at(env, callback->as.identifier, callback->line, callback->column)) {
                Function *transform = env_get_function(env, callback->as.identifier);
                if (transform && transform->return_type == TYPE_STRUCT)
                    return transform->return_struct_type_name;
            }
        }
        Function *function = env_get_function(env, array->as.call.name);
        if (function && function->return_type == TYPE_ARRAY && function->return_element_type == TYPE_STRUCT)
            return function->return_struct_type_name;
    }
    return NULL;
}

/* I compare declarations, including their module identity, at array boundaries. */
static bool check_record_array_contract(Environment *env, Type type, Type element,
                                         const char *name, ASTNode *value) {
    if (type != TYPE_ARRAY || element != TYPE_STRUCT || !name || !value) return true;
    StructDef *expected = env_get_struct(env, name);
    if (!expected) return true; /* Enum and formal-generic contexts have other rules. */
    if (value->type == AST_ARRAY_LITERAL && value->as.array_literal.element_count == 0) return true;
    const char *actual_name = array_record_name(value, env);
    StructDef *actual = actual_name ? env_get_struct(env, actual_name) : NULL;
    if (actual == expected) return true;
    emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
        "I require the declared nominal record type for this array.",
        "Match the array element declaration, including its module identity.");
    return false;
}

/* I retain fixed union field contracts without resolving formal parameters here. */
static void check_union_record_array_contract(Environment *env, UnionDef *def,
                                              int arm, const char *field, ASTNode *value) {
    if (!def->variant_field_type_info || !def->variant_field_type_info[arm]) return;
    for (int i = 0; i < def->variant_field_counts[arm]; ++i) {
        if (strcmp(def->variant_field_names[arm][i], field)) continue;
        TypeInfo *info = def->variant_field_type_info[arm][i];
        if (!info || info->base_type != TYPE_ARRAY || !info->element_type) return;
        TypeInfo *element = info->element_type;
        for (int j = 0; element->generic_name && j < def->generic_param_count; ++j)
            if (!strcmp(element->generic_name, def->generic_params[j])) return;
        check_record_array_contract(env, TYPE_ARRAY, element->base_type,
                                    element->generic_name, value);
        return;
    }
}

/* I retain complete concrete trees for native nested payload substitution. */
static void register_native_union_context(Environment *env, const TypeInfo *info, unsigned depth) {
    if (!info) return;
    if (depth > 128) {
        env->opaque_resolution_failed = true;
        fprintf(stderr, "I cannot retain a native type expansion deeper than 128 edges\n");
        return;
    }
    register_native_union_context(env, info->element_type, depth + 1);
    for (int i = 0; info->type_params && i < info->type_param_count; ++i)
        register_native_union_context(env, info->type_params[i], depth + 1);
    if (info->fn_sig) {
        FunctionSignature *signature = info->fn_sig;
        for (int i = 0; signature->param_type_info && i < signature->param_count; ++i)
            register_native_union_context(env, signature->param_type_info[i], depth + 1);
        register_native_union_context(env, signature->return_type_info, depth + 1);
        if (signature->return_fn_sig) {
            TypeInfo result = {.base_type = TYPE_FUNCTION, .fn_sig = signature->return_fn_sig};
            register_native_union_context(env, &result, depth + 1);
        }
    }
    UnionDef *def = info->generic_name ? env_get_union(env, info->generic_name) : NULL;
    if (!def || !def->generic_param_count || def->generic_param_count != info->type_param_count) return;
    bool added = false;
    if (opaque_type_info_present(info)) {
        if (!env_register_opaque_union_context(env, info, &added)) {
            env->opaque_resolution_failed = true;
            fprintf(stderr, "I cannot retain a complete opaque-bearing union annotation\n");
            return;
        }
    } else {
    char **names = calloc((size_t)info->type_param_count, sizeof(char*));
    if (!names) return;
    bool complete = true;
    for (int i = 0; i < info->type_param_count; ++i) {
        names[i] = typeinfo_to_generic_arg_name(info->type_params[i]);
        if (!names[i]) complete = false;
    }
    if (complete) env_register_union_instantiation(env, info->generic_name,
                                                  (const char**)names, info->type_param_count);
    for (int i = 0; complete && i < env->generic_instance_count; ++i) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (strcmp(inst->generic_name, info->generic_name) || inst->type_arg_count != info->type_param_count) continue;
        bool matches = true;
        for (int j = 0; j < info->type_param_count; ++j)
            if (strcmp(inst->type_arg_names[j], names[j])) matches = false;
        if (matches && !inst->type_info) {
            inst->type_info = copy_payload_type_info(info);
            added = inst->type_info != NULL;
        }
    }
    for (int i = 0; i < info->type_param_count; ++i) free(names[i]);
    free(names);
    }
    if (!added) return;
    for (int arm = 0; arm < def->variant_count; ++arm)
        for (int field = 0; field < def->variant_field_counts[arm]; ++field) {
            TypeInfo *payload = resolve_union_payload_type_info(def, arm, field, info);
            register_native_union_context(env, payload, depth + 1);
            free_payload_type_info(payload);
        }
}

static void register_native_function_context(Environment *env, const Function *function) {
    register_native_union_context(env, function->return_type_info, 0);
    if (function->return_fn_sig) {
        TypeInfo result = {.base_type = TYPE_FUNCTION, .fn_sig = function->return_fn_sig};
        register_native_union_context(env, &result, 0);
    }
    for (int i = 0; i < function->param_count; ++i) {
        register_native_union_context(env, function->params[i].type_info, 0);
        if (function->params[i].fn_sig) {
            TypeInfo parameter = {.base_type = TYPE_FUNCTION, .fn_sig = function->params[i].fn_sig};
            register_native_union_context(env, &parameter, 0);
        }
    }
}

/* I inspect only declaration-bearing opaque leaves; ordinary compatibility
 * remains in the existing checks. All views are borrowed for this call. */
static bool opaque_annotation_present(Environment *, const TypeInfo *, unsigned);
static bool opaque_signature_present(Environment *env, const FunctionSignature *sig, unsigned depth) {
    if (!sig) return false;
    if (depth > 128) return true;
    if ((sig->return_struct_name && env_get_opaque_type(env, sig->return_struct_name)) ||
        opaque_annotation_present(env, sig->return_type_info, depth + 1) ||
        opaque_signature_present(env, sig->return_fn_sig, depth + 1)) return true;
    for (int i = 0; i < sig->param_count; ++i)
        if ((sig->param_struct_names && sig->param_struct_names[i] &&
             env_get_opaque_type(env, sig->param_struct_names[i])) ||
            (sig->param_type_info && opaque_annotation_present(env, sig->param_type_info[i], depth + 1))) return true;
    return false;
}
static bool opaque_annotation_present(Environment *env, const TypeInfo *info, unsigned depth) {
    if (!info) return false;
    if (depth > 128) return true;
    if (info->base_type == TYPE_OPAQUE ||
        (info->generic_name && env_get_opaque_type(env, info->generic_name)) ||
        opaque_annotation_present(env, info->element_type, depth + 1) ||
        opaque_signature_present(env, info->fn_sig, depth + 1)) return true;
    for (int i = 0; info->type_params && i < info->type_param_count; ++i)
        if (opaque_annotation_present(env, info->type_params[i], depth + 1)) return true;
    for (int i = 0; info->tuple_type_names && i < info->tuple_element_count; ++i)
        if (info->tuple_type_names[i] && env_get_opaque_type(env, info->tuple_type_names[i])) return true;
    for (int i = 0; info->row_field_type_names && i < info->row_field_count; ++i)
        if (info->row_field_type_names[i] && env_get_opaque_type(env, info->row_field_type_names[i])) return true;
    return false;
}

/* I resolve payload annotations and retain owned constructor context. */
static void check_concrete_union_arrays(Environment *env, const TypeInfo *expected,
                                        ASTNode *value, unsigned depth) {
    if (!expected || !value) return;
    register_native_union_context(env, expected, 0);
    if (depth > 128) {
        emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
            "I cannot resolve this deeply nested payload context.", "Reduce the nesting depth.");
        return;
    }
    if (value->type == AST_IF) {
        check_concrete_union_arrays(env, expected, value->as.if_stmt.then_branch, depth + 1);
        check_concrete_union_arrays(env, expected, value->as.if_stmt.else_branch, depth + 1);
        return;
    }
    if (value->type == AST_COND) {
        for (int i = 0; i < value->as.cond_expr.clause_count; ++i)
            check_concrete_union_arrays(env, expected, value->as.cond_expr.values[i], depth + 1);
        check_concrete_union_arrays(env, expected, value->as.cond_expr.else_value, depth + 1);
        return;
    }
    if (value->type == AST_MATCH) {
        for (int i = 0; i < value->as.match_expr.arm_count; ++i)
            check_concrete_union_arrays(env, expected, value->as.match_expr.arm_bodies[i], depth + 1);
        return;
    }
    if (value->type == AST_BLOCK) {
        if (value->as.block.count)
            check_concrete_union_arrays(env, expected, value->as.block.statements[value->as.block.count - 1], depth + 1);
        return;
    }
    if (value->type == AST_RETURN) {
        check_concrete_union_arrays(env, expected, value->as.return_stmt.value, depth + 1);
        return;
    }
    if (expected->base_type == TYPE_FUNCTION && expected->fn_sig) {
        FunctionSignature *signature = checked_callable_signature_copy(value, env);
        TypeInfo actual = {.base_type = TYPE_FUNCTION, .fn_sig = signature};
        bool equal = signature && type_infos_equal(expected, &actual);
        free_function_signature(signature);
        if (!equal) emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
            "I require the complete declared callable annotation at this value boundary.",
            "Preserve every parameter, result and nominal declaration identity.");
        return;
    }
    if (expected->base_type == TYPE_TUPLE && value->type == AST_TUPLE_LITERAL) {
        if (!indirect_argument_matches(value, env, expected, TYPE_TUPLE, (int)depth)) {
            emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
                "I require each tuple child to match its complete declared annotation.",
                "Preserve tuple arity, nested annotations and nominal declaration identities.");
            return;
        }
        if (!env_bind_tuple_literal(env, value, expected)) {
            env->opaque_resolution_failed = true;
            emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
                "I cannot retain one complete checked tuple literal context.",
                "Preserve its exact element annotations and allocation boundary.");
            return;
        }
    }
    if ((expected->base_type == TYPE_STRUCT || expected->base_type == TYPE_OPAQUE) &&
        opaque_annotation_present(env, expected, 0)) {
        const char *name = expected->opaque_type_name ? expected->opaque_type_name : expected->generic_name;
        if (name && env_get_opaque_type(env, name)) {
            check_opaque_value(env, expected->base_type, name, value);
            return;
        }
    }
    if (opaque_annotation_present(env, expected, 0) ||
        (expected->base_type == TYPE_ARRAY && type_info_needs_array_context(expected))) {
        if (expected->base_type == TYPE_ARRAY && expected->element_type && value->type == AST_CALL &&
            !value->as.call.func_expr && value->as.call.name && value->as.call.arg_count == 2 &&
            env_native_array_is_builtin(env, value->as.call.name, value->line, value->column) &&
            (!strcmp(value->as.call.name, "array_new") ||
             (!strcmp(value->as.call.name, "array_push") && env_array_push_is_builtin(env, value->line, value->column)))) {
            if (!strcmp(value->as.call.name, "array_push"))
                check_concrete_union_arrays(env, expected, value->as.call.args[0], depth + 1);
            check_concrete_union_arrays(env, expected->element_type, value->as.call.args[1], depth + 1);
            if (!env_bind_array_expression(env, value, expected)) env->opaque_resolution_failed = true;
            return;
        }
        if (expected->base_type == TYPE_ARRAY && value->type == AST_ARRAY_LITERAL) {
            if (!env_bind_array_expression(env, value, expected)) {
                env->opaque_resolution_failed = true; return;
            }
            for (int i = 0; i < value->as.array_literal.element_count; ++i)
                check_concrete_union_arrays(env, expected->element_type,
                    value->as.array_literal.elements[i], depth + 1);
            return;
        }
        if (expected->base_type == TYPE_TUPLE && value->type == AST_TUPLE_LITERAL &&
            expected->tuple_element_count == value->as.tuple_literal.element_count && expected->tuple_types) {
            for (int i = 0; i < expected->tuple_element_count; ++i) {
                TypeInfo element = {.base_type = expected->tuple_types[i],
                    .generic_name = expected->tuple_type_names ? expected->tuple_type_names[i] : NULL};
                const TypeInfo *child = type_info_tuple_element(expected, i, &element);
                if (!child) { env->opaque_resolution_failed = true; return; }
                check_concrete_union_arrays(env, child, value->as.tuple_literal.elements[i], depth + 1);
            }
            return;
        }
        const TypeInfo *actual = try_get_expr_type_info(value, env);
        bool union_constructor = expected->generic_name && env_get_union(env, expected->generic_name) &&
            (value->type == AST_UNION_CONSTRUCT || value->type == AST_STRUCT_LITERAL);
        /* Function declarations have a separate complete signature comparison. */
        bool direct_function = expected->base_type == TYPE_FUNCTION &&
            value->type == AST_IDENTIFIER && env_get_function(env, value->as.identifier);
        if (!direct_function && !union_constructor && (!actual || !type_infos_equal(expected, actual)))
            emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
                "I require complete matching opaque identities inside this annotation.",
                "Preserve the declaring module through each container and callable type.");
        else if (expected->base_type == TYPE_ARRAY && !env_bind_array_expression(env, value, expected))
            env->opaque_resolution_failed = true;
    }
    if (expected->base_type == TYPE_HASHMAP && value->type == AST_CALL &&
        !value->as.call.func_expr && value->as.call.name &&
        strcmp(value->as.call.name, "map_new") == 0) {
        Type key = TYPE_UNKNOWN, item = TYPE_UNKNOWN;
        if (!hashmap_extract_kv((TypeInfo*)expected, &key, &item) ||
            (key != TYPE_INT && key != TYPE_STRING) ||
            (item != TYPE_INT && item != TYPE_STRING)) {
            emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
                "I require int or string map keys and values.", "Declare HashMap<K,V> with supported scalar types.");
            return;
        }
        char *name = typeinfo_to_generic_arg_name((TypeInfo*)expected);
        if (!name) return;
        free(value->as.call.return_struct_type_name);
        value->as.call.return_struct_type_name = name;
        value->as.call.map_key_type = key;
        value->as.call.map_value_type = item;
        value->as.call.map_context_checked = true;
        env_register_hashmap_instantiation(env, type_to_string(key), type_to_string(item));
        return;
    }
    if (expected->base_type == TYPE_HASHMAP) {
        Type expected_key, expected_value, actual_key, actual_value;
        TypeInfo *actual = try_get_expr_type_info(value, env);
        if (hashmap_extract_kv((TypeInfo*)expected, &expected_key, &expected_value) &&
            hashmap_extract_kv(actual, &actual_key, &actual_value) &&
            (expected_key != actual_key || expected_value != actual_value)) {
            char message[256];
            snprintf(message, sizeof(message),
                "I require HashMap<%s,%s>, but this value has HashMap<%s,%s>.",
                type_to_string(expected_key), type_to_string(expected_value),
                type_to_string(actual_key), type_to_string(actual_value));
            emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
                message, "Match both declared map key and value types.");
        }
        return;
    }
    if (expected->base_type == TYPE_ARRAY && expected->element_type) {
        const TypeInfo *element = expected->element_type;
        if (element->base_type == TYPE_STRUCT)
            check_record_array_contract(env, TYPE_ARRAY, TYPE_STRUCT, element->generic_name, value);
        else if (value->type == AST_ARRAY_LITERAL) {
            for (int i = 0; i < value->as.array_literal.element_count; ++i)
                check_concrete_union_arrays(env, element, value->as.array_literal.elements[i], depth + 1);
        } else if (element->base_type == TYPE_ARRAY) {
            const TypeInfo *wanted = expected;
            const TypeInfo *actual = try_get_expr_type_info(value, env);
            while (wanted && wanted->base_type == TYPE_ARRAY) {
                wanted = wanted->element_type;
                actual = actual && actual->base_type == TYPE_ARRAY ? actual->element_type : NULL;
            }
            StructDef *record = wanted && wanted->base_type == TYPE_STRUCT && wanted->generic_name
                ? env_get_struct(env, wanted->generic_name) : NULL;
            if (record && (!actual || actual->base_type != TYPE_STRUCT || !actual->generic_name ||
                           env_get_struct(env, actual->generic_name) != record))
                emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
                    "I require the declared nominal record type for this array.",
                    "Match the nested array element declaration, including its module identity.");
        }
        return;
    }
    if (!expected->generic_name) return;
    UnionDef *def = env_get_union(env, expected->generic_name);
    if (!def) return;
    const char *variant = NULL;
    char **names = NULL;
    ASTNode **values = NULL;
    int count = 0;
    if (value->type == AST_STRUCT_LITERAL && value->as.struct_literal.struct_name) {
        const char *spelling = value->as.struct_literal.struct_name;
        const char *dot = strrchr(spelling, '.');
        if (!dot) return;
        size_t length = (size_t)(dot - spelling);
        char *owner = malloc(length + 1);
        if (!owner) return;
        memcpy(owner, spelling, length);
        owner[length] = '\0';
        bool matches = env_get_union(env, owner) == def;
        free(owner);
        if (!matches) return;
        variant = dot + 1;
        names = value->as.struct_literal.field_names;
        values = value->as.struct_literal.field_values;
        count = value->as.struct_literal.field_count;
    } else if (value->type == AST_UNION_CONSTRUCT) {
        if (env_get_union(env, value->as.union_construct.union_name) != def) return;
        /* I keep explicit arguments before applying an inferred context. */
        const TypeInfo *actual = value->as.union_construct.type_info;
        if (actual && actual->type_param_count > 0) {
            bool exact = actual->type_param_count == expected->type_param_count;
            for (int i = 0; exact && i < actual->type_param_count; ++i)
                exact = actual->type_params && expected->type_params &&
                    type_infos_equal(actual->type_params[i], expected->type_params[i]);
            if (!exact) {
                emit_context_error("E001 TYPE MISMATCH", value->line, value->column, 1,
                    "I require the constructor's explicit generic arguments to match the declared union context.",
                    "Preserve the concrete type arguments, including variants without payloads.");
                return;
            }
        }
        variant = value->as.union_construct.variant_name;
        names = value->as.union_construct.field_names;
        values = value->as.union_construct.field_values;
        count = value->as.union_construct.field_count;
    } else return;
    int arm = env_get_union_variant_index(env, expected->generic_name, variant);
    if (arm < 0) return;
    if (expected->type_param_count > 0) {
        TypeInfo *context = copy_payload_type_info(expected);
        if (!context) return;
        if (value->type == AST_STRUCT_LITERAL) {
            if (value->as.struct_literal.spread_source) { free_payload_type_info(context); return; }
            char *owner = strdup(expected->generic_name);
            char *selected = strdup(variant);
            if (!owner || !selected) { free(owner); free(selected); free_payload_type_info(context); return; }
            free(value->as.struct_literal.struct_name);
            memset(&value->as, 0, sizeof(value->as));
            value->type = AST_UNION_CONSTRUCT;
            value->as.union_construct.union_name = owner;
            value->as.union_construct.variant_name = selected;
            value->as.union_construct.field_names = names;
            value->as.union_construct.field_values = values;
            value->as.union_construct.field_count = count;
        } else free_payload_type_info(value->as.union_construct.type_info);
        value->as.union_construct.type_info = context;
    }
    for (int i = 0; i < count; ++i) {
        for (int field = 0; field < def->variant_field_counts[arm]; ++field) {
            if (strcmp(names[i], def->variant_field_names[arm][field])) continue;
            TypeInfo *payload = resolve_union_payload_type_info(def, arm, field, expected);
            check_concrete_union_arrays(env, payload, values[i], depth + 1);
            free_payload_type_info(payload);
            break;
        }
    }
}

/* Helper: Get the struct type name from an expression (returns NULL if not a struct) */
const char *get_struct_type_name(ASTNode *expr, Environment *env) {
    if (expr && expr->type == AST_IDENTIFIER) {
        Symbol *symbol = env_get_var_visible_at(env, expr->as.identifier, expr->line, expr->column);
        if (symbol && (symbol->type == TYPE_BORROW_SHARED || symbol->type == TYPE_BORROW_MUT)) return symbol->struct_type_name;
    }
    if (!expr) return NULL;
    
    switch (expr->type) {
        case AST_TUPLE_INDEX: {
            TypeInfo *child = try_get_expr_type_info(expr, env);
            return child ? (child->generic_name ? child->generic_name : child->opaque_type_name) : NULL;
        }
        case AST_BLOCK:
            if (expr->as.block.count > 0) {
                ASTNode *tail = expr->as.block.statements[expr->as.block.count - 1];
                if (ast_is_value_expression(tail->type)) return get_struct_type_name(tail, env);
            }
            return NULL;
        case AST_MATCH:
            for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                const char *name = get_struct_type_name(expr->as.match_expr.arm_bodies[i], env);
                if (name) return name;
            }
            return NULL;
        case AST_UNION_CONSTRUCT: {
            TypeInfo *info = expr->as.union_construct.type_info;
            if (info && info->generic_name && info->type_param_count > 0) {
                char *name = typeinfo_to_generic_arg_name(info);
                const char *registered = NULL;
                for (int i = 0; name && i < env->generic_instance_count; i++) {
                    if (strcmp(env->generic_instances[i].concrete_name, name) == 0) {
                        registered = env->generic_instances[i].concrete_name;
                        break;
                    }
                }
                free(name);
                return registered;
            }
            return expr->as.union_construct.union_name;
        }
        case AST_STRUCT_LITERAL: {
            const char *name = expr->as.struct_literal.struct_name;
            /* A dotted constructor produces the union, not its selected payload. */
            for (int i = 0; name && i < env->union_count; ++i) {
                UnionDef *def = &env->unions[i];
                size_t length = strlen(def->name);
                if (strncmp(name, def->name, length) || name[length] != '.') continue;
                for (int v = 0; v < def->variant_count; ++v)
                    if (!strcmp(name + length + 1, def->variant_names[v])) return def->name;
            }
            return name;
        }
            
        case AST_IDENTIFIER: {
            Symbol *sym = env_get_var_visible_at(env, expr->as.identifier, expr->line, expr->column);
            if (sym && (sym->type == TYPE_STRUCT || sym->type == TYPE_UNION)) {
                return sym->struct_type_name;
            }
            return NULL;
        }
        
        case AST_CALL: {
            if (!expr->as.call.func_expr && expr->as.call.name && expr->as.call.arg_count == 2 &&
                (!strcmp(expr->as.call.name, "at") || !strcmp(expr->as.call.name, "array_get")))
                return array_record_name(expr->as.call.args[0], env);
            /* Check if return_struct_type_name was set by type checker (for generic list get) */
            if (expr->as.call.return_struct_type_name) {
                return expr->as.call.return_struct_type_name;
            }
            
            /* Check if function returns a struct */
            Function *func = env_get_function(env, expr->as.call.name);
            if (func && (func->return_type == TYPE_STRUCT || func->return_type == TYPE_UNION)) {
                return func->return_struct_type_name;
            }
            
            /* Special handling for generic list get functions: List_TypeName_get */
            const char *func_name = expr->as.call.name;
            if (func_name && strncmp(func_name, "List_", 5) == 0) {
                const char *func_suffix = strrchr(func_name, '_');
                if (func_suffix && strcmp(func_suffix, "_get") == 0) {
                    /* Extract type name: "List_MyToken_get" -> "MyToken" */
                    const char *type_start = func_name + 5;  /* Skip "List_" */
                    int type_name_len = (int)(func_suffix - type_start);
                    if (type_name_len > 0) {
                        char *type_name = malloc(type_name_len + 1);
                        strncpy(type_name, type_start, type_name_len);
                        type_name[type_name_len] = '\0';
                        
                        /* Check if this type name exists as a struct */
                        StructDef *sdef = env_get_struct(env, type_name);
                        if (sdef) {
                            /* Return a copy that will be used by the caller */
                            char *result = strdup(type_name);
                            free(type_name);
                            return result;
                        }
                        free(type_name);
                    }
                }
            }
            
            return NULL;
        }
        
        case AST_FIELD_ACCESS: {
            /* Get the struct type of the object */
            const char *object_struct_name = get_struct_type_name(expr->as.field_access.object, env);
            if (!object_struct_name) return NULL;

            /* Union variant field access (object type name format: "UnionName.VariantName") */
            const char *dot = strchr(object_struct_name, '.');
            if (dot) {
                int union_name_len = (int)(dot - object_struct_name);
                char *union_name = malloc((size_t)union_name_len + 1);
                strncpy(union_name, object_struct_name, (size_t)union_name_len);
                union_name[union_name_len] = '\0';
                const char *variant_name = dot + 1;

                UnionDef *udef = env_get_union(env, union_name);
                if (udef) {
                    int variant_idx = env_get_union_variant_index(env, union_name, variant_name);
                    if (variant_idx >= 0) {
                        const char *field_name = expr->as.field_access.field_name;
                        for (int i = 0; i < udef->variant_field_counts[variant_idx]; i++) {
                            if (strcmp(udef->variant_field_names[variant_idx][i], field_name) != 0) {
                                continue;
                            }

                            Type field_type = udef->variant_field_types[variant_idx][i];
                            const char *field_type_name = NULL;
                            if (udef->variant_field_type_names) {
                                field_type_name = udef->variant_field_type_names[variant_idx][i];
                            }

                            /* For generic unions, resolve concrete struct/union type name from TypeInfo */
                            if (udef->generic_param_count > 0 && field_type_name &&
                                expr->as.field_access.object->type == AST_IDENTIFIER) {
                                ASTNode *obj = expr->as.field_access.object;
                                Symbol *obj_sym = env_get_var_visible_at(env, obj->as.identifier, obj->line, obj->column);
                                if (obj_sym && obj_sym->type_info && obj_sym->type_info->type_param_count > 0) {
                                    for (int g = 0; g < udef->generic_param_count; g++) {
                                        if (strcmp(udef->generic_params[g], field_type_name) == 0) {
                                            if (g < obj_sym->type_info->type_param_count) {
                                                TypeInfo *concrete = obj_sym->type_info->type_params[g];
                                                if (concrete) {
                                                    if ((concrete->base_type == TYPE_STRUCT || concrete->base_type == TYPE_UNION) &&
                                                        concrete->generic_name) {
                                                        free(union_name);
                                                        return strdup(concrete->generic_name);
                                                    }
                                                }
                                            }
                                            break;
                                        }
                                    }
                                }
                            }

                            /* Non-generic union (or unresolved generic): return declared struct/union name when available */
                            if ((field_type == TYPE_STRUCT || field_type == TYPE_UNION) && field_type_name) {
                                free(union_name);
                                return strdup(field_type_name);
                            }
                        }
                    }
                }

                free(union_name);
            }
            
            /* Look up the struct definition */
            StructDef *sdef = env_get_struct(env, object_struct_name);
            if (!sdef) return NULL;
            
            /* Find the field */
            for (int i = 0; i < sdef->field_count; i++) {
                if (strcmp(sdef->field_names[i], expr->as.field_access.field_name) == 0) {
                    TypeInfo *info = sdef->field_type_info ? sdef->field_type_info[i] : NULL;
                    if (info && info->generic_name && env_get_union(env, info->generic_name))
                        return info->generic_name;
                    /* Check if this field is a struct/union type */
                    if ((sdef->field_types[i] == TYPE_STRUCT || sdef->field_types[i] == TYPE_UNION) &&
                        sdef->field_type_names && sdef->field_type_names[i]) {
                        /* Return the struct type name for this field */
                        return strdup(sdef->field_type_names[i]);
                    }
                    /* Field is not a struct, or type name not available */
                    return NULL;
                }
            }
            return NULL;
        }
        
        default:
            return NULL;
    }
}

static FunctionSignature *function_result_signature(ASTNode *call, Environment *env);

/* I borrow the callback declaration; no temporary signature escapes this query. */
static Type map_callback_type(ASTNode *callback, Environment *env, int *arity, Type *argument) {
    FunctionSignature *sig = NULL;
    *arity = -1;
    *argument = TYPE_UNKNOWN;
    if (callback->type == AST_IDENTIFIER) {
        Symbol *symbol = env_get_var_visible_at(env, callback->as.identifier,
                                               callback->line, callback->column);
        if (symbol) {
            sig = symbol->type_info ? symbol->type_info->fn_sig : NULL;
        } else {
            Function *function = env_get_function(env, callback->as.identifier);
            if (function) {
                *arity = function->param_count;
                if (*arity == 1) *argument = function->params[0].type;
                return function->return_type;
            }
        }
    } else if (callback->type == AST_CALL) {
        sig = function_result_signature(callback, env);
    }
    if (!sig) return TYPE_UNKNOWN;
    *arity = sig->param_count;
    if (*arity == 1) *argument = sig->param_types[0];
    return sig->return_type;
}

Type map_transform_result_type(ASTNode *callback, Environment *env) {
    int arity;
    Type argument;
    return map_callback_type(callback, env, &arity, &argument);
}

Type filter_predicate_element_type(ASTNode *callback, Environment *env) {
    int arity;
    Type argument;
    Type result = map_callback_type(callback, env, &arity, &argument);
    return arity == 1 && result == TYPE_BOOL ? argument : TYPE_UNKNOWN;
}

static Type infer_array_element_type(ASTNode *array_expr, Environment *env) {
    if (!array_expr) return TYPE_UNKNOWN;
    if (array_expr->type == AST_CALL && !array_expr->as.call.func_expr &&
        array_expr->as.call.name && !strcmp(array_expr->as.call.name, "array_new") &&
        env_native_array_is_builtin(env, array_expr->as.call.name, array_expr->line, array_expr->column) &&
        array_expr->as.call.arg_count == 2)
        return check_expression(array_expr->as.call.args[1], env);
    if (array_expr->type == AST_CALL && !array_expr->as.call.func_expr &&
        array_expr->as.call.name && !strcmp(array_expr->as.call.name, "array_push") &&
        array_expr->as.call.arg_count == 2 &&
        env_array_push_is_builtin(env, array_expr->line, array_expr->column))
        return infer_array_element_type(array_expr->as.call.args[0], env);
    if (array_expr->type == AST_CALL && !array_expr->as.call.func_expr &&
        array_expr->as.call.name && strcmp(array_expr->as.call.name, "map") == 0 &&
        env_native_array_is_builtin(env, array_expr->as.call.name, array_expr->line, array_expr->column) &&
        array_expr->as.call.arg_count == 2) {
        int arity;
        Type argument;
        return map_callback_type(array_expr->as.call.args[1], env, &arity, &argument);
    }
    TypeInfo *info = try_get_expr_type_info(array_expr, env);
    if (info && info->base_type == TYPE_ARRAY && info->element_type)
        return resolved_array_element(info->element_type->base_type, info->element_type->generic_name, env);

    if (array_expr->type == AST_CALL && array_expr->as.call.name &&
        !array_expr->as.call.func_expr) {
        Function *producer = env_get_function(env, array_expr->as.call.name);
        if (producer && producer->return_type == TYPE_ARRAY)
            return resolved_array_element(producer->return_element_type, producer->return_struct_type_name, env);
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

    if (array_expr->type == AST_ARRAY_LITERAL) {
        if (array_expr->as.array_literal.element_type != TYPE_UNKNOWN) {
            return array_expr->as.array_literal.element_type;
        }
        if (array_expr->as.array_literal.element_count > 0) {
            return check_expression(array_expr->as.array_literal.elements[0], env);
        }
        return TYPE_UNKNOWN;
    }

    if (array_expr->type == AST_IDENTIFIER) {
        Symbol *sym = env_get_var_visible_at(env, array_expr->as.identifier, array_expr->line, array_expr->column);
        if (sym && sym->type == TYPE_ARRAY && sym->element_type != TYPE_UNKNOWN) {
            return resolved_array_element(sym->element_type, sym->struct_type_name, env);
        }
        return TYPE_UNKNOWN;
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
        return TYPE_UNKNOWN;
    }

    return TYPE_UNKNOWN;
}

/* Internal implementation - do not call directly */
static Type check_expression_impl(ASTNode *expr, Environment *env);

/* I retain emission metadata without extending a local's source visibility. */
static void bound_scope_symbols(Environment *env, int first, ASTNode *scope) {
    if (!env || !scope || scope->scope_end_line <= 0) return;
    for (int i = first; i < env->symbol_count; ++i) {
        Symbol *symbol = &env->symbols[i];
        if (symbol->scope_end_line > 0) continue; /* Inner scopes keep their bound. */
        symbol->scope_end_line = scope->scope_end_line;
        symbol->scope_end_column = scope->scope_end_column;
    }
}

/* Check expression type (wrapper with recursion depth tracking) */
Type check_expression(ASTNode *expr, Environment *env) {
    if (!expr) return TYPE_UNKNOWN;

    /* Check recursion depth to prevent stack overflow */
    g_check_expr_depth++;
    if (g_check_expr_depth > MAX_CHECK_EXPR_DEPTH) {
        fprintf(stderr, "Error: Type checker recursion depth exceeded. "
                        "File too large - consider splitting into modules\n");
        g_check_expr_depth--;
        return TYPE_UNKNOWN;
    }

    int first_symbol = env ? env->symbol_count : 0;
    Type result = check_expression_impl(expr, env);
    bound_scope_symbols(env, first_symbol, expr);
    g_check_expr_depth--;
    return result;
}

/* I use one guard rule in both expression and statement matches. */
static bool check_match_guard(ASTNode *guard, Environment *env) {
    if (!guard) return true;
    Type guard_type = check_expression(guard, env);
    if (guard_type == TYPE_BOOL) return true;

    emit_context_error(
        "E001 TYPE MISMATCH",
        guard->line,
        guard->column,
        1,
        "I require a match guard to have type bool.",
        "Give this guard an exact bool type before I select an arm."
    );
    if (active_statement_checker) active_statement_checker->has_error = true;
    return false;
}

typedef enum {
    MATCH_DOMAIN_INVALID = 0,
    MATCH_DOMAIN_INT,
    MATCH_DOMAIN_UNION
} MatchDomain;

/* Wildcards inherit the checked scrutinee domain; other arms declare a family. */
static void match_arm_families(ASTNode *matched, bool *has_int_patterns,
                               bool *has_variant_patterns) {
    *has_int_patterns = false;
    *has_variant_patterns = false;
    for (int arm = 0; arm < matched->as.match_expr.arm_count; ++arm) {
        const char *pattern = matched->as.match_expr.pattern_variants[arm];
        if (!pattern || strcmp(pattern, "_") == 0) continue;
        if (strncmp(pattern, "INT:", 4) == 0)
            *has_int_patterns = true;
        else
            *has_variant_patterns = true;
    }
}

/* I reject mixed, wrong or unresolved domains before reasoning about coverage. */
static MatchDomain check_match_domain(ASTNode *matched, Environment *env,
                                      Type match_type, bool has_int_patterns,
                                      bool has_variant_patterns,
                                      const char *union_base_name) {
    const char *message = NULL;
    const char *hint = NULL;

    if (has_int_patterns && has_variant_patterns) {
        message = "I do not mix integer and union-variant patterns in one match.";
        hint = "Use only integer patterns for an int, or only named/or-pattern arms for a union.";
    } else if (match_type == TYPE_INT && has_variant_patterns) {
        message = "I require named and or-pattern match arms to inspect a known union.";
        hint = "Use integer patterns for this int scrutinee.";
    } else if (match_type == TYPE_UNION && has_int_patterns) {
        message = "I require integer match patterns to inspect an int.";
        hint = "Use named or or-pattern arms from this union.";
    } else if (match_type == TYPE_INT) {
        return MATCH_DOMAIN_INT;
    } else if (match_type == TYPE_UNION) {
        UnionDef *definition = union_base_name ? env_get_union(env, union_base_name) : NULL;
        if (definition) {
            bool empty_valid = true;
            for (int arm = 0; arm < matched->as.match_expr.arm_count; ++arm) {
                const char *binding = matched->as.match_expr.pattern_bindings[arm];
                if (!binding || *binding) continue;
                bool zero = false;
                for (int variant = 0; variant < definition->variant_count; ++variant)
                    if (!strcmp(matched->as.match_expr.pattern_variants[arm], definition->variant_names[variant]))
                        zero = definition->variant_field_counts[variant] == 0;
                if (!zero) { empty_valid = false; break; }
            }
            if (empty_valid) return MATCH_DOMAIN_UNION;
            message = "I require an exact zero-field variant for an empty match binding.";
            hint = "Bind this payload by name, or explicitly discard it with underscore where permitted.";
        } else {
            message = "I require an exact known union identity before I check match coverage.";
            hint = "Give the scrutinee a declared union type that I can resolve here.";
        }
    } else {
        message = "I require a match to inspect an int or a known union.";
        hint = "Give the scrutinee an exact supported type before matching it.";
    }

    emit_context_error(
        "E001 TYPE MISMATCH",
        matched->line,
        matched->column,
        5,
        message,
        hint
    );
    if (active_statement_checker) active_statement_checker->has_error = true;
    return MATCH_DOMAIN_INVALID;
}

static bool match_guard_is_unconditional(const ASTNode *guard) {
    return !guard || (guard->type == AST_BOOL && guard->as.bool_val);
}

static bool match_pattern_names_variant(const char *pattern, const char *variant) {
    if (!pattern || !variant) return false;
    if (strncmp(pattern, "OR:", 3) != 0) return strcmp(pattern, variant) == 0;

    const char *part = pattern + 3;
    size_t variant_len = strlen(variant);
    while (*part) {
        const char *end = strchr(part, ':');
        size_t part_len = end ? (size_t)(end - part) : strlen(part);
        if (part_len == variant_len && strncmp(part, variant, part_len) == 0)
            return true;
        if (!end) break;
        part = end + 1;
    }
    return false;
}

/*
 * I reject a source match when I cannot prove that one arm must succeed.
 * Runtime terminal backstops remain a separate obligation under task70c5.
 */
static void check_match_totality(ASTNode *matched, Environment *env,
                                 const char *union_base_name,
                                 MatchDomain domain) {
    bool has_unconditional_wildcard = false;
    int unconditional_wildcard = -1;
    for (int i = 0; i < matched->as.match_expr.arm_count; ++i) {
        ASTNode *guard = matched->as.match_expr.guard_exprs
            ? matched->as.match_expr.guard_exprs[i] : NULL;
        if (unconditional_wildcard >= 0) {
            ASTNode *arm = matched->as.match_expr.arm_bodies[i];
            emit_context_error(
                "E036 UNREACHABLE MATCH ARM",
                arm ? arm->line : matched->line,
                arm ? arm->column : matched->column,
                1,
                "I cannot reach a match arm after an unconditional wildcard.",
                "Remove this arm or give the earlier wildcard a non-literal guard."
            );
            if (active_statement_checker) active_statement_checker->has_error = true;
            return;
        }
        if (strcmp(matched->as.match_expr.pattern_variants[i], "_") == 0 &&
            match_guard_is_unconditional(guard)) {
            has_unconditional_wildcard = true;
            unconditional_wildcard = i;
        }
    }

    if (domain == MATCH_DOMAIN_INT) {
        if (!has_unconditional_wildcard) {
            emit_context_error(
                "E035 NON-EXHAUSTIVE MATCH",
                matched->line,
                matched->column,
                5,
                "I require an unconditional wildcard in an integer match.",
                "Add `_ => ...` after the integer cases so every integer is covered."
            );
            if (active_statement_checker) active_statement_checker->has_error = true;
        }
        return;
    }

    if (has_unconditional_wildcard) return;

    UnionDef *union_def = union_base_name ? env_get_union(env, union_base_name) : NULL;
    if (!union_def) {
        emit_context_error(
            "E035 NON-EXHAUSTIVE MATCH",
            matched->line,
            matched->column,
            5,
            "I cannot establish that this match covers every value.",
            "Use a union with known variants or add an unconditional wildcard."
        );
        if (active_statement_checker) active_statement_checker->has_error = true;
        return;
    }

    bool *covered = calloc((size_t)union_def->variant_count, sizeof(bool));
    if (!covered) {
        emit_context_error(
            "E035 NON-EXHAUSTIVE MATCH",
            matched->line,
            matched->column,
            5,
            "I could not allocate match coverage state.",
            "Retry after making memory available."
        );
        if (active_statement_checker) active_statement_checker->has_error = true;
        return;
    }

    for (int arm = 0; arm < matched->as.match_expr.arm_count; ++arm) {
        ASTNode *guard = matched->as.match_expr.guard_exprs
            ? matched->as.match_expr.guard_exprs[arm] : NULL;
        if (!match_guard_is_unconditional(guard)) continue;
        const char *pattern = matched->as.match_expr.pattern_variants[arm];
        for (int variant = 0; variant < union_def->variant_count; ++variant) {
            if (match_pattern_names_variant(pattern, union_def->variant_names[variant]))
                covered[variant] = true;
        }
    }

    char missing[512] = "I require this match to cover every variant; I am missing:";
    size_t used = strlen(missing);
    int missing_count = 0;
    for (int variant = 0; variant < union_def->variant_count; ++variant) {
        if (covered[variant]) continue;
        missing_count++;
        if (used < sizeof(missing)) {
            int written = snprintf(missing + used, sizeof(missing) - used,
                                   " %s", union_def->variant_names[variant]);
            if (written > 0) {
                size_t available = sizeof(missing) - used;
                used += (size_t)written < available ? (size_t)written : available - 1;
            }
        }
    }
    free(covered);

    if (missing_count > 0) {
        emit_context_error(
            "E035 NON-EXHAUSTIVE MATCH",
            matched->line,
            matched->column,
            5,
            missing,
            "Add unconditional arms for the missing variants or one unconditional wildcard."
        );
        if (active_statement_checker) active_statement_checker->has_error = true;
    }
}

/* I borrow declared signatures; the parser/environment owns their storage. */
static FunctionSignature *function_result_signature(ASTNode *call, Environment *env) {
    if (!call || call->type != AST_CALL) return NULL;
    if (call->as.call.checked_signature) return call->as.call.checked_signature->return_fn_sig;
    if (call->as.call.func_expr) {
        FunctionSignature *callee = function_result_signature(call->as.call.func_expr, env);
        return callee ? callee->return_fn_sig : NULL;
    }
    if (!call->as.call.name) return NULL;
    Function *func = env_get_function(env, call->as.call.name);
    if (func) return func->return_fn_sig;
    Symbol *sym = env_get_var_visible_at(env, call->as.call.name, call->line, call->column);
    FunctionSignature *sig = sym && sym->type_info ? sym->type_info->fn_sig : NULL;
    return sig ? sig->return_fn_sig : NULL;
}

/* I retain explicit callable annotations with the AST in every let scope. */
static bool retain_let_function_type(TypeChecker *tc, ASTNode *statement, Type declared) {
    if (statement->as.let.type_info || declared != TYPE_FUNCTION ||
        !statement->as.let.fn_sig) return true;
    TypeInfo *info = calloc(1, sizeof *info);
    if (!info) {
        tc->has_error = true;
        return false;
    }
    info->base_type = TYPE_FUNCTION;
    info->fn_sig = statement->as.let.fn_sig;
    statement->as.let.type_info = info;
    return true;
}

/* I compare complete reduce identities without the general compatibility rules.
 * These views borrow annotations; none escape this check. */
static TypeInfo reduce_type_view(Type type, const char *name, const TypeInfo *info) {
    if (info) return *info;
    TypeInfo view = {.base_type = type, .generic_name = (char *)name};
    return view;
}

/* I recover nominal annotations erased by the legacy declaration pass. */
static Type reduce_identity_kind(const TypeInfo *info, Environment *env) {
    if (info->generic_name) {
        if ((info->base_type == TYPE_STRUCT || info->base_type == TYPE_INT ||
             info->base_type == TYPE_ENUM) && env_get_enum(env, info->generic_name))
            return TYPE_ENUM;
        if (info->base_type == TYPE_STRUCT && env_get_union(env, info->generic_name))
            return TYPE_UNION;
    }
    return info->base_type;
}

static bool reduce_types_exact(const TypeInfo *a, const TypeInfo *b,
                               Environment *env, unsigned depth) {
    if (!a || !b || depth > 128 || a->is_open_row || b->is_open_row ||
        a->type_var_count || b->type_var_count) return false;
    Type at = reduce_identity_kind(a, env);
    Type bt = reduce_identity_kind(b, env);
    if (at != bt) return false;
    switch (at) {
        case TYPE_INT: case TYPE_U8: case TYPE_FLOAT: case TYPE_BOOL:
        case TYPE_STRING: case TYPE_BSTRING:
        case TYPE_LIST_INT: case TYPE_LIST_STRING: case TYPE_LIST_TOKEN:
            return true;
        case TYPE_ARRAY:
            return reduce_types_exact(a->element_type, b->element_type, env, depth + 1);
        case TYPE_STRUCT: case TYPE_ENUM: case TYPE_UNION: {
            if (!a->generic_name || !b->generic_name) return false;
            bool same = false;
            if (at == TYPE_STRUCT) {
                /* My record declarations have no generic parameter list. */
                if (a->type_param_count || b->type_param_count) return false;
                StructDef *left = env_get_struct(env, a->generic_name);
                same = left && left == env_get_struct(env, b->generic_name);
            } else if (at == TYPE_ENUM) {
                if (a->type_param_count || b->type_param_count) return false;
                EnumDef *left = env_get_enum(env, a->generic_name);
                same = left && left == env_get_enum(env, b->generic_name);
            } else {
                UnionDef *left = env_get_union(env, a->generic_name);
                same = left && left == env_get_union(env, b->generic_name);
                if (left && left->generic_param_count != a->type_param_count) return false;
            }
            if (!same || a->type_param_count != b->type_param_count ||
                a->type_param_count < 0) return false;
            for (int i = 0; i < a->type_param_count; ++i)
                if (!a->type_params || !b->type_params ||
                    !reduce_types_exact(a->type_params[i], b->type_params[i], env, depth + 1))
                    return false;
            return true;
        }
        case TYPE_HASHMAP: case TYPE_LIST_GENERIC: {
            int count = at == TYPE_HASHMAP ? 2 : 1;
            if (a->type_param_count != count || b->type_param_count != count ||
                !a->type_params || !b->type_params) return false;
            for (int i = 0; i < count; ++i)
                if (!reduce_types_exact(a->type_params[i], b->type_params[i], env, depth + 1))
                    return false;
            return true;
        }
        case TYPE_TUPLE:
            if (a->tuple_element_count != b->tuple_element_count || a->tuple_element_count < 0)
                return false;
            for (int i = 0; i < a->tuple_element_count; ++i) {
                if (!a->tuple_types || !b->tuple_types) return false;
                TypeInfo left = reduce_type_view(a->tuple_types[i],
                    a->tuple_type_names ? a->tuple_type_names[i] : NULL, NULL);
                TypeInfo right = reduce_type_view(b->tuple_types[i],
                    b->tuple_type_names ? b->tuple_type_names[i] : NULL, NULL);
                const TypeInfo *lc = type_info_tuple_element(a, i, &left);
                const TypeInfo *rc = type_info_tuple_element(b, i, &right);
                if (!lc || !rc || !reduce_types_exact(lc, rc, env, depth + 1)) return false;
            }
            return true;
        case TYPE_FUNCTION: {
            FunctionSignature *left = a->fn_sig, *right = b->fn_sig;
            if (!left || !right || left->param_count != right->param_count ||
                left->param_count < 0) return false;
            for (int i = 0; i < left->param_count; ++i) {
                if (!left->param_types || !right->param_types) return false;
                TypeInfo lp = reduce_type_view(left->param_types[i],
                    left->param_struct_names ? left->param_struct_names[i] : NULL,
                    left->param_type_info ? left->param_type_info[i] : NULL);
                TypeInfo rp = reduce_type_view(right->param_types[i],
                    right->param_struct_names ? right->param_struct_names[i] : NULL,
                    right->param_type_info ? right->param_type_info[i] : NULL);
                if (!reduce_types_exact(&lp, &rp, env, depth + 1)) return false;
            }
            TypeInfo lr = reduce_type_view(left->return_type, left->return_struct_name,
                                            left->return_type_info);
            TypeInfo rr = reduce_type_view(right->return_type, right->return_struct_name,
                                            right->return_type_info);
            if (!lr.fn_sig) lr.fn_sig = left->return_fn_sig;
            if (!rr.fn_sig) rr.fn_sig = right->return_fn_sig;
            if (lr.base_type == TYPE_VOID && rr.base_type == TYPE_VOID) return true;
            return reduce_types_exact(&lr, &rr, env, depth + 1);
        }
        case TYPE_OPAQUE:
            return a->opaque_type_name && b->opaque_type_name &&
                !strcmp(a->opaque_type_name, b->opaque_type_name);
        default:
            return false;
    }
}

static bool reduce_expression_matches(ASTNode *expression, const TypeInfo *expected,
                                      Environment *env, unsigned depth) {
    if (!expression || !expected || depth > 128) return false;
    Type actual = check_expression(expression, env);
    if (actual == TYPE_UNKNOWN || actual == TYPE_VOID) return false;
    if (actual == TYPE_ARRAY && expression->type == AST_ARRAY_LITERAL) {
        if (reduce_identity_kind(expected, env) != TYPE_ARRAY ||
            !reduce_types_exact(expected, expected, env, depth + 1)) return false;
        for (int i = 0; i < expression->as.array_literal.element_count; ++i)
            if (!reduce_expression_matches(expression->as.array_literal.elements[i],
                    expected->element_type, env, depth + 1)) return false;
        return true;
    }
    const char *name = get_struct_type_name(expression, env);
    if (actual == TYPE_ENUM && expression->type == AST_FIELD_ACCESS &&
        expression->as.field_access.object->type == AST_IDENTIFIER)
        name = expression->as.field_access.object->as.identifier;
    TypeInfo view = reduce_type_view(actual, name, try_get_expr_type_info(expression, env));
    return reduce_types_exact(&view, expected, env, depth + 1);
}

static Type check_reduce_call(ASTNode *call, Environment *env) {
    if (call->as.call.arg_count != 3) {
        emit_context_error("E003 ARITY MISMATCH", call->line, call->column, 1,
            "I require exactly three operands for reduce.",
            "Pass an array, an initializer and a binary callback.");
        return TYPE_UNKNOWN;
    }
    ASTNode *array = call->as.call.args[0], *initial = call->as.call.args[1];
    ASTNode *callback = call->as.call.args[2];
    Type array_type = check_expression(array, env);
    Type initial_type = check_expression(initial, env);
    Type callback_type = check_expression(callback, env);
    Function *function = NULL;
    FunctionSignature *signature = NULL;
    if (callback->type == AST_IDENTIFIER) {
        Symbol *value = env_get_var_visible_at(env, callback->as.identifier,
                                               callback->line, callback->column);
        if (value) signature = value->type_info ? value->type_info->fn_sig : NULL;
        else function = env_get_function(env, callback->as.identifier);
    } else if (callback->type == AST_CALL) {
        signature = function_result_signature(callback, env);
    } else {
        TypeInfo *info = try_get_expr_type_info(callback, env);
        signature = info ? info->fn_sig : NULL;
    }
    TypeInfo parameters[2] = {{.base_type = TYPE_UNKNOWN}, {.base_type = TYPE_UNKNOWN}};
    TypeInfo result = {.base_type = TYPE_UNKNOWN};
    if (function && function->param_count == 2 && function->params) {
        for (int i = 0; i < 2; ++i) {
            Parameter *parameter = &function->params[i];
            parameters[i] = reduce_type_view(parameter->type, parameter->struct_type_name,
                                              parameter->type_info);
            if (!parameters[i].fn_sig) parameters[i].fn_sig = parameter->fn_sig;
        }
        result = reduce_type_view(function->return_type, function->return_struct_type_name,
                                  function->return_type_info);
        if (!result.fn_sig) result.fn_sig = function->return_fn_sig;
    } else if (signature && signature->param_count == 2 && signature->param_types) {
        for (int i = 0; i < 2; ++i)
            parameters[i] = reduce_type_view(signature->param_types[i],
                signature->param_struct_names ? signature->param_struct_names[i] : NULL,
                signature->param_type_info ? signature->param_type_info[i] : NULL);
        result = reduce_type_view(signature->return_type, signature->return_struct_name,
                                  signature->return_type_info);
        if (!result.fn_sig) result.fn_sig = signature->return_fn_sig;
    }
    bool valid = array_type == TYPE_ARRAY && callback_type == TYPE_FUNCTION &&
        initial_type != TYPE_UNKNOWN && initial_type != TYPE_VOID &&
        reduce_types_exact(&parameters[0], &result, env, 0) &&
        reduce_types_exact(&parameters[1], &parameters[1], env, 0) &&
        reduce_expression_matches(initial, &parameters[0], env, 0);
    if (valid && array->type == AST_ARRAY_LITERAL) {
        Type element = infer_array_element_type(array, env);
        Type wanted = reduce_identity_kind(&parameters[1], env);
        /* I check nonempty leaves by identity: the container kind can still
         * be the parser's unresolved nominal kind. Empty literals need facts. */
        TypeInfo empty_element = {.base_type = element};
        valid = array->as.array_literal.element_count > 0 ||
            (element == wanted && reduce_types_exact(&empty_element, &parameters[1], env, 0));
        for (int i = 0; valid && i < array->as.array_literal.element_count; ++i)
            valid = reduce_expression_matches(array->as.array_literal.elements[i],
                                               &parameters[1], env, 0);
    } else if (valid) {
        TypeInfo *array_info = try_get_expr_type_info(array, env);
        TypeInfo element = reduce_type_view(infer_array_element_type(array, env), NULL,
            array_info && array_info->base_type == TYPE_ARRAY ? array_info->element_type : NULL);
        valid = reduce_types_exact(&element, &parameters[1], env, 0);
    }
    if (!valid) {
        emit_context_error("E001 TYPE MISMATCH", call->line, call->column, 1,
            "I require reduce to receive array<E>, an initializer A and an exact fn(A,E)->A.",
            "Retain complete types and match both callback parameters and its result exactly.");
        return TYPE_UNKNOWN;
    }
    return initial_type;
}

/* I retain the union identity of the parser's dotted variant literals. */
static const char *inline_variant_union(ASTNode *node, Environment *env) {
    if (!node || node->type != AST_STRUCT_LITERAL || !node->as.struct_literal.struct_name)
        return NULL;
    const char *name = node->as.struct_literal.struct_name;
    const char *dot = strchr(name, '.');
    if (!dot) return NULL;
    char *prefix = strndup(name, (size_t)(dot - name));
    if (!prefix) return NULL;
    UnionDef *definition = env_get_union(env, prefix);
    free(prefix);
    return definition ? definition->name : NULL;
}

/* I check literal leaves against the complete callback annotation. */
static bool indirect_argument_matches(ASTNode *argument, Environment *env,
                                      const TypeInfo *expected, Type fallback,
                                      int depth) {
    if (depth > 128) return false;
    Type actual = check_expression(argument, env);
    if (expected && (expected->base_type == TYPE_STRUCT || expected->base_type == TYPE_OPAQUE)) {
        const char *name = expected->opaque_type_name ? expected->opaque_type_name : expected->generic_name;
        if (name && env_get_opaque_type(env, name))
            return check_opaque_value(env, expected->base_type, name, argument);
    }
    if (!types_match(actual, expected ? expected->base_type : fallback)) return false;
    if (!expected) return true;
    if (expected->base_type == TYPE_TUPLE && argument->type == AST_TUPLE_LITERAL) {
        if (!type_info_tuple_valid(expected) ||
            expected->tuple_element_count != argument->as.tuple_literal.element_count) return false;
        for (int i = 0; i < expected->tuple_element_count; ++i) {
            TypeInfo flat;
            const TypeInfo *child = type_info_tuple_element(expected, i, &flat);
            if (!child || !indirect_argument_matches(argument->as.tuple_literal.elements[i], env,
                    child, child->base_type, depth + 1)) return false;
        }
        return true;
    }
    if (expected->base_type == TYPE_FUNCTION && expected->fn_sig) {
        FunctionSignature *signature = checked_callable_signature_copy(argument, env);
        TypeInfo complete = {.base_type = TYPE_FUNCTION, .fn_sig = signature};
        bool equal = signature && type_infos_equal(expected, &complete);
        free_function_signature(signature);
        return equal;
    }
    if (expected->base_type == TYPE_ARRAY && expected->element_type &&
        argument->type == AST_ARRAY_LITERAL) {
        for (int i = 0; i < argument->as.array_literal.element_count; ++i) {
            if (!indirect_argument_matches(argument->as.array_literal.elements[i], env,
                    expected->element_type, expected->element_type->base_type, depth + 1))
                return false;
        }
        return true;
    }
    TypeInfo *info = try_get_expr_type_info(argument, env);
    if (info) return type_infos_equal(expected, info);
    if (expected->base_type == TYPE_STRUCT && expected->generic_name) {
        const char *actual_name = get_struct_type_name(argument, env);
        if (!actual_name) return false;
        StructDef *wanted = env_get_struct(env, expected->generic_name);
        StructDef *found = env_get_struct(env, actual_name);
        if (wanted || found) return wanted && wanted == found;
        return strcmp(expected->generic_name, actual_name) == 0;
    }
    return true;
}

static Type check_indirect_call(ASTNode *call, Environment *env, FunctionSignature *sig) {
    if (!sig) sig = call->as.call.checked_signature;
    if (!sig) {
        emit_context_error("E001 TYPE MISMATCH", call->line, call->column, 1,
                           "I cannot determine this function value's signature.",
                           "Declare the function value's parameter and return types.");
        return TYPE_UNKNOWN;
    }
    if (sig->param_count != call->as.call.arg_count) {
        emit_context_error("E003 ARITY MISMATCH", call->line, call->column, 1,
                           "I require the declared number of arguments for this function value.",
                           "Match the function signature.");
        return TYPE_UNKNOWN;
    }
    for (int i = 0; i < call->as.call.arg_count; i++) {
        ASTNode *argument = call->as.call.args[i];
        TypeInfo *expected = sig->param_type_info ? sig->param_type_info[i] : NULL;
        check_concrete_union_arrays(env, expected, argument, 0);
        bool matches = indirect_argument_matches(argument, env, expected,
                                                  sig->param_types[i], 0);
        if (!matches) {
            emit_context_error("E001 TYPE MISMATCH", call->as.call.args[i]->line,
                               call->as.call.args[i]->column, 1,
                               "I require the declared argument type for this function value.",
                               "Match the function signature.");
        }
    }
    FunctionSignature *retained = copy_function_signature(sig);
    Type result = sig->return_type;
    char *name = sig->return_struct_name ? strdup(sig->return_struct_name) : NULL;
    free_function_signature(call->as.call.checked_signature);
    call->as.call.checked_signature = retained;
    free(call->as.call.return_struct_type_name);
    call->as.call.return_struct_type_name = name;
    return result;
}

static Type check_perform(ASTNode *expr, Environment *env) {
    const char *effect_name = expr->as.effect_op.effect_name;
    const char *op_name = expr->as.effect_op.op_name;
    EffectDef *effect = effect_name ? env_get_effect(env, effect_name) : NULL;
    EffectOp *op = effect && op_name ? effect_get_op(effect, op_name) : NULL;
    if (!op) {
        emit_context_error("E029 UNKNOWN EFFECT OPERATION", expr->line, expr->column, 7,
                           "I require a declared effect and operation for perform.",
                           "Check the effect and operation names against their declaration.");
        return TYPE_UNKNOWN;
    }
    int count = expr->as.effect_op.arg_count;
    if (count != op->param_count) {
        emit_context_error("E003 ARITY MISMATCH", expr->line, expr->column, 7,
                           "I require the declared operation's argument count for perform.",
                           "Match the effect operation signature.");
        return TYPE_UNKNOWN;
    }
    for (int i = 0; i < count; i++) {
        Type actual = check_expression(expr->as.effect_op.args[i], env);
        if (!types_match(actual, op->params[i].type)) {
            emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 7,
                               "I require the declared operation's argument type for perform.",
                               "Match the effect operation signature.");
            return TYPE_UNKNOWN;
        }
    }
    return op->return_type;
}

static bool check_array_access_arguments(ASTNode *call, Environment *env) {
    if (call->as.call.arg_count != 2) {
        emit_context_error("E003 ARITY MISMATCH", call->line, call->column, 1,
                           "I require an array and an index for array access.",
                           "Pass exactly two arguments.");
        return false;
    }
    ASTNode *array = call->as.call.args[0];
    ASTNode *index = call->as.call.args[1];
    Type array_type = check_expression(array, env);
    Type index_type = check_expression(index, env);
    bool valid = true;
    if (array_type != TYPE_ARRAY) {
        emit_context_error("E001 TYPE MISMATCH", array->line, array->column, 1,
                           "I require an array as the first argument to array access.",
                           "Pass an array before its index.");
        valid = false;
    }
    if (index_type != TYPE_INT && index_type != TYPE_U8) {
        emit_context_error("E001 TYPE MISMATCH", index->line, index->column, 1,
                           "I require an integer array index.",
                           "Pass an int or u8 index.");
        valid = false;
    }
    if (valid) {
        /* I retain element identity while the lexical environment is available.
         * Bytecode field selection runs after these local scopes are gone. */
        const char *name = array_record_name(array, env);
        char *retained = name ? strdup(name) : NULL;
        free(call->as.call.return_struct_type_name);
        call->as.call.return_struct_type_name = retained;
    }
    return valid;
}

/* Internal implementation of check_expression */
static Type check_expression_impl(ASTNode *expr, Environment *env) {
    switch (expr->type) {
        case AST_NUMBER:
            return TYPE_INT;

        case AST_FLOAT:
            return TYPE_FLOAT;

        case AST_STRING:
            return TYPE_STRING;

        case AST_BOOL:
            return TYPE_BOOL;

        case AST_IDENTIFIER: {
            Symbol *sym = env_get_var_visible_at(env, expr->as.identifier, expr->line, expr->column);
            if (!sym) {
                /* Not a variable - check if it's a function name */
                Function *func = env_get_function(env, expr->as.identifier);
                if (func) {
                    /* Function name used as value (for passing/returning) */
                    return TYPE_FUNCTION;
                }
                char message[256];
                snprintf(message, sizeof(message), "I cannot find a variable named `%s`.", expr->as.identifier);
                emit_context_error(
                    "E024 UNDEFINED VARIABLE",
                    expr->line,
                    expr->column,
                    (int)safe_strlen(expr->as.identifier),
                    message,
                    "Check spelling or ensure the variable is in scope."
                );
                return TYPE_UNKNOWN;
            }
            sym->is_used = true;  /* Mark variable as used */
            
            return (sym->type == TYPE_BORROW_SHARED || sym->type == TYPE_BORROW_MUT) ? TYPE_STRUCT : sym->type;
        }

        case AST_QUALIFIED_NAME: {
            /* Handle qualified names: module::symbol or std::io::fs::read_file */
            int part_count = expr->as.qualified_name.part_count;
            char **parts = expr->as.qualified_name.name_parts;
            
            if (part_count < 2) {
                fprintf(stderr, "Error at line %d, column %d: Invalid qualified name (need at least 2 parts)\n",
                        expr->line, expr->column);
                return TYPE_UNKNOWN;
            }
            
            /* For now, handle simple case: module::symbol (2 parts) */
            /* TODO: Handle nested modules (std::io::fs::read_file) */
            if (part_count == 2) {
                char *module_name = parts[0];
                char *symbol_name = parts[1];
                
                /* Look up function in the module namespace */
                /* For now, search all functions with matching name */
                /* TODO: Implement proper module-scoped lookup */
                Function *func = env_get_function(env, symbol_name);
                if (func) {
                    /* TODO: Check if function belongs to the specified module */
                    /* TODO: Check visibility (pub vs private) */
                    return TYPE_FUNCTION;
                }
                
                /* Try looking up as variable (for module-level constants) */
                Symbol *sym = env_get_var_visible_at(env, symbol_name, expr->line, expr->column);
                if (sym) {
                    /* Check if symbol belongs to the specified module */
                    if (module_name && (!sym->struct_type_name || strcmp(sym->struct_type_name, module_name) != 0)) {
                        /* Symbol found but doesn't belong to this module */
                        char message[256];
                        snprintf(message, sizeof(message),
                                "Symbol `%s` does not belong to module `%s`.", symbol_name, module_name);
                        emit_context_error(
                            "E032 WRONG MODULE",
                            expr->line,
                            expr->column,
                            (int)safe_strlen(symbol_name),
                            message,
                            "This symbol belongs to a different module");
                        return TYPE_UNKNOWN;
                    }

                    /* Check if symbol was imported (for selective imports) */
                    if (!is_symbol_imported(symbol_name, module_name, env)) {
                        char message[256];
                        snprintf(message, sizeof(message),
                                "Constant `%s` from module `%s` was not imported.", symbol_name, module_name);
                        emit_context_error(
                            "E033 NOT IMPORTED",
                            expr->line,
                            expr->column,
                            (int)safe_strlen(symbol_name),
                            message,
                            "Add this symbol to your import statement");
                        return TYPE_UNKNOWN;
                    }

                    sym->is_used = true;
                    return sym->type;
                }
                
                char message[256];
                snprintf(message, sizeof(message),
                        "I cannot find `%s` in module `%s`.", symbol_name, module_name);
                emit_context_error(
                    "E025 UNDEFINED SYMBOL",
                    expr->line,
                    expr->column,
                    (int)safe_strlen(symbol_name),
                    message,
                    "Check the module name and exported symbols."
                );
                return TYPE_UNKNOWN;
            }
            
            /* Nested modules not yet supported */
            fprintf(stderr, "Error at line %d, column %d: Nested module paths not yet implemented\n",
                    expr->line, expr->column);
            return TYPE_UNKNOWN;
        }

        case AST_PREFIX_OP: {
            TokenType op = expr->as.prefix_op.op;
            int arg_count = expr->as.prefix_op.arg_count;

            /* Arithmetic operators */
            if (op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR ||
                op == TOKEN_SLASH || op == TOKEN_PERCENT) {
                
                /* Handle unary minus: (- x) */
                if (op == TOKEN_MINUS && arg_count == 1) {
                    Type arg_type = check_expression(expr->as.prefix_op.args[0], env);
                    if (arg_type == TYPE_INT) return TYPE_INT;
                    if (arg_type == TYPE_FLOAT) return TYPE_FLOAT;
                    if (arg_type == TYPE_ARRAY) {
                        Type elem = infer_array_element_type(expr->as.prefix_op.args[0], env);
                        if (elem == TYPE_UNKNOWN || elem == TYPE_INT || elem == TYPE_ENUM || elem == TYPE_FLOAT) {
                            return TYPE_ARRAY;
                        }
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            "Unary minus requires array<int> or array<float>",
                            "Only numeric arrays support element-wise negation");
                        return TYPE_UNKNOWN;
                    }
                    emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                        "Unary minus requires a numeric type (int or float)",
                        "Check that the operand is an int or float variable");
                    return TYPE_UNKNOWN;
                }
                
                /* Binary arithmetic operations */
                if (arg_count != 2) {
                    emit_context_error(
                        "E003 ARITY MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        "Binary arithmetic operators require exactly 2 arguments.",
                        "Provide two numeric operands."
                    );
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                if (left == TYPE_ARRAY || right == TYPE_ARRAY) {
                    ASTNode *left_expr = expr->as.prefix_op.args[0];
                    ASTNode *right_expr = expr->as.prefix_op.args[1];
                    bool left_is_array = (left == TYPE_ARRAY);
                    bool right_is_array = (right == TYPE_ARRAY);

                    Type left_elem = left_is_array ? infer_array_element_type(left_expr, env) : left;
                    Type right_elem = right_is_array ? infer_array_element_type(right_expr, env) : right;

                    /* If both element types are unknown, allow and defer to runtime */
                    if (left_is_array && right_is_array && left_elem == TYPE_UNKNOWN && right_elem == TYPE_UNKNOWN) {
                        return TYPE_ARRAY;
                    }

                    /* If one side's element type is unknown, allow and defer to runtime */
                    if ((left_is_array && left_elem == TYPE_UNKNOWN) || (right_is_array && right_elem == TYPE_UNKNOWN)) {
                        return TYPE_ARRAY;
                    }

                    /* Normalize enums to int for array arithmetic */
                    if (left_elem == TYPE_ENUM) left_elem = TYPE_INT;
                    if (right_elem == TYPE_ENUM) right_elem = TYPE_INT;

                    /* Treat u8 like int for arithmetic purposes. */
                    if (left_elem == TYPE_U8) left_elem = TYPE_INT;
                    if (right_elem == TYPE_U8) right_elem = TYPE_INT;

                    if (op == TOKEN_PLUS && left_elem == TYPE_STRING && right_elem == TYPE_STRING) {
                        return TYPE_ARRAY;
                    }

                    if (op == TOKEN_PERCENT) {
                        if (left_elem == TYPE_INT && right_elem == TYPE_INT) return TYPE_ARRAY;
                        fprintf(stderr, "Error at line %d, column %d: %% only supported on array<int> or array<u8>\n", expr->line, expr->column);
                        return TYPE_UNKNOWN;
                    }

                    if (left_elem == TYPE_INT && right_elem == TYPE_INT) return TYPE_ARRAY;
                    if (left_elem == TYPE_FLOAT && right_elem == TYPE_FLOAT) return TYPE_ARRAY;

                    char message[256];
                    snprintf(message, sizeof(message),
                            "Array arithmetic requires matching numeric element types (got %s and %s).",
                            type_to_string(left), type_to_string(right));
                    emit_context_error(
                        "E001 TYPE MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        message,
                        "Use matching numeric types or convert elements before arithmetic."
                    );
                    return TYPE_UNKNOWN;
                }

                /* String concatenation with + operator
                 *
                 * Self-hosting uses nested concatenations heavily. If one side is
                 * already known to be a string, allow the other side to be
                 * TYPE_UNKNOWN and still treat the result as TYPE_STRING. This
                 * preserves the "string context" through nested (+ a (+ b c)).
                 */
                if (op == TOKEN_PLUS) {
                    if ((left == TYPE_STRING && (right == TYPE_STRING || right == TYPE_UNKNOWN)) ||
                        (right == TYPE_STRING && (left == TYPE_STRING || left == TYPE_UNKNOWN))) {
                        return TYPE_STRING;
                    }
                }

                /* Enums are compatible with ints in arithmetic */
                if ((left == TYPE_INT || left == TYPE_ENUM || left == TYPE_U8) &&
                    (right == TYPE_INT || right == TYPE_ENUM || right == TYPE_U8)) return TYPE_INT;
                if (left == TYPE_FLOAT && right == TYPE_FLOAT) return TYPE_FLOAT;

                char message[256];
                snprintf(message, sizeof(message),
                        "Arithmetic expects numeric types or string concatenation with + (got %s and %s).",
                        type_to_string(left), type_to_string(right));
                emit_context_error(
                    "E001 TYPE MISMATCH",
                    expr->line,
                    expr->column,
                    1,
                    message,
                    "Use matching numeric types, or use + with two strings."
                );
                return TYPE_UNKNOWN;
            }

            /* Comparison operators */
            if (op == TOKEN_LT || op == TOKEN_LE || op == TOKEN_GT || op == TOKEN_GE) {
                if (arg_count != 2) {
                    emit_context_error(
                        "E003 ARITY MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        "Comparison operators require exactly 2 arguments.",
                        "Provide two comparable operands."
                    );
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                if (!types_match(left, right)) {
                    char message[256];
                    snprintf(message, sizeof(message),
                            "Comparison requires both operands to be the same type (got %s and %s).",
                            type_to_string(left), type_to_string(right));
                    emit_context_error(
                        "E001 TYPE MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        message,
                        "Convert operands to the same type before comparing."
                    );
                }
                return TYPE_BOOL;
            }

            /* Equality operators */
            if (op == TOKEN_EQ || op == TOKEN_NE) {
                if (arg_count != 2) {
                    emit_context_error(
                        "E003 ARITY MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        "Equality operators require exactly 2 arguments.",
                        "Provide two operands to compare."
                    );
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                /* Allow comparing opaque types with int (for null checks) */
                bool types_ok = types_match(left, right);
                if (!types_ok) {
                    /* Check if we're comparing an opaque type with int (null check) */
                    if ((left == TYPE_STRUCT || left == TYPE_INT) && (right == TYPE_STRUCT || right == TYPE_INT)) {
                        /* One might be an opaque type - this is allowed for null checks */
                        types_ok = true;
                    }
                }
                
                if (!types_ok) {
                    char message[256];
                    snprintf(message, sizeof(message),
                            "Equality requires both operands to be the same type (got %s and %s).",
                            type_to_string(left), type_to_string(right));
                    emit_context_error(
                        "E001 TYPE MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        message,
                        "Convert operands to the same type before checking equality."
                    );
                }
                return TYPE_BOOL;
            }

            /* Logical operators */
            if (op == TOKEN_AND || op == TOKEN_OR) {
                if (arg_count != 2) {
                    emit_context_error(
                        "E003 ARITY MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        "Logical operators require exactly 2 arguments.",
                        "Provide two boolean operands."
                    );
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                if (left != TYPE_BOOL || right != TYPE_BOOL) {
                    emit_context_error(
                        "E001 TYPE MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        "Logical operators require bool operands.",
                        "Convert operands to bool before using and/or."
                    );
                }
                return TYPE_BOOL;
            }

            if (op == TOKEN_NOT) {
                if (arg_count != 1) {
                    emit_context_error(
                        "E003 ARITY MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        "`not` requires exactly 1 argument.",
                        "Provide a single boolean operand."
                    );
                    return TYPE_UNKNOWN;
                }
                Type arg = check_expression(expr->as.prefix_op.args[0], env);
                if (arg != TYPE_BOOL) {
                    emit_context_error(
                        "E001 TYPE MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        "`not` requires a bool operand.",
                        "Convert the operand to bool before using not."
                    );
                }
                return TYPE_BOOL;
            }

            if (op == TOKEN_QUESTION) {
                /* ? try-propagate operator: expr? returns the Ok value type */
                if (arg_count != 1) {
                    emit_context_error("E008 SYNTAX ERROR", expr->line, expr->column, 1,
                        "? operator requires exactly 1 operand",
                        "Use 'expr?' — the ? operator is a postfix unary operator");
                    return TYPE_UNKNOWN;
                }
                Type inner_type = check_expression(expr->as.prefix_op.args[0], env);
                if (inner_type != TYPE_UNION) {
                    emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                        "? operator requires a Result union type",
                        "Declare your union with Ok and Err variants: 'union Result { Ok { val: T }, Err { msg: string } }'");
                    return TYPE_UNKNOWN;
                }
                /* Look up the union definition to find the Ok variant's first field type */
                const char *union_name = get_struct_type_name(expr->as.prefix_op.args[0], env);
                if (union_name) {
                    UnionDef *udef = env_get_union(env, union_name);
                    if (udef) {
                        for (int vi = 0; vi < udef->variant_count; vi++) {
                            if (strcmp(udef->variant_names[vi], "Ok") == 0) {
                                if (udef->variant_field_counts[vi] > 0) {
                                    return udef->variant_field_types[vi][0];
                                }
                            }
                        }
                    }
                }
                return TYPE_UNKNOWN;
            }

            return TYPE_UNKNOWN;
        }

        case AST_CALL: {
            if (expr->as.call.borrow_mode) {
                emit_context_error("E0036", expr->line, expr->column, 1,
                    "I allow a borrow only as a declared direct-call argument", "Keep the borrow call-scoped.");
                return TYPE_UNKNOWN;
            }
            /* Check if this is a function call returning a function: ((func_call) arg1 arg2) */
            if (expr->as.call.func_expr) {
                /* First, check the inner function call */
                Type inner_type = check_expression(expr->as.call.func_expr, env);
                if (inner_type != TYPE_FUNCTION) {
                    emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                        "Expression does not return a function",
                        "Only function-typed values can be called — check the return type of the inner expression");
                    return TYPE_UNKNOWN;
                }
                
                return check_indirect_call(expr, env, function_result_signature(expr->as.call.func_expr, env));
            }
            
            /* Representation copies never use implicit numeric promotion. */
            if (strcmp(expr->as.call.name, "float_from_bits") == 0 ||
                strcmp(expr->as.call.name, "float_to_bits") == 0) {
                if (env_get_var_visible_at(env, expr->as.call.name, expr->line, expr->column)) {
                    emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                        "I cannot use a bound value as a binary64 bit intrinsic.",
                        "Use an unshadowed intrinsic name.");
                    return TYPE_UNKNOWN;
                }
                bool from = strcmp(expr->as.call.name, "float_from_bits") == 0;
                if (expr->as.call.arg_count != 1 ||
                    check_expression(expr->as.call.args[0], env) != (from ? TYPE_INT : TYPE_FLOAT)) {
                    emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                        "I require one exactly typed operand for binary64 bit transport.",
                        "Use int for float_from_bits and float for float_to_bits.");
                    return TYPE_UNKNOWN;
                }
                return from ? TYPE_FLOAT : TYPE_INT;
            }

            /* I resolve these call spellings through lexical/declaration authority first. */
            if (env_native_array_operation(expr->as.call.name)) {
                Symbol *binding = env_get_var_visible_at(env, expr->as.call.name, expr->line, expr->column);
                if (binding) {
                    binding->is_used = true;
                    if (binding->type != TYPE_FUNCTION) {
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            "I require a function value for this bound array call.",
                            "Call the declared function or a function-typed binding.");
                        return TYPE_UNKNOWN;
                    }
                    return check_indirect_call(expr, env,
                        binding->type_info ? binding->type_info->fn_sig : NULL);
                }
            }

            if (env_native_array_operation(expr->as.call.name) &&
                !env_native_array_is_builtin(env, expr->as.call.name, expr->line, expr->column)) goto checked_array_declared_call;

            /* Regular function call */
            
            /* Special handling for map builtin - check before environment lookup */
            if (strcmp(expr->as.call.name, "map") == 0) {
                if (expr->as.call.arg_count != 2) {
                    emit_context_error("E003 ARITY MISMATCH", expr->line, expr->column, 1,
                        "I require an array and a transform for map.", "Pass exactly two arguments.");
                    return TYPE_UNKNOWN;
                }
                Type array_type = check_expression(expr->as.call.args[0], env);
                Type callback_type = check_expression(expr->as.call.args[1], env);
                int arity;
                Type argument;
                Type result = map_callback_type(expr->as.call.args[1], env, &arity, &argument);
                Type element = infer_array_element_type(expr->as.call.args[0], env);
                if (array_type != TYPE_ARRAY || callback_type != TYPE_FUNCTION || arity != 1 ||
                    result == TYPE_UNKNOWN || result == TYPE_VOID ||
                    (element != TYPE_UNKNOWN && !types_match(element, argument))) {
                    emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                        "I require a unary transform matching the array element type and returning a value.",
                        "Match the transform signature to the source array.");
                    return TYPE_UNKNOWN;
                }
                if (!check_opaque_array_callback(expr, env, false)) return TYPE_UNKNOWN;
                return TYPE_ARRAY;
            }

            /* Special handling for filter builtin - check before environment lookup */
            if (strcmp(expr->as.call.name, "filter") == 0) {
                /* filter(array, predicate_fn) -> array */
                if (expr->as.call.arg_count >= 2) {
                    Type array_type = check_expression(expr->as.call.args[0], env);
                    check_expression(expr->as.call.args[1], env);  /* Check function */
                    if (!check_opaque_array_callback(expr, env, true)) return TYPE_UNKNOWN;
                    return array_type;
                }
                return TYPE_ARRAY;
            }
            
            /* I validate the callback as one exact accumulator/element contract. */
            if (strcmp(expr->as.call.name, "reduce") == 0)
                return check_reduce_call(expr, env);

            /* Special handling for format builtin - variadic string interpolation */
            if (strcmp(expr->as.call.name, "format") == 0) {
                if (expr->as.call.arg_count < 1) {
                    emit_context_error("E003 ARITY MISMATCH", expr->line, expr->column, 1,
                        "format requires at least 1 argument (the template string).",
                        "Usage: format(\"Hello %s\", name)");
                    return TYPE_UNKNOWN;
                }
                Type template_type = check_expression(expr->as.call.args[0], env);
                if (template_type != TYPE_STRING) {
                    ASTNode *template = expr->as.call.args[0];
                    emit_context_error("E001 TYPE MISMATCH", template->line, template->column, 1,
                        "I require a string template for format.",
                        "Pass the template string before its substitution arguments.");
                }
                for (int i = 1; i < expr->as.call.arg_count; i++) {
                    check_expression(expr->as.call.args[i], env);
                }
                return TYPE_STRING;
            }

            /* Result<T, E> helper intrinsics (generic-function stopgap) */
            if (strcmp(expr->as.call.name, "result_is_ok") == 0 || strcmp(expr->as.call.name, "result_is_err") == 0) {
                if (expr->as.call.arg_count != 1) {
                    char message[256];
                    snprintf(message, sizeof(message), "%s requires exactly 1 argument, got %d.",
                             expr->as.call.name, expr->as.call.arg_count);
                    emit_context_error("E003 ARITY MISMATCH", expr->line, expr->column, 1, message,
                                       "Pass a single Result<T, E> value.");
                    return TYPE_UNKNOWN;
                }
                check_expression(expr->as.call.args[0], env);
                return TYPE_BOOL;
            }

            if (strcmp(expr->as.call.name, "result_unwrap") == 0 ||
                strcmp(expr->as.call.name, "result_unwrap_err") == 0 ||
                strcmp(expr->as.call.name, "result_unwrap_or") == 0) {
                int expected = (strcmp(expr->as.call.name, "result_unwrap_or") == 0) ? 2 : 1;
                if (expr->as.call.arg_count != expected) {
                    char message[256];
                    snprintf(message, sizeof(message), "%s requires %d argument(s), got %d.",
                             expr->as.call.name, expected, expr->as.call.arg_count);
                    emit_context_error("E003 ARITY MISMATCH", expr->line, expr->column, 1, message,
                                       "Pass a Result<T, E> value and (for result_unwrap_or) a default value.");
                    return TYPE_UNKNOWN;
                }

                ASTNode *res_expr = expr->as.call.args[0];
                Type res_type = check_expression(res_expr, env);
                if (res_type != TYPE_UNION) {
                    fprintf(stderr, "Error at line %d, column %d: %s requires a Result<T, E> union value\n",
                            expr->line, expr->column, expr->as.call.name);
                    return TYPE_UNKNOWN;
                }

                TypeInfo *res_info = try_get_expr_type_info(res_expr, env);
                if (!res_info || res_info->base_type != TYPE_UNION || !res_info->generic_name || strcmp(res_info->generic_name, "Result") != 0 || res_info->type_param_count < 2) {
                    return TYPE_UNKNOWN;
                }

                int idx = (strcmp(expr->as.call.name, "result_unwrap_err") == 0) ? 1 : 0;
                TypeInfo *param = res_info->type_params[idx];
                const char *struct_name = NULL;
                Type out_type = type_from_typeinfo(param, &struct_name);
                if (out_type == TYPE_STRUCT && struct_name) {
                    if (expr->as.call.return_struct_type_name) free(expr->as.call.return_struct_type_name);
                    expr->as.call.return_struct_type_name = strdup(struct_name);
                }

                if (strcmp(expr->as.call.name, "result_unwrap_or") == 0) {
                    Type default_type = check_expression(expr->as.call.args[1], env);
                    if (default_type != out_type) {
                        char message[256];
                        snprintf(message, sizeof(message),
                                 "result_unwrap_or default value type mismatch: got %s, expected %s.",
                                 type_to_string(default_type), type_to_string(out_type));
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1, message,
                                           "The default value must be the same type as the Ok result.");
                    }
                }

                return out_type;
            }

            if (strcmp(expr->as.call.name, "result_map") == 0 || strcmp(expr->as.call.name, "result_and_then") == 0) {
                if (expr->as.call.arg_count != 2) {
                    fprintf(stderr, "Error at line %d, column %d: %s requires 2 arguments\n",
                            expr->line, expr->column, expr->as.call.name);
                    return TYPE_UNKNOWN;
                }
                check_expression(expr->as.call.args[0], env);
                check_expression(expr->as.call.args[1], env);
                return TYPE_UNION;
            }
            
checked_array_declared_call: ;
            /* Check if function exists */
            Function *func = env_get_function(env, expr->as.call.name);
            
            /* Check visibility */
            if (func && !is_function_accessible(func, env, expr->line, expr->column)) {
                return TYPE_UNKNOWN;
            }

            if (strcmp(expr->as.call.name, "array_push") == 0 &&
                env_array_push_is_builtin(env, expr->line, expr->column) &&
                expr->as.call.arg_count == 2) {
                ASTNode *receiver = expr->as.call.args[0];
                if (receiver->type == AST_ARRAY_LITERAL &&
                    receiver->as.array_literal.element_count == 0) {
                    Type element = check_expression(expr->as.call.args[1], env);
                    if (element != TYPE_UNKNOWN) {
                        receiver->as.array_literal.element_type = element;
                    }
                }
            }
            
            /* If not a function, check if it's a function-typed variable (parameter) */
            /* ALSO: prefer built-in HashMap<K,V> generics when there's a generic type context,
             * even if a user-defined map_* function exists (to avoid conflict with non-generic
             * HashMap module functions like map_new() -> HashMap) */
            bool is_hashmap_generic_context = false;
            
            /* Check return type context (for map_new) */
            if (expr->as.call.return_struct_type_name &&
                strncmp(expr->as.call.return_struct_type_name, "HashMap_", 8) == 0) {
                is_hashmap_generic_context = true;
            }
            
            /* Check first argument type context (for map_put, map_get, etc.) */
            if (!is_hashmap_generic_context && expr->as.call.arg_count >= 1 && 
                expr->as.call.args[0] && expr->as.call.args[0]->type == AST_IDENTIFIER) {
                Symbol *first_arg_sym = env_get_var_visible_at(env, expr->as.call.args[0]->as.identifier,
                    expr->as.call.args[0]->line, expr->as.call.args[0]->column);
                if (first_arg_sym && first_arg_sym->type == TYPE_HASHMAP) {
                    is_hashmap_generic_context = true;
                }
            }
            
            bool is_map_builtin = expr->as.call.name && (
                strcmp(expr->as.call.name, "map_new") == 0 ||
                strcmp(expr->as.call.name, "map_put") == 0 ||
                strcmp(expr->as.call.name, "map_set") == 0 ||
                strcmp(expr->as.call.name, "map_get") == 0 ||
                strcmp(expr->as.call.name, "map_has") == 0 ||
                strcmp(expr->as.call.name, "map_remove") == 0 ||
                strcmp(expr->as.call.name, "map_length") == 0 ||
                strcmp(expr->as.call.name, "map_size") == 0 ||
                strcmp(expr->as.call.name, "map_clear") == 0 ||
                strcmp(expr->as.call.name, "map_free") == 0 ||
                strcmp(expr->as.call.name, "map_keys") == 0 ||
                strcmp(expr->as.call.name, "map_values") == 0);
            if (!func || (is_map_builtin && is_hashmap_generic_context)) {
                Symbol *sym = env_get_var_visible_at(env, expr->as.call.name, expr->line, expr->column);
                if (sym && sym->type == TYPE_FUNCTION) {
                    /* Mark the variable as used */
                    sym->is_used = true;
                    
                    return check_indirect_call(expr, env, sym->type_info ? sym->type_info->fn_sig : NULL);
                }
                
                /* Special handling for dynamic array builtins */
                if (strcmp(expr->as.call.name, "array_push") == 0) {
                    /* array_push(array, value) -> array */
                    if (expr->as.call.arg_count >= 1) {
                        check_expression(expr->as.call.args[0], env);
                        if (expr->as.call.arg_count >= 2) {
                            check_expression(expr->as.call.args[1], env);
                        }
                    }
                    return TYPE_ARRAY;
                }
                
                if (strcmp(expr->as.call.name, "array_pop") == 0) {
                    /* array_pop(array) -> element type (infer from array) */
                    if (expr->as.call.arg_count >= 1) {
                        ASTNode *array_arg = expr->as.call.args[0];
                        check_expression(array_arg, env);
                        
                        const TypeInfo *complete = try_get_expr_type_info(array_arg, env);
                        if (complete && complete->base_type == TYPE_ARRAY && complete->element_type)
                            return complete->element_type->base_type;

                        /* Try to infer element type from array */
                        if (array_arg->type == AST_IDENTIFIER) {
                            Symbol *sym = env_get_var_visible_at(env, array_arg->as.identifier, array_arg->line, array_arg->column);
                            if (sym && sym->element_type != TYPE_UNKNOWN) {
                                return sym->element_type;
                            }
                        }

                        if (array_arg->type == AST_CALL && array_arg->as.call.name) {
                            if (strcmp(array_arg->as.call.name, "file_read_bytes") == 0 ||
                                strcmp(array_arg->as.call.name, "bytes_from_string") == 0) {
                                return TYPE_U8;
                            }
                            if (strcmp(array_arg->as.call.name, "array_slice") == 0 && array_arg->as.call.arg_count >= 1) {
                                ASTNode *inner = array_arg->as.call.args[0];
                                if (inner && inner->type == AST_IDENTIFIER) {
                                    Symbol *sym = env_get_var_visible_at(env, inner->as.identifier, inner->line, inner->column);
                                    if (sym && sym->element_type != TYPE_UNKNOWN) {
                                        return sym->element_type;
                                    }
                                }
                            }
                        }
                    }
                    return TYPE_INT;  /* Default fallback */
                }
                
                if (strcmp(expr->as.call.name, "array_remove_at") == 0) {
                    /* array_remove_at(array, index) -> array */
                    if (expr->as.call.arg_count >= 1) {
                        check_expression(expr->as.call.args[0], env);
                        if (expr->as.call.arg_count >= 2) {
                            check_expression(expr->as.call.args[1], env);
                        }
                    }
                    return TYPE_ARRAY;
                }
                
                /* Special handling for file_read_bytes builtin */
                if (strcmp(expr->as.call.name, "file_read_bytes") == 0) {
                    /* file_read_bytes(filename) -> array<u8> of bytes */
                    if (expr->as.call.arg_count >= 1) {
                        check_expression(expr->as.call.args[0], env);
                    }
                    return TYPE_ARRAY;
                }

                /* Special handling for bytes_from_string/string_from_bytes */
                if (strcmp(expr->as.call.name, "bytes_from_string") == 0) {
                    if (expr->as.call.arg_count >= 1) {
                        check_expression(expr->as.call.args[0], env);
                    }
                    return TYPE_ARRAY;
                }

                if (strcmp(expr->as.call.name, "string_from_bytes") == 0) {
                    if (expr->as.call.arg_count >= 1) {
                        check_expression(expr->as.call.args[0], env);
                    }
                    return TYPE_STRING;
                }

                /* Special handling for array_slice */
                if (strcmp(expr->as.call.name, "array_slice") == 0) {
                    if (expr->as.call.arg_count >= 1) {
                        check_expression(expr->as.call.args[0], env);
                    }
                    if (expr->as.call.arg_count >= 2) {
                        check_expression(expr->as.call.args[1], env);
                    }
                    if (expr->as.call.arg_count >= 3) {
                        check_expression(expr->as.call.args[2], env);
                    }
                    return TYPE_ARRAY;
                }
                
                /* Special handling for bstring operations */
                if (strcmp(expr->as.call.name, "bstr_new") == 0 ||
                    strcmp(expr->as.call.name, "bstr_new_binary") == 0 ||
                    strcmp(expr->as.call.name, "bstr_concat") == 0 ||
                    strcmp(expr->as.call.name, "bstr_substring") == 0) {
                    /* These return bstring */
                    for (int i = 0; i < expr->as.call.arg_count; i++) {
                        check_expression(expr->as.call.args[i], env);
                    }
                    return TYPE_BSTRING;
                }
                
                if (strcmp(expr->as.call.name, "bstr_length") == 0 ||
                    strcmp(expr->as.call.name, "bstr_byte_at") == 0 ||
                    strcmp(expr->as.call.name, "bstr_utf8_length") == 0 ||
                    strcmp(expr->as.call.name, "bstr_utf8_char_at") == 0) {
                    /* These return int */
                    for (int i = 0; i < expr->as.call.arg_count; i++) {
                        check_expression(expr->as.call.args[i], env);
                    }
                    return TYPE_INT;
                }
                
                if (strcmp(expr->as.call.name, "bstr_equals") == 0 ||
                    strcmp(expr->as.call.name, "bstr_validate_utf8") == 0) {
                    /* These return bool */
                    for (int i = 0; i < expr->as.call.arg_count; i++) {
                        check_expression(expr->as.call.args[i], env);
                    }
                    return TYPE_BOOL;
                }
                
                if (strcmp(expr->as.call.name, "bstr_to_cstr") == 0) {
                    /* bstring -> string conversion */
                    if (expr->as.call.arg_count >= 1) {
                        check_expression(expr->as.call.args[0], env);
                    }
                    return TYPE_STRING;
                }
                
                if (strcmp(expr->as.call.name, "bstr_free") == 0) {
                    /* void return */
                    if (expr->as.call.arg_count >= 1) {
                        check_expression(expr->as.call.args[0], env);
                    }
                    return TYPE_VOID;
                }
                
                /* Special handling for array_get builtin */
                if (strcmp(expr->as.call.name, "array_get") == 0) {
                    if (!check_array_access_arguments(expr, env)) return TYPE_UNKNOWN;
                    Type inferred = infer_array_element_type(expr->as.call.args[0], env);
                    if (inferred != TYPE_UNKNOWN) return inferred;
                    /* array_get(array, index) -> element type (same as at()) */
                    if (expr->as.call.arg_count >= 1) {
                        ASTNode *array_arg = expr->as.call.args[0];
                        check_expression(array_arg, env);
                        
                        /* Try to infer element type from array */
                        if (array_arg->type == AST_IDENTIFIER) {
                            Symbol *sym = env_get_var_visible_at(env, array_arg->as.identifier, array_arg->line, array_arg->column);
                            if (sym && sym->element_type != TYPE_UNKNOWN) {
                                return sym->element_type;
                            }
                        }

                        if (array_arg->type == AST_CALL && array_arg->as.call.name) {
                            if (strcmp(array_arg->as.call.name, "file_read_bytes") == 0 ||
                                strcmp(array_arg->as.call.name, "bytes_from_string") == 0) {
                                return TYPE_U8;
                            }
                        }
                    }
                    return TYPE_INT;  /* Default fallback */
                }
                
                /* Special handling for array_length builtin */
                if (strcmp(expr->as.call.name, "array_length") == 0) {
                    /* array_length(array) -> int */
                    if (expr->as.call.arg_count >= 1) {
                        check_expression(expr->as.call.args[0], env);
                    }
                    return TYPE_INT;
                }

                /* HashMap<K,V> core built-ins (only if no user-defined function with same name exists) */
                if (strcmp(expr->as.call.name, "map_new") == 0) {
                    if (expr->as.call.arg_count != 0) {
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                                           "I require zero arguments for map_new.",
                                           "Declare the HashMap<K,V> type on the receiving binding.");
                        return TYPE_UNKNOWN;
                    }

                    /* Requires type context (e.g., let hm: HashMap<K,V> = (map_new)) */
                    if (!expr->as.call.return_struct_type_name) {
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                                           "I require a HashMap<K,V> type annotation for map_new.",
                                           "Use a typed local binding before returning or passing the map.");
                        return TYPE_UNKNOWN;
                    }
                    return TYPE_HASHMAP;
                }

                if (strcmp(expr->as.call.name, "map_put") == 0 || strcmp(expr->as.call.name, "map_set") == 0) {
                    if (expr->as.call.arg_count != 3) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: %s requires 3 arguments.", expr->as.call.name);
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    Type hm_t = check_expression(expr->as.call.args[0], env);
                    Type key_t = check_expression(expr->as.call.args[1], env);
                    Type val_t = check_expression(expr->as.call.args[2], env);
                    if (hm_t != TYPE_HASHMAP) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: %s expects HashMap as first argument.", expr->as.call.name);
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    TypeInfo *hm_info = try_get_expr_type_info(expr->as.call.args[0], env);
                    Type exp_k = TYPE_UNKNOWN;
                    Type exp_v = TYPE_UNKNOWN;
                    if (!hashmap_extract_kv(hm_info, &exp_k, &exp_v)) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: Cannot infer HashMap<K,V> type arguments.");
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    if (!types_match(key_t, exp_k) || !types_match(val_t, exp_v)) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: %s expects key %s and value %s.", expr->as.call.name, type_to_string(exp_k), type_to_string(exp_v));
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    return TYPE_VOID;
                }

                if (strcmp(expr->as.call.name, "map_get") == 0) {
                    if (expr->as.call.arg_count != 2) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: map_get requires 2 arguments.");
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    Type hm_t = check_expression(expr->as.call.args[0], env);
                    Type key_t = check_expression(expr->as.call.args[1], env);
                    if (hm_t != TYPE_HASHMAP) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: map_get expects HashMap as first argument.");
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    TypeInfo *hm_info = try_get_expr_type_info(expr->as.call.args[0], env);
                    Type exp_k = TYPE_UNKNOWN;
                    Type exp_v = TYPE_UNKNOWN;
                    if (!hashmap_extract_kv(hm_info, &exp_k, &exp_v)) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: Cannot infer HashMap<K,V> type arguments.");
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    if (!types_match(key_t, exp_k)) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: map_get expects key type %s.", type_to_string(exp_k));
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    return exp_v;
                }

                if (strcmp(expr->as.call.name, "map_has") == 0) {
                    if (expr->as.call.arg_count != 2) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: map_has requires 2 arguments.");
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    Type hm_t = check_expression(expr->as.call.args[0], env);
                    Type key_t = check_expression(expr->as.call.args[1], env);
                    if (hm_t != TYPE_HASHMAP) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: map_has expects HashMap as first argument.");
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    TypeInfo *hm_info = try_get_expr_type_info(expr->as.call.args[0], env);
                    Type exp_k = TYPE_UNKNOWN;
                    if (!hashmap_extract_kv(hm_info, &exp_k, NULL)) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: Cannot infer HashMap<K,V> type arguments.");
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    if (!types_match(key_t, exp_k)) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: map_has expects key type %s.", type_to_string(exp_k));
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    return TYPE_BOOL;
                }

                if (strcmp(expr->as.call.name, "map_remove") == 0) {
                    if (expr->as.call.arg_count != 2) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: map_remove requires 2 arguments.");
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    Type hm_t = check_expression(expr->as.call.args[0], env);
                    Type key_t = check_expression(expr->as.call.args[1], env);
                    if (hm_t != TYPE_HASHMAP) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: map_remove expects HashMap as first argument.");
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    TypeInfo *hm_info = try_get_expr_type_info(expr->as.call.args[0], env);
                    Type exp_k = TYPE_UNKNOWN;
                    if (!hashmap_extract_kv(hm_info, &exp_k, NULL)) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: Cannot infer HashMap<K,V> type arguments.");
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    if (!types_match(key_t, exp_k)) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: map_remove expects key type %s.", type_to_string(exp_k));
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    return TYPE_VOID;
                }

                if (strcmp(expr->as.call.name, "map_length") == 0 || strcmp(expr->as.call.name, "map_size") == 0) {
                    if (expr->as.call.arg_count != 1) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: %s requires 1 argument.", expr->as.call.name);
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    Type hm_t = check_expression(expr->as.call.args[0], env);
                    if (hm_t != TYPE_HASHMAP) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: %s expects HashMap as first argument.", expr->as.call.name);
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    return TYPE_INT;
                }

                if (strcmp(expr->as.call.name, "map_clear") == 0 || strcmp(expr->as.call.name, "map_free") == 0) {
                    if (expr->as.call.arg_count != 1) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: %s requires 1 argument.", expr->as.call.name);
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    Type hm_t = check_expression(expr->as.call.args[0], env);
                    if (hm_t != TYPE_HASHMAP) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: %s expects HashMap as first argument.", expr->as.call.name);
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    return TYPE_VOID;
                }

                if (strcmp(expr->as.call.name, "map_keys") == 0 || strcmp(expr->as.call.name, "map_values") == 0) {
                    if (expr->as.call.arg_count != 1) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: %s requires 1 argument.", expr->as.call.name);
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    Type hm_t = check_expression(expr->as.call.args[0], env);
                    if (hm_t != TYPE_HASHMAP) {
                        char message[256];
                        snprintf(message, sizeof(message), "I cannot accept this map call: %s expects HashMap as first argument.", expr->as.call.name);
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            message, "Match the map operation arity and declared key/value types.");
                        return TYPE_UNKNOWN;
                    }
                    return TYPE_ARRAY;
                }
                
                /* Special handling for generic list functions: list_TypeName_operation */
                const char *func_name = expr->as.call.name;
                if (func_name && strncmp(func_name, "list_", 5) == 0) {
                    /* Find the last underscore to identify the operation */
                    const char *func_suffix = strrchr(func_name, '_');
                    if (func_suffix) {
                        /* Extract type name: "list_MyType_new" -> "MyType" */
                        const char *type_start = func_name + 5;  /* Skip "list_" */
                        int type_name_len = (int)(func_suffix - type_start);
                        if (type_name_len > 0) {
                            char *type_name = malloc(type_name_len + 1);
                            strncpy(type_name, type_start, type_name_len);
                            type_name[type_name_len] = '\0';
                            
                            /* Check if this type name exists as a struct or enum */
                            StructDef *sdef = env_get_struct(env, type_name);
                            EnumDef *edef = env_get_enum(env, type_name);
                            
                            if (sdef || edef) {
                                /* Type check based on the operation */
                                const char *operation = func_suffix + 1;  /* Skip the '_' */
                                
                                /* Type check arguments */
                                for (int i = 0; i < expr->as.call.arg_count; i++) {
                                    check_expression(expr->as.call.args[i], env);
                                }
                                
                                /* Return appropriate type based on operation */
                                if (strcmp(operation, "new") == 0 || strcmp(operation, "with_capacity") == 0) {
                                    /* Register this instantiation for code generation */
                                    env_register_list_instantiation(env, type_name);
                                    free(type_name);
                                    return TYPE_LIST_GENERIC;  /* Returns List<Type> */
                                } else if (strcmp(operation, "get") == 0) {
                                    /* Set struct type name on the call node for field access */
                                    if (!edef && sdef) {
                                        expr->as.call.return_struct_type_name = strdup(type_name);
                                    }
                                    free(type_name);
                                    return edef ? TYPE_ENUM : TYPE_STRUCT;  /* Returns element type */
                                } else if (strcmp(operation, "length") == 0 || strcmp(operation, "capacity") == 0) {
                                    free(type_name);
                                    return TYPE_INT;
                                } else if (strcmp(operation, "is_empty") == 0) {
                                    free(type_name);
                                    return TYPE_BOOL;
                                } else if (strcmp(operation, "pop") == 0) {
                                    free(type_name);
                                    return edef ? TYPE_ENUM : TYPE_STRUCT;  /* Returns element type */
                                } else {
                                    /* push, set, insert, remove, clear, free return void */
                                    free(type_name);
                                    return TYPE_VOID;
                                }
                            } else {
                                /* Better error message: list function for unknown type */
                                char message[256];
                                snprintf(message, sizeof(message),
                                        "Unknown list element type `%s` in `%s`.",
                                        type_name, expr->as.call.name);
                                emit_context_error(
                                    "E026 UNDEFINED IDENTIFIER",
                                    expr->line,
                                    expr->column,
                                    (int)safe_strlen(expr->as.call.name),
                                    message,
                                    "Define the struct/enum before using it in List<T>."
                                );
                                free(type_name);
                                return TYPE_UNKNOWN;
                            }
                        }
                    }
                }
                
                char message[256];
                snprintf(message, sizeof(message),
                        "I cannot find a function named `%s`.",
                        safe_format_string(expr->as.call.name));
                emit_context_error(
                    "E027 UNDEFINED FUNCTION",
                    expr->line,
                    expr->column,
                    (int)safe_strlen(expr->as.call.name),
                    message,
                    "Check spelling or ensure the function is defined/imported."
                );
                return TYPE_UNKNOWN;
            }

            /* Check argument count */
            if (expr->as.call.arg_count != func->param_count) {
                char message[256];
                snprintf(message, sizeof(message),
                        "Function `%s` expects %d argument(s), but got %d.",
                        safe_format_string(expr->as.call.name), func->param_count, expr->as.call.arg_count);
                emit_context_error(
                    "E003 ARITY MISMATCH",
                    expr->line,
                    expr->column,
                    (int)safe_strlen(expr->as.call.name),
                    message,
                    "Add or remove arguments to match the function signature."
                );
                return TYPE_UNKNOWN;
            }

            /* ── Generic function handling ─────────────────────────────────────── */
            /* If the function has type-variable parameters (T, E, etc.), collect
             * the concrete types from the call site, register a monomorphized
             * instance, and store the concrete function name on the call node.   */
            if (func_is_generic(func)) {
                /* Collect unique type variables in first-appearance order */
                char *var_names_buf[16];
                Type  bound_types_buf[16];
                char *bound_names_buf[16];
                int   binding_count = 0;

                for (int i = 0; i < func->param_count && binding_count < 16; i++) {
                    if (func->params[i].type != TYPE_STRUCT ||
                        !is_type_variable_name(func->params[i].struct_type_name)) continue;
                    const char *var = func->params[i].struct_type_name;
                    bool already = false;
                    for (int k = 0; k < binding_count; k++) {
                        if (strcmp(var_names_buf[k], var) == 0) { already = true; break; }
                    }
                    if (!already) {
                        var_names_buf[binding_count] = (char *)var;
                        bound_types_buf[binding_count] = TYPE_UNKNOWN;
                        bound_names_buf[binding_count] = NULL;
                        binding_count++;
                    }
                }

                /* Resolve each type variable from the argument types */
                for (int i = 0; i < func->param_count && i < expr->as.call.arg_count; i++) {
                    if (func->params[i].type != TYPE_STRUCT ||
                        !is_type_variable_name(func->params[i].struct_type_name)) continue;
                    const char *var = func->params[i].struct_type_name;
                    Type arg_type = check_expression(expr->as.call.args[i], env);
                    for (int k = 0; k < binding_count; k++) {
                        if (strcmp(var_names_buf[k], var) == 0) {
                            if (bound_types_buf[k] == TYPE_UNKNOWN) {
                                bound_types_buf[k] = arg_type;
                                /* Capture struct name for struct-typed args */
                                if (arg_type == TYPE_STRUCT) {
                                    ASTNode *a = expr->as.call.args[i];
                                    if (a->type == AST_IDENTIFIER) {
                                        Symbol *sym = env_get_var_visible_at(env, a->as.identifier, a->line, a->column);
                                        if (sym) bound_names_buf[k] = sym->struct_type_name;
                                    }
                                }
                            } else if (bound_types_buf[k] != arg_type) {
                                char message[256];
                                snprintf(message, sizeof(message),
                                        "Type variable `%s` is bound to %s but argument %d has type %s.",
                                        var, type_to_string(bound_types_buf[k]), i + 1, type_to_string(arg_type));
                                emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1, message,
                                        "All uses of the same type variable must have the same concrete type.");
                            }
                            break;
                        }
                    }
                }

                /* Build monomorphized name and register instance */
                char mono_name[256];
                build_generic_mono_name(mono_name, sizeof(mono_name), func->name,
                                        var_names_buf, bound_types_buf, bound_names_buf, binding_count);
                env_register_generic_func_instance(env, func->name, mono_name,
                                                    (const char **)var_names_buf,
                                                    bound_types_buf,
                                                    (const char **)bound_names_buf,
                                                    binding_count);
                if (expr->as.call.concrete_func_name) free(expr->as.call.concrete_func_name);
                expr->as.call.concrete_func_name = strdup(mono_name);

                /* Determine return type — substitute type variable if needed */
                if (func->return_type == TYPE_STRUCT && is_type_variable_name(func->return_struct_type_name)) {
                    for (int k = 0; k < binding_count; k++) {
                        if (strcmp(var_names_buf[k], func->return_struct_type_name) == 0)
                            return bound_types_buf[k];
                    }
                }
                return func->return_type;
            }

            /* Check argument types (skip for built-ins with NULL params like range) */
            if (func->params) {
                for (int i = 0; i < expr->as.call.arg_count; i++) {
                    ASTNode *arg = expr->as.call.args[i];
                    
                    if (func->params[i].type == TYPE_BORROW_SHARED || func->params[i].type == TYPE_BORROW_MUT) {
                        int borrow_mode = func->params[i].type == TYPE_BORROW_MUT ? 2 : 1;
                        const TypeInfo *inner = func->params[i].type_info ? func->params[i].type_info->element_type : NULL;
                        ASTNode *place = arg->type == AST_CALL && arg->as.call.borrow_mode == borrow_mode && arg->as.call.arg_count == 1 ? arg->as.call.args[0] : NULL;
                        ASTNode *root = place;
                        while (root && root->type == AST_FIELD_ACCESS) root = root->as.field_access.object;
                        Symbol *owner = root && root->type == AST_IDENTIFIER ? env_get_var_visible_at(env, root->as.identifier, root->line, root->column) : NULL;
                        Type referent = owner ? check_expression(place, env) : TYPE_UNKNOWN;
                        const char *actual = referent == TYPE_STRUCT ? get_struct_type_name(place, env) : NULL;
                        if (!owner || !inner || !actual || !inner->generic_name || strcmp(actual, inner->generic_name) ||
                            (owner->type != TYPE_STRUCT && owner->type != TYPE_BORROW_SHARED && owner->type != TYPE_BORROW_MUT) ||
                            (borrow_mode == 2 && !(owner->type == TYPE_BORROW_MUT || (owner->type == TYPE_STRUCT && owner->is_mut)))) {
                            emit_context_error("E0036", arg->line, arg->column, 1,
                                "I require an explicit borrow of the declared owner with matching mutability", "Use &owner or &mut owner with the declared capability and nominal record.");
                            return TYPE_UNKNOWN;
                        }
                        check_expression(place, env);
                        continue;
                    }
                    /* Special handling for function-typed parameters */
                    if (func->params[i].type == TYPE_FUNCTION) {
                        /* Argument must be an identifier (function name or function-typed variable) */
                        if (arg->type != AST_IDENTIFIER) {
                            emit_context_error(
                                "E001 TYPE MISMATCH",
                                arg->line,
                                arg->column,
                                1,
                                "Function parameter expects a function name.",
                                "Pass a function identifier or a function-typed variable."
                            );
                            return TYPE_UNKNOWN;
                        }
                        
                        /* First check if it's a function-typed variable */
                        Symbol *sym = env_get_var_visible_at(env, arg->as.identifier, arg->line, arg->column);
                        if (sym && sym->type == TYPE_FUNCTION) {
                            /* It's a function-typed variable - mark as used and allow it */
                            sym->is_used = true;
                            FunctionSignature *actual = sym->type_info ? sym->type_info->fn_sig : NULL;
                            if (!function_signatures_equal(func->params[i].fn_sig, actual)) {
                                emit_context_error("E001 TYPE MISMATCH", arg->line, arg->column, 1,
                                    "I require the complete declared function signature.",
                                    "Match the callback parameter and result annotations.");
                                return TYPE_UNKNOWN;
                            }
                            continue;
                        }
                        
                        /* Look up the function */
                        Function *passed_func = env_get_function(env, arg->as.identifier);
                        if (!passed_func) {
                            char message[256];
                            snprintf(message, sizeof(message),
                                    "I cannot find a function named `%s`.",
                                    arg->as.identifier);
                            emit_context_error(
                                "E027 UNDEFINED FUNCTION",
                                arg->line,
                                arg->column,
                                (int)safe_strlen(arg->as.identifier),
                                message,
                                "Check spelling or ensure the function is defined/imported."
                            );
                            return TYPE_UNKNOWN;
                        }
                        
                        FunctionSignature *passed_sig = function_signature_from_function(passed_func);
                        
                        /* Compare signatures */
                        if (!function_signatures_equal(func->params[i].fn_sig, passed_sig)) {
                            char message[256];
                            snprintf(message, sizeof(message),
                                    "Argument %d expects a function with a different signature.",
                                    i + 1);
                            emit_context_error(
                                "E001 TYPE MISMATCH",
                                arg->line,
                                arg->column,
                                (int)safe_strlen(arg->as.identifier),
                                message,
                                "Match the parameter's expected function signature."
                            );
                            free_function_signature(passed_sig);
                            return TYPE_UNKNOWN;
                        }
                        
                        /* Clean up temporary signature */
                        free_function_signature(passed_sig);
                    } else {
                        /* Handle anonymous struct literals: infer struct name from parameter type */
                        if (arg->type == AST_STRUCT_LITERAL && arg->as.struct_literal.struct_name == NULL) {
                            if (func->params[i].type == TYPE_STRUCT && func->params[i].struct_type_name) {
                                arg->as.struct_literal.struct_name = strdup(func->params[i].struct_type_name);
                            } else {
                                fprintf(stderr, "Error at line %d, column %d: Cannot infer struct type for anonymous literal in function argument\n",
                                        arg->line, arg->column);
                            }
                        }
                        
                        /* Regular argument type checking */
                        check_concrete_union_arrays(env, func->params[i].type_info, arg, 0);
                        Type arg_type = check_expression(arg, env);
                        if (arg->type == AST_ARRAY_LITERAL &&
                            arg->as.array_literal.element_count == 0 &&
                            func->params[i].type == TYPE_ARRAY &&
                            func->params[i].element_type != TYPE_UNKNOWN) {
                            arg->as.array_literal.element_type = resolved_array_element(func->params[i].element_type, func->params[i].struct_type_name, env);
                        }
                        
                        /* Check for opaque type parameters. I accept only the
                         * literal integer zero as the source spelling of null;
                         * arbitrary integers are not pointer values. */
                        bool is_opaque_param = false;
                        bool named_opaque =
                            func->params[i].type == TYPE_STRUCT &&
                            func->params[i].struct_type_name &&
                            env_get_opaque_type(env, func->params[i].struct_type_name);
                        if (func->params[i].type == TYPE_OPAQUE || named_opaque) {
                                const char *opaque_name = func->params[i].struct_type_name
                                    ? func->params[i].struct_type_name : "opaque";
                                bool null_literal = arg_type == TYPE_INT &&
                                    arg->type == AST_NUMBER && arg->as.number == 0;
                                is_opaque_param = true;
                                if (!null_literal && !check_opaque_value(env, func->params[i].type,
                                        func->params[i].struct_type_name, arg)) {
                                    char message[256];
                                    snprintf(message, sizeof(message),
                                            "Argument %d expects opaque type `%s` or 0 (null), got %s.",
                                            i + 1,
                                            opaque_name,
                                            type_to_string(arg_type));
                                    emit_context_error(
                                        "E001 TYPE MISMATCH",
                                        expr->line,
                                        expr->column,
                                        1,
                                        message,
                                        "Pass the opaque handle or 0 (null)."
                                    );
                                }
                        }
                        
                        /* Check for reverse case: parameter is int but argument is opaque type
                         * Opaque types are stored as int64_t, so they can be passed to int parameters */
                        bool is_opaque_arg = false;
                        if (func->params[i].type == TYPE_INT && arg_type == TYPE_STRUCT) {
                            /* Check if the argument is a variable with an opaque type */
                            if (arg->type == AST_IDENTIFIER) {
                                Symbol *sym = env_get_var_visible_at(env, arg->as.identifier, arg->line, arg->column);
                                if (sym && sym->type == TYPE_STRUCT && sym->struct_type_name) {
                                    OpaqueTypeDef *opaque = env_get_opaque_type(env, sym->struct_type_name);
                                    if (opaque) {
                                        is_opaque_arg = true;  /* Allow opaque type to be passed as int */
                                    }
                                }
                            }
                            /* Also check if the argument is a function call that returns an opaque type */
                            else if (arg->type == AST_CALL) {
                                Function *called_func = env_get_function(env, arg->as.call.name);
                                if (called_func && called_func->return_type == TYPE_STRUCT && called_func->return_struct_type_name) {
                                    OpaqueTypeDef *opaque = env_get_opaque_type(env, called_func->return_struct_type_name);
                                    if (opaque) {
                                        is_opaque_arg = true;  /* Allow opaque type to be passed as int */
                                    }
                                }
                            }
                        }
                        
                        if (!is_opaque_param)
                            check_opaque_value(env, func->params[i].type, func->params[i].struct_type_name, arg);
                        check_record_array_contract(env, func->params[i].type,
                            func->params[i].element_type, func->params[i].struct_type_name, arg);
                        if (!is_opaque_param && !is_opaque_arg && !types_match(arg_type, func->params[i].type)) {
                            char message[256];
                            snprintf(message, sizeof(message),
                                    "Argument %d expects %s, got %s.",
                                    i + 1,
                                    type_to_string(func->params[i].type),
                                    type_to_string(arg_type));
                            emit_context_error(
                                "E001 TYPE MISMATCH",
                                expr->line,
                                expr->column,
                                1,
                                message,
                                "Convert the argument to the expected type."
                            );
                        }
                    }
                }
            } else {
                /* For built-ins without param info, just check that arguments are valid expressions */
                for (int i = 0; i < expr->as.call.arg_count; i++) {
                    check_expression(expr->as.call.args[i], env);
                }
            }

            /* Special handling for array operations that need element type inference */
            if (strcmp(expr->as.call.name, "at") == 0 || strcmp(expr->as.call.name, "array_get") == 0) {
                if (!check_array_access_arguments(expr, env)) return TYPE_UNKNOWN;
                /* at(array, index) returns the element type of the array */
                if (expr->as.call.arg_count >= 1) {
                    ASTNode *array_arg = expr->as.call.args[0];
                    Type inferred_element = infer_array_element_type(array_arg, env);
                    if (inferred_element != TYPE_UNKNOWN) return inferred_element;
                    
                    /* Check if it's an array literal - get element type from it */
                    if (array_arg->type == AST_ARRAY_LITERAL && array_arg->as.array_literal.element_count > 0) {
                        Type elem_type = check_expression(array_arg->as.array_literal.elements[0], env);
                        return elem_type;
                    }
                    
                    /* Check if it's a variable - look up its element type */
                    if (array_arg->type == AST_IDENTIFIER) {
                        Symbol *sym = env_get_var_visible_at(env, array_arg->as.identifier, array_arg->line, array_arg->column);
                        if (sym && sym->type == TYPE_ARRAY) {
                            /* Get element type from Symbol if stored */
                            if (sym->element_type != TYPE_UNKNOWN) {
                                return sym->element_type;
                            }
                            /* Fallback: try to get from array value if available */
                            if (sym->value.type == VAL_ARRAY && sym->value.as.array_val) {
                                ValueType elem_val_type = sym->value.as.array_val->element_type;
                                /* Convert ValueType to Type */
                                switch (elem_val_type) {
                                    case VAL_INT: return TYPE_INT;
                                    case VAL_FLOAT: return TYPE_FLOAT;
                                    case VAL_BOOL: return TYPE_BOOL;
                                    case VAL_STRING: return TYPE_STRING;
                                    default: break;
                                }
                            }
                        }
                    }
                    
                    /* Check if it's a field access - look up element type from struct */
                    if (array_arg->type == AST_FIELD_ACCESS) {
                        const char *struct_name = get_struct_type_name(array_arg->as.field_access.object, env);
                        if (struct_name) {
                            StructDef *sdef = env_get_struct(env, struct_name);
                            if (sdef && sdef->field_element_types) {
                                const char *field_name = array_arg->as.field_access.field_name;
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

                    /* Calls that return byte arrays */
                    if (array_arg->type == AST_CALL && array_arg->as.call.name) {
                        if (strcmp(array_arg->as.call.name, "file_read_bytes") == 0 ||
                            strcmp(array_arg->as.call.name, "bytes_from_string") == 0) {
                            return TYPE_U8;
                        }

                        /* array_slice(arr, start, length) preserves element type */
                        if (strcmp(array_arg->as.call.name, "array_slice") == 0 && array_arg->as.call.arg_count >= 1) {
                            ASTNode *inner = array_arg->as.call.args[0];
                            if (inner) {
                                if (inner->type == AST_IDENTIFIER) {
                                    Symbol *sym = env_get_var_visible_at(env, inner->as.identifier, inner->line, inner->column);
                                    if (sym && sym->type == TYPE_ARRAY && sym->element_type != TYPE_UNKNOWN) {
                                        return sym->element_type;
                                    }
                                } else if (inner->type == AST_ARRAY_LITERAL) {
                                    if (inner->as.array_literal.element_type != TYPE_UNKNOWN) {
                                        return inner->as.array_literal.element_type;
                                    }
                                    if (inner->as.array_literal.element_count > 0) {
                                        return check_expression(inner->as.array_literal.elements[0], env);
                                    }
                                } else if (inner->type == AST_FIELD_ACCESS) {
                                    const char *struct_name = get_struct_type_name(inner->as.field_access.object, env);
                                    if (struct_name) {
                                        StructDef *sdef = env_get_struct(env, struct_name);
                                        if (sdef && sdef->field_element_types) {
                                            const char *field_name = inner->as.field_access.field_name;
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
                                } else if (inner->type == AST_CALL && inner->as.call.name &&
                                           (strcmp(inner->as.call.name, "file_read_bytes") == 0 ||
                                            strcmp(inner->as.call.name, "bytes_from_string") == 0)) {
                                    return TYPE_U8;
                                }
                            }
                        }
                    }
                    
                    /* Fallback: try to infer from the array_literal's stored element_type */
                    Type array_type = check_expression(array_arg, env);
                    if (array_type == TYPE_ARRAY && array_arg->type == AST_ARRAY_LITERAL) {
                        return array_arg->as.array_literal.element_type;
                    }
                }
            }

            /* Special handling for file_read_bytes - returns array<u8> */
            if (strcmp(expr->as.call.name, "file_read_bytes") == 0) {
                /* Check arguments */
                if (expr->as.call.arg_count >= 1) {
                    check_expression(expr->as.call.args[0], env);
                }
                return TYPE_ARRAY;  /* Returns array<u8> of bytes */
            }

            if (strcmp(expr->as.call.name, "bytes_from_string") == 0) {
                if (expr->as.call.arg_count >= 1) {
                    check_expression(expr->as.call.args[0], env);
                }
                return TYPE_ARRAY;
            }

            if (strcmp(expr->as.call.name, "string_from_bytes") == 0) {
                if (expr->as.call.arg_count >= 1) {
                    check_expression(expr->as.call.args[0], env);
                }
                return TYPE_STRING;
            }

            if (strcmp(expr->as.call.name, "array_slice") == 0) {
                if (expr->as.call.arg_count >= 1) {
                    check_expression(expr->as.call.args[0], env);
                }
                if (expr->as.call.arg_count >= 2) {
                    check_expression(expr->as.call.args[1], env);
                }
                if (expr->as.call.arg_count >= 3) {
                    check_expression(expr->as.call.args[2], env);
                }
                return TYPE_ARRAY;
            }
            
            /* Special handling for polymorphic built-in functions (abs, min, max) */
            /* These functions return the same type as their input arguments */
            if (strcmp(expr->as.call.name, "abs") == 0 ||
                strcmp(expr->as.call.name, "min") == 0 ||
                strcmp(expr->as.call.name, "max") == 0) {
                /* Check the type of the first argument */
                if (expr->as.call.arg_count >= 1) {
                    Type arg_type = check_expression(expr->as.call.args[0], env);
                    if (arg_type == TYPE_FLOAT) {
                        return TYPE_FLOAT;
                    }
                    /* For int or other types, return int */
                    return TYPE_INT;
                }
            }

            /* Check if function returns a function type */
            if (func->return_type == TYPE_FUNCTION) {
                return TYPE_FUNCTION;
            }
            
            return func->return_type;
        }

        case AST_MODULE_QUALIFIED_CALL: {
            /* Module-qualified function call: (Module.function args...) */
            const char *module_alias = expr->as.module_qualified_call.module_alias;
            const char *function_name = expr->as.module_qualified_call.function_name;
            
            /* For now, construct a qualified name "Module.function" and look it up */
            /* TODO: Use proper module namespace lookup */
            size_t qname_len = strlen(module_alias) + strlen(function_name) + 2;
            char *qualified_name = malloc(qname_len);
            if (!qualified_name) return TYPE_UNKNOWN;
            snprintf(qualified_name, qname_len, "%s.%s", module_alias, function_name);

            /* Look up function in environment */
            Function *func = env_get_function(env, qualified_name);
            if (!func) {
                char message[256];
                snprintf(message, sizeof(message),
                        "I cannot find a function named `%s`.",
                        qualified_name);
                emit_context_error(
                    "E027 UNDEFINED FUNCTION",
                    expr->line,
                    expr->column,
                    (int)safe_strlen(qualified_name),
                    message,
                    "Check the module alias and exported functions."
                );
                free(qualified_name);
                return TYPE_UNKNOWN;
            }
            
            /* Visibility check: private functions cannot be called from other modules */
            if (func->module_name && !func->is_pub) {
                bool same_module = env->current_module &&
                                   strcmp(func->module_name, env->current_module) == 0;
                if (!same_module) {
                    char priv_msg[512];
                    snprintf(priv_msg, sizeof(priv_msg),
                             "Function '%s' is private to module '%s'.",
                             func->name, func->module_name);
                    char priv_hint[512];
                    snprintf(priv_hint, sizeof(priv_hint),
                             "Use 'pub fn %s(...)' to make it accessible from other modules.",
                             func->name);
                    emit_context_error("E009 PRIVATE ACCESS", expr->line, expr->column,
                                       (int)safe_strlen(function_name), priv_msg, priv_hint);
                    free(qualified_name);
                    return TYPE_UNKNOWN;
                }
            }

            /* Phase 3: Warn on calls to functions from unsafe modules if --warn-unsafe-calls is set */
            if (env->warn_unsafe_calls && func->module_name) {
                /* Check if the function's module is unsafe */
                ModuleInfo *mod = env_get_module(env, func->module_name);
                if (mod && mod->is_unsafe) {
                    fprintf(stderr, "Warning at line %d, column %d: Calling function '%s.%s' from unsafe module '%s'\n",
                            expr->line, expr->column, module_alias, function_name, func->module_name);
                    fprintf(stderr, "  Note: Functions from unsafe modules may have safety implications\n");
                }
            }

            /* Phase 3: Warn on FFI calls if --warn-ffi is set */
            if (func->is_extern && env->warn_ffi) {
                fprintf(stderr, "Warning at line %d, column %d: FFI call to extern function '%s.%s'\n",
                        expr->line, expr->column, module_alias, function_name);
                fprintf(stderr, "  Note: Extern functions perform arbitrary operations\n");
            }

            /* I share the ordinary call checker, including complete callback
             * signatures. The argument ASTs remain owned by the qualified node. */
            ASTNode call = {0};
            call.type = AST_CALL;
            call.line = expr->line;
            call.column = expr->column;
            call.as.call.name = qualified_name;
            call.as.call.args = expr->as.module_qualified_call.args;
            call.as.call.arg_count = expr->as.module_qualified_call.arg_count;
            Type result = check_expression(&call, env);
            free(call.as.call.return_struct_type_name);
            free(call.as.call.concrete_func_name);
            free_function_signature(call.as.call.checked_signature);
            free(qualified_name);
            return result;
        }

        case AST_ARRAY_LITERAL: {
            /* Type check array literal */
            int element_count = expr->as.array_literal.element_count;
            
            /* Empty array - type will be inferred from context */
            if (element_count == 0) {
                return TYPE_ARRAY;
            }
            
            /* Check first element to determine array type */
            Type first_type = check_expression(expr->as.array_literal.elements[0], env);
            if (first_type == TYPE_UNKNOWN) {
                return TYPE_UNKNOWN;
            }
            
            /* Check all remaining elements match first element's type */
            for (int i = 1; i < element_count; i++) {
                Type elem_type = check_expression(expr->as.array_literal.elements[i], env);
                const char *first_record = first_type == TYPE_STRUCT
                    ? get_struct_type_name(expr->as.array_literal.elements[0], env) : NULL;
                const char *next_record = elem_type == TYPE_STRUCT
                    ? get_struct_type_name(expr->as.array_literal.elements[i], env) : NULL;
                if (first_type == TYPE_STRUCT && elem_type == TYPE_STRUCT &&
                    (!first_record || !next_record || strcmp(first_record, next_record))) {
                    emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                        "I require the same nominal record type for every array element.",
                        "Use one declared record type for this array.");
                    return TYPE_UNKNOWN;
                }
                if (elem_type != first_type) {
                    char message[256];
                    snprintf(message, sizeof(message),
                            "Array elements must all have the same type (expected %s, got %s).",
                            type_to_string(first_type), type_to_string(elem_type));
                    emit_context_error(
                        "E001 TYPE MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        message,
                        "Use consistent element types or convert values."
                    );
                    return TYPE_UNKNOWN;
                }
            }
            
            /* Store the element type in the AST for later use */
            expr->as.array_literal.element_type = first_type;
            
            return TYPE_ARRAY;
        }

        case AST_IF: {
            Type cond_type = check_expression(expr->as.if_stmt.condition, env);
            if (cond_type != TYPE_BOOL) {
                emit_context_error(
                    "E001 TYPE MISMATCH",
                    expr->line,
                    expr->column,
                    1,
                    "If condition must be a bool.",
                    "Ensure the condition expression evaluates to bool."
                );
            }

            /* For if expressions, we need to infer the type from the blocks */
            /* This is simplified - just return UNKNOWN for now */
            /* A proper implementation would need to analyze the blocks */
            return TYPE_UNKNOWN;
        }

        case AST_COND: {
            /* Type check all conditions (must be bool) */
            for (int i = 0; i < expr->as.cond_expr.clause_count; i++) {
                Type cond_type = check_expression(expr->as.cond_expr.conditions[i], env);
                if (cond_type != TYPE_BOOL) {
                    emit_context_error(
                        "E001 TYPE MISMATCH",
                        expr->line,
                        expr->column,
                        1,
                        "Cond clause condition must be a bool.",
                        "Ensure each cond clause condition is boolean."
                    );
                }
            }
            
            /* Type check all values and ensure they have the same type */
            Type result_type = TYPE_UNKNOWN;
            if (expr->as.cond_expr.clause_count > 0) {
                result_type = check_expression(expr->as.cond_expr.values[0], env);
                for (int i = 1; i < expr->as.cond_expr.clause_count; i++) {
                    Type val_type = check_expression(expr->as.cond_expr.values[i], env);
                    if (val_type != result_type && result_type != TYPE_UNKNOWN && val_type != TYPE_UNKNOWN) {
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                            "I require all cond clause values to have the same type.",
                            "Use the same type in every cond value arm.");
                    }
                }
            }
            
            /* Type check else value (must match clause values) */
            Type else_type = check_expression(expr->as.cond_expr.else_value, env);
            if (else_type != result_type && result_type != TYPE_UNKNOWN && else_type != TYPE_UNKNOWN) {
                emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                    "I require the cond else value to match the clause value type.",
                    "Use the same type in every cond value arm.");
            }
            
            return result_type != TYPE_UNKNOWN ? result_type : else_type;
        }

        case AST_STRUCT_LITERAL: {
            /* Check if struct name was inferred (should happen in let/return/call context) */
            if (expr->as.struct_literal.struct_name == NULL) {
                fprintf(stderr, "Error at line %d, column %d: Anonymous struct literal requires type context\n",
                        expr->line, expr->column);
                return TYPE_UNKNOWN;
            }
            
            /* Check if this is a union variant construction (format: "UnionName.VariantName") */
            const char *dot = strchr(expr->as.struct_literal.struct_name, '.');
            if (dot) {
                /* This may be a union variant construction OR a module-qualified struct literal.
                 * Only treat it as a union if the prefix is a known union type.
                 */
                int union_name_len = dot - expr->as.struct_literal.struct_name;
                char *union_name = malloc(union_name_len + 1);
                strncpy(union_name, expr->as.struct_literal.struct_name, union_name_len);
                union_name[union_name_len] = '\0';
                const char *variant_name = dot + 1;
                
                /* Look up union definition */
                UnionDef *udef = env_get_union(env, union_name);
                if (!udef) {
                    free(union_name);
                    goto not_a_union_variant;
                }
                
                /* Find the variant index */
                int variant_idx = env_get_union_variant_index(env, union_name, variant_name);
                if (variant_idx < 0) {
                    char hint[512];
                    int hint_off = snprintf(hint, sizeof(hint), "Available variants:");
                    for (int vi = 0; vi < udef->variant_count && hint_off < (int)sizeof(hint) - 3; vi++) {
                        hint_off += snprintf(hint + hint_off, sizeof(hint) - hint_off, " %s", udef->variant_names[vi]);
                    }
                    char message[256];
                    snprintf(message, sizeof(message), "Unknown variant '%s' in union '%s'.", variant_name, union_name);
                    emit_context_error("E005 UNKNOWN VARIANT", expr->line, expr->column, 1, message, hint);
                    free(union_name);
                    return TYPE_UNKNOWN;
                }

                /* Verify field count matches */
                if (expr->as.struct_literal.field_count != udef->variant_field_counts[variant_idx]) {
                    char message[256];
                    snprintf(message, sizeof(message), "Variant '%s.%s' expects %d field(s), got %d.",
                             union_name, variant_name,
                             udef->variant_field_counts[variant_idx], expr->as.struct_literal.field_count);
                    emit_context_error("E003 ARITY MISMATCH", expr->line, expr->column, 1, message,
                                       "Provide all required fields for the variant.");
                    free(union_name);
                    return TYPE_UNKNOWN;
                }
                
                /* I resolve each supplied name before selecting its declared type. */
                for (int i = 0; i < expr->as.struct_literal.field_count; i++) {
                    const char *field_name = expr->as.struct_literal.field_names[i];
                    int field_index = -1;
                    for (int j = 0; j < udef->variant_field_counts[variant_idx]; ++j)
                        if (!strcmp(field_name, udef->variant_field_names[variant_idx][j])) field_index = j;
                    bool duplicate = false;
                    for (int j = 0; j < i; ++j)
                        if (!strcmp(field_name, expr->as.struct_literal.field_names[j])) duplicate = true;
                    if (field_index < 0 || duplicate) {
                        emit_context_error("E004 UNKNOWN FIELD", expr->line, expr->column, 1,
                                           "I require every declared union field exactly once.",
                                           "Use distinct field names from this variant.");
                        free(union_name);
                        return TYPE_UNKNOWN;
                    }
                    Type field_type = check_expression(expr->as.struct_literal.field_values[i], env);
                    Type expected = udef->variant_field_types[variant_idx][field_index];
                    check_union_record_array_contract(env, udef, variant_idx,
                        expr->as.struct_literal.field_names[i], expr->as.struct_literal.field_values[i]);

                    /* If the expected field type refers to a generic parameter name (T, E, etc.),
                     * treat it as a wildcard here. This struct-literal path does not carry the
                     * concrete instantiation needed for substitution.
                     */
                    if (udef->generic_param_count > 0 &&
                        udef->variant_field_type_names &&
                        udef->variant_field_type_names[variant_idx] &&
                        udef->variant_field_type_names[variant_idx][field_index]) {
                        const char *expected_name = udef->variant_field_type_names[variant_idx][field_index];
                        for (int gp = 0; gp < udef->generic_param_count; gp++) {
                            if (udef->generic_params && udef->generic_params[gp] &&
                                strcmp(expected_name, udef->generic_params[gp]) == 0) {
                                goto next_union_field;
                            }
                        }
                    }

                    /* Generic unions (e.g., Result<T, E>) store TYPE_GENERIC for variant fields.
                     * If we don't have concrete substitution info at this node, treat TYPE_GENERIC
                     * as a wildcard to avoid spurious type mismatch errors.
                     */
                    if (expected == TYPE_GENERIC) {
                        goto next_union_field;
                    }

                    if (!types_match(field_type, expected)) {
                        char message[256];
                        snprintf(message, sizeof(message),
                                 "Field type mismatch in variant '%s.%s': got %s, expected %s.",
                                 union_name, variant_name,
                                 type_to_string(field_type), type_to_string(expected));
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1, message,
                                           "Ensure each field value matches the variant's declared type.");
                    }

                next_union_field:
                    ;
                }
                
                free(union_name);
                return TYPE_UNION;
            }

        not_a_union_variant:
            ;
            
            /* Check that struct is defined */
            StructDef *sdef = env_get_struct(env, expr->as.struct_literal.struct_name);
            if (!sdef) {
                char message[256];
                snprintf(message, sizeof(message), "Undefined struct '%s'.",
                         expr->as.struct_literal.struct_name);
                emit_context_error("E006 UNDEFINED STRUCT", expr->line, expr->column,
                                   (int)safe_strlen(expr->as.struct_literal.struct_name),
                                   message,
                                   "Define 'struct Name { ... }' before using it in a literal.");
                return TYPE_UNKNOWN;
            }

            /* Normalize qualified name (Math.Point) to actual struct name (Point) */
            if (strcmp(expr->as.struct_literal.struct_name, sdef->name) != 0) {
                free(expr->as.struct_literal.struct_name);
                expr->as.struct_literal.struct_name = strdup(sdef->name);
            }

            /* Check that all fields are provided and types match.
             * Spread literals ({..base, overrides}) supply missing fields from
             * the spread source, so the explicit field_count may be less than
             * sdef->field_count; skip the arity check in that case. */
            if (!expr->as.struct_literal.spread_source &&
                expr->as.struct_literal.field_count != sdef->field_count) {
                char hint[512];
                int hint_off = snprintf(hint, sizeof(hint), "Expected fields:");
                for (int fi = 0; fi < sdef->field_count && hint_off < (int)sizeof(hint) - 3; fi++) {
                    hint_off += snprintf(hint + hint_off, sizeof(hint) - hint_off, " %s", sdef->field_names[fi]);
                }
                char message[256];
                snprintf(message, sizeof(message), "Struct '%s' expects %d field(s), got %d.",
                         expr->as.struct_literal.struct_name, sdef->field_count, expr->as.struct_literal.field_count);
                emit_context_error("E003 ARITY MISMATCH", expr->line, expr->column, 1, message, hint);
                return TYPE_UNKNOWN;
            }
            
            /* Check each field */
            for (int i = 0; i < expr->as.struct_literal.field_count; i++) {
                const char *field_name = expr->as.struct_literal.field_names[i];
                
                /* Find matching field in struct definition */
                int field_index = -1;
                for (int j = 0; j < sdef->field_count; j++) {
                    if (strcmp(field_name, sdef->field_names[j]) == 0) {
                        field_index = j;
                        break;
                    }
                }
                
                if (field_index == -1) {
                    char hint[512];
                    int hint_off = snprintf(hint, sizeof(hint), "Available fields:");
                    for (int j = 0; j < sdef->field_count && hint_off < (int)sizeof(hint) - 3; j++) {
                        hint_off += snprintf(hint + hint_off, sizeof(hint) - hint_off, " %s", sdef->field_names[j]);
                    }
                    char message[256];
                    snprintf(message, sizeof(message), "Unknown field '%s' in struct '%s'.",
                             field_name, expr->as.struct_literal.struct_name);
                    emit_context_error("E004 UNKNOWN FIELD", expr->line, expr->column,
                                       (int)safe_strlen(field_name), message, hint);
                    continue;
                }

                /* I apply the complete field annotation before checking its constructor. */
                if (sdef->field_type_info)
                    check_concrete_union_arrays(env, sdef->field_type_info[field_index],
                        expr->as.struct_literal.field_values[i], 0);
                /* Check field type */
                Type field_type = check_expression(expr->as.struct_literal.field_values[i], env);
                ASTNode *field_value = expr->as.struct_literal.field_values[i];
                check_opaque_value(env, sdef->field_types[field_index],
                    sdef->field_type_names ? sdef->field_type_names[field_index] : NULL, field_value);
                check_record_array_contract(env, sdef->field_types[field_index],
                    sdef->field_element_types ? sdef->field_element_types[field_index] : TYPE_UNKNOWN,
                    sdef->field_type_names ? sdef->field_type_names[field_index] : NULL, field_value);
                if (sdef->field_types[field_index] == TYPE_ARRAY &&
                    sdef->field_element_types &&
                    field_value->type == AST_ARRAY_LITERAL &&
                    field_value->as.array_literal.element_count == 0) {
                    /* An empty field has no element from which to infer its
                     * runtime representation. Preserve its declaration. */
                    field_value->as.array_literal.element_type = resolved_array_element(
                        sdef->field_element_types[field_index],
                        sdef->field_type_names ? sdef->field_type_names[field_index] : NULL, env);
                }
                if (!types_match(field_type, sdef->field_types[field_index])) {
                    char message[256];
                    snprintf(message, sizeof(message),
                             "Field '%s' type mismatch in struct '%s': got %s, expected %s.",
                             field_name, expr->as.struct_literal.struct_name,
                             type_to_string(field_type), type_to_string(sdef->field_types[field_index]));
                    emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column,
                                       (int)safe_strlen(field_name), message,
                                       "Ensure the field value matches the declared field type.");
                }
            }
            
            return TYPE_STRUCT;
        }

        case AST_FIELD_ACCESS: {
            /* Check object is not NULL */
            assert(expr->as.field_access.object != NULL);
            if (!expr->as.field_access.object) {
                safe_fprintf(stderr, "Error at line %d, column %d: NULL object in field access\n",
                        expr->line, expr->column);
                return TYPE_UNKNOWN;
            }
            
            /* Special case: Check if this is an enum variant access */
            if (expr->as.field_access.object->type == AST_IDENTIFIER) {
                const char *enum_name = expr->as.field_access.object->as.identifier;
                assert(enum_name != NULL);
                if (!enum_name) {
                    safe_fprintf(stderr, "Error at line %d, column %d: NULL enum name in field access\n",
                            expr->line, expr->column);
                    return TYPE_UNKNOWN;
                }
                EnumDef *enum_def = env_get_enum(env, enum_name);
                
                if (enum_def && enum_def->variant_names) {
                    /* This is an enum variant access (e.g., Color.Red) */
                    const char *variant_name = expr->as.field_access.field_name;
                    assert(variant_name != NULL);
                    
                    if (!variant_name) {
                        safe_fprintf(stderr, "Error at line %d, column %d: NULL variant name in enum access\n",
                                expr->line, expr->column);
                        return TYPE_UNKNOWN;
                    }
                    
                    /* Verify variant exists */
                    for (int i = 0; i < enum_def->variant_count; i++) {
                        if (safe_strcmp(enum_def->variant_names[i], variant_name) == 0) {
                            return TYPE_ENUM;  /* Return TYPE_ENUM for proper type checking */
                        }
                    }
                    
                    safe_fprintf(stderr, "Error at line %d, column %d: Enum '%s' has no variant '%s'\n",
                            expr->line, expr->column, safe_format_string(enum_name), safe_format_string(variant_name));
                    return TYPE_UNKNOWN;
                }
            }
            
            /* Regular struct field access */
            /* Check the object type */
            Type object_type = check_expression(expr->as.field_access.object, env);
            /* Open-record (row-poly) parameters are compatible with struct field access */
            if (object_type == TYPE_OPEN_RECORD) return TYPE_UNKNOWN;
            if (object_type != TYPE_STRUCT) {
                emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 1,
                                   "Field access requires a struct value.",
                                   "Ensure the object before '.' is a struct type.");
                return TYPE_UNKNOWN;
            }
            
            /* Get the specific struct type name */
            const char *struct_name = get_struct_type_name(expr->as.field_access.object, env);
            if (!struct_name) {
                fprintf(stderr, "Error at line %d, column %d: Cannot determine struct type for field access\n",
                        expr->line, expr->column);
                return TYPE_UNKNOWN;
            }
            
            /* Check if this is a union variant (format: "UnionName.VariantName") */
            const char *dot = strchr(struct_name, '.');
            if (dot) {
                /* This is a union variant field access */
                int union_name_len = dot - struct_name;
                char *union_name = malloc(union_name_len + 1);
                strncpy(union_name, struct_name, union_name_len);
                union_name[union_name_len] = '\0';
                const char *variant_name = dot + 1;
                
                /* Look up the union definition */
                UnionDef *udef = env_get_union(env, union_name);
                if (!udef) {
                    /* Likely a module-qualified struct name (e.g., Math.Point). */
                    free(union_name);
                    goto not_a_union_variant_field_access;
                }
                
                /* Find the variant */
                int variant_idx = env_get_union_variant_index(env, union_name, variant_name);
                if (variant_idx < 0) {
                    char hint[512];
                    int hint_off = snprintf(hint, sizeof(hint), "Available variants:");
                    for (int vi = 0; vi < udef->variant_count && hint_off < (int)sizeof(hint) - 3; vi++) {
                        hint_off += snprintf(hint + hint_off, sizeof(hint) - hint_off, " %s", udef->variant_names[vi]);
                    }
                    char message[256];
                    snprintf(message, sizeof(message), "Unknown variant '%s' in union '%s'.", variant_name, union_name);
                    emit_context_error("E005 UNKNOWN VARIANT", expr->line, expr->column, 1, message, hint);
                    free(union_name);
                    return TYPE_UNKNOWN;
                }

                /* Find the field in the variant */
                const char *field_name = expr->as.field_access.field_name;
                for (int i = 0; i < udef->variant_field_counts[variant_idx]; i++) {
                    if (strcmp(udef->variant_field_names[variant_idx][i], field_name) == 0) {
                        Type field_type = udef->variant_field_types[variant_idx][i];
                        TypeInfo *arguments = try_get_expr_type_info(expr->as.field_access.object, env);
                        TypeInfo *payload = resolve_union_payload_type_info(udef, variant_idx, i, arguments);
                        if (payload) {
                            free_payload_type_info(expr->as.field_access.resolved_type_info);
                            expr->as.field_access.resolved_type_info = payload;
                            free(union_name);
                            return payload->base_type;
                        }

                        /* For generic unions, resolve the concrete type using TypeInfo */
                        if (udef->generic_param_count > 0 && expr->as.field_access.object->type == AST_IDENTIFIER) {
                            Symbol *obj_sym = env_get_var_visible_at(env, expr->as.field_access.object->as.identifier,
                                                                      expr->as.field_access.object->line,
                                                                      expr->as.field_access.object->column);
                            if (obj_sym && obj_sym->type_info && obj_sym->type_info->type_param_count > 0) {
                                /* Check if the field type is a generic parameter (e.g., "T", "E") */
                                const char *field_type_name = udef->variant_field_type_names[variant_idx][i];
                                if (field_type_name) {
                                    /* Try to match it to a generic parameter name */
                                    for (int g = 0; g < udef->generic_param_count; g++) {
                                        if (strcmp(udef->generic_params[g], field_type_name) == 0) {
                                            /* Found a match - use the concrete type from TypeInfo */
                                            if (g < obj_sym->type_info->type_param_count) {
                                                TypeInfo *concrete_type_info = obj_sym->type_info->type_params[g];
                                                if (concrete_type_info) {
                                                    field_type = concrete_type_info->base_type;
                                                }
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        
                        free(union_name);
                        return field_type;
                    }
                }
                
                /* Field not found */
                {
                    char hint[512];
                    int hint_off = snprintf(hint, sizeof(hint), "Available fields:");
                    for (int vi = 0; vi < udef->variant_field_counts[variant_idx] && hint_off < (int)sizeof(hint) - 3; vi++) {
                        hint_off += snprintf(hint + hint_off, sizeof(hint) - hint_off,
                                             " %s", udef->variant_field_names[variant_idx][vi]);
                    }
                    if (udef->variant_field_counts[variant_idx] == 0) {
                        snprintf(hint, sizeof(hint), "Variant '%s' has no fields.", variant_name);
                    }
                    char message[256];
                    snprintf(message, sizeof(message), "Variant '%s' of union '%s' has no field '%s'.",
                             variant_name, union_name, field_name);
                    emit_context_error("E004 UNKNOWN FIELD", expr->line, expr->column,
                                       (int)safe_strlen(field_name), message, hint);
                }
                free(union_name);
                return TYPE_UNKNOWN;
            }

        not_a_union_variant_field_access:
            ;

            /* Look up the struct definition */
            StructDef *sdef = env_get_struct(env, struct_name);
            if (!sdef) {
                char message[256];
                snprintf(message, sizeof(message), "Undefined struct '%s'.", struct_name);
                emit_context_error("E006 UNDEFINED STRUCT", expr->line, expr->column,
                                   (int)safe_strlen(struct_name), message,
                                   "Define 'struct Name { ... }' before accessing its fields.");
                return TYPE_UNKNOWN;
            }

            /* Find the field and return its type */
            const char *field_name = expr->as.field_access.field_name;
            for (int i = 0; i < sdef->field_count; i++) {
                if (strcmp(sdef->field_names[i], field_name) == 0) {
                    if (sdef->field_type_info && sdef->field_type_info[i]) {
                        free_payload_type_info(expr->as.field_access.resolved_type_info);
                        TypeInfo *info = copy_payload_type_info(sdef->field_type_info[i]);
                        expr->as.field_access.resolved_type_info = info;
                        if (info && info->generic_name && env_get_union(env, info->generic_name)) {
                            info->base_type = TYPE_UNION;
                            return TYPE_UNION;
                        }
                    }
                    return sdef->field_types[i];
                }
            }

            /* Field not found */
            {
                char hint[512];
                int hint_off = snprintf(hint, sizeof(hint), "Available fields:");
                for (int i = 0; i < sdef->field_count && hint_off < (int)sizeof(hint) - 3; i++) {
                    hint_off += snprintf(hint + hint_off, sizeof(hint) - hint_off, " %s", sdef->field_names[i]);
                }
                char message[256];
                snprintf(message, sizeof(message), "Struct '%s' has no field '%s'.", struct_name, field_name);
                emit_context_error("E004 UNKNOWN FIELD", expr->line, expr->column,
                                   (int)safe_strlen(field_name), message, hint);
            }
            return TYPE_UNKNOWN;
        }

        case AST_UNION_CONSTRUCT: {
            /* Check that union is defined */
            UnionDef *udef = env_get_union(env, expr->as.union_construct.union_name);
            if (!udef) {
                char message[256];
                snprintf(message, sizeof(message), "Undefined union '%s'.",
                         expr->as.union_construct.union_name);
                emit_context_error("E007 UNDEFINED UNION", expr->line, expr->column,
                                   (int)safe_strlen(expr->as.union_construct.union_name),
                                   message, "Define 'union Name { ... }' before constructing it.");
                return TYPE_UNKNOWN;
            }

            /* Check that variant exists */
            int variant_idx = env_get_union_variant_index(env,
                expr->as.union_construct.union_name,
                expr->as.union_construct.variant_name);
            if (variant_idx < 0) {
                char hint[512];
                int hint_off = snprintf(hint, sizeof(hint), "Available variants:");
                for (int vi = 0; vi < udef->variant_count && hint_off < (int)sizeof(hint) - 3; vi++) {
                    hint_off += snprintf(hint + hint_off, sizeof(hint) - hint_off, " %s", udef->variant_names[vi]);
                }
                char message[256];
                snprintf(message, sizeof(message), "Unknown variant '%s' in union '%s'.",
                         expr->as.union_construct.variant_name, expr->as.union_construct.union_name);
                emit_context_error("E005 UNKNOWN VARIANT", expr->line, expr->column, 1, message, hint);
                return TYPE_UNKNOWN;
            }

            /* Check that field count matches */
            int expected_field_count = udef->variant_field_counts[variant_idx];
            if (expr->as.union_construct.field_count != expected_field_count) {
                char message[256];
                snprintf(message, sizeof(message), "Variant '%s' expects %d field(s), got %d.",
                         expr->as.union_construct.variant_name,
                         expected_field_count, expr->as.union_construct.field_count);
                emit_context_error("E003 ARITY MISMATCH", expr->line, expr->column, 1, message,
                                   "Provide all required fields for the variant.");
                return TYPE_UNKNOWN;
            }

            /* Check each field type */
            for (int i = 0; i < expr->as.union_construct.field_count; i++) {
                const char *field_name = expr->as.union_construct.field_names[i];

                /* Find matching field in variant definition */
                int field_index = -1;
                for (int j = 0; j < expected_field_count; j++) {
                    if (strcmp(udef->variant_field_names[variant_idx][j], field_name) == 0) {
                        field_index = j;
                        break;
                    }
                }

                if (field_index < 0) {
                    char hint[512];
                    int hint_off = snprintf(hint, sizeof(hint), "Available fields:");
                    for (int j = 0; j < expected_field_count && hint_off < (int)sizeof(hint) - 3; j++) {
                        hint_off += snprintf(hint + hint_off, sizeof(hint) - hint_off,
                                             " %s", udef->variant_field_names[variant_idx][j]);
                    }
                    char message[256];
                    snprintf(message, sizeof(message), "Unknown field '%s' in variant '%s'.",
                             field_name, expr->as.union_construct.variant_name);
                    emit_context_error("E004 UNKNOWN FIELD", expr->line, expr->column,
                                       (int)safe_strlen(field_name), message, hint);
                    return TYPE_UNKNOWN;
                }

                /* Check field type */
                Type expected_type = udef->variant_field_types[variant_idx][field_index];
                Type actual_type = check_expression(expr->as.union_construct.field_values[i], env);
                check_union_record_array_contract(env, udef, variant_idx, field_name,
                    expr->as.union_construct.field_values[i]);

                /* For generic unions, accept any type for generic type parameters */
                /* TODO: Proper type substitution for generic instantiations */
                bool is_generic_param = (expected_type == TYPE_GENERIC || expected_type == TYPE_STRUCT);
                if (is_generic_param && udef->generic_param_count > 0) {
                    /* This is likely a generic type parameter - accept it for now */
                    /* The transpiler will handle concrete type generation */
                } else if (actual_type != expected_type) {
                    char message[256];
                    snprintf(message, sizeof(message), "Field '%s' expects type '%s', got '%s'.",
                             field_name, type_to_string(expected_type), type_to_string(actual_type));
                    emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column,
                                       (int)safe_strlen(field_name), message,
                                       "Ensure the field value matches the variant's declared field type.");
                    return TYPE_UNKNOWN;
                }
            }

            return TYPE_UNION;
        }

        case AST_MATCH: {
            /* Code generation may ask again without a function-checking context. */
            if (!active_statement_checker && expr->as.match_expr.result_type_checked)
                return expr->as.match_expr.result_type;
            expr->as.match_expr.checked_scrutinee_type = TYPE_UNKNOWN;
            expr->as.match_expr.scrutinee_type_checked = false;
            /* Check the expression being matched */
            Type match_type = check_expression(expr->as.match_expr.expr, env);
            bool has_int_patterns_expr;
            bool has_variant_patterns_expr;
            match_arm_families(expr, &has_int_patterns_expr, &has_variant_patterns_expr);
            
            /* Infer and store union type name for transpiler */
            const char *union_type_name = NULL;      /* base name for variant-field lookup: Result */
            const char *union_base_name = NULL;      /* kept for binding metadata */
            char *union_concrete_name = NULL;        /* for transpiler: Result_int_string */
            TypeInfo *union_type_info = NULL;        /* For generic unions: Result<int, string> */
            ASTNode *match_expr_node = expr->as.match_expr.expr;
            
            if (match_expr_node->type == AST_IDENTIFIER) {
                Symbol *sym = env_get_var_visible_at(env, match_expr_node->as.identifier, match_expr_node->line, match_expr_node->column);
                if (sym && sym->struct_type_name) {
                    union_type_name = sym->struct_type_name;
                }
                /* For generic unions, also extract TypeInfo */
                if (sym && sym->type_info) {
                    union_type_info = sym->type_info;
                    /* Prefer the generic name as the base for variant lookup */
                    if (union_type_info->generic_name) union_type_name = union_type_info->generic_name;
                }
            } else if (match_expr_node->type == AST_UNION_CONSTRUCT) {
                union_type_name = match_expr_node->as.union_construct.union_name;
            } else if (match_expr_node->type == AST_STRUCT_LITERAL) {
                union_type_name = inline_variant_union(match_expr_node, env);
            } else if (match_expr_node->type == AST_CALL) {
                Function *func = env_get_function(env, match_expr_node->as.call.name);
                if (func && func->return_struct_type_name) {
                    union_type_name = func->return_struct_type_name;
                }
            } else if (match_expr_node->type == AST_FIELD_ACCESS) {
                /* Handle field access expressions like resp.status */
                const char *struct_name = get_struct_type_name(match_expr_node->as.field_access.object, env);
                if (struct_name) {
                    /* Look up the struct definition to find the field's type name */
                    StructDef *sdef = env_get_struct(env, struct_name);
                    if (sdef && sdef->field_type_names) {
                        const char *field_name = match_expr_node->as.field_access.field_name;
                        for (int i = 0; i < sdef->field_count; i++) {
                            if (strcmp(sdef->field_names[i], field_name) == 0) {
                                if (sdef->field_types[i] == TYPE_UNION && sdef->field_type_names[i]) {
                                    union_type_name = sdef->field_type_names[i];
                                }
                                break;
                            }
                        }
                    }
                }
            }

            if (match_expr_node->type == AST_FIELD_ACCESS || match_expr_node->type == AST_CALL) {
                TypeInfo *field_info = try_get_expr_type_info(match_expr_node, env);
                if (field_info && field_info->generic_name && env_get_union(env, field_info->generic_name)) {
                    union_type_info = field_info;
                    union_type_name = field_info->generic_name;
                }
            }

            union_base_name = union_type_name;
            if (union_type_info && union_type_info->generic_name && union_type_info->type_param_count > 0) {
                union_base_name = union_type_info->generic_name;
                union_concrete_name = typeinfo_to_monomorphized_generic_name(union_type_info);
            }

            MatchDomain match_domain = check_match_domain(
                expr, env, match_type, has_int_patterns_expr,
                has_variant_patterns_expr, union_base_name);
            if (match_domain == MATCH_DOMAIN_INVALID) {
                free(union_concrete_name);
                return TYPE_UNKNOWN;
            }
            expr->as.match_expr.checked_scrutinee_type = match_type;
            expr->as.match_expr.scrutinee_type_checked = true;
            
            if (expr->as.match_expr.union_type_name) {
                free(expr->as.match_expr.union_type_name);
                expr->as.match_expr.union_type_name = NULL;
            }
            if (union_concrete_name) {
                expr->as.match_expr.union_type_name = union_concrete_name;
            } else if (union_base_name) {
                expr->as.match_expr.union_type_name = strdup(union_base_name);
            }
            
            /* Check each arm and infer return type from first arm */
            Type return_type = TYPE_UNKNOWN;
            for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                int arm_first_symbol = env->symbol_count;
                /* Save symbol count for scope */
                int saved_symbol_count __attribute__((unused)) = env->symbol_count;
                const char *variant_name_i = expr->as.match_expr.pattern_variants[i];

                /* Wildcard arm: _ => { body }  — no binding to add; also skip or-patterns */
                if (expr->as.match_expr.pattern_bindings[i][0] &&
                    strcmp(expr->as.match_expr.pattern_bindings[i], "_") != 0 &&
                    strcmp(variant_name_i, "_") != 0 &&
                    strncmp(variant_name_i, "INT:", 4) != 0 &&
                    strncmp(variant_name_i, "OR:", 3) != 0) {
                    /* Add pattern binding to environment - bind as STRUCT type with "UnionName.VariantName"
                     * This allows us to distinguish union variant fields from regular struct fields
                     */
                    Value binding_val = create_void();
                    env_define_var_with_type_info(env,
                        expr->as.match_expr.pattern_bindings[i],
                        TYPE_STRUCT, TYPE_UNKNOWN, union_type_info, false, binding_val);

                    /* Store "UnionName.VariantName" as the struct type name for the binding
                     * This will be used by field access type checking to look up variant fields
                     */
                    if (union_base_name && env->symbol_count > 0) {
                        Symbol *binding_sym = &env->symbols[env->symbol_count - 1];
                        /* Format: "UnionName.VariantName" */
                        char *type_name = malloc(strlen(union_base_name) + strlen(variant_name_i) + 2);
                        sprintf(type_name, "%s.%s", union_base_name, variant_name_i);
                        binding_sym->struct_type_name = type_name;

                        /* Ensure bindings participate in visibility disambiguation */
                        binding_sym->def_line = expr->line;
                        binding_sym->def_column = expr->column;
                    }
                }

                /* Unknown is not permission to emit a guard. */
                if (expr->as.match_expr.guard_exprs)
                    check_match_guard(expr->as.match_expr.guard_exprs[i], env);

                /* Type check arm body (which is now an expression) */
                Type arm_type = check_expression(expr->as.match_expr.arm_bodies[i], env);
                
                /* I retain emission metadata within its lexical arm only. */
                bound_scope_symbols(env, arm_first_symbol, expr->as.match_expr.arm_bodies[i]);
                
                /* A definite function exit contributes no match value. */
                if (ast_always_returns(expr->as.match_expr.arm_bodies[i])) continue;
                if (return_type == TYPE_UNKNOWN) {
                    return_type = arm_type;
                } else if (arm_type != return_type) {
                    fprintf(stderr, "Error at line %d, column %d: Match arms must all return the same type\n",
                            expr->line, expr->column);
                    if (active_statement_checker) active_statement_checker->has_error = true;
                }
            }

            check_match_totality(expr, env, union_base_name, match_domain);

            expr->as.match_expr.result_type = return_type;
            expr->as.match_expr.result_type_checked = true;
            return return_type;
        }

        case AST_BLOCK: {
            /* Blocks can be used as expressions in match arms
             * I check statements in function context; only the final expression yields a value.
             */
            Type block_type = TYPE_VOID;
            
            /* I inherit return/unsafe context, rather than inventing a function. */
            TypeChecker temp_tc = active_statement_checker ? *active_statement_checker : (TypeChecker){0};
            temp_tc.env = env;
            temp_tc.has_error = false;
            
            for (int i = 0; i < expr->as.block.count; i++) {
                ASTNode *stmt = expr->as.block.statements[i];
                if (i == expr->as.block.count - 1 && ast_is_value_expression(stmt->type)) {
                    block_type = check_expression(stmt, env);
                } else {
                    check_statement(&temp_tc, stmt);
                }
            }
            if (temp_tc.has_error) {
                if (active_statement_checker) active_statement_checker->has_error = true;
                return TYPE_UNKNOWN;
            }
            return block_type;
        }

        case AST_RETURN: {
            /* Return statements can appear in blocks that are used as expressions */
            if (expr->as.return_stmt.value) {
                return check_expression(expr->as.return_stmt.value, env);
            }
            return TYPE_VOID;
        }

        case AST_TUPLE_LITERAL: {
            /* Type check tuple literal: (expr1, expr2, expr3) */
            int element_count = expr->as.tuple_literal.element_count;
            
            /* Empty tuple is valid */
            if (element_count == 0) {
                expr->as.tuple_literal.element_types = NULL;
                return TYPE_TUPLE;
            }
            
            /* Allocate space for element types */
            expr->as.tuple_literal.element_types = malloc(sizeof(Type) * element_count);
            
            /* Type check each element */
            for (int i = 0; i < element_count; i++) {
                Type elem_type = check_expression(expr->as.tuple_literal.elements[i], env);
                if (elem_type == TYPE_UNKNOWN) {
                    fprintf(stderr, "Error at line %d, column %d: Tuple element %d has unknown type\n",
                            expr->line, expr->column, i);
                    return TYPE_UNKNOWN;
                }
                expr->as.tuple_literal.element_types[i] = elem_type;
            }
            
            return TYPE_TUPLE;
        }

        case AST_TUPLE_INDEX: {
            /* Type check tuple index access: tuple.0, tuple.1 */
            Type tuple_type = check_expression(expr->as.tuple_index.tuple, env);
            
            if (tuple_type != TYPE_TUPLE) {
                fprintf(stderr, "Error at line %d, column %d: Tuple index access on non-tuple type\n",
                        expr->line, expr->column);
                return TYPE_UNKNOWN;
            }
            
            /* Get the tuple expression to check bounds */
            ASTNode *tuple_expr = expr->as.tuple_index.tuple;
            int index = expr->as.tuple_index.index;
            
            /* If the tuple is a literal, we can check bounds and get exact type */
            if (tuple_expr->type == AST_TUPLE_LITERAL) {
                int element_count = tuple_expr->as.tuple_literal.element_count;
                if (index < 0 || index >= element_count) {
                    fprintf(stderr, "Error at line %d, column %d: Tuple index %d out of bounds (tuple has %d elements)\n",
                            expr->line, expr->column, index, element_count);
                    return TYPE_UNKNOWN;
                }
                
                /* Return the type of the indexed element */
                if (tuple_expr->as.tuple_literal.element_types) {
                    return tuple_expr->as.tuple_literal.element_types[index];
                }
            }
            /* If the tuple is a variable, look up TypeInfo */
            else if (tuple_expr->type == AST_IDENTIFIER) {
                Symbol *sym = env_get_var_visible_at(env, tuple_expr->as.identifier, tuple_expr->line, tuple_expr->column);
                if (sym && sym->type == TYPE_TUPLE && sym->type_info) {
                    TypeInfo *type_info = sym->type_info;
                    
                    /* Check bounds */
                    if (index < 0 || index >= type_info->tuple_element_count) {
                        fprintf(stderr, "Error at line %d, column %d: Tuple index %d out of bounds (tuple has %d elements)\n",
                                expr->line, expr->column, index, type_info->tuple_element_count);
                        return TYPE_UNKNOWN;
                    }
                    
                    /* Return the type of the indexed element */
                    return type_info->tuple_types[index];
                }
            }
            
            TypeInfo *complete = try_get_expr_type_info(tuple_expr, env);
            if (complete && complete->base_type == TYPE_TUPLE) {
                TypeInfo flat;
                const TypeInfo *child = type_info_tuple_element(complete, index, &flat);
                if (child) return child->base_type;
                fprintf(stderr, "I require an in-range complete tuple index annotation\n");
                return TYPE_UNKNOWN;
            }
            /* For function returns or other complex expressions, we can't statically determine the type.
             * Return TYPE_INT as a conservative estimate.
             * TODO: Store TypeInfo in function return types for complete type checking.
             */
            return TYPE_INT;
        }

        case AST_TRY_OP: {
            /* Postfix ? try operator: expr?
             * The operand must be a union type with Ok and Err variants.
             * Returns the type of the first field of the Ok variant. */
            Type inner_type = check_expression(expr->as.try_op.operand, env);
            if (inner_type != TYPE_UNION) {
                fprintf(stderr, "Error at line %d, column %d: '?' operator requires a union type (got %d)\n",
                        expr->line, expr->column, inner_type);
                return TYPE_UNKNOWN;
            }

            /* Resolve the union type name from the inner expression */
            const char *union_name = NULL;
            ASTNode *inner = expr->as.try_op.operand;
            if (inner->type == AST_IDENTIFIER) {
                Symbol *sym = env_get_var_visible_at(env, inner->as.identifier, inner->line, inner->column);
                if (sym && sym->struct_type_name) union_name = sym->struct_type_name;
            } else if (inner->type == AST_CALL) {
                Function *func = env_get_function(env, inner->as.call.name);
                if (func && func->return_struct_type_name) union_name = func->return_struct_type_name;
            } else if (inner->type == AST_UNION_CONSTRUCT) {
                union_name = inner->as.union_construct.union_name;
            }

            if (!union_name) {
                fprintf(stderr, "Error at line %d, column %d: '?' operator: cannot determine union type name\n",
                        expr->line, expr->column);
                return TYPE_UNKNOWN;
            }

            /* Look up union def to find Ok variant and its first field */
            UnionDef *udef = env_get_union(env, union_name);
            if (!udef) {
                fprintf(stderr, "Error at line %d, column %d: '?' operator: union '%s' not found\n",
                        expr->line, expr->column, union_name);
                return TYPE_UNKNOWN;
            }

            /* Find Ok variant */
            int ok_idx = -1;
            for (int i = 0; i < udef->variant_count; i++) {
                if (strcmp(udef->variant_names[i], "Ok") == 0) {
                    ok_idx = i;
                    break;
                }
            }
            if (ok_idx < 0 || udef->variant_field_counts[ok_idx] == 0) {
                fprintf(stderr, "Error at line %d, column %d: '?' operator: union '%s' has no Ok variant with fields\n",
                        expr->line, expr->column, union_name);
                return TYPE_UNKNOWN;
            }

            /* Store metadata in node for the transpiler */
            if (expr->as.try_op.union_type_name) free(expr->as.try_op.union_type_name);
            if (expr->as.try_op.ok_field_name) free(expr->as.try_op.ok_field_name);
            expr->as.try_op.union_type_name = strdup(union_name);
            expr->as.try_op.ok_field_name = strdup(udef->variant_field_names[ok_idx][0]);

            /* Return type is the Ok variant's first field type */
            return udef->variant_field_types[ok_idx][0];
        }

        case AST_HANDLE_EXPR: {
            /* handle { body } with { op args -> handler_body ... }
             *
             * Typecheck the body first to determine what effects it may perform.
             * Then verify each handler matches an operation in a known effect.
             * The overall type is the type of the body expression.
             *
             * Effect inference: we scan handler op names against registered effects
             * to identify which effect this handle block handles.
             */
            if (expr->as.handle_expr.handler_count == 0) {
                emit_context_error("E028 EMPTY HANDLER", expr->line, expr->column, 6,
                    "Handler block has no operation handlers",
                    "E014: handle...with requires at least one operation handler");
                g_typecheck_error_count++;
                return TYPE_UNKNOWN;
            }

            /* Identify which effect is being handled by matching op names */
            EffectDef *matched_effect = NULL;
            for (int i = 0; i < expr->as.handle_expr.handler_count; i++) {
                for (int j = 0; j < i; j++) {
                    if (!strcmp(expr->as.handle_expr.handler_op_names[i],
                                expr->as.handle_expr.handler_op_names[j])) {
                        emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 6,
                            "I require one handler clause per operation.",
                            "Remove the duplicate operation handler.");
                        return TYPE_UNKNOWN;
                    }
                }
            }
            for (int i = 0; i < env->effect_count; i++) {
                bool matches = true;
                for (int j = 0; j < expr->as.handle_expr.handler_count; j++) {
                    if (!effect_get_op(&env->effects[i], expr->as.handle_expr.handler_op_names[j])) {
                        matches = false;
                        break;
                    }
                }
                if (!matches) continue;
                if (matched_effect) {
                    emit_context_error("E001 TYPE MISMATCH", expr->line, expr->column, 6,
                        "I cannot infer a unique effect for this handler.",
                        "Use operation names that identify one effect.");
                    return TYPE_UNKNOWN;
                }
                matched_effect = &env->effects[i];
            }

            if (!matched_effect) {
                emit_context_error("E029 UNKNOWN EFFECT OPERATION", expr->line, expr->column,
                    6,
                    "I found no effect declaring all these handler operations.",
                    "E015: define the effect before using handle...with");
                g_typecheck_error_count++;
                return TYPE_UNKNOWN;
            }

            /* Store resolved effect name for transpiler */
            if (expr->as.handle_expr.effect_name) free(expr->as.handle_expr.effect_name);
            expr->as.handle_expr.effect_name = strdup(matched_effect->name);

            /* Verify every handler op exists in the matched effect */
            for (int i = 0; i < expr->as.handle_expr.handler_count; i++) {
                const char *op_name = expr->as.handle_expr.handler_op_names[i];
                if (!effect_get_op(matched_effect, op_name)) {
                    emit_context_error("E030 UNKNOWN HANDLER OPERATION", expr->line, expr->column,
                        (int)strlen(op_name),
                        "This operation is not declared in the matched effect",
                        "E015: check the effect declaration for valid operation names");
                    g_typecheck_error_count++;
                }
            }

            /* Typecheck each handler body — introduce params as symbols */
            for (int i = 0; i < expr->as.handle_expr.handler_count; i++) {
                const char *op_name = expr->as.handle_expr.handler_op_names[i];
                EffectOp *op = effect_get_op(matched_effect, op_name);
                int saved_count = env->symbol_count;

                /* Bind handler parameters to their declared types */
                if (op) {
                    int param_count = expr->as.handle_expr.handler_param_counts[i];
                    if (param_count != op->param_count) {
                        emit_context_error("E003 ARITY MISMATCH", expr->line, expr->column, 6,
                            "I require the declared operation's parameter count in its handler.",
                            "Match the effect operation signature.");
                        return TYPE_UNKNOWN;
                    }
                    int bind_count = param_count < op->param_count ? param_count : op->param_count;
                    for (int k = 0; k < bind_count; k++) {
                        const char *pname = expr->as.handle_expr.handler_param_names[i][k];
                        if (pname) {
                            Value dummy = create_void();
                            Parameter *param = &op->params[k];
                            env_define_var_with_type_info(env, pname, param->type,
                                param->element_type, param->type_info, false, dummy);
                            Symbol *symbol = env_get_var(env, pname);
                            if (symbol) {
                                free(symbol->struct_type_name);
                                symbol->struct_type_name = param->struct_type_name
                                    ? strdup(param->struct_type_name) : NULL;
                                symbol->def_line = expr->line;
                                symbol->def_column = expr->column;
                            }
                        }
                    }
                }

                check_expression(expr->as.handle_expr.handler_bodies[i], env);

                /* Pop handler-local symbols */
                env->symbol_count = saved_count;
            }

            /* Type of handle expression = type of the handled body */
            return check_expression(expr->as.handle_expr.body, env);
        }

        case AST_AWAIT:
            /* await expr: transparent in typechecker — returns the type of the inner expression */
            return check_expression_impl(expr->as.await_expr.expr, env);

        case AST_EFFECT_DECL:
            return TYPE_VOID;

        case AST_EFFECT_HANDLER: {
            if (expr->as.effect_handler.body)
                check_expression(expr->as.effect_handler.body, env);
            /* Type-check handler arm bodies with the param pre-defined. */
            for (int i = 0; i < expr->as.effect_handler.handler_count; i++) {
                int saved_sym = env ? env->symbol_count : 0;
                const char *param = expr->as.effect_handler.handler_param_names
                                    ? expr->as.effect_handler.handler_param_names[i]
                                    : NULL;
                if (param && param[0] != '\0' && env) {
                    Value dummy = {0};
                    env_define_var(env, param, TYPE_UNKNOWN, false, dummy);
                }
                if (expr->as.effect_handler.handler_bodies[i])
                    check_expression(expr->as.effect_handler.handler_bodies[i], env);
                if (env) env->symbol_count = saved_sym;
            }
            return TYPE_VOID;
        }

        case AST_EFFECT_OP: {
            return check_perform(expr, env);
        }

        default:
            fprintf(stderr, "Error at line %d, column %d: Invalid expression type\n", expr->line, expr->column);
            return TYPE_UNKNOWN;
    }
}

/* I apply the same declared map context to local and global initializers. */
static void prepare_map_initializer(TypeChecker *tc, ASTNode *stmt) {
    /* Handle HashMap<K,V> (register instantiation for code generation) */
    if (stmt->as.let.var_type == TYPE_HASHMAP && stmt->as.let.type_info) {
        TypeInfo *info = stmt->as.let.type_info;
        if (!info->generic_name || strcmp(info->generic_name, "HashMap") != 0) {
            fprintf(stderr, "Error at line %d, column %d: Invalid HashMap type annotation\n",
                    stmt->line, stmt->column);
            tc->has_error = true;
        } else if (info->type_param_count != 2) {
            fprintf(stderr, "Error at line %d, column %d: HashMap expects 2 type parameter(s), got %d\n",
                    stmt->line, stmt->column, info->type_param_count);
            tc->has_error = true;
        } else {
            Type key_t = TYPE_UNKNOWN;
            Type val_t = TYPE_UNKNOWN;
            if (!hashmap_extract_kv(info, &key_t, &val_t)) {
                fprintf(stderr, "Error at line %d, column %d: Invalid HashMap type annotation\n",
                        stmt->line, stmt->column);
                tc->has_error = true;
            }

            /* Current runtime supports hashing for int and string keys only */
            if (!(key_t == TYPE_INT || key_t == TYPE_STRING)) {
                fprintf(stderr, "Error at line %d, column %d: HashMap key type must be int or string (got %s)\n",
                        stmt->line, stmt->column, type_to_string(key_t));
                tc->has_error = true;
            }
            if (!(val_t == TYPE_INT || val_t == TYPE_STRING)) {
                fprintf(stderr, "Error at line %d, column %d: HashMap value type must be int or string (got %s)\n",
                        stmt->line, stmt->column, type_to_string(val_t));
                tc->has_error = true;
            }

            /* Register instantiation for codegen */
            char *key_name = typeinfo_to_generic_arg_name(info->type_params[0]);
            char *val_name = typeinfo_to_generic_arg_name(info->type_params[1]);
            env_register_hashmap_instantiation(tc->env, key_name, val_name);

            /* If RHS is (map_new), annotate call with monomorphized return type for transpiler.
             * ALWAYS set this, even if a user-defined map_new exists - the check_expression
             * handler will use this to prefer the built-in generic over user-defined. */
            if (stmt->as.let.value && stmt->as.let.value->type == AST_CALL &&
                stmt->as.let.value->as.call.name &&
                strcmp(stmt->as.let.value->as.call.name, "map_new") == 0) {
                char mono[512];
                snprintf(mono, sizeof(mono), "HashMap_%s_%s", key_name, val_name);
                if (stmt->as.let.value->as.call.return_struct_type_name) {
                    free(stmt->as.let.value->as.call.return_struct_type_name);
                }
                stmt->as.let.value->as.call.return_struct_type_name = strdup(mono);
            }

            free(key_name);
            free(val_name);
        }
    }
}

/* Internal implementation - do not call directly */
static Type check_statement_impl(TypeChecker *tc, ASTNode *stmt);

/* Check statement and return its type (for blocks) (wrapper with recursion depth tracking) */
static Type check_statement(TypeChecker *tc, ASTNode *stmt) {
    if (!stmt) return TYPE_VOID;

    /* Check recursion depth to prevent stack overflow */
    g_check_stmt_depth++;
    if (g_check_stmt_depth > MAX_CHECK_STMT_DEPTH) {
        fprintf(stderr, "Error: Type checker recursion depth exceeded. "
                        "File too large - consider splitting into modules\n");
        tc->has_error = true;
        g_check_stmt_depth--;
        return TYPE_VOID;
    }

    TypeChecker *previous = active_statement_checker;
    active_statement_checker = tc;
    int first_symbol = tc->env->symbol_count;
    Type result = check_statement_impl(tc, stmt);
    ASTNode *scope = stmt->type == AST_FOR ? stmt->as.for_stmt.body : stmt;
    bound_scope_symbols(tc->env, first_symbol, scope);
    active_statement_checker = previous;
    g_check_stmt_depth--;
    return result;
}

/* Internal implementation of check_statement */
static Type check_statement_impl(TypeChecker *tc, ASTNode *stmt) {
    switch (stmt->type) {
        case AST_LET: {
            if (stmt->as.let.is_destructure) {
                const char *pattern = stmt->as.let.type_name;
                StructDef *record = env_get_struct(tc->env, pattern);
                int field_count = record ? record->field_count : -1;
                char **field_names = record ? record->field_names : NULL;
                bool variant_pattern = false;
                if (!record && pattern) {
                    for (int u = 0; u < tc->env->union_count; ++u) {
                        UnionDef *def = &tc->env->unions[u];
                        size_t length = strlen(def->name);
                        if (strncmp(pattern, def->name, length) || pattern[length] != '.') continue;
                        for (int v = 0; v < def->variant_count; ++v) {
                            if (strcmp(pattern + length + 1, def->variant_names[v])) continue;
                            field_count = def->variant_field_counts[v];
                            field_names = def->variant_field_names[v];
                            variant_pattern = true;
                        }
                    }
                }
                bool complete = field_count >= 0 && field_count == stmt->as.let.destructure_count;
                Type actual_type = check_expression(stmt->as.let.value, tc->env);
                const char *actual_name = get_struct_type_name(stmt->as.let.value, tc->env);
                if (actual_type != TYPE_STRUCT || !actual_name ||
                    (variant_pattern ? strcmp(actual_name, pattern) != 0 :
                        env_get_struct(tc->env, actual_name) != record)) complete = false;
                for (int i = 0; complete && i < stmt->as.let.destructure_count; i++) {
                    const char *name = stmt->as.let.destructure_names[i];
                    bool found = false;
                    for (int j = 0; j < field_count; j++)
                        if (strcmp(name, field_names[j]) == 0) found = true;
                    for (int j = 0; j < i; j++)
                        if (strcmp(name, stmt->as.let.destructure_names[j]) == 0) found = false;
                    complete = found;
                }
                if (!complete) {
                    fprintf(stderr, "I require every record or selected variant field exactly once in an owned pattern at line %d.\n", stmt->line);
                    tc->has_error = true;
                    return TYPE_VOID;
                }
                /* The hidden whole-payload binding owns a copy of the concrete
                 * arguments. Its projected fields use the same substitution as
                 * direct selected-variant field access. */
                TypeInfo *selected = try_get_expr_type_info(stmt->as.let.value, tc->env);
                if (variant_pattern && selected) {
                    free_payload_type_info(stmt->as.let.type_info);
                    stmt->as.let.type_info = copy_payload_type_info(selected);
                }
            }
            /* INVARIANT (bead nl-ico): declared_type is a local working copy
             * of stmt->as.let.var_type. Any place that reclassifies the
             * inferred/declared type (struct→union, struct→enum, etc.) must
             * write back to BOTH declared_type AND stmt->as.let.var_type so
             * downstream backends see the corrected type. The original bug
             * fixed in edf4ceb was that backends saw a stale var_type while
             * declared_type drifted ahead.
             *
             * The dual state is kept (rather than always reading
             * stmt->as.let.var_type) because the local is a natural
             * "working type" abstraction; eliminating it would touch ~20
             * sites with no clear safety gain over enforcing this invariant. */
            Type declared_type = stmt->as.let.var_type;

            /* Type inference: let x = expr  (no declared type annotation) */
            if (declared_type == TYPE_UNKNOWN && stmt->as.let.value) {
                Type inferred = check_expression(stmt->as.let.value, tc->env);
                stmt->as.let.var_type = inferred;
                declared_type = inferred;
                /* Preserve array representation for fields, aliases and calls too. */
                if (inferred == TYPE_ARRAY && stmt->as.let.element_type == TYPE_UNKNOWN) {
                    stmt->as.let.element_type = infer_array_element_type(stmt->as.let.value, tc->env);
                }
                /* Infer struct/union type_name from expression where possible */
                if ((inferred == TYPE_STRUCT || inferred == TYPE_UNION) && !stmt->as.let.type_name) {
                    const char *name = get_struct_type_name(stmt->as.let.value, tc->env);
                    if (name) stmt->as.let.type_name = strdup(name);
                }
                if (!stmt->as.let.type_info) {
                    stmt->as.let.type_info = copy_payload_type_info(
                        try_get_expr_type_info(stmt->as.let.value, tc->env));
                }
                if (opaque_type_info_present(stmt->as.let.type_info) || type_info_needs_array_context(stmt->as.let.type_info))
                    check_concrete_union_arrays(tc->env, stmt->as.let.type_info, stmt->as.let.value, 0);
                /* Register and add to env */
                Value val = create_void();
                env_define_var_with_type_info(tc->env, stmt->as.let.name, inferred,
                    stmt->as.let.element_type, stmt->as.let.type_info,
                    stmt->as.let.is_mut, val);
                Symbol *isym = env_get_var(tc->env, stmt->as.let.name);
                if (isym) {
                    isym->def_line = stmt->line;
                    isym->def_column = stmt->column;
                    if (stmt->as.let.type_name) {
                        isym->struct_type_name = strdup(stmt->as.let.type_name);
                    }
                }
                return TYPE_VOID;
            }

            Type original_declared_type = declared_type;  /* Save original before modifications */

            /* Handle generic lists: List<UserType> - Register BEFORE checking expression */
            if (declared_type == TYPE_LIST_GENERIC && stmt->as.let.type_name) {
                const char *element_type = stmt->as.let.type_name;
                
                /* Verify element type exists (struct or enum must be defined) */
                if (!env_get_struct(tc->env, element_type) && !env_get_enum(tc->env, element_type)) {
                    fprintf(stderr, "Error at line %d, column %d: Unknown type '%s' in List<%s>\n",
                            stmt->line, stmt->column, element_type, element_type);
                    tc->has_error = true;
                } else {
                    /* Register this instantiation for code generation */
                    env_register_list_instantiation(tc->env, element_type);
                }
            }
            
            /* Handle generic unions: Result<int, string>, Option<T>, etc. */
            if (declared_type == TYPE_UNION && stmt->as.let.type_info) {
                TypeInfo *info = stmt->as.let.type_info;
                if (info->generic_name) {
                    /* Verify the union definition exists */
                    UnionDef *union_def = env_get_union(tc->env, info->generic_name);
                    if (!union_def) {
                        fprintf(stderr, "Error at line %d, column %d: Unknown union '%s'\n",
                                stmt->line, stmt->column, info->generic_name);
                        tc->has_error = true;
                    } else if (union_def->generic_param_count != info->type_param_count) {
                        fprintf(stderr, "Error at line %d, column %d: Union '%s' expects %d type parameter(s), got %d\n",
                                stmt->line, stmt->column, info->generic_name,
                                union_def->generic_param_count, info->type_param_count);
                        tc->has_error = true;
                    } else {
                        if (opaque_type_info_present(info)) {
                            register_native_union_context(tc->env, info, 0);
                        } else {
                        /* Build concrete type names for registration */
                        char **type_names = malloc(sizeof(char*) * info->type_param_count);
                        for (int i = 0; i < info->type_param_count; i++) {
                            type_names[i] = typeinfo_to_generic_arg_name(info->type_params[i]);
                        }
                        
                        /* Register this instantiation for code generation */
                        env_register_union_instantiation(tc->env, info->generic_name,
                                                        (const char**)type_names,
                                                        info->type_param_count);
                        
                        /* Free type names */
                        for (int i = 0; i < info->type_param_count; i++) {
                            free(type_names[i]);
                        }
                        free(type_names);
                        }
                    }
                }
            }

            prepare_map_initializer(tc, stmt);

            /* Handle anonymous struct literals: infer struct name from declared type */
            if (stmt->as.let.value && stmt->as.let.value->type == AST_STRUCT_LITERAL) {
                ASTNode *struct_lit = stmt->as.let.value;
                if (struct_lit->as.struct_literal.struct_name == NULL) {
                    /* Anonymous struct literal - infer name from declared type */
                    if (declared_type == TYPE_STRUCT && stmt->as.let.type_name) {
                        /* Fill in the struct name for type checking */
                        struct_lit->as.struct_literal.struct_name = strdup(stmt->as.let.type_name);
                    } else {
                        fprintf(stderr, "Error at line %d, column %d: Cannot infer struct type for anonymous literal\n",
                                struct_lit->line, struct_lit->column);
                        tc->has_error = true;
                    }
                }
            }
            
            /* Now check the expression - the specialized functions are registered */
            check_concrete_union_arrays(tc->env, stmt->as.let.type_info, stmt->as.let.value, 0);
            Type value_type = check_expression(stmt->as.let.value, tc->env);
            if (!check_opaque_value(tc->env, declared_type, stmt->as.let.type_name, stmt->as.let.value))
                tc->has_error = true;
            /* A projected concrete union retains its complete annotation; I
             * do not accept a different instantiation merely because both are unions. */
            if (stmt->as.let.value->type == AST_FIELD_ACCESS && stmt->as.let.type_info) {
                TypeInfo *actual = try_get_expr_type_info(stmt->as.let.value, tc->env);
                TypeInfo *expected = stmt->as.let.type_info;
                if (actual && actual->generic_name && expected->generic_name &&
                    env_get_union(tc->env, actual->generic_name) &&
                    env_get_union(tc->env, expected->generic_name)) {
                    TypeInfo concrete_actual = *actual, concrete_expected = *expected;
                    concrete_actual.base_type = concrete_expected.base_type = TYPE_UNION;
                    if (!type_infos_equal(&concrete_actual, &concrete_expected)) {
                        emit_context_error("E001 TYPE MISMATCH", stmt->line, stmt->column, 1,
                            "I require the record field's concrete union type to match the binding annotation.",
                            "Preserve the union declaration and all concrete type arguments.");
                        tc->has_error = true;
                    }
                }
            }

            if (!check_record_array_contract(tc->env, stmt->as.let.var_type,
                    stmt->as.let.element_type, stmt->as.let.type_name, stmt->as.let.value))
                tc->has_error = true;
            
            /* If declared type is STRUCT, check if it's actually an enum or union */
            if (declared_type == TYPE_STRUCT && stmt->as.let.type_name) {
                /* Normalize qualified name (Math.Point) to actual struct name (Point) */
                StructDef *sdef = env_get_struct(tc->env, stmt->as.let.type_name);
                if (sdef && strcmp(stmt->as.let.type_name, sdef->name) != 0) {
                    free((char*)stmt->as.let.type_name);
                    stmt->as.let.type_name = strdup(sdef->name);
                }
                
                /* Check if this is actually a union */
                if (env_get_union(tc->env, stmt->as.let.type_name)) {
                    declared_type = TYPE_UNION;
                    stmt->as.let.var_type = TYPE_UNION;
                }
                /* Check if this is actually an enum */
                else if (env_get_enum(tc->env, stmt->as.let.type_name)) {
                    declared_type = TYPE_ENUM;
                    stmt->as.let.var_type = TYPE_ENUM;
                }
            }
            /* Legacy handling for enums without type_name */
            else if (declared_type == TYPE_STRUCT && value_type == TYPE_INT) {
                /* This is okay - enums are compatible with ints */
                declared_type = TYPE_ENUM;
                stmt->as.let.var_type = TYPE_ENUM;
            }
            
            /* Special handling for function types - need to check signatures match */
            /* Also handle case where value_type is TYPE_INT (function-typed parameter placeholder) */
            if (declared_type == TYPE_FUNCTION && (value_type == TYPE_FUNCTION || value_type == TYPE_INT)) {
                /* Both are function types - check if signatures match */
                FunctionSignature *declared_sig = stmt->as.let.fn_sig;
                FunctionSignature *value_sig = NULL;
                bool owns_value_sig = false;
                
                /* Get function signature from value expression */
                if (stmt->as.let.value->type == AST_CALL) {
                    /* Function call - could be calling a function-typed parameter */
                    Function *func = env_get_function(tc->env, stmt->as.let.value->as.call.name);
                    if (func && func->return_type == TYPE_FUNCTION) {
                        /* Function that returns a function */
                        value_sig = func->return_fn_sig;
                    } else if (!func) {
                        /* Check if it's a function-typed parameter being called */
                        Symbol *sym = env_get_var(tc->env, stmt->as.let.value->as.call.name);
                        if (sym && sym->type == TYPE_FUNCTION) {
                            /* Function-typed parameter - if called with 0 args, it's the function itself */
                            if (stmt->as.let.value->as.call.arg_count == 0) {
                                /* TODO: Get function signature from parameter - for now allow it */
                                /* The signature should match the declared type */
                            }
                        }
                    }
                } else if (stmt->as.let.value->type == AST_IDENTIFIER) {
                    /* Could be function name or function-typed variable */
                    Function *func = env_get_function(tc->env, stmt->as.let.value->as.identifier);
                    if (func) {
                        value_sig = function_signature_from_function(func);
                        owns_value_sig = true;
                    } else {
                        /* Check if it's a function-typed variable */
                        Symbol *sym = env_get_var(tc->env, stmt->as.let.value->as.identifier);
                        if (sym && sym->type == TYPE_FUNCTION) {
                            value_sig = sym->type_info ? sym->type_info->fn_sig : NULL;
                        }
                    }
                } else if (stmt->as.let.value->type == AST_CALL && stmt->as.let.value->as.call.func_expr) {
                    /* Function call returning function: ((func_call) arg1 arg2) */
                    /* The return type will be determined at runtime */
                    /* For now, allow it if declared type is TYPE_FUNCTION */
                }
                
                /* Check if signatures match */
                if (declared_sig && value_sig) {
                    if (!function_signatures_equal(declared_sig, value_sig)) {
                        fprintf(stderr, "Error at line %d, column %d: Function signature mismatch in let statement\n", stmt->line, stmt->column);
                        tc->has_error = true;
                    }
                } else if (!declared_sig || !value_sig) {
                    /* One or both signatures missing - allow for now (runtime will handle) */
                    /* This happens when function signatures aren't fully parsed yet, or when */
                    /* dealing with function-typed parameters where we don't have full signature info */
                }
                if (owns_value_sig) free_function_signature(value_sig);
            } else if (!types_match(value_type, declared_type)) {
                char message[256];
                snprintf(message, sizeof(message),
                        "Let binding expects %s but got %s.",
                        type_to_string(declared_type), type_to_string(value_type));
                emit_context_error(
                    "E001 TYPE MISMATCH",
                    stmt->line,
                    stmt->column,
                    1,
                    message,
                    "Ensure the assigned expression matches the declared type."
                );
                tc->has_error = true;
            }

            /* Extract element type if this is an array */
            Type element_type = stmt->as.let.element_type;  /* Get from type annotation if available */
            if (declared_type == TYPE_ARRAY && stmt->as.let.value->type == AST_CALL &&
                !stmt->as.let.value->as.call.func_expr && stmt->as.let.value->as.call.name &&
                strcmp(stmt->as.let.value->as.call.name, "map") == 0) {
                Type mapped = infer_array_element_type(stmt->as.let.value, tc->env);
                if (element_type == TYPE_UNKNOWN) {
                    element_type = mapped;
                    stmt->as.let.element_type = mapped;
                } else if (mapped != TYPE_UNKNOWN && !types_match(mapped, element_type)) {
                    emit_context_error("E001 TYPE MISMATCH", stmt->line, stmt->column, 1,
                        "I require the array annotation to match the transform result type.",
                        "Use the transform's return type as the mapped element type.");
                    tc->has_error = true;
                }
            }
            if (declared_type == TYPE_ARRAY && element_type == TYPE_UNKNOWN) {
                /* Fallback: infer from array literal if not specified in type annotation */
                if (stmt->as.let.value->type == AST_ARRAY_LITERAL) {
                    ASTNode *array_lit = stmt->as.let.value;
                    if (array_lit->as.array_literal.element_count > 0) {
                        element_type = check_expression(array_lit->as.array_literal.elements[0], tc->env);
                    } else if (array_lit->as.array_literal.element_type != TYPE_UNKNOWN) {
                        element_type = array_lit->as.array_literal.element_type;
                    }
                } else if (stmt->as.let.value->type == AST_CALL && stmt->as.let.value->as.call.name) {
                    const char *name = stmt->as.let.value->as.call.name;
                    if (strcmp(name, "file_read_bytes") == 0 || strcmp(name, "bytes_from_string") == 0) {
                        element_type = TYPE_U8;
                    } else if (strcmp(name, "array_slice") == 0 && stmt->as.let.value->as.call.arg_count >= 1) {
                        ASTNode *inner = stmt->as.let.value->as.call.args[0];
                        if (inner && inner->type == AST_IDENTIFIER) {
                            Symbol *sym = env_get_var_visible_at(tc->env, inner->as.identifier, inner->line, inner->column);
                            if (sym && sym->type == TYPE_ARRAY && sym->element_type != TYPE_UNKNOWN) {
                                element_type = sym->element_type;
                            }
                        }
                    }
                }
            }
            
            /* Propagate element type to empty array literals for correct transpilation */
            if (declared_type == TYPE_ARRAY && element_type != TYPE_UNKNOWN) {
                if (stmt->as.let.value->type == AST_ARRAY_LITERAL) {
                    ASTNode *array_lit = stmt->as.let.value;
                    check_array_literal_annotation(tc, array_lit, element_type, stmt->as.let.type_name);
                }
            }
            
            /* Create TypeInfo for tuples or use existing from parser for generic types */
            if (!retain_let_function_type(tc, stmt, declared_type)) return TYPE_UNKNOWN;
            TypeInfo *type_info = stmt->as.let.type_info;  /* Use parser's TypeInfo if available */
            if (!type_info && declared_type == TYPE_TUPLE && stmt->as.let.value->type == AST_TUPLE_LITERAL) {
                /* Create TypeInfo from tuple literal */
                ASTNode *tuple_lit = stmt->as.let.value;
                type_info = calloc(1, sizeof(TypeInfo));
                if (!type_info) { tc->has_error = true; return TYPE_UNKNOWN; }
                type_info->base_type = TYPE_TUPLE;
                type_info->element_type = NULL;
                type_info->generic_name = NULL;
                type_info->type_params = NULL;
                type_info->type_param_count = 0;
                type_info->tuple_element_count = tuple_lit->as.tuple_literal.element_count;
                
                /* Copy tuple element types */
                if (type_info->tuple_element_count > 0) {
                    type_info->tuple_types = malloc(sizeof(Type) * type_info->tuple_element_count);
                    type_info->tuple_type_names = malloc(sizeof(char*) * type_info->tuple_element_count);
                    
                    for (int i = 0; i < type_info->tuple_element_count; i++) {
                        if (tuple_lit->as.tuple_literal.element_types) {
                            type_info->tuple_types[i] = tuple_lit->as.tuple_literal.element_types[i];
                        } else {
                            type_info->tuple_types[i] = TYPE_UNKNOWN;
                        }
                        type_info->tuple_type_names[i] = NULL;  /* TODO: Handle struct/union types */
                    }
                } else {
                    type_info->tuple_types = NULL;
                    type_info->tuple_type_names = NULL;
                }
            }
            
            /* Add to environment */
            /* Use declared_type which has been corrected for unions and enums */
            Type env_type = declared_type;
            Value val = create_void(); /* Placeholder */
            env_define_var_with_type_info(tc->env, stmt->as.let.name, env_type, element_type, type_info, stmt->as.let.is_mut, val);
            
            /* Store definition location and type metadata for unused variable warnings */
            /* IMPORTANT: Look up the symbol FRESH each time we need to modify it,
             * because the symbol array may get reallocated! */
            
            /* Set definition location */
            Symbol *sym = env_get_var(tc->env, stmt->as.let.name);
            if (sym) {
                sym->def_line = stmt->line;
                sym->def_column = stmt->column;
            }
            
            /* Set struct type name - look up symbol again to be safe */
            sym = env_get_var(tc->env, stmt->as.let.name);
            if (sym && stmt->as.let.type_name) {
                if (original_declared_type == TYPE_STRUCT || original_declared_type == TYPE_UNION) {
                    /* Use the declared type name */
                    if (sym->struct_type_name) free(sym->struct_type_name);  /* Free old value if any */
                    sym->struct_type_name = strdup(stmt->as.let.type_name);
                    
                    /* Mark as resource if the struct type is a resource */
                    mark_variable_as_resource_if_needed(tc->env, stmt->as.let.name, stmt->as.let.type_name);
                }
            }
            
            /* Also try to infer from value expression if struct_type_name not set */
            sym = env_get_var(tc->env, stmt->as.let.name);
            if (sym && !sym->struct_type_name && value_type == TYPE_STRUCT) {
                /* Infer struct type name from the value expression */
                const char *struct_name = get_struct_type_name(stmt->as.let.value, tc->env);
                if (struct_name) {
                    sym->struct_type_name = strdup(struct_name);
                }
            }
            
            /* If this is an array of structs, store the struct type name for the elements */
            sym = env_get_var(tc->env, stmt->as.let.name);
            if (sym && declared_type == TYPE_ARRAY && element_type == TYPE_STRUCT && stmt->as.let.type_name) {
                if (sym->struct_type_name) free(sym->struct_type_name);
                sym->struct_type_name = strdup(stmt->as.let.type_name);
            }
            
            /* If this is a union, store the union type name */
            sym = env_get_var(tc->env, stmt->as.let.name);
            if (sym && declared_type == TYPE_UNION && stmt->as.let.type_name) {
                if (sym->struct_type_name) free(sym->struct_type_name);
                sym->struct_type_name = strdup(stmt->as.let.type_name);
            }
            
            return TYPE_VOID;
        }

        case AST_SET: {
            Symbol *sym = env_get_var_visible_at(tc->env, stmt->as.set.name, stmt->line, stmt->column);
            if (!sym) {
                char message[256];
                snprintf(message, sizeof(message), "I cannot find a variable named `%s`.", stmt->as.set.name);
                emit_context_error(
                    "E024 UNDEFINED VARIABLE",
                    stmt->line,
                    stmt->column,
                    (int)safe_strlen(stmt->as.set.name),
                    message,
                    "Check spelling or ensure the variable is in scope."
                );
                tc->has_error = true;
                return TYPE_VOID;
            }

            if (stmt->as.set.field_name) {
                StructDef *record = sym->struct_type_name ? env_get_struct(tc->env, sym->struct_type_name) : NULL;
                if (sym->type != TYPE_BORROW_MUT || !record) {
                    fprintf(stderr, "I require an exclusive borrowed owner for field mutation\n");
                    tc->has_error = true;
                    return TYPE_VOID;
                }
                for (int i = 0; i < record->field_count; ++i) {
                    if (strcmp(record->field_names[i], stmt->as.set.field_name)) continue;
                    Type actual = check_expression(stmt->as.set.value, tc->env);
                    if (!types_match(actual, record->field_types[i])) {
                        fprintf(stderr, "I require the declared field type for borrowed mutation\n");
                        tc->has_error = true;
                    }
                    return TYPE_VOID;
                }
                fprintf(stderr, "I cannot find the named field in this borrowed owner\n");
                tc->has_error = true;
                return TYPE_VOID;
            }

            if (!sym->is_mut) {
                fprintf(stderr, "Error at line %d, column %d: Cannot assign to immutable variable '%s'\n",
                        stmt->line, stmt->column, stmt->as.set.name);
                tc->has_error = true;
            }

            check_concrete_union_arrays(tc->env, sym->type_info, stmt->as.set.value, 0);
            Type value_type = check_expression(stmt->as.set.value, tc->env);
            if (!check_opaque_value(tc->env, sym->type, sym->struct_type_name, stmt->as.set.value))
                tc->has_error = true;
            if (!check_record_array_contract(tc->env, sym->type, sym->element_type,
                    sym->struct_type_name, stmt->as.set.value)) tc->has_error = true;

            /* Propagate element type to array literals for correct transpilation */
            if (sym->type == TYPE_ARRAY && sym->element_type != TYPE_UNKNOWN) {
                if (stmt->as.set.value->type == AST_ARRAY_LITERAL) {
                    ASTNode *array_lit = stmt->as.set.value;
                    check_array_literal_annotation(tc, array_lit, sym->element_type, sym->struct_type_name);
                }
            }

            if (!types_match(value_type, sym->type)) {
                char message[256];
                snprintf(message, sizeof(message),
                        "Assignment expects %s but got %s.",
                        type_to_string(sym->type), type_to_string(value_type));
                emit_context_error(
                    "E001 TYPE MISMATCH",
                    stmt->line,
                    stmt->column,
                    1,
                    message,
                    "Convert the value to the variable's type before assignment."
                );
                tc->has_error = true;
            }

            return TYPE_VOID;
        }

        case AST_WHILE: {
            Type cond_type = check_expression(stmt->as.while_stmt.condition, tc->env);
            if (cond_type != TYPE_BOOL) {
                emit_context_error(
                    "E001 TYPE MISMATCH",
                    stmt->line,
                    stmt->column,
                    1,
                    "While condition must be a bool.",
                    "Ensure the loop condition evaluates to bool."
                );
                tc->has_error = true;
            }

            /* Increment loop depth for break/continue validation */
            tc->loop_depth++;
            check_statement(tc, stmt->as.while_stmt.body);
            tc->loop_depth--;
            return TYPE_VOID;
        }

        case AST_FOR: {
            /* Determine loop variable type from iterable */
            Type iter_type = check_expression(stmt->as.for_stmt.range_expr, tc->env);

            Type loop_var_type = TYPE_INT;  /* default for range(start, end) */
            const char *loop_var_struct_name = NULL;

            if (iter_type == TYPE_ARRAY) {
                /* Look up array variable to get element type */
                ASTNode *rng = stmt->as.for_stmt.range_expr;
                if (rng && rng->type == AST_IDENTIFIER) {
                    Symbol *arr_sym = env_get_var(tc->env, rng->as.identifier);
                    if (arr_sym && arr_sym->element_type != TYPE_UNKNOWN) {
                        loop_var_type = arr_sym->element_type;
                    }
                    if (loop_var_type == TYPE_STRUCT && arr_sym && arr_sym->struct_type_name) {
                        loop_var_struct_name = arr_sym->struct_type_name;
                    }
                }
            } else if (iter_type == TYPE_LIST_STRING) {
                loop_var_type = TYPE_STRING;
            } else if (iter_type == TYPE_LIST_GENERIC) {
                /* Look up list variable to get element type */
                ASTNode *rng = stmt->as.for_stmt.range_expr;
                if (rng && rng->type == AST_IDENTIFIER) {
                    Symbol *list_sym = env_get_var(tc->env, rng->as.identifier);
                    if (list_sym && list_sym->element_type != TYPE_UNKNOWN) {
                        loop_var_type = list_sym->element_type;
                    }
                    if (loop_var_type == TYPE_STRUCT && list_sym && list_sym->struct_type_name) {
                        loop_var_struct_name = list_sym->struct_type_name;
                    }
                }
            }
            /* TYPE_LIST_INT, TYPE_LIST_TOKEN -> TYPE_INT (default already set) */

            Value val = create_void();
            const TypeInfo *array_info = iter_type == TYPE_ARRAY
                ? try_get_expr_type_info(stmt->as.for_stmt.range_expr, tc->env) : NULL;
            const TypeInfo *element_info = array_info && array_info->base_type == TYPE_ARRAY
                ? array_info->element_type : NULL;
            if (element_info && type_info_exact_array_element(element_info)) {
                loop_var_type = element_info->base_type;
                loop_var_struct_name = element_info->opaque_type_name
                    ? element_info->opaque_type_name : element_info->generic_name;
                env_define_var_with_type_info(tc->env, stmt->as.for_stmt.var_name, loop_var_type,
                    element_info->element_type ? element_info->element_type->base_type : TYPE_UNKNOWN,
                    (TypeInfo *)element_info, false, val);
            } else env_define_var(tc->env, stmt->as.for_stmt.var_name, loop_var_type, false, val);

            /* Set definition location and struct type name */
            Symbol *loop_var_sym = env_get_var(tc->env, stmt->as.for_stmt.var_name);
            if (loop_var_sym) {
                loop_var_sym->def_line = stmt->line;
                loop_var_sym->def_column = stmt->column;
                if (loop_var_struct_name) {
                    loop_var_sym->struct_type_name = strdup(loop_var_struct_name);
                }
            }

            /* Check the loop body (increment loop depth for break/continue validation) */
            tc->loop_depth++;
            check_statement(tc, stmt->as.for_stmt.body);
            tc->loop_depth--;

            /* DON'T restore environment - transpiler needs loop variable symbols! */
            /* The old code removed loop variables after typechecking:
             *   int old_symbol_count = tc->env->symbol_count;
             *   ...
             *   for (int i = old_symbol_count; i < tc->env->symbol_count; i++) {
             *       free(tc->env->symbols[i].name);
             *       ...
             *   }
             *   tc->env->symbol_count = old_symbol_count;
             * This caused undefined variable warnings when transpiler/post-processing
             * needed to look up loop variables. Loop variables are scoped by C's block
             * scope rules, so keeping them in the environment doesn't cause collisions.
             */

            return TYPE_VOID;
        }

        case AST_BREAK: {
            if (tc->loop_depth == 0) {
                fprintf(stderr, "Error at line %d, column %d: 'break' outside loop\n", stmt->line, stmt->column);
                tc->has_error = true;
            }
            return TYPE_VOID;
        }

        case AST_CONTINUE: {
            if (tc->loop_depth == 0) {
                fprintf(stderr, "Error at line %d, column %d: 'continue' outside loop\n", stmt->line, stmt->column);
                tc->has_error = true;
            }
            return TYPE_VOID;
        }

        case AST_RETURN: {
            if (stmt->as.return_stmt.value) {
                /* Handle anonymous struct literals: infer struct name from function return type */
                if (stmt->as.return_stmt.value->type == AST_STRUCT_LITERAL) {
                    ASTNode *struct_lit = stmt->as.return_stmt.value;
                    if (struct_lit->as.struct_literal.struct_name == NULL) {
                        /* Anonymous struct literal - infer name from function return type */
                        if (tc->current_function_return_type == TYPE_STRUCT && tc->current_function_return_struct_name) {
                            /* Fill in the struct name for type checking */
                            struct_lit->as.struct_literal.struct_name = strdup(tc->current_function_return_struct_name);
                        } else {
                            emit_context_error("E001 TYPE MISMATCH", struct_lit->line, struct_lit->column, 1,
                                               "Cannot infer struct type for anonymous literal in return.",
                                               "Specify the struct name explicitly, e.g. 'StructName { field: value }'.");
                            tc->has_error = true;
                        }
                    }
                }
                
                check_concrete_union_arrays(tc->env, tc->current_function_return_info, stmt->as.return_stmt.value, 0);
                Type return_type = check_expression(stmt->as.return_stmt.value, tc->env);
                if (!check_opaque_value(tc->env, tc->current_function_return_type,
                        tc->current_function_return_struct_name, stmt->as.return_stmt.value)) tc->has_error = true;
                if (!check_record_array_contract(tc->env, tc->current_function_return_type,
                        tc->current_function_return_element_type, tc->current_function_return_struct_name,
                        stmt->as.return_stmt.value)) tc->has_error = true;
                if (!types_match(return_type, tc->current_function_return_type)) {
                    char message[256];
                    snprintf(message, sizeof(message), "Return type mismatch: got %s, expected %s.",
                             type_to_string(return_type), type_to_string(tc->current_function_return_type));
                    emit_context_error("E001 TYPE MISMATCH", stmt->line, stmt->column, 1, message,
                                       "Ensure the returned value matches the function's declared return type.");
                    tc->has_error = true;
                }
            } else {
                if (tc->current_function_return_type != TYPE_VOID) {
                    char hint[256];
                    snprintf(hint, sizeof(hint), "Add 'return <value>' of type %s.",
                             type_to_string(tc->current_function_return_type));
                    emit_context_error("E010 MISSING RETURN", stmt->line, stmt->column, 1,
                                       "Empty return in a non-void function.", hint);
                    tc->has_error = true;
                }
            }
            return tc->current_function_return_type;
        }

        case AST_BLOCK: {
            Type last_type = TYPE_VOID;
            bool returned = false;
            for (int i = 0; i < stmt->as.block.count; i++) {
                ASTNode *s = stmt->as.block.statements[i];
                if (returned) {
                    fprintf(stderr, "Warning at line %d, column %d: Unreachable code after return\n",
                            s->line, s->column);
                }
                last_type = check_statement(tc, s);
                if (s->type == AST_RETURN) returned = true;
            }
            return last_type;
        }

        case AST_PRINT: {
            check_expression(stmt->as.print.expr, tc->env);
            return TYPE_VOID;
        }

        case AST_ASSERT: {
            Type cond_type = check_expression(stmt->as.assert.condition, tc->env);
            if (cond_type != TYPE_BOOL) {
                fprintf(stderr, "Error at line %d, column %d: Assert condition must be bool\n", stmt->line, stmt->column);
                tc->has_error = true;
            }
            return TYPE_VOID;
        }

        case AST_PAR_BLOCK: {
            if (stmt->as.par_block.is_flow) {
                int *order = passive_binding_order(stmt);
                if (!order) {
                    emit_context_error("E0036 PASSIVE FLOW", stmt->line, stmt->column, 3,
                        "I cannot establish distinct immutable flow bindings and acyclic dependencies.",
                        "Use a nonempty scalar graph with a stable serial order.");
                    tc->has_error = true;
                    return TYPE_VOID;
                }
                for (int step = 0; step < stmt->as.par_block.count; ++step) {
                    ASTNode *binding = stmt->as.par_block.bindings[order[step]];
                    if (!par_scalar_expression(binding->as.let.value, tc->env)) {
                        emit_context_error("E0036 PASSIVE FLOW", binding->line, binding->column, 3,
                            "I require immutable scalar inputs and checked closed scalar calls in flow.",
                            "Keep mutation, captures and unsupported effects outside this graph.");
                        tc->has_error = true;
                        break;
                    }
                    check_statement(tc, binding);
                    /* Only completed graph dependencies enter the environment.
                     * Later emitters reuse graph visibility without changing the
                     * declaration coordinates used in source diagnostics. */
                    Symbol *symbol = env_get_var(tc->env, binding->as.let.name);
                    if (symbol) {
                        symbol->flow_start_line = stmt->line;
                        symbol->flow_start_column = stmt->column;
                    }
                }
                free(order);
                return TYPE_VOID;
            }
            int count = stmt->as.par_block.count;
            bool valid = count > 0;
            for (int i = 0; i < count; ++i) {
                ASTNode *binding = stmt->as.par_block.bindings[i];
                if (!binding || binding->type != AST_LET || binding->as.let.is_mut) {
                    valid = false;
                    continue;
                }
                if (!par_scalar_expression(binding->as.let.value, tc->env)) valid = false;
                for (int j = 0; j < count; ++j) {
                    ASTNode *other = stmt->as.par_block.bindings[j];
                    if (!other || other->type != AST_LET) continue;
                    if ((i != j && !strcmp(binding->as.let.name, other->as.let.name)) ||
                        ast_references_name(binding->as.let.value, other->as.let.name)) valid = false;
                }
            }
            if (!valid) {
                emit_context_error("E0036 PASSIVE PAR", stmt->line, stmt->column, 3,
                    "I require nonempty independent immutable scalar let bindings in par and checked closed scalar calls.",
                    "Use independent scalar expressions without calls or mutation.");
                tc->has_error = true;
                return TYPE_VOID;
            }
            /* Only after checking every initializer do bindings enter the scope. */
            for (int i = 0; i < count; ++i)
                check_statement(tc, stmt->as.par_block.bindings[i]);
            return TYPE_VOID;
        }

        case AST_PAR_LET: {
            int n = stmt->as.par_let.count;
            if (n == 0) return TYPE_VOID;
            if (n == 1) {
                fprintf(stderr, "Warning at line %d, column %d: par-let with a single binding; consider using regular let\n",
                        stmt->line, stmt->column);
            }
            /* Validate: no binding may reference a sibling binding (would create a data dependency) */
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (j == i) continue;
                    if (ast_references_name(stmt->as.par_let.values[i], stmt->as.par_let.names[j])) {
                        fprintf(stderr, "Error at line %d, column %d: par-let binding '%s' depends on sibling binding '%s' — use sequential let for dependent bindings\n",
                                stmt->line, stmt->column,
                                stmt->as.par_let.names[i], stmt->as.par_let.names[j]);
                        return TYPE_VOID;
                    }
                }
            }
            /* Type-check each binding's RHS independently (siblings not yet in scope) */
            for (int i = 0; i < n; i++) {
                Type t = check_expression(stmt->as.par_let.values[i], tc->env);
                /* Register binding in environment with inferred type */
                Value dummy = create_void();
                env_define_var_with_type_info(tc->env, stmt->as.par_let.names[i],
                    t, TYPE_UNKNOWN, NULL, false, dummy);
            }
            /* Type-check body with all bindings in scope */
            return check_expression(stmt->as.par_let.body, tc->env);
        }

        case AST_UNSAFE_BLOCK: {
            /* Mark that we're entering an unsafe block */
            bool prev_unsafe = tc->in_unsafe_block;
            tc->in_unsafe_block = true;

            /* Type check all statements in the unsafe block */
            for (int i = 0; i < stmt->as.unsafe_block.count; i++) {
                check_statement(tc, stmt->as.unsafe_block.statements[i]);
            }

            /* Restore previous unsafe state */
            tc->in_unsafe_block = prev_unsafe;
            return TYPE_VOID;
        }

        case AST_EFFECT_DECL:
            /* Already registered in the first pass — nothing to type-check here */
            return TYPE_VOID;

        case AST_EFFECT_HANDLER: {
            /* Type-check the body expression */
            if (stmt->as.effect_handler.body)
                check_statement(tc, stmt->as.effect_handler.body);
            /* Type-check each handler arm body.
             * Pre-define the arm's parameter so the body can reference it. */
            for (int i = 0; i < stmt->as.effect_handler.handler_count; i++) {
                int saved_sym = tc->env ? tc->env->symbol_count : 0;
                const char *param = stmt->as.effect_handler.handler_param_names
                                    ? stmt->as.effect_handler.handler_param_names[i]
                                    : NULL;
                if (param && param[0] != '\0' && tc->env) {
                    Value dummy = {0};
                    env_define_var(tc->env, param, TYPE_UNKNOWN, false, dummy);
                }
                if (stmt->as.effect_handler.handler_bodies[i])
                    check_statement(tc, stmt->as.effect_handler.handler_bodies[i]);
                /* Remove the temp param binding */
                if (tc->env) tc->env->symbol_count = saved_sym;
            }
            return TYPE_VOID;
        }

        case AST_EFFECT_OP: {
            return check_perform(stmt, tc->env);
        }

        case AST_IF: {
            /* Type check if statement */
            Type cond_type = check_expression(stmt->as.if_stmt.condition, tc->env);
            if (cond_type != TYPE_BOOL) {
                emit_context_error(
                    "E001 TYPE MISMATCH",
                    stmt->line,
                    stmt->column,
                    1,
                    "If condition must be a bool.",
                    "Ensure the condition expression evaluates to bool."
                );
                tc->has_error = true;
            }
            
            /* Type check then branch */
            if (stmt->as.if_stmt.then_branch) {
                check_statement(tc, stmt->as.if_stmt.then_branch);
            }
            
            /* Type check else branch if present */
            if (stmt->as.if_stmt.else_branch) {
                check_statement(tc, stmt->as.if_stmt.else_branch);
            }
            
            return TYPE_VOID;
        }

        case AST_COND: {
            /* Type check cond statement (just check as expression) */
            check_expression(stmt, tc->env);
            return TYPE_VOID;
        }

        case AST_MATCH: {
            /* Match used as a statement: type check arms as statements so return statements
             * inside match arms are checked against the current function's return type.
             * (The expression-mode match checker uses a temporary TypeChecker without
             * current_function_return_type initialized, which can produce spurious errors.)
            */
            stmt->as.match_expr.checked_scrutinee_type = TYPE_UNKNOWN;
            stmt->as.match_expr.scrutinee_type_checked = false;
            Type match_type = check_expression(stmt->as.match_expr.expr, tc->env);
            bool has_int_patterns_stmt;
            bool has_variant_patterns_stmt;
            match_arm_families(stmt, &has_int_patterns_stmt, &has_variant_patterns_stmt);

            /* Infer and store union type name for transpiler + variant binding metadata */
            const char *union_type_name = NULL;      /* base name for variant-field lookup */
            const char *union_base_name = NULL;      /* kept for binding metadata */
            char *union_concrete_name = NULL;        /* for transpiler: Result_int_string */
            TypeInfo *union_type_info = NULL;        /* For generic unions: Result<int, string> */
            ASTNode *match_expr_node = stmt->as.match_expr.expr;

            if (match_expr_node->type == AST_IDENTIFIER) {
                Symbol *sym = env_get_var(tc->env, match_expr_node->as.identifier);
                if (sym && sym->struct_type_name) {
                    union_type_name = sym->struct_type_name;
                }
                /* For generic unions, also extract TypeInfo */
                if (sym && sym->type_info) {
                    union_type_info = sym->type_info;
                    if (union_type_info->generic_name) union_type_name = union_type_info->generic_name;
                }
            } else if (match_expr_node->type == AST_UNION_CONSTRUCT) {
                union_type_name = match_expr_node->as.union_construct.union_name;
            } else if (match_expr_node->type == AST_STRUCT_LITERAL) {
                union_type_name = inline_variant_union(match_expr_node, tc->env);
            } else if (match_expr_node->type == AST_CALL) {
                Function *func = env_get_function(tc->env, match_expr_node->as.call.name);
                if (func && func->return_struct_type_name) {
                    union_type_name = func->return_struct_type_name;
                }
            } else if (match_expr_node->type == AST_FIELD_ACCESS) {
                const char *struct_name = get_struct_type_name(match_expr_node->as.field_access.object, tc->env);
                if (struct_name) {
                    StructDef *sdef = env_get_struct(tc->env, struct_name);
                    if (sdef && sdef->field_type_names) {
                        const char *field_name = match_expr_node->as.field_access.field_name;
                        for (int i = 0; i < sdef->field_count; i++) {
                            if (strcmp(sdef->field_names[i], field_name) == 0) {
                                if (sdef->field_types[i] == TYPE_UNION && sdef->field_type_names[i]) {
                                    union_type_name = sdef->field_type_names[i];
                                }
                                break;
                            }
                        }
                    }
                }
            }

            if (match_expr_node->type == AST_FIELD_ACCESS || match_expr_node->type == AST_CALL) {
                TypeInfo *field_info = try_get_expr_type_info(match_expr_node, tc->env);
                if (field_info && field_info->generic_name && env_get_union(tc->env, field_info->generic_name)) {
                    union_type_info = field_info;
                    union_type_name = field_info->generic_name;
                }
            }

            union_base_name = union_type_name;
            if (union_type_info && union_type_info->generic_name && union_type_info->type_param_count > 0) {
                union_base_name = union_type_info->generic_name;
                union_concrete_name = typeinfo_to_monomorphized_generic_name(union_type_info);
            }

            MatchDomain match_domain = check_match_domain(
                stmt, tc->env, match_type, has_int_patterns_stmt,
                has_variant_patterns_stmt, union_base_name);
            if (match_domain == MATCH_DOMAIN_INVALID) {
                free(union_concrete_name);
                return TYPE_VOID;
            }
            stmt->as.match_expr.checked_scrutinee_type = match_type;
            stmt->as.match_expr.scrutinee_type_checked = true;

            if (stmt->as.match_expr.union_type_name) {
                free(stmt->as.match_expr.union_type_name);
                stmt->as.match_expr.union_type_name = NULL;
            }
            if (union_concrete_name) {
                stmt->as.match_expr.union_type_name = union_concrete_name;
            } else if (union_base_name) {
                stmt->as.match_expr.union_type_name = strdup(union_base_name);
            }

            for (int i = 0; i < stmt->as.match_expr.arm_count; i++) {
                int arm_first_symbol = tc->env->symbol_count;
                const char *variant_name_s = stmt->as.match_expr.pattern_variants[i];

                /* Only add binding for non-wildcard, non-int-pattern, non-or-pattern arms */
                if (stmt->as.match_expr.pattern_bindings[i][0] &&
                    strcmp(stmt->as.match_expr.pattern_bindings[i], "_") != 0 &&
                    strcmp(variant_name_s, "_") != 0 &&
                    strncmp(variant_name_s, "INT:", 4) != 0 &&
                    strncmp(variant_name_s, "OR:", 3) != 0) {
                    Value binding_val = create_void();
                    env_define_var_with_type_info(tc->env,
                        stmt->as.match_expr.pattern_bindings[i],
                        TYPE_STRUCT, TYPE_UNKNOWN, union_type_info, false, binding_val);

                    if (union_base_name && tc->env->symbol_count > 0) {
                        Symbol *binding_sym = &tc->env->symbols[tc->env->symbol_count - 1];
                        char *type_name = malloc(strlen(union_base_name) + strlen(variant_name_s) + 2);
                        sprintf(type_name, "%s.%s", union_base_name, variant_name_s);
                        binding_sym->struct_type_name = type_name;

                        /* Ensure bindings participate in visibility disambiguation */
                        binding_sym->def_line = stmt->line;
                        binding_sym->def_column = stmt->column;
                    }
                }

                /* Unknown is not permission to emit a guard. */
                if (stmt->as.match_expr.guard_exprs)
                    check_match_guard(stmt->as.match_expr.guard_exprs[i], tc->env);

                ASTNode *arm = stmt->as.match_expr.arm_bodies[i];
                if (arm && arm->type == AST_BLOCK) {
                    check_statement(tc, arm);
                } else {
                    check_expression(arm, tc->env);
                }

                /* I retain emission metadata within its lexical arm only. */
                bound_scope_symbols(tc->env, arm_first_symbol, arm);
            }

            check_match_totality(stmt, tc->env, union_base_name, match_domain);

            return TYPE_VOID;
        }
        
        case AST_PREFIX_OP:
            /* Expression statement - operator expression used as statement has no effect */
            check_expression(stmt, tc->env);
            emit_context_error(
                "E034 EXPRESSION HAS NO EFFECT",
                stmt->line,
                stmt->column,
                1,
                "This operator expression is used as a statement but produces no side effect. The result is discarded.",
                "If you meant to call a function, use (function_name args). If you meant to assign, use set."
            );
            tc->has_error = true;
            return TYPE_VOID;
            
        case AST_CALL: {
            /* Check if this is a call to an extern function outside unsafe context */
            if (stmt->as.call.name) {
                Function *func = env_get_function(tc->env, stmt->as.call.name);
                if (func) {
                    /* Phase 3: Warn on calls to functions from unsafe modules if --warn-unsafe-calls is set */
                    if (tc->env->warn_unsafe_calls && func->module_name) {
                        /* Check if the function's module is unsafe */
                        ModuleInfo *mod = env_get_module(tc->env, func->module_name);
                        if (mod && mod->is_unsafe) {
                            fprintf(stderr, "Warning at line %d, column %d: Calling function '%s' from unsafe module '%s'\n",
                                    stmt->line, stmt->column, stmt->as.call.name, func->module_name);
                            fprintf(stderr, "  Note: Functions from unsafe modules may have safety implications\n");
                        }
                    }
                    
                    if (func->is_extern) {
                        /* Phase 3: Warn on FFI calls if --warn-ffi is set */
                        if (tc->env->warn_ffi) {
                            fprintf(stderr, "Warning at line %d, column %d: FFI call to extern function '%s'\n",
                                    stmt->line, stmt->column, stmt->as.call.name);
                            fprintf(stderr, "  Note: Extern functions perform arbitrary operations\n");
                        }
                        
                        /* Check if unsafe context is required */
                        if (!tc->in_unsafe_block && !tc->env->current_module_is_unsafe) {
                            fprintf(stderr, "Error at line %d, column %d: Call to extern function '%s' requires unsafe block or unsafe module\n",
                                    stmt->line, stmt->column, stmt->as.call.name);
                            fprintf(stderr, "  Note: Extern functions can perform arbitrary operations.\n");
                            fprintf(stderr, "  Hint: Either wrap the call in 'unsafe { ... }' or declare the module as 'unsafe module name { ... }'\n");
                            tc->has_error = true;
                        }
                    }
                }
            }
            /* Type check the call expression */
            check_expression(stmt, tc->env);
            return TYPE_VOID;
        }

        case AST_MODULE_QUALIFIED_CALL:
            check_expression(stmt, tc->env);
            return TYPE_VOID;

        case AST_FUNCTION: {
            /* Nested function definition (for closures).
             * Register it in the environment and type-check its body. */
            if (!stmt->as.function.is_extern) {
                Function func = {0};
                func.name = (char *)stmt->as.function.name;
                func.params = stmt->as.function.params;
                func.param_count = stmt->as.function.param_count;
                func.return_type = stmt->as.function.return_type;
                func.return_element_type = stmt->as.function.return_element_type;
                func.return_struct_type_name = stmt->as.function.return_struct_type_name;
                func.return_fn_sig = stmt->as.function.return_fn_sig;
                func.return_type_info = stmt->as.function.return_type_info;
                func.body = stmt->as.function.body;
                func.is_extern = false;
                func.is_pub = false;
                env_define_function(tc->env, func);

                /* Type-check the function body */
                if (stmt->as.function.body) {
                    for (int p = 0; p < stmt->as.function.param_count; p++) {
                        Value dummy_val = {0};
                        Parameter *param = &stmt->as.function.params[p];
                        env_define_var_with_type_info(tc->env, param->name,
                                      param->type, param->element_type,
                                      param->type_info, false, dummy_val);
                        Symbol *symbol = env_get_var(tc->env, param->name);
                        if (symbol) {
                            symbol->def_line = stmt->line;
                            symbol->def_column = stmt->column;
                            /* I use this declaration, not metadata copied from
                             * an unrelated earlier parameter with this name. */
                            free(symbol->struct_type_name);
                            symbol->struct_type_name = param->struct_type_name
                                ? strdup(param->struct_type_name) : NULL;
                        }
                    }
                    TypeChecker nested = *tc;
                    nested.loop_depth = 0;
                    nested.current_function_return_type = func.return_type;
                    nested.current_function_return_element_type = func.return_element_type;
                    nested.current_function_return_info = func.return_type_info;
                    nested.current_function_return_struct_name = func.return_struct_type_name;
                    check_statement(&nested, stmt->as.function.body);
                    tc->has_error = tc->has_error || nested.has_error;
                }
            }
            return TYPE_VOID;
        }

        case AST_HANDLE_EXPR:
            /* handle expression used as a statement — typecheck it */
            check_expression(stmt, tc->env);
            return TYPE_VOID;

        default: {
            /* Literals, identifiers, and other pure expressions used as statements */
            check_expression(stmt, tc->env);
            const char *kind = "expression";
            switch (stmt->type) {
                case AST_NUMBER:    kind = "numeric literal"; break;
                case AST_FLOAT:     kind = "float literal"; break;
                case AST_STRING:    kind = "string literal"; break;
                case AST_BOOL:      kind = "boolean literal"; break;
                case AST_IDENTIFIER: kind = "variable reference"; break;
                case AST_ARRAY_LITERAL: kind = "array literal"; break;
                case AST_STRUCT_LITERAL: kind = "struct literal"; break;
                case AST_FIELD_ACCESS: kind = "field access"; break;
                case AST_TUPLE_LITERAL: kind = "tuple literal"; break;
                case AST_TUPLE_INDEX: kind = "tuple index"; break;
                case AST_QUALIFIED_NAME: kind = "qualified name"; break;
                case AST_TRY_OP: kind = "try expression"; break;
                default: break;
            }
            char message[256];
            snprintf(message, sizeof(message),
                     "This %s is used as a statement but produces no side effect. The result is discarded.",
                     kind);
            emit_context_error(
                "E034 EXPRESSION HAS NO EFFECT",
                stmt->line,
                stmt->column,
                1,
                message,
                "If you meant to call a function, use (function_name args). If you meant to assign, use set."
            );
            tc->has_error = true;
            return TYPE_VOID;
        }
    }
}

/* Check program */
/* List of all built-in function names */
static const char *builtin_function_names[] = {
    /* Core */
    "range", "print", "println", "assert",
    /* Math */
    "abs", "min", "max", "sqrt", "pow", "floor", "ceil", "round",
    "sin", "cos", "tan", "atan2",
    /* Type casting */
    "float_from_bits", "float_to_bits", "cast_int", "cast_float", "cast_bool", "cast_string", "cast_bstring", "to_string", "null_opaque",
    /* String (C strings) */
    "str_length", "str_concat", "str_substring", "str_contains", "str_equals", "format",
    /* Bytes (array<u8>) */
    "bytes_from_string", "string_from_bytes",
    /* Binary strings (nl_string_t) */
    "bstr_new", "bstr_new_binary", "bstr_length", "bstr_concat", "bstr_substring",
    "bstr_equals", "bstr_byte_at", "bstr_validate_utf8", "bstr_utf8_length",
    "bstr_utf8_char_at", "bstr_to_cstr", "bstr_free",
    /* Advanced string operations */
    "char_at", "string_from_char",
    /* Character classification */
    "is_digit", "is_alpha", "is_alnum", "is_whitespace", "is_upper", "is_lower",
    /* Type conversions */
    "int_to_string", "string_to_int", "digit_value", "char_to_lower", "char_to_upper",
    /* Array */
    "at", "array_get", "array_length", "array_new", "array_set",
    "array_slice",
    /* Higher-order array functions */
    "map", "reduce",
    /* OS */
    "getcwd", "getenv", "exit",
    /* File I/O (stdlib functions) */
    "file_read", "file_read_bytes", "file_write", "file_append", "file_remove", "file_rename",
    "file_exists", "file_size",
    /* Temp helpers */
    "tmp_dir", "mktemp", "mktemp_dir",
    /* Directory operations */
    "dir_create", "dir_remove", "dir_list", "dir_exists", "chdir",
    /* Path operations */
    "path_isfile", "path_isdir", "path_join", "path_basename", "path_dirname",
    /* Process operations */
    "system",
    "process_run",
    /* List operations - list_int */
    "list_int_new", "list_int_with_capacity", "list_int_push", "list_int_pop",
    "list_int_get", "list_int_set", "list_int_insert", "list_int_remove",
    "list_int_length", "list_int_capacity", "list_int_is_empty", "list_int_clear",
    "list_int_free",
    /* List operations - list_string */
    "list_string_new", "list_string_with_capacity", "list_string_push", "list_string_pop",
    "list_string_get", "list_string_set", "list_string_insert", "list_string_remove",
    "list_string_length", "list_string_capacity", "list_string_is_empty", "list_string_clear",
    "list_string_free",
    /* List operations - list_Token */
    "nl_list_Token_new", "nl_list_Token_with_capacity", "nl_list_Token_push", "nl_list_Token_pop",
    "nl_list_Token_get", "nl_list_Token_set", "nl_list_Token_insert", "nl_list_Token_remove",
    "nl_list_Token_length", "nl_list_Token_capacity", "nl_list_Token_is_empty", "nl_list_Token_clear",
    "nl_list_Token_free"
};

static const int builtin_function_name_count = sizeof(builtin_function_names) / sizeof(char*);

static const char *g_typecheck_current_file = NULL;
static int g_typecheck_error_count = 0;

void typecheck_set_current_file(const char *path) {
    g_typecheck_current_file = path;
}

/* The environment needs the same notion for symbol visibility, but it is not
 * reachable from here -- callers that own the environment set it directly.
 * Kept as one concept with two owners rather than two concepts. */

static void emit_context_error(
    const char *title,
    int line,
    int column,
    int caret_len,
    const char *message,
    const char *hint
);

static bool read_source_line(const char *path, int target_line, char *buffer, size_t buffer_size) {
    if (!path || target_line <= 0 || !buffer || buffer_size == 0) {
        return false;
    }
    FILE *f = fopen(path, "r");
    if (!f) {
        return false;
    }
    int line = 1;
    while (fgets(buffer, (int)buffer_size, f)) {
        if (line == target_line) {
            fclose(f);
            return true;
        }
        line++;
    }
    fclose(f);
    return false;
}

static void print_error_header(const char *title, const char *file_path) {
    const char *file = file_path ? file_path : "";
    /* Use red for errors, yellow for warnings */
    const char *color_start = CSTART_ERROR;
    const char *color_end = CEND;

    /* Check if this is a warning */
    if (strstr(title, "WARNING") || strstr(title, "Warning")) {
        color_start = CSTART_WARNING;
    }

    fprintf(stderr, "%s-- %s %s", color_start, title, color_end);
    int title_len = (int)strlen(title);
    int file_len = (int)strlen(file);
    int dash_count = 54 - title_len - file_len;
    if (dash_count < 3) {
        dash_count = 3;
    }
    for (int i = 0; i < dash_count; i++) {
        fputc('-', stderr);
    }
    if (file_len > 0) {
        fprintf(stderr, " %s%s%s", CSTART_DIM, file, CEND);
    }
    fputc('\n', stderr);
}

static void print_error_context_line(int line, int column, int caret_len, const char *line_text) {
    if (!line_text) {
        return;
    }
    char line_prefix[32];
    int prefix_len = snprintf(line_prefix, sizeof(line_prefix), "%d| ", line);
    /* Print line number in dim color */
    fprintf(stderr, "%s%s%s%s", CSTART_DIM, line_prefix, CEND, line_text);
    size_t line_len = strlen(line_text);
    if (line_len == 0 || line_text[line_len - 1] != '\n') {
        fputc('\n', stderr);
    }

    if (column < 1) {
        column = 1;
    }
    if (caret_len < 1) {
        caret_len = 1;
    }

    /* Print spacing before caret */
    for (int i = 0; i < prefix_len + column - 1; i++) {
        fputc(' ', stderr);
    }
    /* Print caret in red */
    fprintf(stderr, "%s", CSTART_ERROR);
    for (int i = 0; i < caret_len; i++) {
        fputc('^', stderr);
    }
    fprintf(stderr, "%s\n", CEND);
}

static void emit_context_error(
    const char *title,
    int line,
    int column,
    int caret_len,
    const char *message,
    const char *hint
) {
    bool is_warning = title && (strstr(title, "WARNING") || strstr(title, "Warning"));

    /* Track error count for non-warning emissions */
    if (!is_warning) {
        g_typecheck_error_count++;
    }

    /* Populate JSON diagnostics for LSP / tooling when enabled */
    if (g_json_output_enabled) {
        /* Extract a stable id: "[E003] ..." or leading "E001 TYPE ...". */
        char code_buf[16] = "";
        if (title) {
            const char *lb = strchr(title, '[');
            if (lb) {
                const char *rb = strchr(lb, ']');
                if (rb && (rb - lb - 1) < (int)sizeof(code_buf)) {
                    size_t clen = (size_t)(rb - lb - 1);
                    memcpy(code_buf, lb + 1, clen);
                    code_buf[clen] = '\0';
                }
            }
            if (!code_buf[0] && title[0] == 'E') {
                size_t n = 1;
                while (title[n] >= '0' && title[n] <= '9') n++;
                if (n > 1 && n < sizeof(code_buf) && (title[n] == '\0' || title[n] == ' ')) {
                    memcpy(code_buf, title, n);
                    code_buf[n] = '\0';
                }
            }
        }
        const char *code = code_buf[0] ? code_buf : "E035";
        const char *msg  = message ? message : "";
        const char *hint_str = (hint && hint[0]) ? hint : NULL;
        if (is_warning)
            json_warning(code, msg, g_typecheck_current_file, line, column, hint_str);
        else
            json_error(code, msg, g_typecheck_current_file, line, column, hint_str);
    }

    if (g_typecheck_current_file) {
        print_error_header(title, g_typecheck_current_file);
    } else {
        fprintf(stderr, "%sError at line %d, column %d:%s %s\n",
                CSTART_ERROR, line, column, CEND, message);
        if (hint && hint[0] != '\0') {
            fprintf(stderr, "%sHint:%s %s\n", CSTART_HINT, CEND, hint);
        }
        return;
    }

    fprintf(stderr, "%s\n\n", message);
    char source_line[1024];
    if (read_source_line(g_typecheck_current_file, line, source_line, sizeof(source_line))) {
        print_error_context_line(line, column, caret_len, source_line);
    }
    if (hint && hint[0] != '\0') {
        fprintf(stderr, "%sHint:%s %s\n", CSTART_HINT, CEND, hint);
    }
}

/* Check if a function name is a built-in */
static bool is_builtin_name(const char *name) {
    for (int i = 0; i < builtin_function_name_count; i++) {
        if (strcmp(builtin_function_names[i], name) == 0) {
            return true;
        }
    }
    return false;
}

/* Register built-in functions in environment */
static void register_builtin_functions(Environment *env) {
    /* Check if already initialized to avoid duplicate registration */
    if (env->builtins_registered) {
        return;
    }
    env->builtins_registered = true;

    Function func = (Function){0};
    /* Important: zero-init so visibility/module_name pointers don't contain garbage. */
    func.is_pub = true;      /* Builtins are always accessible */
    func.module_name = NULL; /* Builtins are global */
    
    /* range(start: int, end: int) -> void (special - only for for-loops) */
    func.name = "range";
    func.params = NULL;  /* Special handling */
    func.param_count = 2;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* abs(x: int|float) -> int|float */
    func.name = "abs";
    func.params = NULL;  /* Accept int or float */
    func.param_count = 1;
    func.return_type = TYPE_INT;  /* Can also be float */
    func.return_type_info = NULL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* min(a: int|float, b: int|float) -> int|float */
    func.name = "min";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_INT;  /* Can also be float */
    func.return_type_info = NULL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* max(a: int|float, b: int|float) -> int|float */
    func.name = "max";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_INT;  /* Can also be float */
    func.return_type_info = NULL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* print(x: any) -> void */
    func.name = "print";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* println(x: any) -> void */
    func.name = "println";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* Advanced math functions */
    /* sqrt(x: int|float) -> float */
    func.name = "sqrt";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* pow(base: int|float, exponent: int|float) -> float */
    func.name = "pow";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_FLOAT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* floor(x: int|float) -> float */
    func.name = "floor";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* ceil(x: int|float) -> float */
    func.name = "ceil";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* round(x: int|float) -> float */
    func.name = "round";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* Trigonometric functions */
    /* sin(x: int|float) -> float */
    func.name = "sin";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* cos(x: int|float) -> float */
    func.name = "cos";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* tan(x: int|float) -> float */
    func.name = "tan";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* String operations */
    /* str_length(s: string) -> int */
    func.name = "str_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* str_concat(s1: string, s2: string) -> string */
    func.name = "str_concat";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* str_substring(s: string, start: int, length: int) -> string */
    func.name = "str_substring";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* str_contains(s: string, substr: string) -> bool */
    func.name = "str_contains";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* str_equals(s1: string, s2: string) -> bool */
    func.name = "str_equals";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);

    /* format(template: string, ...args) -> string  [variadic; handled specially above] */
    func.name = "format";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);

    /* Array operations */
    /* at(arr: array<T>, index: int) -> T */
    func.name = "at";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_UNKNOWN;  /* Will be determined by array element type */
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* array_length(arr: array<T>) -> int */
    func.name = "array_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* array_new(size: int, default: T) -> array<T> */
    func.name = "array_new";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_ARRAY;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* array_set(arr: mut array<T>, index: int, value: T) -> void */
    func.name = "array_set";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* map(arr: array<T>, transform: fn(T) -> T) -> array<T> */
    func.name = "map";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_ARRAY;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* reduce(arr: array<T>, initial: T, combine: fn(T, T) -> T) -> T */
    func.name = "reduce";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_UNKNOWN;  /* Will be type of initial value */
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* OS built-ins */
    func.name = "getcwd";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "getenv";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);

    /* dir_list(path: string) -> array<string>
     * Registered here so bytecode/VM programs (which lower dir_list to the
     * vm_dir_list extern) type-check without an unsafe FFI declaration. */
    func.name = "dir_list";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_ARRAY;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);

    /* process_run(command: string) -> array<string> ([exit_code, stdout, stderr])
     * Registered here so bytecode/VM programs can shell out through the
     * vm_process_run extern without an unsafe FFI declaration. */
    func.name = "process_run";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_ARRAY;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);

    /* Temp helpers */
    func.name = "tmp_dir";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);

    func.name = "mktemp";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);

    func.name = "mktemp_dir";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* Advanced string operations */
    func.name = "char_at";
    func.params = NULL;
    func.param_count = 2;  /* string, index */
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "string_from_char";
    func.params = NULL;
    func.param_count = 1;  /* char code */
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* Character classification */
    func.name = "is_digit";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "is_alpha";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "is_alnum";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "is_whitespace";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "is_upper";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "is_lower";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* Type conversions */
    func.name = "int_to_string";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "string_to_int";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "digit_value";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "char_to_lower";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "char_to_upper";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* list_int operations */
    func.name = "list_int_new";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_LIST_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_with_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_LIST_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_push";
    func.params = NULL;
    func.param_count = 2;  /* list, value */
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_pop";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_get";
    func.params = NULL;
    func.param_count = 2;  /* list, index */
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_set";
    func.params = NULL;
    func.param_count = 3;  /* list, index, value */
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_insert";
    func.params = NULL;
    func.param_count = 3;  /* list, index, value */
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_remove";
    func.params = NULL;
    func.param_count = 2;  /* list, index */
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_is_empty";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_clear";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_free";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* list_string operations */
    func.name = "list_string_new";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_LIST_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_with_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_LIST_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_push";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_pop";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_get";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_set";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_insert";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_remove";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRING;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_is_empty";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_clear";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_free";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* list_Token operations */
    func.name = "nl_list_Token_new";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_LIST_TOKEN;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_with_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_LIST_TOKEN;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_push";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_pop";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRUCT;  /* Returns Token struct */
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_get";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRUCT;  /* Returns Token struct */
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_set";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_insert";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_remove";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRUCT;  /* Returns Token struct */
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_is_empty";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_clear";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "nl_list_Token_free";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
}

/* Check if two functions have matching signatures */
static bool functions_match(Function *f1, Function *f2) {
    if (f1->param_count != f2->param_count) return false;
    if (f1->return_type != f2->return_type) return false;
    
    for (int i = 0; i < f1->param_count; i++) {
        if (f1->params[i].type != f2->params[i].type) return false;
    }
    
    return true;
}

/* I check only the root's shadows here, after its declarations and functions.
 * I retain inferred metadata just as the ordinary function checker does;
 * the bytecode emitter still needs it when choosing operand/array kinds. */
bool type_check_root_shadows(ASTNode *program, Environment *env) {
    if (ast_has_service_declaration(program)) {
        fprintf(stderr, "I have not resolved File service declarations for this consumer.\n");
        return false;
    }
    if (!program || program->type != AST_PROGRAM || !env) return false;
    bool ok = true;
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type != AST_SHADOW) continue;
        TypeChecker tc = {0};
        tc.env = env;
        tc.current_function_return_type = TYPE_VOID;
        check_statement(&tc, item->as.shadow.body);
        if (tc.has_error) ok = false;
    }
    return ok && g_typecheck_error_count == 0;
}

/* I check selected shadows in their source and authority context, then restore
 * the root context. Module declarations have already been checked by loading. */
bool type_check_shadow_scope(ASTNode *program, Environment *env, ModuleList *modules,
                             const char *input_file, bool include_imports) {
    if (!env) return false;
    char *root_owner = env->current_module;
    bool root_unsafe = env->current_module_is_unsafe;
    env_set_current_file(env, input_file);
    typecheck_set_current_file(input_file);
    bool typed = type_check_root_shadows(program, env);
    int count = include_imports && modules ? modules->count : 0;
    for (int i = 0; i < count && typed; i++) {
        const char *file = modules->module_paths[i];
        ASTNode *dependency = get_cached_module_ast(file);
        char *owner = module_program_name(dependency, file);
        env->current_module = owner;
        env->current_module_is_unsafe = false;
        if (dependency) {
            for (int j = 0; j < dependency->as.program.count; j++) {
                ASTNode *item = dependency->as.program.items[j];
                if (item->type == AST_IMPORT && item->as.import_stmt.is_unsafe)
                    env->current_module_is_unsafe = true;
            }
        }
        env_set_current_file(env, file);
        typecheck_set_current_file(file);
        typed = owner && type_check_root_shadows(dependency, env);
        env->current_module = root_owner;
        free(owner);
    }
    env_set_current_file(env, input_file);
    env->current_module_is_unsafe = root_unsafe;
    typecheck_set_current_file(input_file);
    return typed;
}

static bool register_effect_declaration(ASTNode *item, Environment *env) {
    /* Register algebraic effect definition */
    const char *eff_name = item->as.effect_decl.effect_name;
    if (env_get_effect(env, eff_name)) {
        emit_context_error("E031 DUPLICATE EFFECT", item->line, item->column, (int)strlen(eff_name),
            "Effect is already defined in this scope",
            "E013: effect names must be unique");
        return false;
    }

    EffectDef edef;
    edef.name = strdup(eff_name);
    edef.op_count = item->as.effect_decl.op_count;
    edef.is_pub = item->as.effect_decl.is_pub;
    edef.module_name = env->current_module ? strdup(env->current_module) : NULL;
    edef.ops = edef.op_count > 0 ? malloc(sizeof(EffectOp) * edef.op_count) : NULL;

    for (int j = 0; j < edef.op_count; j++) {
        edef.ops[j].name = strdup(item->as.effect_decl.op_names[j]);
        edef.ops[j].return_type = item->as.effect_decl.op_return_types[j];
        edef.ops[j].return_type_name = item->as.effect_decl.op_return_type_names[j]
            ? strdup(item->as.effect_decl.op_return_type_names[j]) : NULL;
        edef.ops[j].param_count = item->as.effect_decl.op_param_counts[j];
        if (edef.ops[j].param_count > 0) {
            edef.ops[j].params = malloc(sizeof(Parameter) * edef.ops[j].param_count);
            for (int k = 0; k < edef.ops[j].param_count; k++) {
                edef.ops[j].params[k] = item->as.effect_decl.op_params[j][k];
                if (item->as.effect_decl.op_params[j][k].name)
                    edef.ops[j].params[k].name = strdup(item->as.effect_decl.op_params[j][k].name);
            }
        } else {
            edef.ops[j].params = NULL;
        }
    }
    env_define_effect(env, edef);

    return true;
}

bool type_check(ASTNode *program, Environment *env) {
    if (ast_has_service_declaration(program)) {
        fprintf(stderr, "I have not resolved File service declarations for this consumer.\n");
        return false;
    }
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program AST\n");
        return false;
    }

    g_typecheck_error_count = 0;

    TypeChecker tc = {0};
    tc.env = env;
    tc.has_error = false;
    tc.warnings_enabled = true;  /* Enable unused variable warnings */
    tc.in_unsafe_block = false;  /* Start outside unsafe blocks */
    tc.loop_depth = 0;           /* Start outside loops */
    tc.current_function_return_type = TYPE_VOID;
    tc.current_function_return_struct_name = NULL;
    tc.current_function_return_element_type = TYPE_UNKNOWN;

    /* Register built-in functions */
    register_builtin_functions(env);

    /* Pre-pass: Process imports and register modules for introspection */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        if (item->type == AST_IMPORT) {
            /* Extract module name from path (e.g., "modules/sdl/sdl.nano" -> "sdl") */
            const char *path = item->as.import_stmt.module_path;
            char *module_name = NULL;
            
            /* Find last '/' and last '.' */
            const char *last_slash = strrchr(path, '/');
            const char *last_dot = strrchr(path, '.');
            
            if (last_slash && last_dot && last_dot > last_slash) {
                /* Extract between last slash and last dot */
                size_t name_len = last_dot - (last_slash + 1);
                module_name = strndup(last_slash + 1, name_len);
            } else if (last_slash) {
                /* No extension, use everything after last slash */
                module_name = strdup(last_slash + 1);
            } else if (last_dot) {
                /* No slash, use everything before dot */
                size_t name_len = last_dot - path;
                module_name = strndup(path, name_len);
            } else {
                /* No slash or dot, use entire path */
                module_name = strdup(path);
            }
            
            /* Register module for introspection */
            env_register_module(env, module_name, path, item->as.import_stmt.is_unsafe);
            
            /* Phase 3: Check warning flags for unsafe imports */
            if (item->as.import_stmt.is_unsafe) {
                if (env->forbid_unsafe) {
                    /* --forbid-unsafe: Error on unsafe module imports */
                    fprintf(stderr, "Error at line %d, column %d: Unsafe module import forbidden: '%s'\n",
                            item->line, item->column, path);
                    fprintf(stderr, "  Note: Compiled with --forbid-unsafe flag\n");
                    fprintf(stderr, "  Hint: Remove --forbid-unsafe or use safe modules only\n");
                    tc.has_error = true;
                } else if (env->warn_unsafe_imports) {
                    /* --warn-unsafe-imports: Warn on unsafe module imports */
                    fprintf(stderr, "Warning at line %d, column %d: Importing unsafe module: '%s'\n",
                            item->line, item->column, path);
                    fprintf(stderr, "  Note: This module requires unsafe context for FFI calls\n");
                }
            }
            
            free(module_name);
            
            /* Mark current context as unsafe if we're importing unsafe modules */
            if (item->as.import_stmt.is_unsafe) {
                env->current_module_is_unsafe = true;
            }
        }
    }

    if (!bind_nominal_records(program, env)) return false;

    /* First pass: collect all struct, enum, and function definitions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        /* Handle module declaration */
        if (item->type == AST_MODULE_DECL) {
            /* Set current module context */
            if (env->current_module) {
                free(env->current_module);
            }
            env->current_module = strdup(item->as.module_decl.name);
            /* TODO: Check if module is declared as unsafe */
            continue;
        }
        
        /* Skip imports - they're handled in pre-pass */
        if (item->type == AST_IMPORT) {
            continue;
        }
        
        if (item->type == AST_STRUCT_DEF) {
            const char *struct_name = item->as.struct_def.name;
            
            /* Check if struct already defined */
            if (env_get_struct_owned(env, struct_name, env->current_module)) {
                fprintf(stderr, "Error at line %d, column %d: Struct '%s' is already defined\n",
                        item->line, item->column, struct_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register the struct */
            StructDef sdef = {0};
            sdef.field_type_info = item->as.struct_def.field_type_info;
            sdef.name = strdup(struct_name);
            sdef.original_name = item->as.struct_def.original_name ? strdup(item->as.struct_def.original_name) : NULL;
            sdef.field_count = item->as.struct_def.field_count;
            
            /* Duplicate field names (AST will be freed) */
            sdef.field_names = malloc(sizeof(char*) * sdef.field_count);
            for (int j = 0; j < sdef.field_count; j++) {
                sdef.field_names[j] = strdup(item->as.struct_def.field_names[j]);
            }
            
            /* Duplicate field types */
            sdef.field_types = malloc(sizeof(Type) * sdef.field_count);
            for (int j = 0; j < sdef.field_count; j++) {
                sdef.field_types[j] = item->as.struct_def.field_types[j];
            }
            
            /* Duplicate field type names (for struct/union/enum types) */
            sdef.field_type_names = malloc(sizeof(char*) * sdef.field_count);
            for (int j = 0; j < sdef.field_count; j++) {
                if (item->as.struct_def.field_type_names && item->as.struct_def.field_type_names[j]) {
                    sdef.field_type_names[j] = strdup(item->as.struct_def.field_type_names[j]);
                    
                    /* Fix type if this is actually an enum (parser can't distinguish at parse time) */
                    if (sdef.field_types[j] == TYPE_STRUCT) {
                        /* Check if this name is an enum */
                        if (env_get_enum(env, item->as.struct_def.field_type_names[j])) {
                            sdef.field_types[j] = TYPE_ENUM;
                        }
                        /* Check if this name is a union */
                        else if (env_get_union(env, item->as.struct_def.field_type_names[j])) {
                            sdef.field_types[j] = TYPE_UNION;
                        }
                    }
                } else {
                    sdef.field_type_names[j] = NULL;
                }
            }
            
            /* Duplicate field element types (for array types) */
            sdef.field_element_types = malloc(sizeof(Type) * sdef.field_count);
            for (int j = 0; j < sdef.field_count; j++) {
                sdef.field_element_types[j] = item->as.struct_def.field_element_types[j];
                
                /* Register generic list instantiation for List<T> fields */
                if (sdef.field_types[j] == TYPE_LIST_GENERIC && sdef.field_type_names[j] != NULL) {
                    env_register_list_instantiation(env, sdef.field_type_names[j]);
                }
            }
            
            sdef.is_resource = item->as.struct_def.is_resource;  /* Propagate resource flag */
            sdef.is_extern = item->as.struct_def.is_extern;      /* Propagate extern flag */
sdef.is_pub = item->as.struct_def.is_pub;            /* Propagate public visibility flag */
            sdef.module_name = env->current_module ? strdup(env->current_module) : NULL;  /* Set module context */
            
            env_define_struct(env, sdef);

            /* Module introspection: track exported structs (public only) */
            if (sdef.is_pub && env->current_module) {
                env_add_module_exported_struct(env, env->current_module, sdef.original_name ? sdef.original_name : struct_name);
            }
            
        } else if (item->type == AST_UNION_DEF) {
            const char *union_name = item->as.union_def.name;
            
            /* Check if union already defined */
            if (env_get_union(env, union_name)) {
                fprintf(stderr, "Error at line %d, column %d: Union '%s' is already defined\n",
                        item->line, item->column, union_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register the union */
            UnionDef udef = {0};
            udef.name = strdup(union_name);
            udef.variant_count = item->as.union_def.variant_count;
            
            /* Duplicate variant names */
            udef.variant_names = malloc(sizeof(char*) * udef.variant_count);
            for (int j = 0; j < udef.variant_count; j++) {
                udef.variant_names[j] = strdup(item->as.union_def.variant_names[j]);
            }
            
            /* Duplicate variant field counts */
            udef.variant_field_counts = malloc(sizeof(int) * udef.variant_count);
            for (int j = 0; j < udef.variant_count; j++) {
                udef.variant_field_counts[j] = item->as.union_def.variant_field_counts[j];
            }
            
            /* Duplicate variant field names */
            udef.variant_field_names = malloc(sizeof(char**) * udef.variant_count);
            for (int j = 0; j < udef.variant_count; j++) {
                int field_count = udef.variant_field_counts[j];
                udef.variant_field_names[j] = malloc(sizeof(char*) * field_count);
                for (int k = 0; k < field_count; k++) {
                    udef.variant_field_names[j][k] = strdup(item->as.union_def.variant_field_names[j][k]);
                }
            }
            
            /* Duplicate variant field types */
            udef.variant_field_types = malloc(sizeof(Type*) * udef.variant_count);
            for (int j = 0; j < udef.variant_count; j++) {
                int field_count = udef.variant_field_counts[j];
                udef.variant_field_types[j] = malloc(sizeof(Type) * field_count);
                for (int k = 0; k < field_count; k++) {
                    udef.variant_field_types[j][k] = item->as.union_def.variant_field_types[j][k];
                }
            }
            
            /* Duplicate variant field type names */
            udef.variant_field_type_names = malloc(sizeof(char**) * udef.variant_count);
            for (int j = 0; j < udef.variant_count; j++) {
                int field_count = udef.variant_field_counts[j];
                udef.variant_field_type_names[j] = malloc(sizeof(char*) * field_count);
                for (int k = 0; k < field_count; k++) {
                    if (item->as.union_def.variant_field_type_names[j][k]) {
                        udef.variant_field_type_names[j][k] = strdup(item->as.union_def.variant_field_type_names[j][k]);
                    } else {
                        udef.variant_field_type_names[j][k] = NULL;
                    }
                }
            }
            
            /* Each registered declaration owns an independent payload tree. */
            udef.variant_field_type_info = calloc((size_t)udef.variant_count, sizeof(TypeInfo **));
            for (int j = 0; j < udef.variant_count; ++j) {
                int fields = udef.variant_field_counts[j];
                udef.variant_field_type_info[j] = calloc((size_t)fields, sizeof(TypeInfo *));
                for (int k = 0; k < fields; ++k)
                    if (item->as.union_def.variant_field_type_info && item->as.union_def.variant_field_type_info[j])
                        udef.variant_field_type_info[j][k] = copy_payload_type_info(item->as.union_def.variant_field_type_info[j][k]);
            }

            /* Copy generic parameters if present */
            udef.generic_param_count = item->as.union_def.generic_param_count;
            if (udef.generic_param_count > 0) {
                udef.generic_params = malloc(sizeof(char*) * udef.generic_param_count);
                for (int j = 0; j < udef.generic_param_count; j++) {
                    udef.generic_params[j] = strdup(item->as.union_def.generic_params[j]);
                }
            } else {
                udef.generic_params = NULL;
            }
            
            /* Set module visibility */
            udef.is_pub = item->as.union_def.is_pub;
            udef.module_name = env->current_module ? strdup(env->current_module) : NULL;
            udef.is_extern = item->as.union_def.is_extern;
            
            env_define_union(env, udef);
            
        } else if (item->type == AST_OPAQUE_TYPE) {
            /* Register opaque type */
            const char *type_name = item->as.opaque_type.name;
            
            /* The nominal prepass checked local duplicates and staged all rows. */
            if (!env_define_opaque_type(env, type_name)) tc.has_error = true;

        } else if (item->type == AST_EFFECT_DECL) {
            if (!register_effect_declaration(item, env)) tc.has_error = true;

        } else if (item->type == AST_ENUM_DEF) {
            /* Defensive check: ensure item and enum_def fields are valid */
            if (!item) {
                fprintf(stderr, "Error: NULL AST item in enum processing\n");
                tc.has_error = true;
                continue;
            }
            
            const char *enum_name = item->as.enum_def.name;
            assert(enum_name != NULL); /* Parser should never create enum with NULL name */
            
            if (!enum_name) {
                safe_fprintf(stderr, "Error at line %d, column %d: Enum definition has NULL name\n",
                        item->line, item->column);
                tc.has_error = true;
                continue;
            }
            
            /* Check if enum already defined - defensive check for NULL name */
            if (enum_name && env_get_enum(env, enum_name)) {
                safe_fprintf(stderr, "Error at line %d, column %d: Enum '%s' is already defined\n",
                        item->line, item->column, safe_format_string(enum_name));
                tc.has_error = true;
                continue;
            }
            
            /* Register the enum */
            EnumDef edef;
            edef.name = enum_name ? strdup(enum_name) : NULL;
            if (!edef.name) {
                fprintf(stderr, "Error: Failed to allocate memory for enum name\n");
                tc.has_error = true;
                continue;
            }
            edef.variant_count = item->as.enum_def.variant_count;
            if (edef.variant_count <= 0) {
                safe_fprintf(stderr, "Error: Enum '%s' has invalid variant count: %d\n", safe_format_string(enum_name), edef.variant_count);
                free(edef.name);
                tc.has_error = true;
                continue;
            }
            
            /* Check if variant_names array exists in AST */
            if (!item->as.enum_def.variant_names) {
                safe_fprintf(stderr, "Error: Enum '%s' has NULL variant_names array\n", safe_format_string(enum_name));
                free(edef.name);
                tc.has_error = true;
                continue;
            }
            
            /* Duplicate variant names (AST will be freed) */
            edef.variant_names = malloc(sizeof(char*) * edef.variant_count);
            if (!edef.variant_names) {
                fprintf(stderr, "Error: Failed to allocate memory for enum variant names\n");
                free(edef.name);
                tc.has_error = true;
                continue;
            }
            for (int j = 0; j < edef.variant_count; j++) {
                if (j < item->as.enum_def.variant_count && 
                    item->as.enum_def.variant_names && 
                    item->as.enum_def.variant_names[j]) {
                    const char *src_name = item->as.enum_def.variant_names[j];
                    if (src_name) {
                        edef.variant_names[j] = strdup(src_name);
                        if (!edef.variant_names[j]) {
                            fprintf(stderr, "Error: Failed to duplicate variant name at index %d\n", j);
                            edef.variant_names[j] = NULL;
                        }
                    } else {
                        safe_fprintf(stderr, "Error: Enum '%s' has NULL variant name at index %d\n", safe_format_string(enum_name), j);
                        edef.variant_names[j] = NULL;
                    }
                } else {
                    safe_fprintf(stderr, "Error: Enum '%s' has NULL variant name at index %d\n", safe_format_string(enum_name), j);
                    edef.variant_names[j] = NULL;
                }
            }
            
            /* Duplicate variant values */
            edef.variant_values = malloc(sizeof(int) * edef.variant_count);
            if (!edef.variant_values) {
                fprintf(stderr, "Error: Failed to allocate memory for enum variant values\n");
                free(edef.name);
                for (int j = 0; j < edef.variant_count; j++) {
                    free(edef.variant_names[j]);
                }
                free(edef.variant_names);
                tc.has_error = true;
                continue;
            }
            if (item->as.enum_def.variant_values) {
                for (int j = 0; j < edef.variant_count; j++) {
                    edef.variant_values[j] = item->as.enum_def.variant_values[j];
                }
            } else {
                /* No explicit values - use index as value */
                for (int j = 0; j < edef.variant_count; j++) {
                    edef.variant_values[j] = j;
                }
            }
            
            /* Set module visibility */
            edef.is_pub = item->as.enum_def.is_pub;
            edef.module_name = env->current_module ? strdup(env->current_module) : NULL;
            edef.is_extern = item->as.enum_def.is_extern;
            
            env_define_enum(env, edef);
            
        } else if (item->type == AST_ASYNC_FN && item->as.async_fn.function &&
                   item->as.async_fn.function->type == AST_FUNCTION) {
            /* async fn: unwrap to inner function for registration */
            item = item->as.async_fn.function;
            /* fall through to function registration below */
            goto register_function_pass1;
        } else if (item->type == AST_EFFECT_DECL) {
            /* Register algebraic effect in the environment's effect registry */
            env_effect_register(env, item);


        } else if (item->type == AST_FUNCTION) {
register_function_pass1:;
            const char *func_name = item->as.function.name;

            /* Check if function name collides with built-in (but allow extern functions) */
            if (!item->as.function.is_extern && is_builtin_name(func_name)) {
                fprintf(stderr, "Error at line %d, column %d: Cannot redefine built-in function '%s'\n",
                        item->line, item->column, func_name);
                fprintf(stderr, "  Built-in functions cannot be shadowed\n");
                fprintf(stderr, "  Choose a different function name\n");
                tc.has_error = true;
                continue;  /* Skip this function */
            }
            
            /* Check if function is already defined */
            Function *existing = env_get_function(env, func_name);
            /* I distinguish an imported name from a duplicate in this module. */
            if (existing && !existing->is_extern && !item->as.function.is_extern && existing->module_name &&
                (!env->current_module || strcmp(existing->module_name, env->current_module) != 0)) {
                existing = NULL;
            }
            if (existing) {
                /* If both are extern and signatures match, it's fine (idempotent) */
                if (item->as.function.is_extern && existing->is_extern) {
                    /* Create a temporary function object for matching */
                    Function current = (Function){0};
                    current.param_count = item->as.function.param_count;
                    current.params = item->as.function.params;
                    current.return_type = item->as.function.return_type;
                    /* Note: return_struct_type_name match not fully implemented here */
                    
                    if (functions_match(&current, existing)) {
                        continue; /* Skip registration, already there and matches */
                    }
                }

                /* Extern functions cannot be redefined or shadowed */
                if (existing->is_extern) {
                    fprintf(stderr, "Error at line %d, column %d: Extern function '%s' cannot be redefined\n",
                            item->line, item->column, func_name);
                    fprintf(stderr, "  Extern functions are first-class and cannot be shadowed\n");
                    fprintf(stderr, "  Previous extern declaration at line %d, column %d\n",
                            item->line, item->column);  /* Note: we don't track extern line numbers well */
                    tc.has_error = true;
                    continue;
                }
                /* Regular functions cannot be redefined */
                if (existing->body != NULL) {
                    fprintf(stderr, "Error at line %d, column %d: Function '%s' is already defined\n",
                            item->line, item->column, func_name);
                    fprintf(stderr, "  Previous definition at line %d, column %d\n",
                            existing->body->line, existing->body->column);
                    tc.has_error = true;
                    continue;
                }
                /* Regular functions cannot shadow extern functions */
                if (!item->as.function.is_extern && existing->is_extern) {
                    fprintf(stderr, "Error at line %d, column %d: Function '%s' cannot shadow extern function\n",
                            item->line, item->column, func_name);
                    fprintf(stderr, "  Extern functions are first-class and cannot be shadowed\n");
                    fprintf(stderr, "  Choose a different function name\n");
                    tc.has_error = true;
                    continue;
                }
            }
            
            /* Define the function */
            /* First, resolve parameter types (enum -> int, union -> TYPE_UNION) */
            for (int j = 0; j < item->as.function.param_count; j++) {
                if (item->as.function.params[j].type == TYPE_STRUCT &&
                    item->as.function.params[j].struct_type_name) {
                    /* Check if this is actually a union */
                    if (env_get_union(env, item->as.function.params[j].struct_type_name)) {
                        item->as.function.params[j].type = TYPE_UNION;
                    }
                    /* Check if this is actually an enum */
                    else if (env_get_enum(env, item->as.function.params[j].struct_type_name)) {
                        /* This is an enum, treat as int */
                        item->as.function.params[j].type = TYPE_INT;
                    }
                }
            }
            
            /* Resolve return type */
            Type return_type = item->as.function.return_type;
            if (return_type == TYPE_STRUCT && item->as.function.return_struct_type_name) {
                /* Check if this is actually a union */
                if (env_get_union(env, item->as.function.return_struct_type_name)) {
                    return_type = TYPE_UNION;
                }
                /* Check if this is actually an enum */
                else if (env_get_enum(env, item->as.function.return_struct_type_name)) {
                    return_type = TYPE_INT;
                }
            }
            
            Function func = (Function){0};
            func.source_file = env_current_file(env);
            func.name = env_own_checker_allocation(env, strdup(func_name));  /* Create copy to avoid const qualifier warning */
            func.params = item->as.function.params;
            func.param_count = item->as.function.param_count;
            func.return_type = return_type;
            func.return_element_type = item->as.function.return_element_type;
            func.return_type_info = NULL;
            func.return_struct_type_name = item->as.function.return_struct_type_name;
            func.return_fn_sig = item->as.function.return_fn_sig;  /* Store function signature for TYPE_FUNCTION returns */
            func.return_type_info = item->as.function.return_type_info;  /* Store tuple type info for TYPE_TUPLE returns */
            func.body = item->as.function.body;
            func.shadow_test = NULL;
            func.is_extern = item->as.function.is_extern;
            func.is_gpu = item->as.function.is_gpu;  /* Store GPU annotation */
            func.is_pure = item->as.function.is_pure;  /* Store purity annotation */
            func.is_pub = item->as.function.is_pub;  /* Store visibility */
            
            /* Store module context with independent copy */
            func.module_name = NULL;
            
            /* Try to find module declaration in this program to get fresh copy from AST */
            const char *module_name_from_ast = NULL;
            for (int m = 0; m < program->as.program.count && !module_name_from_ast; m++) {
                if (program->as.program.items[m]->type == AST_MODULE_DECL) {
                    module_name_from_ast = program->as.program.items[m]->as.module_decl.name;
                    /* Validate AST string */
                    bool valid = true;
                    for (int c = 0; c < 64 && module_name_from_ast[c]; c++) {
                        if ((unsigned char)module_name_from_ast[c] < 32 || 
                            (unsigned char)module_name_from_ast[c] >= 127) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        func.module_name = env_own_checker_allocation(env, strdup(module_name_from_ast));
                    }
                    break;
                }
            }
            
            /* Fallback to env->current_module if no valid AST module name */
            if (!func.module_name && env->current_module) {
                bool valid_module_name = true;
                for (const char *p = env->current_module; *p && valid_module_name; p++) {
                    unsigned char c = (unsigned char)*p;
                    if (c < 32 || c >= 127) {
                        valid_module_name = false;
                    }
                }
                if (valid_module_name) {
                    func.module_name = env_own_checker_allocation(env, strdup(env->current_module));
                }
            }

            env_define_function(env, func);
            register_native_function_context(env, &func);

            /* Module introspection: track exported functions (public only) */
            if (item->as.function.is_pub && env->current_module) {
                env_add_module_exported_function(env, env->current_module, func_name);
            }
            
            /* Trace function definition */
            if (!func.is_extern) {
                trace_function_def(func_name, func.params, func.param_count,
                                  func.return_type, item->line, item->column);
            }
        }
    }

    /* I retain concrete field-only union instances after all declarations
     * are visible, including forward declarations in the same module. */
    for (int record = 0; record < env->struct_count; ++record) {
        StructDef *definition = &env->structs[record];
        char *saved_module = env->current_module;
        env->current_module = definition->module_name;
        for (int field = 0; definition->field_type_info && field < definition->field_count; ++field)
            register_native_union_context(env, definition->field_type_info[field], 0);
        env->current_module = saved_module;
    }
    for (int index = 0; index < env->union_count; ++index) {
        UnionDef *definition = &env->unions[index];
        if (definition->generic_param_count) continue;
        char *saved_module = env->current_module;
        env->current_module = definition->module_name;
        for (int arm = 0; definition->variant_field_type_info && arm < definition->variant_count; ++arm)
            for (int field = 0; definition->variant_field_type_info[arm] && field < definition->variant_field_counts[arm]; ++field)
                register_native_union_context(env, definition->variant_field_type_info[arm][field], 0);
        env->current_module = saved_module;
    }

    /* Second pass: link shadow tests to functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        /* Skip imports - they're handled separately */
        if (item->type == AST_IMPORT) {
            continue;
        }
        
        if (item->type == AST_SHADOW) {
            Function *func = env_get_function(env, item->as.shadow.function_name);
            if (!func) {
                fprintf(stderr, "Error at line %d, column %d: Shadow test for undefined function '%s'\n",
                        item->line, item->column, item->as.shadow.function_name);
                tc.has_error = true;
            } else if (func->is_extern) {
                /* Extern functions cannot have shadow tests - they're C functions */
                fprintf(stderr, "Error at line %d, column %d: Shadow test cannot be attached to extern function '%s'\n",
                        item->line, item->column, item->as.shadow.function_name);
                fprintf(stderr, "  Extern functions are C functions and cannot be tested in the interpreter\n");
                fprintf(stderr, "  Remove the shadow test or test a wrapper function instead\n");
                tc.has_error = true;
            } else {
                func->shadow_test = item->as.shadow.body;
            }
        }
    }
    
    /* Process top-level constants/variables (before type checking functions) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_LET) {
            /* I provide declared constructor context before checking the initializer. */
            prepare_map_initializer(&tc, item);
            check_concrete_union_arrays(env, item->as.let.type_info, item->as.let.value, 0);
            Type value_type = check_expression(item->as.let.value, env);
            check_global_ownership(env, item, &tc.has_error);
            if (!check_record_array_contract(env, item->as.let.var_type, item->as.let.element_type,
                    item->as.let.type_name, item->as.let.value)) tc.has_error = true;
            if (item->as.let.var_type == TYPE_ARRAY &&
                item->as.let.element_type != TYPE_UNKNOWN &&
                item->as.let.value->type == AST_ARRAY_LITERAL) {
                check_array_literal_annotation(&tc, item->as.let.value,
                                               item->as.let.element_type, item->as.let.type_name);
            }
            
            /* Verify it matches the declared type */
            if (item->as.let.var_type != value_type) {
                fprintf(stderr, "Error at line %d, column %d: Constant '%s' type mismatch (declared %s, got %s)\n",
                        item->line, item->column,
                        item->as.let.name,
                        type_to_string(item->as.let.var_type),
                        type_to_string(value_type));
                tc.has_error = true;
                continue;
            }
            
            /* Add constant to environment */
            Value val = create_void();  /* Placeholder value for type checking */

            /* I preserve explicit function signatures just as in local bindings. */
            if (!retain_let_function_type(&tc, item, item->as.let.var_type)) continue;
            /* Preserve element type / generic type info for arrays and other complex types */
            env_define_var_with_type_info(env,
                                         item->as.let.name,
                                         item->as.let.var_type,
                                         item->as.let.element_type,
                                         item->as.let.type_info,
                                         item->as.let.is_mut,
                                         val);

            /* Preserve struct/union type name metadata for globals */
            Symbol *sym = env_get_var(env, item->as.let.name);
            if (sym) {
                sym->is_global = true;
                sym->def_line = item->line;
                sym->def_column = item->column;
            }
            if (sym && item->as.let.type_name) {
                if (sym->struct_type_name) free(sym->struct_type_name);
                sym->struct_type_name = NULL;

                if (item->as.let.var_type == TYPE_STRUCT || item->as.let.var_type == TYPE_UNION) {
                    sym->struct_type_name = strdup(item->as.let.type_name);
                    mark_variable_as_resource_if_needed(env, item->as.let.name, item->as.let.type_name);
                } else if (item->as.let.var_type == TYPE_ARRAY && item->as.let.element_type == TYPE_STRUCT) {
                    sym->struct_type_name = strdup(item->as.let.type_name);
                }
            }
        }
    }

    /* Third pass: type check all functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        /* Unwrap async fn to inner function for body checking */
        if (item->type == AST_ASYNC_FN && item->as.async_fn.function &&
            item->as.async_fn.function->type == AST_FUNCTION)
            item = item->as.async_fn.function;
        if (item->type == AST_FUNCTION) {
            /* Skip extern functions - they have no body to check */
            if (item->as.function.is_extern) {
                check_function_ownership(env, item, &tc.has_error);
                continue;
            }
            
            /* Get the resolved return type from the function definition in env */
            Function *func_def = env_get_function(env, item->as.function.name);
            tc.current_function_return_type = func_def ? func_def->return_type : item->as.function.return_type;
            tc.current_function_return_struct_name = func_def ? func_def->return_struct_type_name : NULL;
            tc.current_function_return_element_type = func_def ? func_def->return_element_type : TYPE_UNKNOWN;
            tc.current_function_return_info = func_def ? func_def->return_type_info : NULL;
            
            /* Register generic union instantiation for function return type */
            if (item->as.function.return_type == TYPE_UNION &&
                item->as.function.return_type_info &&
                item->as.function.return_type_info->generic_name &&
                item->as.function.return_type_info->type_param_count > 0) {
                
                TypeInfo *info = item->as.function.return_type_info;
                if (opaque_type_info_present(info)) {
                    register_native_union_context(tc.env, info, 0);
                } else {
                char **type_names = malloc(sizeof(char*) * info->type_param_count);
                
                for (int ti = 0; ti < info->type_param_count; ti++) {
                    type_names[ti] = typeinfo_to_generic_arg_name(info->type_params[ti]);
                }
                
                /* Register this instantiation for code generation */
                env_register_union_instantiation(tc.env, info->generic_name,
                                                (const char**)type_names,
                                                info->type_param_count);
                
                /* Free type names */
                for (int ti = 0; ti < info->type_param_count; ti++) {
                    free(type_names[ti]);
                }
                free(type_names);
                }
            }

            /* Register HashMap<K,V> instantiation for function return type */
            if (item->as.function.return_type == TYPE_HASHMAP &&
                item->as.function.return_type_info) {
                TypeInfo *info = item->as.function.return_type_info;
                Type key_t = TYPE_UNKNOWN;
                Type val_t = TYPE_UNKNOWN;
                if (!hashmap_extract_kv(info, &key_t, &val_t)) {
                    fprintf(stderr, "Error at line %d, column %d: Invalid HashMap return type annotation\n",
                            item->line, item->column);
                    tc.has_error = true;
                } else {
                    if (!(key_t == TYPE_INT || key_t == TYPE_STRING)) {
                        fprintf(stderr, "Error at line %d, column %d: HashMap key type must be int or string (got %s)\n",
                                item->line, item->column, type_to_string(key_t));
                        tc.has_error = true;
                    }
                    if (!(val_t == TYPE_INT || val_t == TYPE_STRING)) {
                        fprintf(stderr, "Error at line %d, column %d: HashMap value type must be int or string (got %s)\n",
                                item->line, item->column, type_to_string(val_t));
                        tc.has_error = true;
                    }
                    if (!tc.has_error) {
                        char *key_name = typeinfo_to_generic_arg_name(info->type_params[0]);
                        char *val_name = typeinfo_to_generic_arg_name(info->type_params[1]);
                        env_register_hashmap_instantiation(tc.env, key_name, val_name);
                        free(key_name);
                        free(val_name);
                    }
                }
            }

            /* Save current symbol count for scope restoration */
            int saved_symbol_count = env->symbol_count;

            /* Add parameters to environment (create a scope) */
            for (int j = 0; j < item->as.function.param_count; j++) {
                Value val = create_void();
                Type param_type = item->as.function.params[j].type;
                Type element_type = item->as.function.params[j].element_type;  /* Get actual element type from parameter */
                TypeInfo *param_type_info = item->as.function.params[j].type_info;  /* Get TypeInfo for generic types */

                /* Register HashMap<K,V> instantiation for parameters */
                if (param_type == TYPE_HASHMAP && param_type_info) {
                    Type key_t = TYPE_UNKNOWN;
                    Type val_t = TYPE_UNKNOWN;
                    if (!hashmap_extract_kv(param_type_info, &key_t, &val_t)) {
                        fprintf(stderr, "Error at line %d, column %d: Invalid HashMap parameter type annotation\n",
                                item->line, item->column);
                        tc.has_error = true;
                    } else {
                        if (!(key_t == TYPE_INT || key_t == TYPE_STRING)) {
                            fprintf(stderr, "Error at line %d, column %d: HashMap key type must be int or string (got %s)\n",
                                    item->line, item->column, type_to_string(key_t));
                            tc.has_error = true;
                        }
                        if (!(val_t == TYPE_INT || val_t == TYPE_STRING)) {
                            fprintf(stderr, "Error at line %d, column %d: HashMap value type must be int or string (got %s)\n",
                                    item->line, item->column, type_to_string(val_t));
                            tc.has_error = true;
                        }
                        if (!tc.has_error) {
                            char *key_name = typeinfo_to_generic_arg_name(param_type_info->type_params[0]);
                            char *val_name = typeinfo_to_generic_arg_name(param_type_info->type_params[1]);
                            env_register_hashmap_instantiation(tc.env, key_name, val_name);
                            free(key_name);
                            free(val_name);
                        }
                    }
                }
                
                /* For array parameters, use the element type from the parameter definition */
                if (param_type == TYPE_ARRAY && element_type == TYPE_UNKNOWN) {
                    element_type = TYPE_INT;  /* Fallback to TYPE_INT if not specified */
                }
                
                /* For parameters with TypeInfo (generics, function types, etc.), use full type info */
                if (param_type_info) {
                    env_define_var_with_type_info(env, item->as.function.params[j].name, param_type, element_type, param_type_info, false, val);
                }
                /* For function parameters, create TypeInfo with signature */
                else if (param_type == TYPE_FUNCTION && item->as.function.params[j].fn_sig) {
                    TypeInfo *type_info = env_own_checker_allocation(env, malloc(sizeof(TypeInfo)));
                    memset(type_info, 0, sizeof(TypeInfo));
                    type_info->base_type = TYPE_FUNCTION;
                    type_info->fn_sig = item->as.function.params[j].fn_sig;
                    env_define_var_with_type_info(env, item->as.function.params[j].name, param_type, TYPE_UNKNOWN, type_info, false, val);
                } else {
                    env_define_var_with_element_type(env, item->as.function.params[j].name,
                                 param_type, element_type, false, val);
                }
                
                /* Store type name for struct/union parameters */
                Symbol *param_sym = env_get_var(env, item->as.function.params[j].name);
                if (param_sym) {
                    /* Mark parameter as defined at the function definition line so later passes
                     * (e.g., transpilation) can disambiguate identical local names across functions.
                     */
                    param_sym->def_line = item->line;
                    param_sym->def_column = item->column;

                    if ((param_type == TYPE_STRUCT || param_type == TYPE_UNION || param_type == TYPE_BORROW_SHARED || param_type == TYPE_BORROW_MUT) &&
                        item->as.function.params[j].struct_type_name) {
                        param_sym->struct_type_name = strdup(item->as.function.params[j].struct_type_name);
                    }
                    /* For generic unions with TypeInfo, use the generic_name as struct_type_name */
                    else if (param_type == TYPE_UNION && param_type_info && param_type_info->generic_name) {
                        param_sym->struct_type_name = strdup(param_type_info->generic_name);
                    }
                }
            }

            /* Check function body */
            check_statement(&tc, item->as.function.body);
            bound_scope_symbols(env, saved_symbol_count, item->as.function.body);
            check_function_ownership(env, item, &tc.has_error);

            /* Purity check: verify pure fn body obeys purity rules */
            if (item->as.function.is_pure) {
                check_purity(item->as.function.body, env, item->as.function.name);
            }

            /* Extern parameters describe the C ABI; they have no body in which
             * they could be used. */
            if (!item->as.function.is_extern) {
                check_unused_variables(&tc, saved_symbol_count);
            }

            /* DON'T restore environment - transpiler needs these symbols! */
            /* The old code removed function-local symbols after typechecking:
             *   env->symbol_count = saved_symbol_count;
             * This caused array<struct> to fail because transpiler couldn't find
             * the struct_type_name metadata. Now we keep all symbols so transpiler
             * can access type information. C's function-local scope prevents collisions.
             */

            /* Verify function has shadow test (skip for extern functions, main, and functions that use extern functions) */
            Function *func = env_get_function(env, item->as.function.name);
            if (!env->suppress_shadow_warnings && !env->gpu_target &&
                !func->is_extern && !func->shadow_test &&
                strcmp(item->as.function.name, "main") != 0 &&
                strncmp(item->as.function.name, "__lambda_", 9) != 0) {
                /* Check if function body uses extern functions - if so, shadow test is optional */
                bool uses_extern = func->body && contains_extern_calls(func->body, env);
                if (!uses_extern) {
                    fprintf(stderr, "%sWarning:%s Function '%s' is missing a shadow test\n",
                            CSTART_WARNING, CEND, item->as.function.name);
                    /* Don't fail - just warn */
                    /* tc.has_error = true; */
                }
            }
        }
    }

    /* Post-pass: Update module FFI tracking based on loaded functions */
    for (int i = 0; i < env->module_count; i++) {
        ModuleInfo *mod = &env->modules[i];
        if (!mod || !mod->name) continue;
        
        /* Check if any functions in this module are extern */
        for (int f = 0; f < env->function_count; f++) {
            Function *func = &env->functions[f];
            if (func->is_extern && func->module_name && 
                strcmp(func->module_name, mod->name) == 0) {
                mod->has_ffi = true;
                break;
            }
        }
    }
    
    /* Verify main function exists (not required for GPU/PTX targets) */
    Function *main_func = env_get_function(env, "main");
    if (!main_func && !(env && env->gpu_target)) {
        fprintf(stderr, "Error: Program must define a 'main' function\n");
        fprintf(stderr, "  Hint: Add 'fn main() -> int { ... return 0 }' to your program.\n");
        tc.has_error = true;
    } else if (main_func && main_func->return_type != TYPE_INT) {
        fprintf(stderr, "Error: 'main' function must return int\n");
        fprintf(stderr, "  Hint: Change the return type declaration to '-> int'.\n");
        tc.has_error = true;
    }

    return !tc.has_error && !env->opaque_resolution_failed && g_typecheck_error_count == 0;
}

/* Type check a module (without requiring main function) */
bool type_check_module(ASTNode *program, Environment *env) {
    if (ast_has_service_declaration(program)) {
        fprintf(stderr, "I have not resolved File service declarations for this consumer.\n");
        return false;
    }
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program AST\n");
        return false;
    }

    g_typecheck_error_count = 0;

    TypeChecker tc = {0};
    tc.env = env;
    tc.has_error = false;
    tc.warnings_enabled = true;
    tc.in_unsafe_block = false;  /* Start outside unsafe blocks */
    tc.loop_depth = 0;           /* Start outside loops */

    /* Register built-in functions */
    register_builtin_functions(env);

    if (!bind_nominal_records(program, env)) return false;

    /* First pass: collect all struct, enum, and function definitions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        /* Handle module declaration */
        if (item->type == AST_MODULE_DECL) {
            /* Set current module context */
            if (env->current_module) {
                free(env->current_module);
            }
            env->current_module = strdup(item->as.module_decl.name);
            continue;
        }
        
        /* Skip imports - they're handled separately */
        if (item->type == AST_IMPORT) {
            continue;
        }
        
        if (item->type == AST_STRUCT_DEF) {
            const char *struct_name = item->as.struct_def.name;
            
            /* Check if struct already defined */
            if (env_get_struct_owned(env, struct_name, env->current_module)) {
                fprintf(stderr, "Error at line %d, column %d: Struct '%s' is already defined\n",
                        item->line, item->column, struct_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register the struct */
            StructDef sdef = {0};
            sdef.field_type_info = item->as.struct_def.field_type_info;
            sdef.name = strdup(struct_name);
            sdef.original_name = item->as.struct_def.original_name ? strdup(item->as.struct_def.original_name) : NULL;
            sdef.field_count = item->as.struct_def.field_count;
            
            /* Duplicate field names (AST will be freed) */
            sdef.field_names = malloc(sizeof(char*) * sdef.field_count);
            for (int j = 0; j < sdef.field_count; j++) {
                sdef.field_names[j] = strdup(item->as.struct_def.field_names[j]);
            }
            
            /* Duplicate field types */
            sdef.field_types = malloc(sizeof(Type) * sdef.field_count);
            for (int j = 0; j < sdef.field_count; j++) {
                sdef.field_types[j] = item->as.struct_def.field_types[j];
            }
            
            /* Duplicate field type names (for struct/union/enum types) */
            sdef.field_type_names = malloc(sizeof(char*) * sdef.field_count);
            for (int j = 0; j < sdef.field_count; j++) {
                if (item->as.struct_def.field_type_names && item->as.struct_def.field_type_names[j]) {
                    sdef.field_type_names[j] = strdup(item->as.struct_def.field_type_names[j]);
                    
                    /* Fix type if this is actually an enum (parser can't distinguish at parse time) */
                    if (sdef.field_types[j] == TYPE_STRUCT) {
                        /* Check if this name is an enum */
                        if (env_get_enum(env, item->as.struct_def.field_type_names[j])) {
                            sdef.field_types[j] = TYPE_ENUM;
                        }
                        /* Check if this name is a union */
                        else if (env_get_union(env, item->as.struct_def.field_type_names[j])) {
                            sdef.field_types[j] = TYPE_UNION;
                        }
                    }
                } else {
                    sdef.field_type_names[j] = NULL;
                }
            }
            
            /* Duplicate field element types (for array types) */
            sdef.field_element_types = malloc(sizeof(Type) * sdef.field_count);
            for (int j = 0; j < sdef.field_count; j++) {
                sdef.field_element_types[j] = item->as.struct_def.field_element_types[j];
                
                /* Register generic list instantiation for List<T> fields */
                if (sdef.field_types[j] == TYPE_LIST_GENERIC && sdef.field_type_names[j] != NULL) {
                    env_register_list_instantiation(env, sdef.field_type_names[j]);
                }
            }
            
            sdef.is_resource = item->as.struct_def.is_resource;  /* Propagate resource flag */
            sdef.is_extern = item->as.struct_def.is_extern;      /* Propagate extern flag */
sdef.is_pub = item->as.struct_def.is_pub;            /* Propagate public visibility flag */
            sdef.module_name = env->current_module ? strdup(env->current_module) : NULL;  /* Set module context */
            
            env_define_struct(env, sdef);

            /* Module introspection: track exported structs (public only) */
            if (sdef.is_pub && env->current_module) {
                env_add_module_exported_struct(env, env->current_module, sdef.original_name ? sdef.original_name : struct_name);
            }
            
        } else if (item->type == AST_UNION_DEF) {
            const char *union_name = item->as.union_def.name;
            
            /* Check if union already defined */
            if (env_get_union(env, union_name)) {
                fprintf(stderr, "Error at line %d, column %d: Union '%s' is already defined\n",
                        item->line, item->column, union_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register the union */
            UnionDef udef = {0};
            udef.name = strdup(union_name);
            udef.variant_count = item->as.union_def.variant_count;
            
            /* Allocate variant names */
            udef.variant_names = malloc(sizeof(char*) * udef.variant_count);
            for (int j = 0; j < udef.variant_count; j++) {
                udef.variant_names[j] = strdup(item->as.union_def.variant_names[j]);
            }
            
            /* Allocate field counts */
            udef.variant_field_counts = malloc(sizeof(int) * udef.variant_count);
            for (int j = 0; j < udef.variant_count; j++) {
                udef.variant_field_counts[j] = item->as.union_def.variant_field_counts[j];
            }
            
            /* Allocate field names */
            udef.variant_field_names = malloc(sizeof(char**) * udef.variant_count);
            for (int j = 0; j < udef.variant_count; j++) {
                int field_count = udef.variant_field_counts[j];
                udef.variant_field_names[j] = malloc(sizeof(char*) * field_count);
                for (int k = 0; k < field_count; k++) {
                    udef.variant_field_names[j][k] = strdup(item->as.union_def.variant_field_names[j][k]);
                }
            }
            
            /* Allocate field types */
            udef.variant_field_types = malloc(sizeof(Type*) * udef.variant_count);
            for (int j = 0; j < udef.variant_count; j++) {
                int field_count = udef.variant_field_counts[j];
                udef.variant_field_types[j] = malloc(sizeof(Type) * field_count);
                for (int k = 0; k < field_count; k++) {
                    udef.variant_field_types[j][k] = item->as.union_def.variant_field_types[j][k];
                }
            }
            
            /* Allocate field type names */
            udef.variant_field_type_names = malloc(sizeof(char**) * udef.variant_count);
            for (int j = 0; j < udef.variant_count; j++) {
                int field_count = udef.variant_field_counts[j];
                udef.variant_field_type_names[j] = malloc(sizeof(char*) * field_count);
                for (int k = 0; k < field_count; k++) {
                    if (item->as.union_def.variant_field_type_names[j][k]) {
                        udef.variant_field_type_names[j][k] = strdup(item->as.union_def.variant_field_type_names[j][k]);
                    } else {
                        udef.variant_field_type_names[j][k] = NULL;
                    }
                }
            }
            
            /* Each registered declaration owns an independent payload tree. */
            udef.variant_field_type_info = calloc((size_t)udef.variant_count, sizeof(TypeInfo **));
            for (int j = 0; j < udef.variant_count; ++j) {
                int fields = udef.variant_field_counts[j];
                udef.variant_field_type_info[j] = calloc((size_t)fields, sizeof(TypeInfo *));
                for (int k = 0; k < fields; ++k)
                    if (item->as.union_def.variant_field_type_info && item->as.union_def.variant_field_type_info[j])
                        udef.variant_field_type_info[j][k] = copy_payload_type_info(item->as.union_def.variant_field_type_info[j][k]);
            }

            /* Copy generic parameters if present */
            udef.generic_param_count = item->as.union_def.generic_param_count;
            if (udef.generic_param_count > 0) {
                udef.generic_params = malloc(sizeof(char*) * udef.generic_param_count);
                for (int j = 0; j < udef.generic_param_count; j++) {
                    udef.generic_params[j] = strdup(item->as.union_def.generic_params[j]);
                }
            } else {
                udef.generic_params = NULL;
            }
            
            /* Set module visibility */
            udef.is_pub = item->as.union_def.is_pub;
            udef.module_name = env->current_module ? strdup(env->current_module) : NULL;
            udef.is_extern = item->as.union_def.is_extern;
            
            env_define_union(env, udef);
            
        } else if (item->type == AST_OPAQUE_TYPE) {
            /* Register opaque type */
            const char *type_name = item->as.opaque_type.name;
            
            /* The nominal prepass checked local duplicates and staged all rows. */
            if (!env_define_opaque_type(env, type_name)) tc.has_error = true;
            
        } else if (item->type == AST_EFFECT_DECL) {
            if (!register_effect_declaration(item, env)) tc.has_error = true;
        } else if (item->type == AST_ENUM_DEF) {
            /* Defensive check: ensure item and enum_def fields are valid */
            if (!item) {
                fprintf(stderr, "Error: NULL AST item in enum processing\n");
                tc.has_error = true;
                continue;
            }
            
            const char *enum_name = item->as.enum_def.name;
            assert(enum_name != NULL); /* Parser should never create enum with NULL name */
            
            if (!enum_name) {
                safe_fprintf(stderr, "Error at line %d, column %d: Enum definition has NULL name\n",
                        item->line, item->column);
                tc.has_error = true;
                continue;
            }
            
            /* Check if enum already defined - defensive check for NULL name */
            if (enum_name && env_get_enum(env, enum_name)) {
                safe_fprintf(stderr, "Error at line %d, column %d: Enum '%s' is already defined\n",
                        item->line, item->column, safe_format_string(enum_name));
                tc.has_error = true;
                continue;
            }
            
            /* Register the enum */
            EnumDef edef;
            edef.name = enum_name ? strdup(enum_name) : NULL;
            if (!edef.name) {
                fprintf(stderr, "Error: Failed to allocate memory for enum name\n");
                tc.has_error = true;
                continue;
            }
            edef.variant_count = item->as.enum_def.variant_count;
            if (edef.variant_count <= 0) {
                safe_fprintf(stderr, "Error: Enum '%s' has invalid variant count: %d\n", safe_format_string(enum_name), edef.variant_count);
                free(edef.name);
                tc.has_error = true;
                continue;
            }
            edef.variant_names = malloc(sizeof(char*) * edef.variant_count);
            if (!edef.variant_names) {
                fprintf(stderr, "Error: Failed to allocate memory for enum variant names\n");
                free(edef.name);
                tc.has_error = true;
                continue;
            }
            edef.variant_values = malloc(sizeof(int) * edef.variant_count);
            if (!edef.variant_values) {
                fprintf(stderr, "Error: Failed to allocate memory for enum variant values\n");
                free(edef.name);
                free(edef.variant_names);
                tc.has_error = true;
                continue;
            }
            
            for (int j = 0; j < edef.variant_count; j++) {
                if (item->as.enum_def.variant_names && 
                    j < item->as.enum_def.variant_count && 
                    item->as.enum_def.variant_names[j]) {
                    const char *src_name = item->as.enum_def.variant_names[j];
                    if (src_name) {
                        edef.variant_names[j] = strdup(src_name);
                        if (!edef.variant_names[j]) {
                            fprintf(stderr, "Error: Failed to duplicate variant name at index %d\n", j);
                            edef.variant_names[j] = NULL;
                        }
                    } else {
                        safe_fprintf(stderr, "Error: Enum '%s' has NULL variant name at index %d\n", safe_format_string(enum_name), j);
                        edef.variant_names[j] = NULL;
                    }
                } else {
                    safe_fprintf(stderr, "Error: Enum '%s' has NULL variant name at index %d\n", safe_format_string(enum_name), j);
                    edef.variant_names[j] = NULL;
                }
            }
            
            /* Duplicate variant values */
            edef.variant_values = malloc(sizeof(int) * edef.variant_count);
            if (item->as.enum_def.variant_values) {
                for (int j = 0; j < edef.variant_count; j++) {
                    edef.variant_values[j] = item->as.enum_def.variant_values[j];
                }
            } else {
                /* No explicit values - use index as value */
                for (int j = 0; j < edef.variant_count; j++) {
                    edef.variant_values[j] = j;
                }
            }
            
            /* Set module visibility */
            edef.is_pub = item->as.enum_def.is_pub;
            edef.module_name = env->current_module ? strdup(env->current_module) : NULL;
            edef.is_extern = item->as.enum_def.is_extern;
            
            env_define_enum(env, edef);
            
        } else if (item->type == AST_ASYNC_FN && item->as.async_fn.function &&
                   item->as.async_fn.function->type == AST_FUNCTION) {
            item = item->as.async_fn.function;
            goto register_function_pass2;
        } else if (item->type == AST_FUNCTION) {
register_function_pass2:;
            const char *func_name = item->as.function.name;
            
            /* Check for duplicate function definitions */
            Function *existing = env_get_function(env, func_name);
            /* I distinguish an imported name from a duplicate in this module. */
            if (existing && !existing->is_extern && !item->as.function.is_extern && existing->module_name &&
                (!env->current_module || strcmp(existing->module_name, env->current_module) != 0)) {
                existing = NULL;
            }
            if (existing) {
                /* If both are extern and signatures match, it's fine (idempotent) */
                if (item->as.function.is_extern && existing->is_extern) {
                    Function current;
                    current.param_count = item->as.function.param_count;
                    current.params = item->as.function.params;
                    current.return_type = item->as.function.return_type;
                    
                    if (functions_match(&current, existing)) {
                        continue; /* Skip registration, already there and matches */
                    }
                }

                /* Extern functions cannot be redefined or shadowed */
                if (existing->is_extern) {
                    fprintf(stderr, "Error at line %d, column %d: Extern function '%s' cannot be redefined\n",
                            item->line, item->column, func_name);
                    fprintf(stderr, "  Extern functions are first-class and cannot be shadowed\n");
                    tc.has_error = true;
                    continue;
                }
                /* Regular functions cannot be redefined */
                fprintf(stderr, "Error at line %d, column %d: Function '%s' is already defined\n",
                        item->line, item->column, func_name);
                tc.has_error = true;
                continue;
            }
            
            /* Regular functions cannot shadow extern functions */
            /* Note: This check happens after we've registered extern functions */
            /* We check this in the first pass, but also here for module type checking */
            
            /* Check if function name shadows a built-in */
            if (is_builtin_function(func_name)) {
                fprintf(stderr, "Error at line %d, column %d: Function '%s' shadows a built-in function\n",
                        item->line, item->column, func_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register function signature */
            Function f = (Function){0};
            f.source_file = env_current_file(env);
            f.name = env_own_checker_allocation(env, strdup(func_name));
            f.param_count = item->as.function.param_count;
            f.params = env_own_checker_allocation(env, malloc(sizeof(Parameter) * f.param_count));
            for (int j = 0; j < f.param_count; j++) {
                /* I preserve borrowed generic/tuple metadata before copying owned names. */
                f.params[j] = item->as.function.params[j];
                f.params[j].name = env_own_checker_allocation(env, strdup(item->as.function.params[j].name));
                f.params[j].type = item->as.function.params[j].type;
                f.params[j].struct_type_name = item->as.function.params[j].struct_type_name ? 
                    env_own_checker_allocation(env, strdup(item->as.function.params[j].struct_type_name)) : NULL;
                f.params[j].element_type = item->as.function.params[j].element_type;
                f.params[j].fn_sig = item->as.function.params[j].fn_sig;
            }
            f.return_type = item->as.function.return_type;
            f.return_element_type = item->as.function.return_element_type;
            f.return_struct_type_name = item->as.function.return_struct_type_name ? 
                env_own_checker_allocation(env, strdup(item->as.function.return_struct_type_name)) : NULL;
            f.return_fn_sig = item->as.function.return_fn_sig;
            f.return_type_info = item->as.function.return_type_info;
            f.body = item->as.function.body;
            f.shadow_test = NULL;  /* Will be linked in second pass */
            f.is_extern = item->as.function.is_extern;
            f.is_pub = item->as.function.is_pub;  /* Store visibility */
            f.is_pure = item->as.function.is_pure;  /* Propagate purity annotation */
            f.module_name = env->current_module ? env_own_checker_allocation(env, strdup(env->current_module)) : NULL;

            env_define_function(env, f);
            register_native_function_context(env, &f);

            /* Module introspection: track exported functions (public only) */
            if (item->as.function.is_pub && env->current_module) {
                env_add_module_exported_function(env, env->current_module, func_name);
            }
        }
    }

    /* I retain concrete field-only union instances after all declarations
     * are visible, including forward declarations in the same module. */
    for (int record = 0; record < env->struct_count; ++record) {
        StructDef *definition = &env->structs[record];
        char *saved_module = env->current_module;
        env->current_module = definition->module_name;
        for (int field = 0; definition->field_type_info && field < definition->field_count; ++field)
            register_native_union_context(env, definition->field_type_info[field], 0);
        env->current_module = saved_module;
    }
    for (int index = 0; index < env->union_count; ++index) {
        UnionDef *definition = &env->unions[index];
        if (definition->generic_param_count) continue;
        char *saved_module = env->current_module;
        env->current_module = definition->module_name;
        for (int arm = 0; definition->variant_field_type_info && arm < definition->variant_count; ++arm)
            for (int field = 0; definition->variant_field_type_info[arm] && field < definition->variant_field_counts[arm]; ++field)
                register_native_union_context(env, definition->variant_field_type_info[arm][field], 0);
        env->current_module = saved_module;
    }

    /* Second pass: link shadow tests to functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        /* Skip imports - they're handled separately */
        if (item->type == AST_IMPORT) {
            continue;
        }
        
        if (item->type == AST_SHADOW) {
            Function *func = env_get_function(env, item->as.shadow.function_name);
            if (!func) {
                fprintf(stderr, "Error at line %d, column %d: Shadow test for undefined function '%s'\n",
                        item->line, item->column, item->as.shadow.function_name);
                tc.has_error = true;
                continue;
            } else if (func->is_extern) {
                /* Extern functions cannot have shadow tests - they're C functions */
                fprintf(stderr, "Error at line %d, column %d: Shadow test cannot be attached to extern function '%s'\n",
                        item->line, item->column, item->as.shadow.function_name);
                fprintf(stderr, "  Extern functions are C functions and cannot be tested in the interpreter\n");
                fprintf(stderr, "  Remove the shadow test or test a wrapper function instead\n");
                tc.has_error = true;
                continue;
            }
            
            func->shadow_test = item->as.shadow.body;
        }
    }

    /* Process top-level constants/variables (before type checking functions) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_LET) {
            /* I provide declared constructor context before checking the initializer. */
            prepare_map_initializer(&tc, item);
            check_concrete_union_arrays(env, item->as.let.type_info, item->as.let.value, 0);
            Type value_type = check_expression(item->as.let.value, env);
            check_global_ownership(env, item, &tc.has_error);
            if (!check_record_array_contract(env, item->as.let.var_type, item->as.let.element_type,
                    item->as.let.type_name, item->as.let.value)) tc.has_error = true;
            if (item->as.let.var_type == TYPE_ARRAY &&
                item->as.let.element_type != TYPE_UNKNOWN &&
                item->as.let.value->type == AST_ARRAY_LITERAL) {
                check_array_literal_annotation(&tc, item->as.let.value,
                                               item->as.let.element_type, item->as.let.type_name);
            }
            
            /* Verify it matches the declared type */
            if (item->as.let.var_type != value_type) {
                fprintf(stderr, "Error at line %d, column %d: Constant '%s' type mismatch (declared %s, got %s)\n",
                        item->line, item->column,
                        item->as.let.name,
                        type_to_string(item->as.let.var_type),
                        type_to_string(value_type));
                tc.has_error = true;
                continue;
            }
            
            /* Add constant to environment */
            Value val = create_void();  /* Placeholder value for type checking */

            /* I preserve explicit function signatures just as in local bindings. */
            if (!retain_let_function_type(&tc, item, item->as.let.var_type)) continue;
            /* Preserve element type / generic type info for arrays and other complex types */
            env_define_var_with_type_info(env,
                                         item->as.let.name,
                                         item->as.let.var_type,
                                         item->as.let.element_type,
                                         item->as.let.type_info,
                                         item->as.let.is_mut,
                                         val);

            /* Preserve struct/union type name metadata for globals */
            Symbol *sym = env_get_var(env, item->as.let.name);
            if (sym) {
                sym->is_global = true;
                sym->def_line = item->line;
                sym->def_column = item->column;
            }
            if (sym && item->as.let.type_name) {
                if (sym->struct_type_name) free(sym->struct_type_name);
                sym->struct_type_name = NULL;

                if (item->as.let.var_type == TYPE_STRUCT || item->as.let.var_type == TYPE_UNION) {
                    sym->struct_type_name = strdup(item->as.let.type_name);
                    mark_variable_as_resource_if_needed(env, item->as.let.name, item->as.let.type_name);
                } else if (item->as.let.var_type == TYPE_ARRAY && item->as.let.element_type == TYPE_STRUCT) {
                    sym->struct_type_name = strdup(item->as.let.type_name);
                }
            }
        }
    }

    /* Third pass: type check all statements and expressions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        /* Skip imports, struct/enum/union/effect definitions, and shadow tests */
        if (item->type == AST_IMPORT || 
            item->type == AST_STRUCT_DEF ||
            item->type == AST_ENUM_DEF ||
            item->type == AST_UNION_DEF ||
            item->type == AST_EFFECT_DECL ||
            item->type == AST_SHADOW) {
            continue;
        }
        
        /* Unwrap async fn */
        if (item->type == AST_ASYNC_FN && item->as.async_fn.function &&
            item->as.async_fn.function->type == AST_FUNCTION)
            item = item->as.async_fn.function;

        if (item->type == AST_FUNCTION) {
            /* Save current symbol count */
            int saved_symbol_count = env->symbol_count;
            
            /* Set current function return type for return statement checking */
            tc.current_function_return_type = item->as.function.return_type;
            tc.current_function_return_element_type = item->as.function.return_element_type;
            tc.current_function_return_info = item->as.function.return_type_info;
            tc.current_function_return_struct_name = item->as.function.return_struct_type_name;

            /* Register generic union instantiation for function return type */
            if (item->as.function.return_type == TYPE_UNION &&
                item->as.function.return_type_info &&
                item->as.function.return_type_info->generic_name &&
                item->as.function.return_type_info->type_param_count > 0) {
                TypeInfo *info = item->as.function.return_type_info;

                /* Best-effort: only register if the generic union exists */
                UnionDef *udef = env_get_union(env, info->generic_name);
                if (udef && udef->generic_param_count == info->type_param_count) {
                    if (opaque_type_info_present(info)) {
                        register_native_union_context(env, info, 0);
                    } else {
                    char **type_names = malloc(sizeof(char*) * info->type_param_count);
                    for (int ti = 0; ti < info->type_param_count; ti++) {
                        type_names[ti] = typeinfo_to_generic_arg_name(info->type_params[ti]);
                    }

                    env_register_union_instantiation(env, info->generic_name,
                                                    (const char**)type_names,
                                                    info->type_param_count);

                    for (int ti = 0; ti < info->type_param_count; ti++) {
                        free(type_names[ti]);
                    }
                    free(type_names);
                    }
                }
            }

            /* Register HashMap<K,V> instantiation for function return type */
            if (item->as.function.return_type == TYPE_HASHMAP &&
                item->as.function.return_type_info) {
                TypeInfo *info = item->as.function.return_type_info;
                Type key_t = TYPE_UNKNOWN;
                Type val_t = TYPE_UNKNOWN;
                if (!hashmap_extract_kv(info, &key_t, &val_t)) {
                    fprintf(stderr, "Error at line %d, column %d: Invalid HashMap return type annotation\n",
                            item->line, item->column);
                    tc.has_error = true;
                } else {
                    if (!(key_t == TYPE_INT || key_t == TYPE_STRING)) {
                        fprintf(stderr, "Error at line %d, column %d: HashMap key type must be int or string (got %s)\n",
                                item->line, item->column, type_to_string(key_t));
                        tc.has_error = true;
                    }
                    if (!(val_t == TYPE_INT || val_t == TYPE_STRING)) {
                        fprintf(stderr, "Error at line %d, column %d: HashMap value type must be int or string (got %s)\n",
                                item->line, item->column, type_to_string(val_t));
                        tc.has_error = true;
                    }
                    if (!tc.has_error) {
                        char *key_name = typeinfo_to_generic_arg_name(info->type_params[0]);
                        char *val_name = typeinfo_to_generic_arg_name(info->type_params[1]);
                        env_register_hashmap_instantiation(env, key_name, val_name);
                        free(key_name);
                        free(val_name);
                    }
                }
            }
            
            /* Add function parameters to environment */
            for (int j = 0; j < item->as.function.param_count; j++) {
                Type param_type = item->as.function.params[j].type;
                Type element_type = item->as.function.params[j].element_type;
                TypeInfo *param_type_info = item->as.function.params[j].type_info;
                Value val;

                /* Register HashMap<K,V> instantiation for parameters */
                if (param_type == TYPE_HASHMAP && param_type_info) {
                    Type key_t = TYPE_UNKNOWN;
                    Type val_t = TYPE_UNKNOWN;
                    if (!hashmap_extract_kv(param_type_info, &key_t, &val_t)) {
                        fprintf(stderr, "Error at line %d, column %d: Invalid HashMap parameter type annotation\n",
                                item->line, item->column);
                        tc.has_error = true;
                    } else {
                        if (!(key_t == TYPE_INT || key_t == TYPE_STRING)) {
                            fprintf(stderr, "Error at line %d, column %d: HashMap key type must be int or string (got %s)\n",
                                    item->line, item->column, type_to_string(key_t));
                            tc.has_error = true;
                        }
                        if (!(val_t == TYPE_INT || val_t == TYPE_STRING)) {
                            fprintf(stderr, "Error at line %d, column %d: HashMap value type must be int or string (got %s)\n",
                                    item->line, item->column, type_to_string(val_t));
                            tc.has_error = true;
                        }
                        if (!tc.has_error) {
                            char *key_name = typeinfo_to_generic_arg_name(param_type_info->type_params[0]);
                            char *val_name = typeinfo_to_generic_arg_name(param_type_info->type_params[1]);
                            env_register_hashmap_instantiation(env, key_name, val_name);
                            free(key_name);
                            free(val_name);
                        }
                    }
                }
                if (param_type == TYPE_INT) val = create_int(0);
                else if (param_type == TYPE_FLOAT) val = create_float(0.0);
                else if (param_type == TYPE_BOOL) val = create_bool(false);
                else if (param_type == TYPE_STRING) val = create_string("");
                else if (param_type == TYPE_ARRAY) {
                    val = create_array((ValueType)element_type, 0, 0);
                    env_own_checker_allocation(env, val.as.array_val);
                    env_own_checker_allocation(env, val.as.array_val->data);
                } else if (param_type == TYPE_STRUCT) {
                    val = create_struct(item->as.function.params[j].struct_type_name, NULL, NULL, 0);
                } else if (param_type == TYPE_UNION) {
                    /* For union parameters, create empty union value */
                    val = create_void();  /* Placeholder */
                } else val = create_void();
                
                /* For function parameters with TypeInfo (generics, function types, etc.), use full type info */
                if (param_type_info) {
                    /* Already have TypeInfo from parser - use it directly */
                    env_define_var_with_type_info(env, item->as.function.params[j].name, param_type, element_type, param_type_info, false, val);
                }
                /* For function parameters, create TypeInfo with signature */
                else if (param_type == TYPE_FUNCTION && item->as.function.params[j].fn_sig) {
                    TypeInfo *type_info = env_own_checker_allocation(env, malloc(sizeof(TypeInfo)));
                    memset(type_info, 0, sizeof(TypeInfo));
                    type_info->base_type = TYPE_FUNCTION;
                    type_info->fn_sig = item->as.function.params[j].fn_sig;
                    env_define_var_with_type_info(env, item->as.function.params[j].name, param_type, TYPE_UNKNOWN, type_info, false, val);
                }
                /* Use env_define_var_with_element_type for arrays to preserve element type */
                else if (param_type == TYPE_ARRAY && element_type != TYPE_UNKNOWN) {
                    env_define_var_with_element_type(env, item->as.function.params[j].name, param_type, element_type, false, val);
                } else {
                    env_define_var(env, item->as.function.params[j].name, param_type, false, val);
                }
                
                /* If parameter is a struct or union, store the type name */
                Symbol *param_sym = env_get_var(env, item->as.function.params[j].name);
                if (param_sym) {
                    param_sym->def_line = item->line;
                    param_sym->def_column = item->column;

                    if ((param_type == TYPE_STRUCT || param_type == TYPE_UNION || param_type == TYPE_BORROW_SHARED || param_type == TYPE_BORROW_MUT) &&
                        item->as.function.params[j].struct_type_name) {
                        param_sym->struct_type_name = strdup(item->as.function.params[j].struct_type_name);
                    }
                    /* For generic unions with TypeInfo, use the generic_name as struct_type_name */
                    else if (param_type == TYPE_UNION && param_type_info && param_type_info->generic_name) {
                        param_sym->struct_type_name = strdup(param_type_info->generic_name);
                    }
                }
            }

            /* Check function body */
            check_statement(&tc, item->as.function.body);
            bound_scope_symbols(env, saved_symbol_count, item->as.function.body);
            check_function_ownership(env, item, &tc.has_error);

            /* Purity check: verify pure fn body obeys purity rules */
            if (item->as.function.is_pure) {
                check_purity(item->as.function.body, env, item->as.function.name);
            }

            /* Extern parameters describe the C ABI; they have no body in which
             * they could be used. */
            if (!item->as.function.is_extern) {
                check_unused_variables(&tc, saved_symbol_count);
            }

            /* DON'T restore environment - transpiler needs these symbols! */
            /* The old code removed function-local symbols after typechecking:
             *   env->symbol_count = saved_symbol_count;
             * This caused array<struct> to fail because transpiler couldn't find
             * the struct_type_name metadata. Now we keep all symbols so transpiler
             * can access type information. C's function-local scope prevents collisions.
             */

            /* Verify function has shadow test (skip for extern functions, main, and functions that use extern functions) */
            Function *func = env_get_function(env, item->as.function.name);
            if (!env->suppress_shadow_warnings && !env->gpu_target &&
                !func->is_extern && !func->shadow_test &&
                strcmp(item->as.function.name, "main") != 0 &&
                strncmp(item->as.function.name, "__lambda_", 9) != 0) {
                /* Check if function body uses extern functions - if so, shadow test is optional */
                bool uses_extern = func->body && contains_extern_calls(func->body, env);
                if (!uses_extern) {
                    fprintf(stderr, "%sWarning:%s Function '%s' is missing a shadow test\n",
                            CSTART_WARNING, CEND, item->as.function.name);
                    /* Don't fail - just warn */
                    /* tc.has_error = true; */
                }
            }
        }
    }

    /* Note: Modules don't require a main function */
    /* Main function check is skipped for modules */

    return !tc.has_error && !env->opaque_resolution_failed && g_typecheck_error_count == 0;
}
