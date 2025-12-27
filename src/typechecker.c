#include "nanolang.h"
#include "tracing.h"

/* Type checking context */
typedef struct {
    Environment *env;
    Type current_function_return_type;
    bool has_error;
    bool warnings_enabled;
} TypeChecker;

static char *typeinfo_to_generic_arg_name(TypeInfo *param) {
    if (!param) return strdup("unknown");

    switch (param->base_type) {
        case TYPE_INT:
            return strdup("int");
        case TYPE_U8:
            return strdup("u8");
        case TYPE_STRING:
            return strdup("string");
        case TYPE_BOOL:
            return strdup("bool");
        case TYPE_FLOAT:
            return strdup("float");
        case TYPE_STRUCT:
        case TYPE_UNION:
        case TYPE_ENUM:
            if (param->generic_name) return strdup(param->generic_name);
            return strdup("unknown");
        case TYPE_ARRAY: {
            char *elem = typeinfo_to_generic_arg_name(param->element_type);
            if (!elem) return strdup("array_unknown");
            size_t n = strlen(elem) + 7;
            char *out = malloc(n);
            if (!out) {
                free(elem);
                return strdup("array_unknown");
            }
            snprintf(out, n, "array_%s", elem);
            free(elem);
            return out;
        }
        default:
            return strdup("unknown");
    }
}

static char *typeinfo_to_monomorphized_generic_name(TypeInfo *info) {
    if (!info || !info->generic_name) return NULL;
    if (info->type_param_count <= 0) return strdup(info->generic_name);

    size_t cap = 256;
    char *out = malloc(cap);
    if (!out) return NULL;
    out[0] = '\0';

    snprintf(out, cap, "%s", info->generic_name);
    for (int i = 0; i < info->type_param_count; i++) {
        char *arg = typeinfo_to_generic_arg_name(info->type_params[i]);
        if (!arg) arg = strdup("unknown");
        size_t need = strlen(out) + 1 + strlen(arg) + 1;
        if (need > cap) {
            cap = need * 2;
            char *bigger = realloc(out, cap);
            if (!bigger) {
                free(arg);
                free(out);
                return NULL;
            }
            out = bigger;
        }
        strcat(out, "_");
        strcat(out, arg);
        free(arg);
    }

    return out;
}

/* Helper: Check if symbol was explicitly imported via selective import
 * NOTE: Currently unused - reserved for future selective import enforcement
 */
#if 0  /* Disabled - not yet needed */
static bool is_symbol_imported(const char *symbol_name, const char *module_path, Environment *env) {
    if (!env->import_tracker) return true;  /* No tracking = allow all */
    
    for (int i = 0; i < env->import_tracker->import_count; i++) {
        SelectiveImport *imp = &env->import_tracker->imports[i];
        
        /* Check if this import is from the right module */
        if (imp->module_path && strcmp(imp->module_path, module_path) == 0) {
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
    
    /* Not found in any selective import - not accessible */
    return false;
}
#endif  /* Disabled - not yet needed */

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
        fprintf(stderr, "Error at line %d, column %d: Function '%s' is private to module '%s'\n",
                line, column, func->name, func->module_name);
        fprintf(stderr, "  Note: Use 'pub fn %s(...)' to make it accessible from other modules\n",
                func->name);
        fprintf(stderr, "  Hint: Private functions are only accessible within their defining module\n");
        return false;
    }
    
    /* TODO: Check if symbol was explicitly imported via selective import */
    /* For now, if public and visible, allow access */
    
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

static TypeInfo *try_get_expr_type_info(ASTNode *expr, Environment *env) {
    if (!expr) return NULL;
    if (expr->type == AST_IDENTIFIER) {
        Symbol *sym = env_get_var_visible_at(env, expr->as.identifier, expr->line, expr->column);
        if (sym) return sym->type_info;
    }
    if (expr->type == AST_CALL && expr->as.call.name) {
        Function *func = env_get_function(env, expr->as.call.name);
        if (func && func->return_type_info) return func->return_type_info;
    }
    return NULL;
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

/* Check if types are compatible */
static bool types_match(Type t1, Type t2) {
    if (t1 == t2) return true;

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
    
    return false;
}

/* Helper: Get the struct type name from an expression (returns NULL if not a struct) */
const char *get_struct_type_name(ASTNode *expr, Environment *env) {
    if (!expr) return NULL;
    
    switch (expr->type) {
        case AST_STRUCT_LITERAL:
            return expr->as.struct_literal.struct_name;
            
        case AST_IDENTIFIER: {
            Symbol *sym = env_get_var_visible_at(env, expr->as.identifier, expr->line, expr->column);
            if (sym && (sym->type == TYPE_STRUCT || sym->type == TYPE_UNION)) {
                return sym->struct_type_name;
            }
            return NULL;
        }
        
        case AST_CALL: {
            /* Check if return_struct_type_name was set by type checker (for generic list get) */
            if (expr->as.call.return_struct_type_name) {
                return expr->as.call.return_struct_type_name;
            }
            
            /* Check if function returns a struct */
            Function *func = env_get_function(env, expr->as.call.name);
            if (func && func->return_type == TYPE_STRUCT) {
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

static Type infer_array_element_type(ASTNode *array_expr, Environment *env) {
    if (!array_expr) return TYPE_UNKNOWN;

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
            return sym->element_type;
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
                            return sdef->field_element_types[i];
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

    Type result = check_expression_impl(expr, env);
    g_check_expr_depth--;
    return result;
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
                fprintf(stderr, "Error at line %d, column %d: Undefined variable '%s'\n",
                        expr->line, expr->column, expr->as.identifier);
                return TYPE_UNKNOWN;
            }
            sym->is_used = true;  /* Mark variable as used */
            return sym->type;
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
                    /* TODO: Check if symbol belongs to the specified module */
                    /* TODO: Check visibility */
                    sym->is_used = true;
                    return sym->type;
                }
                
                fprintf(stderr, "Error at line %d, column %d: Undefined symbol '%s' in module '%s'\n",
                        expr->line, expr->column, symbol_name, module_name);
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
                        fprintf(stderr, "Error at line %d, column %d: Unary minus requires array<int> or array<float>\n", expr->line, expr->column);
                        return TYPE_UNKNOWN;
                    }
                    fprintf(stderr, "Error at line %d, column %d: Unary minus requires numeric type\n", expr->line, expr->column);
                    return TYPE_UNKNOWN;
                }
                
                /* Binary arithmetic operations */
                if (arg_count != 2) {
                    fprintf(stderr, "Error at line %d, column %d: Binary arithmetic operators require 2 arguments\n", expr->line, expr->column);
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

                    fprintf(stderr, "Error at line %d, column %d: Type mismatch in array arithmetic operation\n", expr->line, expr->column);
                    fprintf(stderr, "  Got: %s and %s\n", type_to_string(left), type_to_string(right));
                    return TYPE_UNKNOWN;
                }

                /* Enums are compatible with ints in arithmetic */
                if ((left == TYPE_INT || left == TYPE_ENUM || left == TYPE_U8) &&
                    (right == TYPE_INT || right == TYPE_ENUM || right == TYPE_U8)) return TYPE_INT;
                if (left == TYPE_FLOAT && right == TYPE_FLOAT) return TYPE_FLOAT;

                fprintf(stderr, "Error at line %d, column %d: Type mismatch in arithmetic operation\n", expr->line, expr->column);
                fprintf(stderr, "  Expected: numeric types (int, u8, or float)\n");
                fprintf(stderr, "  Got: %s and %s\n", type_to_string(left), type_to_string(right));
                fprintf(stderr, "  Hint: Both operands must be the same numeric type\n");
                return TYPE_UNKNOWN;
            }

            /* Comparison operators */
            if (op == TOKEN_LT || op == TOKEN_LE || op == TOKEN_GT || op == TOKEN_GE) {
                if (arg_count != 2) {
                    fprintf(stderr, "Error at line %d, column %d: Comparison operators require 2 arguments\n", expr->line, expr->column);
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                if (!types_match(left, right)) {
                    fprintf(stderr, "Error at line %d, column %d: Type mismatch in comparison\n", expr->line, expr->column);
                    fprintf(stderr, "  Left operand: %s\n", type_to_string(left));
                    fprintf(stderr, "  Right operand: %s\n", type_to_string(right));
                    fprintf(stderr, "  Hint: Comparison requires both operands to be the same type\n");
                }
                return TYPE_BOOL;
            }

            /* Equality operators */
            if (op == TOKEN_EQ || op == TOKEN_NE) {
                if (arg_count != 2) {
                    fprintf(stderr, "Error at line %d, column %d: Equality operators require 2 arguments\n", expr->line, expr->column);
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
                    fprintf(stderr, "Error at line %d, column %d: Type mismatch in equality check\n", expr->line, expr->column);
                }
                return TYPE_BOOL;
            }

            /* Logical operators */
            if (op == TOKEN_AND || op == TOKEN_OR) {
                if (arg_count != 2) {
                    fprintf(stderr, "Error at line %d, column %d: Logical operators require 2 arguments\n", expr->line, expr->column);
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                if (left != TYPE_BOOL || right != TYPE_BOOL) {
                    fprintf(stderr, "Error at line %d, column %d: Logical operators require bool operands\n", expr->line, expr->column);
                }
                return TYPE_BOOL;
            }

            if (op == TOKEN_NOT) {
                if (arg_count != 1) {
                    fprintf(stderr, "Error at line %d, column %d: 'not' requires 1 argument\n", expr->line, expr->column);
                    return TYPE_UNKNOWN;
                }
                Type arg = check_expression(expr->as.prefix_op.args[0], env);
                if (arg != TYPE_BOOL) {
                    fprintf(stderr, "Error at line %d, column %d: 'not' requires bool operand\n", expr->line, expr->column);
                }
                return TYPE_BOOL;
            }

            return TYPE_UNKNOWN;
        }

        case AST_CALL: {
            /* Check if this is a function call returning a function: ((func_call) arg1 arg2) */
            if (expr->as.call.func_expr) {
                /* First, check the inner function call */
                Type inner_type = check_expression(expr->as.call.func_expr, env);
                if (inner_type != TYPE_FUNCTION) {
                    fprintf(stderr, "Error at line %d, column %d: Expression does not return a function\n",
                            expr->line, expr->column);
                    return TYPE_UNKNOWN;
                }
                
                /* Check argument types */
                for (int i = 0; i < expr->as.call.arg_count; i++) {
                    check_expression(expr->as.call.args[i], env);
                }
                
                /* Return type will be determined at runtime */
                /* For now, assume it returns int */
                return TYPE_INT;
            }
            
            /* Regular function call */
            
            /* Special handling for map builtin - check before environment lookup */
            if (strcmp(expr->as.call.name, "map") == 0) {
                /* map(array, transform_fn) -> array */
                if (expr->as.call.arg_count >= 2) {
                    Type array_type = check_expression(expr->as.call.args[0], env);
                    check_expression(expr->as.call.args[1], env);  /* Check function */
                    return array_type;  /* Return same type as input array */
                }
                return TYPE_ARRAY;
            }

            /* Special handling for filter builtin - check before environment lookup */
            if (strcmp(expr->as.call.name, "filter") == 0) {
                /* filter(array, predicate_fn) -> array */
                if (expr->as.call.arg_count >= 2) {
                    Type array_type = check_expression(expr->as.call.args[0], env);
                    check_expression(expr->as.call.args[1], env);  /* Check function */
                    return array_type;
                }
                return TYPE_ARRAY;
            }
            
            /* Special handling for reduce builtin - check before environment lookup */
            if (strcmp(expr->as.call.name, "reduce") == 0) {
                /* reduce(array, initial, combine_fn) -> type_of_initial */
                if (expr->as.call.arg_count >= 3) {
                    check_expression(expr->as.call.args[0], env);  /* Check array */
                    Type initial_type = check_expression(expr->as.call.args[1], env);  /* Check initial value */
                    check_expression(expr->as.call.args[2], env);  /* Check function */
                    return initial_type;  /* Return same type as initial value */
                }
                return TYPE_UNKNOWN;
            }

            /* Result<T, E> helper intrinsics (generic-function stopgap) */
            if (strcmp(expr->as.call.name, "result_is_ok") == 0 || strcmp(expr->as.call.name, "result_is_err") == 0) {
                if (expr->as.call.arg_count != 1) {
                    fprintf(stderr, "Error at line %d, column %d: %s requires 1 argument\n",
                            expr->line, expr->column, expr->as.call.name);
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
                    fprintf(stderr, "Error at line %d, column %d: %s requires %d argument(s)\n",
                            expr->line, expr->column, expr->as.call.name, expected);
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
                        fprintf(stderr, "Error at line %d, column %d: result_unwrap_or default value type mismatch\n",
                                expr->line, expr->column);
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
            
            /* Check if function exists */
            Function *func = env_get_function(env, expr->as.call.name);
            
            /* Check visibility */
            if (func && !is_function_accessible(func, env, expr->line, expr->column)) {
                return TYPE_UNKNOWN;
            }
            
            /* If not a function, check if it's a function-typed variable (parameter) */
            if (!func) {
                Symbol *sym = env_get_var_visible_at(env, expr->as.call.name, expr->line, expr->column);
                if (sym && sym->type == TYPE_FUNCTION) {
                    /* Mark the variable as used */
                    sym->is_used = true;
                    
                    /* This is a call to a function parameter - check arguments */
                    for (int i = 0; i < expr->as.call.arg_count; i++) {
                        check_expression(expr->as.call.args[i], env);
                    }
                    /* If this is a call with no arguments (just getting the function), return TYPE_FUNCTION */
                    if (expr->as.call.arg_count == 0) {
                        return TYPE_FUNCTION;
                    }
                    /* Get return type from the function signature in type_info */
                    if (sym->type_info && sym->type_info->fn_sig) {
                        return sym->type_info->fn_sig->return_type;
                    }
                    /* Fallback if no signature available */
                    /* Return TYPE_INT for legacy compatibility (used in let statement checks) */
                    return TYPE_INT;
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
                                safe_fprintf(stderr, "Error at line %d, column %d: Undefined function '%s'\n",
                                        expr->line, expr->column, safe_format_string(expr->as.call.name));
                                safe_fprintf(stderr, "  Note: struct or enum '%s' is not defined\n", type_name);
                                safe_fprintf(stderr, "  Hint: Define 'struct %s { ... }' before using List<%s>\n", 
                                        type_name, type_name);
                                free(type_name);
                                return TYPE_UNKNOWN;
                            }
                        }
                    }
                }
                
                safe_fprintf(stderr, "Error at line %d, column %d: Undefined function '%s'\n",
                        expr->line, expr->column, safe_format_string(expr->as.call.name));
                return TYPE_UNKNOWN;
            }

            /* Check argument count */
            if (expr->as.call.arg_count != func->param_count) {
                safe_fprintf(stderr, "Error at line %d, column %d: Function '%s' expects %d arguments, got %d\n",
                        expr->line, expr->column, safe_format_string(expr->as.call.name), func->param_count, expr->as.call.arg_count);
                return TYPE_UNKNOWN;
            }

            /* Check argument types (skip for built-ins with NULL params like range) */
            if (func->params) {
                for (int i = 0; i < expr->as.call.arg_count; i++) {
                    ASTNode *arg = expr->as.call.args[i];
                    
                    /* Special handling for function-typed parameters */
                    if (func->params[i].type == TYPE_FUNCTION) {
                        /* Argument must be an identifier (function name or function-typed variable) */
                        if (arg->type != AST_IDENTIFIER) {
                            fprintf(stderr, "Error at line %d, column %d: Function parameter expects a function name\n",
                                    arg->line, arg->column);
                            return TYPE_UNKNOWN;
                        }
                        
                        /* First check if it's a function-typed variable */
                        Symbol *sym = env_get_var_visible_at(env, arg->as.identifier, arg->line, arg->column);
                        if (sym && sym->type == TYPE_FUNCTION) {
                            /* It's a function-typed variable - mark as used and allow it */
                            sym->is_used = true;
                            /* TODO: Check signature match when we store signatures in Symbol */
                            continue;  /* Skip to next argument */
                        }
                        
                        /* Look up the function */
                        Function *passed_func = env_get_function(env, arg->as.identifier);
                        if (!passed_func) {
                            fprintf(stderr, "Error at line %d, column %d: Undefined function '%s'\n",
                                    arg->line, arg->column, arg->as.identifier);
                            return TYPE_UNKNOWN;
                        }
                        
                        /* Create signature from passed function */
                        FunctionSignature passed_sig;
                        passed_sig.param_count = passed_func->param_count;
                        passed_sig.param_types = malloc(sizeof(Type) * passed_func->param_count);
                        passed_sig.param_struct_names = malloc(sizeof(char*) * passed_func->param_count);
                        for (int j = 0; j < passed_func->param_count; j++) {
                            passed_sig.param_types[j] = passed_func->params[j].type;
                            passed_sig.param_struct_names[j] = passed_func->params[j].struct_type_name;
                        }
                        passed_sig.return_type = passed_func->return_type;
                        passed_sig.return_struct_name = passed_func->return_struct_type_name;
                        
                        /* Compare signatures */
                        if (!function_signatures_equal(func->params[i].fn_sig, &passed_sig)) {
                            fprintf(stderr, "Error at line %d, column %d: Function signature mismatch for argument %d in call to '%s'\n",
                                    arg->line, arg->column, i + 1, expr->as.call.name);
                            fprintf(stderr, "  Expected function with different signature\n");
                            free(passed_sig.param_types);
                            free(passed_sig.param_struct_names);
                            return TYPE_UNKNOWN;
                        }
                        
                        /* Clean up temporary signature */
                        free(passed_sig.param_types);
                        free(passed_sig.param_struct_names);
                    } else {
                        /* Regular argument type checking */
                        Type arg_type = check_expression(arg, env);
                        
                        /* Check for opaque type parameters - allow 0 (null) as argument */
                        bool is_opaque_param = false;
                        if (func->params[i].type == TYPE_STRUCT && func->params[i].struct_type_name) {
                            OpaqueTypeDef *opaque = env_get_opaque_type(env, func->params[i].struct_type_name);
                            if (opaque) {
                                is_opaque_param = true;
                                /* For opaque types, allow TYPE_INT (for passing 0 as NULL) */
                                if (arg_type != TYPE_INT && arg_type != TYPE_STRUCT) {
                                    fprintf(stderr, "Error at line %d, column %d: Argument %d type mismatch in call to '%s'\n",
                                            expr->line, expr->column, i + 1, expr->as.call.name);
                                    fprintf(stderr, "  Expected opaque type '%s' or 0 (null), got %s\n",
                                            func->params[i].struct_type_name, type_to_string(arg_type));
                                }
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
                        
                        if (!is_opaque_param && !is_opaque_arg && !types_match(arg_type, func->params[i].type)) {
                            fprintf(stderr, "Error at line %d, column %d: Argument %d type mismatch in call to '%s'\n",
                                    expr->line, expr->column, i + 1, expr->as.call.name);
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
                /* at(array, index) returns the element type of the array */
                if (expr->as.call.arg_count >= 1) {
                    ASTNode *array_arg = expr->as.call.args[0];
                    
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
                if (elem_type != first_type) {
                    fprintf(stderr, "Error at line %d, column %d: Array elements must all have the same type. Expected %d, got %d\n",
                            expr->line, expr->column, first_type, elem_type);
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
                fprintf(stderr, "Error at line %d, column %d: If condition must be bool\n", expr->line, expr->column);
            }

            /* For if expressions, we need to infer the type from the blocks */
            /* This is simplified - just return UNKNOWN for now */
            /* A proper implementation would need to analyze the blocks */
            return TYPE_UNKNOWN;
        }

        case AST_STRUCT_LITERAL: {
            /* Check that struct is defined */
            StructDef *sdef = env_get_struct(env, expr->as.struct_literal.struct_name);
            if (!sdef) {
                fprintf(stderr, "Error at line %d, column %d: Undefined struct '%s'\n",
                        expr->line, expr->column, expr->as.struct_literal.struct_name);
                return TYPE_UNKNOWN;
            }
            
            /* Check that all fields are provided and types match */
            if (expr->as.struct_literal.field_count != sdef->field_count) {
                fprintf(stderr, "Error at line %d, column %d: Struct '%s' expects %d fields, got %d\n",
                        expr->line, expr->column, expr->as.struct_literal.struct_name,
                        sdef->field_count, expr->as.struct_literal.field_count);
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
                    fprintf(stderr, "Error at line %d, column %d: Unknown field '%s' in struct '%s'\n",
                            expr->line, expr->column, field_name, expr->as.struct_literal.struct_name);
                    continue;
                }
                
                /* Check field type */
                Type field_type = check_expression(expr->as.struct_literal.field_values[i], env);
                if (!types_match(field_type, sdef->field_types[field_index])) {
                    fprintf(stderr, "Error at line %d, column %d: Field '%s' type mismatch in struct '%s'\n",
                            expr->line, expr->column, field_name, expr->as.struct_literal.struct_name);
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
            if (object_type != TYPE_STRUCT) {
                safe_fprintf(stderr, "Error at line %d, column %d: Field access requires a struct\n",
                        expr->line, expr->column);
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
                    fprintf(stderr, "Error at line %d, column %d: Undefined union '%s'\n",
                            expr->line, expr->column, union_name);
                    free(union_name);
                    return TYPE_UNKNOWN;
                }
                
                /* Find the variant */
                int variant_idx = env_get_union_variant_index(env, union_name, variant_name);
                if (variant_idx < 0) {
                    fprintf(stderr, "Error at line %d, column %d: Unknown variant '%s' in union '%s'\n",
                            expr->line, expr->column, variant_name, union_name);
                    free(union_name);
                    return TYPE_UNKNOWN;
                }
                
                /* Find the field in the variant */
                const char *field_name = expr->as.field_access.field_name;
                for (int i = 0; i < udef->variant_field_counts[variant_idx]; i++) {
                    if (strcmp(udef->variant_field_names[variant_idx][i], field_name) == 0) {
                        Type field_type = udef->variant_field_types[variant_idx][i];
                        
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
                fprintf(stderr, "Error at line %d, column %d: Variant '%s' of union '%s' has no field '%s'\n",
                        expr->line, expr->column, variant_name, union_name, field_name);
                free(union_name);
                return TYPE_UNKNOWN;
            }
            
            /* Look up the struct definition */
            StructDef *sdef = env_get_struct(env, struct_name);
            if (!sdef) {
                fprintf(stderr, "Error at line %d, column %d: Undefined struct '%s'\n",
                        expr->line, expr->column, struct_name);
                return TYPE_UNKNOWN;
            }
            
            /* Find the field and return its type */
            const char *field_name = expr->as.field_access.field_name;
            for (int i = 0; i < sdef->field_count; i++) {
                if (strcmp(sdef->field_names[i], field_name) == 0) {
                    return sdef->field_types[i];
                }
            }
            
            /* Field not found */
            fprintf(stderr, "Error at line %d, column %d: Struct '%s' has no field '%s'\n",
                    expr->line, expr->column, struct_name, field_name);
            return TYPE_UNKNOWN;
        }

        case AST_UNION_CONSTRUCT: {
            /* Check that union is defined */
            UnionDef *udef = env_get_union(env, expr->as.union_construct.union_name);
            if (!udef) {
                fprintf(stderr, "Error at line %d, column %d: Undefined union '%s'\n",
                        expr->line, expr->column, expr->as.union_construct.union_name);
                return TYPE_UNKNOWN;
            }
            
            /* Check that variant exists */
            int variant_idx = env_get_union_variant_index(env, 
                expr->as.union_construct.union_name, 
                expr->as.union_construct.variant_name);
            if (variant_idx < 0) {
                fprintf(stderr, "Error at line %d, column %d: Unknown variant '%s' in union '%s'\n",
                        expr->line, expr->column, 
                        expr->as.union_construct.variant_name,
                        expr->as.union_construct.union_name);
                return TYPE_UNKNOWN;
            }
            
            /* Check that field count matches */
            int expected_field_count = udef->variant_field_counts[variant_idx];
            if (expr->as.union_construct.field_count != expected_field_count) {
                fprintf(stderr, "Error at line %d, column %d: Variant '%s' expects %d fields, got %d\n",
                        expr->line, expr->column,
                        expr->as.union_construct.variant_name,
                        expected_field_count,
                        expr->as.union_construct.field_count);
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
                    fprintf(stderr, "Error at line %d, column %d: Unknown field '%s' in variant '%s'\n",
                            expr->line, expr->column, field_name,
                            expr->as.union_construct.variant_name);
                    return TYPE_UNKNOWN;
                }
                
                /* Check field type */
                Type expected_type = udef->variant_field_types[variant_idx][field_index];
                Type actual_type = check_expression(expr->as.union_construct.field_values[i], env);
                
                /* For generic unions, accept any type for generic type parameters */
                /* TODO: Proper type substitution for generic instantiations */
                bool is_generic_param = (expected_type == TYPE_GENERIC || expected_type == TYPE_STRUCT);
                if (is_generic_param && udef->generic_param_count > 0) {
                    /* This is likely a generic type parameter - accept it for now */
                    /* The transpiler will handle concrete type generation */
                } else if (actual_type != expected_type) {
                    fprintf(stderr, "Error at line %d, column %d: Field '%s' expects type '%s', got '%s'\n",
                            expr->line, expr->column, field_name,
                            type_to_string(expected_type),
                            type_to_string(actual_type));
                    return TYPE_UNKNOWN;
                }
            }
            
            return TYPE_UNION;
        }

        case AST_MATCH: {
            /* Check the expression being matched */
            Type match_type = check_expression(expr->as.match_expr.expr, env);
            if (match_type != TYPE_UNION) {
                fprintf(stderr, "Error at line %d, column %d: Match expression must be a union type\n",
                        expr->line, expr->column);
                return TYPE_UNKNOWN;
            }
            
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

            union_base_name = union_type_name;
            if (union_type_info && union_type_info->generic_name && union_type_info->type_param_count > 0) {
                union_base_name = union_type_info->generic_name;
                union_concrete_name = typeinfo_to_monomorphized_generic_name(union_type_info);
            }
            
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
                /* Save symbol count for scope */
                int saved_symbol_count __attribute__((unused)) = env->symbol_count;
                
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
                    const char *variant_name = expr->as.match_expr.pattern_variants[i];
                    /* Format: "UnionName.VariantName" */
                    char *type_name = malloc(strlen(union_base_name) + strlen(variant_name) + 2);
                    sprintf(type_name, "%s.%s", union_base_name, variant_name);
                    binding_sym->struct_type_name = type_name;
                }
                
                /* Type check arm body (which is now an expression) */
                Type arm_type = check_expression(expr->as.match_expr.arm_bodies[i], env);
                
                /* NOTE: We do NOT restore symbol_count here because the transpiler needs these bindings
                 * later when it re-typechecks expressions for code generation. Match arm bindings need
                 * to remain in the environment for the lifetime of the compilation unit.
                 * This is safe because each arm's binding uses a unique name from the source code.
                 */
                
                /* First arm determines return type */
                if (i == 0) {
                    return_type = arm_type;
                } else if (arm_type != return_type && arm_type != TYPE_VOID) {
                    fprintf(stderr, "Error at line %d, column %d: Match arms must all return the same type\n",
                            expr->line, expr->column);
                }
            }
            
            return return_type;
        }

        case AST_BLOCK: {
            /* Blocks can be used as expressions in match arms
             * Type check all statements and return the type of the last expression/return
             */
            Type block_type = TYPE_VOID;
            
            /* Create a temporary TypeChecker for statement type checking */
            TypeChecker temp_tc;
            temp_tc.env = env;
            temp_tc.has_error = false;
            
            for (int i = 0; i < expr->as.block.count; i++) {
                ASTNode *stmt = expr->as.block.statements[i];
                if (stmt->type == AST_RETURN && stmt->as.return_stmt.value) {
                    block_type = check_expression(stmt->as.return_stmt.value, env);
                } else {
                    /* Type check the statement (for side effects) */
                    Type stmt_type = check_statement(&temp_tc, stmt);
                    /* If it's the last statement and not a return, use its type */
                    if (i == expr->as.block.count - 1 && stmt_type != TYPE_VOID) {
                        block_type = stmt_type;
                    }
                }
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
            
            /* For function returns or other complex expressions, we can't statically determine the type.
             * Return TYPE_INT as a conservative estimate.
             * TODO: Store TypeInfo in function return types for complete type checking.
             */
            return TYPE_INT;
        }

        default:
            fprintf(stderr, "Error at line %d, column %d: Invalid expression type\n", expr->line, expr->column);
            return TYPE_UNKNOWN;
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

    Type result = check_statement_impl(tc, stmt);
    g_check_stmt_depth--;
    return result;
}

/* Internal implementation of check_statement */
static Type check_statement_impl(TypeChecker *tc, ASTNode *stmt) {
    switch (stmt->type) {
        case AST_LET: {
            Type declared_type = stmt->as.let.var_type;
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
            
            /* Now check the expression - the specialized functions are registered */
            Type value_type = check_expression(stmt->as.let.value, tc->env);
            
            /* If declared type is STRUCT, check if it's actually an enum or union */
            if (declared_type == TYPE_STRUCT && stmt->as.let.type_name) {
                /* Check if this is actually a union */
                if (env_get_union(tc->env, stmt->as.let.type_name)) {
                    declared_type = TYPE_UNION;
                }
                /* Check if this is actually an enum */
                else if (env_get_enum(tc->env, stmt->as.let.type_name)) {
                    declared_type = TYPE_ENUM;
                }
            }
            /* Legacy handling for enums without type_name */
            else if (declared_type == TYPE_STRUCT && value_type == TYPE_INT) {
                /* This is okay - enums are compatible with ints */
                declared_type = TYPE_ENUM;
            }
            
            /* Special handling for function types - need to check signatures match */
            /* Also handle case where value_type is TYPE_INT (function-typed parameter placeholder) */
            if (declared_type == TYPE_FUNCTION && (value_type == TYPE_FUNCTION || value_type == TYPE_INT)) {
                /* Both are function types - check if signatures match */
                FunctionSignature *declared_sig = stmt->as.let.fn_sig;
                FunctionSignature *value_sig = NULL;
                
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
                    if (func && func->return_type == TYPE_FUNCTION) {
                        /* Function that returns a function */
                        value_sig = func->return_fn_sig;
                    } else if (func) {
                        /* Function name used as value - create signature from function definition */
                        Type *param_types = NULL;
                        if (func->params && func->param_count > 0) {
                            param_types = malloc(sizeof(Type) * func->param_count);
                            for (int i = 0; i < func->param_count; i++) {
                                param_types[i] = func->params[i].type;
                            }
                        }
                        value_sig = create_function_signature(param_types, func->param_count, func->return_type);
                        if (param_types) free(param_types);
                    } else {
                        /* Check if it's a function-typed variable */
                        Symbol *sym = env_get_var(tc->env, stmt->as.let.value->as.identifier);
                        if (sym && sym->type == TYPE_FUNCTION) {
                            /* TODO: Store function signature in Symbol for function-typed variables */
                            /* For now, allow it - runtime will handle */
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
            } else if (!types_match(value_type, declared_type)) {
                fprintf(stderr, "Error at line %d, column %d: Type mismatch in let statement\n", stmt->line, stmt->column);
                tc->has_error = true;
            }

            /* Extract element type if this is an array */
            Type element_type = stmt->as.let.element_type;  /* Get from type annotation if available */
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
                    /* Set element type on array literal so transpiler knows what to generate */
                    array_lit->as.array_literal.element_type = element_type;
                }
            }
            
            /* Create TypeInfo for tuples or use existing from parser for generic types */
            TypeInfo *type_info = stmt->as.let.type_info;  /* Use parser's TypeInfo if available */
            if (!type_info && declared_type == TYPE_TUPLE && stmt->as.let.value->type == AST_TUPLE_LITERAL) {
                /* Create TypeInfo from tuple literal */
                ASTNode *tuple_lit = stmt->as.let.value;
                type_info = malloc(sizeof(TypeInfo));
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
                }
            }
            
            /* Also try to infer from value expression if struct_type_name not set */
            sym = env_get_var(tc->env, stmt->as.let.name);
            if (sym && !sym->struct_type_name && value_type == TYPE_STRUCT) {
                /* Infer struct type name from the value expression */
                const char *struct_name = get_struct_type_name(stmt->as.let.value, tc->env);
                if (struct_name) {
                    sym->struct_type_name = strdup(struct_name);
                    /* Free the temporary string returned by get_struct_type_name */
                    free((void*)struct_name);
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
                fprintf(stderr, "Error at line %d, column %d: Undefined variable '%s'\n",
                        stmt->line, stmt->column, stmt->as.set.name);
                tc->has_error = true;
                return TYPE_VOID;
            }

            if (!sym->is_mut) {
                fprintf(stderr, "Error at line %d, column %d: Cannot assign to immutable variable '%s'\n",
                        stmt->line, stmt->column, stmt->as.set.name);
                tc->has_error = true;
            }

            Type value_type = check_expression(stmt->as.set.value, tc->env);

            /* Propagate element type to array literals for correct transpilation */
            if (sym->type == TYPE_ARRAY && sym->element_type != TYPE_UNKNOWN) {
                if (stmt->as.set.value->type == AST_ARRAY_LITERAL) {
                    ASTNode *array_lit = stmt->as.set.value;
                    array_lit->as.array_literal.element_type = sym->element_type;
                }
            }

            if (!types_match(value_type, sym->type)) {
                fprintf(stderr, "Error at line %d, column %d: Type mismatch in assignment\n", stmt->line, stmt->column);
                tc->has_error = true;
            }

            return TYPE_VOID;
        }

        case AST_WHILE: {
            Type cond_type = check_expression(stmt->as.while_stmt.condition, tc->env);
            if (cond_type != TYPE_BOOL) {
                fprintf(stderr, "Error at line %d, column %d: While condition must be bool\n", stmt->line, stmt->column);
                tc->has_error = true;
            }

            check_statement(tc, stmt->as.while_stmt.body);
            return TYPE_VOID;
        }

        case AST_FOR: {
            /* Save current symbol count to restore after loop */
            int old_symbol_count = tc->env->symbol_count;

            /* For loop variable has type int */
            Value val = create_void();
            env_define_var(tc->env, stmt->as.for_stmt.var_name, TYPE_INT, false, val);

            /* Range expression should return a range */
            check_expression(stmt->as.for_stmt.range_expr, tc->env);

            /* Check the loop body */
            check_statement(tc, stmt->as.for_stmt.body);

            /* Restore symbol count (remove loop variable and any vars declared in body) */
            /* Free the names that were allocated */
            for (int i = old_symbol_count; i < tc->env->symbol_count; i++) {
                free(tc->env->symbols[i].name);
                if (tc->env->symbols[i].value.type == VAL_STRING) {
                    free(tc->env->symbols[i].value.as.string_val);
                }
            }
            tc->env->symbol_count = old_symbol_count;

            return TYPE_VOID;
        }

        case AST_RETURN: {
            if (stmt->as.return_stmt.value) {
                Type return_type = check_expression(stmt->as.return_stmt.value, tc->env);
                if (!types_match(return_type, tc->current_function_return_type)) {
                    fprintf(stderr, "Error at line %d, column %d: Return type mismatch\n", stmt->line, stmt->column);
                    tc->has_error = true;
                }
            } else {
                if (tc->current_function_return_type != TYPE_VOID) {
                    fprintf(stderr, "Error at line %d, column %d: Function must return a value\n", stmt->line, stmt->column);
                    tc->has_error = true;
                }
            }
            return tc->current_function_return_type;
        }

        case AST_BLOCK: {
            Type last_type = TYPE_VOID;
            for (int i = 0; i < stmt->as.block.count; i++) {
                last_type = check_statement(tc, stmt->as.block.statements[i]);
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

        case AST_IF: {
            /* Type check if statement */
            Type cond_type = check_expression(stmt->as.if_stmt.condition, tc->env);
            if (cond_type != TYPE_BOOL) {
                fprintf(stderr, "Error at line %d, column %d: If condition must be bool\n", 
                        stmt->line, stmt->column);
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

        case AST_MATCH: {
            /* Match used as a statement: type check arms as statements so return statements
             * inside match arms are checked against the current function's return type.
             * (The expression-mode match checker uses a temporary TypeChecker without
             * current_function_return_type initialized, which can produce spurious errors.)
             */
            Type match_type = check_expression(stmt->as.match_expr.expr, tc->env);
            if (match_type != TYPE_UNION) {
                fprintf(stderr, "Error at line %d, column %d: Match expression must be a union type\n",
                        stmt->line, stmt->column);
                tc->has_error = true;
                return TYPE_VOID;
            }

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

            union_base_name = union_type_name;
            if (union_type_info && union_type_info->generic_name && union_type_info->type_param_count > 0) {
                union_base_name = union_type_info->generic_name;
                union_concrete_name = typeinfo_to_monomorphized_generic_name(union_type_info);
            }

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
                Value binding_val = create_void();
                env_define_var_with_type_info(tc->env,
                    stmt->as.match_expr.pattern_bindings[i],
                    TYPE_STRUCT, TYPE_UNKNOWN, union_type_info, false, binding_val);

                if (union_base_name && tc->env->symbol_count > 0) {
                    Symbol *binding_sym = &tc->env->symbols[tc->env->symbol_count - 1];
                    const char *variant_name = stmt->as.match_expr.pattern_variants[i];
                    char *type_name = malloc(strlen(union_base_name) + strlen(variant_name) + 2);
                    sprintf(type_name, "%s.%s", union_base_name, variant_name);
                    binding_sym->struct_type_name = type_name;
                }

                ASTNode *arm = stmt->as.match_expr.arm_bodies[i];
                if (arm && arm->type == AST_BLOCK) {
                    check_statement(tc, arm);
                } else {
                    check_expression(arm, tc->env);
                }

                /* NOTE: We do NOT restore symbol_count here because the transpiler needs these bindings
                 * later when it re-typechecks expressions for code generation. Match arm bindings need
                 * to remain in the environment for the lifetime of the compilation unit.
                 * This is safe because each arm's binding uses a unique name from the source code.
                 */
            }

            return TYPE_VOID;
        }
        
        case AST_PREFIX_OP:
        case AST_CALL:
            /* Expression statements */
            check_expression(stmt, tc->env);
            return TYPE_VOID;

        default:
            /* Literals and identifiers as statements */
            check_expression(stmt, tc->env);
            return TYPE_VOID;
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
    "cast_int", "cast_float", "cast_bool", "cast_string", "cast_bstring", "to_string",
    /* String (C strings) */
    "str_length", "str_concat", "str_substring", "str_contains", "str_equals",
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
    /* List operations - list_token */
    "list_token_new", "list_token_with_capacity", "list_token_push", "list_token_pop",
    "list_token_get", "list_token_set", "list_token_insert", "list_token_remove",
    "list_token_length", "list_token_capacity", "list_token_is_empty", "list_token_clear",
    "list_token_free"
};

static const int builtin_function_name_count = sizeof(builtin_function_names) / sizeof(char*);

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
    /* Important: zero-init so visibility/module_name pointers don't contain garbage.
     * Module-aware typechecking calls strcmp() on module_name. */
    Function func = (Function){0};
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
    
    /* list_token operations */
    func.name = "list_token_new";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_LIST_TOKEN;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_with_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_LIST_TOKEN;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_push";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_pop";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRUCT;  /* Returns Token struct */
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_get";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRUCT;  /* Returns Token struct */
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_set";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_insert";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_remove";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRUCT;  /* Returns Token struct */
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_is_empty";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_clear";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_free";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
}

bool type_check(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program AST\n");
        return false;
    }

    TypeChecker tc;
    tc.env = env;
    tc.has_error = false;
    tc.warnings_enabled = true;  /* Enable unused variable warnings */

    /* Register built-in functions */
    register_builtin_functions(env);

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
            if (env_get_struct(env, struct_name)) {
                fprintf(stderr, "Error at line %d, column %d: Struct '%s' is already defined\n",
                        item->line, item->column, struct_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register the struct */
            StructDef sdef;
            sdef.name = strdup(struct_name);
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
            
            env_define_struct(env, sdef);
            
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
            UnionDef udef;
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
            
            env_define_union(env, udef);
            
        } else if (item->type == AST_OPAQUE_TYPE) {
            /* Register opaque type */
            const char *type_name = item->as.opaque_type.name;
            
            /* Check if opaque type already defined */
            if (env_get_opaque_type(env, type_name)) {
                fprintf(stderr, "Error at line %d, column %d: Opaque type '%s' is already defined\n",
                        item->line, item->column, type_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register the opaque type in environment */
            env_define_opaque_type(env, type_name);
            
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
            
            env_define_enum(env, edef);
            
        } else if (item->type == AST_FUNCTION) {
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
            if (existing) {
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
            
            Function func;
            func.name = strdup(func_name);  /* Create copy to avoid const qualifier warning */
            func.params = item->as.function.params;
            func.param_count = item->as.function.param_count;
            func.return_type = return_type;
            func.return_type_info = NULL;
            func.return_struct_type_name = item->as.function.return_struct_type_name;
            func.return_fn_sig = item->as.function.return_fn_sig;  /* Store function signature for TYPE_FUNCTION returns */
            func.return_type_info = item->as.function.return_type_info;  /* Store tuple type info for TYPE_TUPLE returns */
            func.body = item->as.function.body;
            func.shadow_test = NULL;
            func.is_extern = item->as.function.is_extern;
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
                        func.module_name = strdup(module_name_from_ast);
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
                    func.module_name = strdup(env->current_module);
                }
            }

            env_define_function(env, func);
            
            /* Trace function definition */
            if (!func.is_extern) {
                trace_function_def(func_name, func.params, func.param_count,
                                  func.return_type, item->line, item->column);
            }
        }
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
    
    /* Process top-level constants (before type checking functions) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_LET) {
            /* Type check the constant's initial value */
            Type value_type = check_expression(item->as.let.value, env);
            
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
            env_define_var(env, item->as.let.name, item->as.let.var_type, false, val);
        }
    }

    /* Third pass: type check all functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            /* Skip extern functions - they have no body to check */
            if (item->as.function.is_extern) {
                continue;
            }
            
            /* Get the resolved return type from the function definition in env */
            Function *func_def = env_get_function(env, item->as.function.name);
            tc.current_function_return_type = func_def ? func_def->return_type : item->as.function.return_type;
            
            /* Register generic union instantiation for function return type */
            if (item->as.function.return_type == TYPE_UNION &&
                item->as.function.return_type_info &&
                item->as.function.return_type_info->generic_name &&
                item->as.function.return_type_info->type_param_count > 0) {
                
                TypeInfo *info = item->as.function.return_type_info;
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

            /* Save current symbol count for scope restoration */
            int saved_symbol_count = env->symbol_count;

            /* Add parameters to environment (create a scope) */
            for (int j = 0; j < item->as.function.param_count; j++) {
                Value val = create_void();
                Type param_type = item->as.function.params[j].type;
                Type element_type = item->as.function.params[j].element_type;  /* Get actual element type from parameter */
                TypeInfo *param_type_info = item->as.function.params[j].type_info;  /* Get TypeInfo for generic types */
                
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
                    TypeInfo *type_info = malloc(sizeof(TypeInfo));
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

                    if ((param_type == TYPE_STRUCT || param_type == TYPE_UNION) && 
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

            /* Check for unused variables before leaving scope */
            check_unused_variables(&tc, saved_symbol_count);

            /* DON'T restore environment - transpiler needs these symbols! */
            /* The old code removed function-local symbols after typechecking:
             *   env->symbol_count = saved_symbol_count;
             * This caused array<struct> to fail because transpiler couldn't find
             * the struct_type_name metadata. Now we keep all symbols so transpiler
             * can access type information. C's function-local scope prevents collisions.
             */

            /* Verify function has shadow test (skip for extern functions and functions that use extern functions) */
            Function *func = env_get_function(env, item->as.function.name);
            if (!func->is_extern && !func->shadow_test) {
                /* Check if function body uses extern functions - if so, shadow test is optional */
                bool uses_extern = func->body && contains_extern_calls(func->body, env);
                if (!uses_extern) {
                    fprintf(stderr, "Warning: Function '%s' is missing a shadow test\n",
                            item->as.function.name);
                    /* Don't fail - just warn */
                    /* tc.has_error = true; */
                }
            }
        }
    }

    /* Verify main function exists */
    Function *main_func = env_get_function(env, "main");
    if (!main_func) {
        fprintf(stderr, "Error: Program must define a 'main' function\n");
        tc.has_error = true;
    } else if (main_func->return_type != TYPE_INT) {
        fprintf(stderr, "Error: 'main' function must return int\n");
        tc.has_error = true;
    }

    return !tc.has_error;
}

/* Type check a module (without requiring main function) */
bool type_check_module(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program AST\n");
        return false;
    }

    TypeChecker tc;
    tc.env = env;
    tc.has_error = false;
    tc.warnings_enabled = true;

    /* Register built-in functions */
    register_builtin_functions(env);

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
            if (env_get_struct(env, struct_name)) {
                fprintf(stderr, "Error at line %d, column %d: Struct '%s' is already defined\n",
                        item->line, item->column, struct_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register the struct */
            StructDef sdef;
            sdef.name = strdup(struct_name);
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
            
            env_define_struct(env, sdef);
            
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
            UnionDef udef;
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
            
            env_define_union(env, udef);
            
        } else if (item->type == AST_OPAQUE_TYPE) {
            /* Register opaque type */
            const char *type_name = item->as.opaque_type.name;
            
            /* Check if opaque type already defined */
            if (env_get_opaque_type(env, type_name)) {
                fprintf(stderr, "Error at line %d, column %d: Opaque type '%s' is already defined\n",
                        item->line, item->column, type_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register the opaque type in environment */
            env_define_opaque_type(env, type_name);
            
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
            
            env_define_enum(env, edef);
            
        } else if (item->type == AST_FUNCTION) {
            const char *func_name = item->as.function.name;
            
            /* Check for duplicate function definitions */
            Function *existing = env_get_function(env, func_name);
            if (existing) {
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
            Function f;
            f.name = strdup(func_name);
            f.param_count = item->as.function.param_count;
            f.params = malloc(sizeof(Parameter) * f.param_count);
            for (int j = 0; j < f.param_count; j++) {
                f.params[j].name = strdup(item->as.function.params[j].name);
                f.params[j].type = item->as.function.params[j].type;
                f.params[j].struct_type_name = item->as.function.params[j].struct_type_name ? 
                    strdup(item->as.function.params[j].struct_type_name) : NULL;
                f.params[j].element_type = item->as.function.params[j].element_type;
                f.params[j].fn_sig = item->as.function.params[j].fn_sig;
            }
            f.return_type = item->as.function.return_type;
            f.return_struct_type_name = item->as.function.return_struct_type_name ? 
                strdup(item->as.function.return_struct_type_name) : NULL;
            f.return_fn_sig = item->as.function.return_fn_sig;
            f.return_type_info = item->as.function.return_type_info;
            f.body = item->as.function.body;
            f.shadow_test = NULL;  /* Will be linked in second pass */
            f.is_extern = item->as.function.is_extern;
            f.is_pub = item->as.function.is_pub;  /* Store visibility */
            f.module_name = env->current_module ? strdup(env->current_module) : NULL;
            
            env_define_function(env, f);
        }
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

    /* Process top-level constants (before type checking functions) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_LET) {
            /* Type check the constant's initial value */
            Type value_type = check_expression(item->as.let.value, env);
            
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
            env_define_var(env, item->as.let.name, item->as.let.var_type, false, val);
        }
    }

    /* Third pass: type check all statements and expressions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        /* Skip imports, struct/enum/union definitions, and shadow tests */
        if (item->type == AST_IMPORT || 
            item->type == AST_STRUCT_DEF ||
            item->type == AST_ENUM_DEF ||
            item->type == AST_UNION_DEF ||
            item->type == AST_SHADOW) {
            continue;
        }
        
        if (item->type == AST_FUNCTION) {
            /* Save current symbol count */
            int saved_symbol_count = env->symbol_count;
            
            /* Set current function return type for return statement checking */
            tc.current_function_return_type = item->as.function.return_type;

            /* Register generic union instantiation for function return type */
            if (item->as.function.return_type == TYPE_UNION &&
                item->as.function.return_type_info &&
                item->as.function.return_type_info->generic_name &&
                item->as.function.return_type_info->type_param_count > 0) {
                TypeInfo *info = item->as.function.return_type_info;

                /* Best-effort: only register if the generic union exists */
                UnionDef *udef = env_get_union(env, info->generic_name);
                if (udef && udef->generic_param_count == info->type_param_count) {
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
            
            /* Add function parameters to environment */
            for (int j = 0; j < item->as.function.param_count; j++) {
                Type param_type = item->as.function.params[j].type;
                Type element_type = item->as.function.params[j].element_type;
                TypeInfo *param_type_info = item->as.function.params[j].type_info;
                Value val;
                if (param_type == TYPE_INT) val = create_int(0);
                else if (param_type == TYPE_FLOAT) val = create_float(0.0);
                else if (param_type == TYPE_BOOL) val = create_bool(false);
                else if (param_type == TYPE_STRING) val = create_string("");
                else if (param_type == TYPE_ARRAY) {
                    val = create_array((ValueType)element_type, 0, 0);
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
                    TypeInfo *type_info = malloc(sizeof(TypeInfo));
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

                    if ((param_type == TYPE_STRUCT || param_type == TYPE_UNION) && 
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

            /* Check for unused variables before leaving scope */
            check_unused_variables(&tc, saved_symbol_count);

            /* DON'T restore environment - transpiler needs these symbols! */
            /* The old code removed function-local symbols after typechecking:
             *   env->symbol_count = saved_symbol_count;
             * This caused array<struct> to fail because transpiler couldn't find
             * the struct_type_name metadata. Now we keep all symbols so transpiler
             * can access type information. C's function-local scope prevents collisions.
             */

            /* Verify function has shadow test (skip for extern functions and functions that use extern functions) */
            Function *func = env_get_function(env, item->as.function.name);
            if (!func->is_extern && !func->shadow_test) {
                /* Check if function body uses extern functions - if so, shadow test is optional */
                bool uses_extern = func->body && contains_extern_calls(func->body, env);
                if (!uses_extern) {
                    fprintf(stderr, "Warning: Function '%s' is missing a shadow test\n",
                            item->as.function.name);
                    /* Don't fail - just warn */
                    /* tc.has_error = true; */
                }
            }
        }
    }

    /* Note: Modules don't require a main function */
    /* Main function check is skipped for modules */

    return !tc.has_error;
}