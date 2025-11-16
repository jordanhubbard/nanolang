#include "nanolang.h"
#include "tracing.h"

/* Type checking context */
typedef struct {
    Environment *env;
    Type current_function_return_type;
    bool has_error;
    bool warnings_enabled;
} TypeChecker;

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

/* Utility functions */
Type token_to_type(TokenType token) {
    switch (token) {
        case TOKEN_TYPE_INT: return TYPE_INT;
        case TOKEN_TYPE_FLOAT: return TYPE_FLOAT;
        case TOKEN_TYPE_BOOL: return TYPE_BOOL;
        case TOKEN_TYPE_STRING: return TYPE_STRING;
        case TOKEN_TYPE_VOID: return TYPE_VOID;
        default: return TYPE_UNKNOWN;
    }
}

const char *type_to_string(Type type) {
    switch (type) {
        case TYPE_INT: return "int";
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
    
    /* Generic lists match with int (list functions return int handles) */
    if ((t1 == TYPE_LIST_GENERIC && t2 == TYPE_INT) ||
        (t1 == TYPE_INT && t2 == TYPE_LIST_GENERIC)) {
        return true;
    }
    
    return false;
}

/* Helper: Get the struct type name from an expression (returns NULL if not a struct) */
static const char *get_struct_type_name(ASTNode *expr, Environment *env) {
    if (!expr) return NULL;
    
    switch (expr->type) {
        case AST_STRUCT_LITERAL:
            return expr->as.struct_literal.struct_name;
            
        case AST_IDENTIFIER: {
            Symbol *sym = env_get_var(env, expr->as.identifier);
            if (sym && sym->type == TYPE_STRUCT) {
                return sym->struct_type_name;
            }
            return NULL;
        }
        
        case AST_CALL: {
            /* Check if function returns a struct */
            Function *func = env_get_function(env, expr->as.call.name);
            if (func && func->return_type == TYPE_STRUCT) {
                return func->return_struct_type_name;
            }
            return NULL;
        }
        
        case AST_FIELD_ACCESS: {
            /* Get the struct type of the object */
            const char *object_struct_name = get_struct_type_name(expr->as.field_access.object, env);
            if (!object_struct_name) return NULL;
            
            /* Look up the struct definition */
            StructDef *sdef = env_get_struct(env, object_struct_name);
            if (!sdef) return NULL;
            
            /* Find the field */
            for (int i = 0; i < sdef->field_count; i++) {
                if (strcmp(sdef->field_names[i], expr->as.field_access.field_name) == 0) {
                    /* If the field is a struct, we'd need to track its type name */
                    /* For now, we only track top-level struct names */
                    return NULL;
                }
            }
            return NULL;
        }
        
        default:
            return NULL;
    }
}

/* Check expression type */
Type check_expression(ASTNode *expr, Environment *env) {
    if (!expr) return TYPE_UNKNOWN;

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
            Symbol *sym = env_get_var(env, expr->as.identifier);
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

        case AST_PREFIX_OP: {
            TokenType op = expr->as.prefix_op.op;
            int arg_count = expr->as.prefix_op.arg_count;

            /* Arithmetic operators */
            if (op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR ||
                op == TOKEN_SLASH || op == TOKEN_PERCENT) {
                if (arg_count != 2) {
                    fprintf(stderr, "Error at line %d, column %d: Arithmetic operators require 2 arguments\n", expr->line, expr->column);
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                if (left == TYPE_INT && right == TYPE_INT) return TYPE_INT;
                if (left == TYPE_FLOAT && right == TYPE_FLOAT) return TYPE_FLOAT;

                fprintf(stderr, "Error at line %d, column %d: Type mismatch in arithmetic operation\n", expr->line, expr->column);
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

                if (!types_match(left, right)) {
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
            /* Check if function exists */
            Function *func = env_get_function(env, expr->as.call.name);
            
            /* If not a function, check if it's a function-typed variable (parameter) */
            if (!func) {
                Symbol *sym = env_get_var(env, expr->as.call.name);
                if (sym && sym->type == TYPE_FUNCTION) {
                    /* This is a call to a function parameter - we can't fully type check it */
                    /* Just check that arguments are valid expressions */
                    for (int i = 0; i < expr->as.call.arg_count; i++) {
                        check_expression(expr->as.call.args[i], env);
                    }
                    /* Return type is unknown - function signatures don't store this info yet */
                    /* TODO: Store full function signature in Symbol for better type checking */
                    return TYPE_INT;  /* Assume int for now */
                }
                
                fprintf(stderr, "Error at line %d, column %d: Undefined function '%s'\n",
                        expr->line, expr->column, expr->as.call.name);
                return TYPE_UNKNOWN;
            }

            /* Check argument count */
            if (expr->as.call.arg_count != func->param_count) {
                fprintf(stderr, "Error at line %d, column %d: Function '%s' expects %d arguments, got %d\n",
                        expr->line, expr->column, expr->as.call.name, func->param_count, expr->as.call.arg_count);
                return TYPE_UNKNOWN;
            }

            /* Check argument types (skip for built-ins with NULL params like range) */
            if (func->params) {
                for (int i = 0; i < expr->as.call.arg_count; i++) {
                    ASTNode *arg = expr->as.call.args[i];
                    
                    /* Special handling for function-typed parameters */
                    if (func->params[i].type == TYPE_FUNCTION) {
                        /* Argument must be an identifier (function name) */
                        if (arg->type != AST_IDENTIFIER) {
                            fprintf(stderr, "Error at line %d, column %d: Function parameter expects a function name\n",
                                    arg->line, arg->column);
                            return TYPE_UNKNOWN;
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
                        if (!types_match(arg_type, func->params[i].type)) {
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
            if (strcmp(expr->as.call.name, "at") == 0) {
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
                        Symbol *sym = env_get_var(env, array_arg->as.identifier);
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
                    
                    /* Fallback: try to infer from the array_literal's stored element_type */
                    Type array_type = check_expression(array_arg, env);
                    if (array_type == TYPE_ARRAY && array_arg->type == AST_ARRAY_LITERAL) {
                        return array_arg->as.array_literal.element_type;
                    }
                }
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
            if (!expr->as.field_access.object) {
                fprintf(stderr, "Error at line %d, column %d: NULL object in field access\n",
                        expr->line, expr->column);
                return TYPE_UNKNOWN;
            }
            
            /* Special case: Check if this is an enum variant access */
            if (expr->as.field_access.object->type == AST_IDENTIFIER) {
                const char *enum_name = expr->as.field_access.object->as.identifier;
                if (!enum_name) {
                    fprintf(stderr, "Error at line %d, column %d: NULL enum name in field access\n",
                            expr->line, expr->column);
                    return TYPE_UNKNOWN;
                }
                EnumDef *enum_def = env_get_enum(env, enum_name);
                
                if (enum_def && enum_def->variant_names) {
                    /* This is an enum variant access (e.g., Color.Red) */
                    const char *variant_name = expr->as.field_access.field_name;
                    
                    if (!variant_name) {
                        fprintf(stderr, "Error at line %d, column %d: NULL variant name in enum access\n",
                                expr->line, expr->column);
                        return TYPE_UNKNOWN;
                    }
                    
                    /* Verify variant exists */
                    for (int i = 0; i < enum_def->variant_count; i++) {
                        if (enum_def->variant_names[i] && strcmp(enum_def->variant_names[i], variant_name) == 0) {
                            return TYPE_INT;  /* Enums are represented as integers */
                        }
                    }
                    
                    fprintf(stderr, "Error at line %d, column %d: Enum '%s' has no variant '%s'\n",
                            expr->line, expr->column, enum_name, variant_name);
                    return TYPE_UNKNOWN;
                }
            }
            
            /* Regular struct field access */
            /* Check the object type */
            Type object_type = check_expression(expr->as.field_access.object, env);
            if (object_type != TYPE_STRUCT) {
                fprintf(stderr, "Error at line %d, column %d: Field access requires a struct\n",
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
                
                if (actual_type != expected_type) {
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
            const char *union_type_name = NULL;
            ASTNode *match_expr_node = expr->as.match_expr.expr;
            
            if (match_expr_node->type == AST_IDENTIFIER) {
                Symbol *sym = env_get_var(env, match_expr_node->as.identifier);
                if (sym && sym->struct_type_name) {
                    union_type_name = sym->struct_type_name;
                }
            } else if (match_expr_node->type == AST_UNION_CONSTRUCT) {
                union_type_name = match_expr_node->as.union_construct.union_name;
            } else if (match_expr_node->type == AST_CALL) {
                Function *func = env_get_function(env, match_expr_node->as.call.name);
                if (func && func->return_struct_type_name) {
                    union_type_name = func->return_struct_type_name;
                }
            }
            
            if (union_type_name) {
                expr->as.match_expr.union_type_name = strdup(union_type_name);
            }
            
            /* Check each arm and infer return type from first arm */
            Type return_type = TYPE_UNKNOWN;
            for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                /* Save symbol count for scope */
                int saved_symbol_count = env->symbol_count;
                
                /* Add pattern binding to environment */
                Value binding_val = create_void();
                env_define_var_with_element_type(env, 
                    expr->as.match_expr.pattern_bindings[i],
                    TYPE_UNION, TYPE_UNKNOWN, false, binding_val);
                
                /* Type check arm body (which is now an expression) */
                Type arm_type = check_expression(expr->as.match_expr.arm_bodies[i], env);
                
                /* Restore scope */
                env->symbol_count = saved_symbol_count;
                
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

        default:
            fprintf(stderr, "Error at line %d, column %d: Invalid expression type\n", expr->line, expr->column);
            return TYPE_UNKNOWN;
    }
}

/* Check statement and return its type (for blocks) */
static Type check_statement(TypeChecker *tc, ASTNode *stmt) {
    if (!stmt) return TYPE_VOID;

    switch (stmt->type) {
        case AST_LET: {
            Type declared_type = stmt->as.let.var_type;
            
            /* Handle generic lists: List<UserType> - Register BEFORE checking expression */
            if (declared_type == TYPE_LIST_GENERIC && stmt->as.let.type_name) {
                const char *element_type = stmt->as.let.type_name;
                
                /* Verify element type exists (struct must be defined) */
                if (!env_get_struct(tc->env, element_type)) {
                    fprintf(stderr, "Error at line %d, column %d: Unknown type '%s' in List<%s>\n",
                            stmt->line, stmt->column, element_type, element_type);
                    tc->has_error = true;
                } else {
                    /* Register this instantiation for code generation */
                    env_register_list_instantiation(tc->env, element_type);
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
                /* Check if this is actually an enum (treat enums as int) */
                else if (env_get_enum(tc->env, stmt->as.let.type_name)) {
                    declared_type = TYPE_INT;
                }
            }
            /* Legacy handling for enums without type_name */
            else if (declared_type == TYPE_STRUCT && value_type == TYPE_INT) {
                /* This is okay - enums are represented as ints */
                declared_type = TYPE_INT;
            }
            
            if (!types_match(value_type, declared_type)) {
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
                }
            }
            
            /* Add to environment */
            Value val = create_void(); /* Placeholder */
            env_define_var_with_element_type(tc->env, stmt->as.let.name, declared_type, element_type, stmt->as.let.is_mut, val);
            
            /* Store definition location for unused variable warnings */
            Symbol *sym = env_get_var(tc->env, stmt->as.let.name);
            if (sym) {
                sym->def_line = stmt->line;
                sym->def_column = stmt->column;
                
                /* If this is a struct, store the struct type name */
                if (value_type == TYPE_STRUCT) {
                    const char *struct_name = get_struct_type_name(stmt->as.let.value, tc->env);
                    if (struct_name) {
                        sym->struct_type_name = strdup(struct_name);
                    }
                }
                
                /* If this is a union, store the union type name */
                if (declared_type == TYPE_UNION && stmt->as.let.type_name) {
                    sym->struct_type_name = strdup(stmt->as.let.type_name);
                }
            }
            
            return TYPE_VOID;
        }

        case AST_SET: {
            Symbol *sym = env_get_var(tc->env, stmt->as.set.name);
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

        case AST_IF:
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
    "sin", "cos", "tan",
    /* String */
    "str_length", "str_concat", "str_substring", "str_contains", "str_equals",
    /* Advanced string operations */
    "char_at", "string_from_char",
    /* Character classification */
    "is_digit", "is_alpha", "is_alnum", "is_whitespace", "is_upper", "is_lower",
    /* Type conversions */
    "int_to_string", "string_to_int", "digit_value", "char_to_lower", "char_to_upper",
    /* Array */
    "at", "array_length", "array_new", "array_set",
    /* OS */
    "getcwd", "getenv", "exit",
    /* File I/O (stdlib functions) */
    "file_read", "file_write", "file_append", "file_remove", "file_rename",
    "file_exists", "file_size",
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
    Function func;
    
    /* range(start: int, end: int) -> void (special - only for for-loops) */
    func.name = "range";
    func.params = NULL;  /* Special handling */
    func.param_count = 2;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* abs(x: int|float) -> int|float */
    func.name = "abs";
    func.params = NULL;  /* Accept int or float */
    func.param_count = 1;
    func.return_type = TYPE_INT;  /* Can also be float */
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* min(a: int|float, b: int|float) -> int|float */
    func.name = "min";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_INT;  /* Can also be float */
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* max(a: int|float, b: int|float) -> int|float */
    func.name = "max";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_INT;  /* Can also be float */
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* print(x: any) -> void */
    func.name = "print";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* println(x: any) -> void */
    func.name = "println";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
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
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* pow(base: int|float, exponent: int|float) -> float */
    func.name = "pow";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* floor(x: int|float) -> float */
    func.name = "floor";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* ceil(x: int|float) -> float */
    func.name = "ceil";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* round(x: int|float) -> float */
    func.name = "round";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
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
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* cos(x: int|float) -> float */
    func.name = "cos";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* tan(x: int|float) -> float */
    func.name = "tan";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
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
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* str_concat(s1: string, s2: string) -> string */
    func.name = "str_concat";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* str_substring(s: string, start: int, length: int) -> string */
    func.name = "str_substring";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* str_contains(s: string, substr: string) -> bool */
    func.name = "str_contains";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* str_equals(s1: string, s2: string) -> bool */
    func.name = "str_equals";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_BOOL;
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
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* array_length(arr: array<T>) -> int */
    func.name = "array_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* array_new(size: int, default: T) -> array<T> */
    func.name = "array_new";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_ARRAY;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* array_set(arr: mut array<T>, index: int, value: T) -> void */
    func.name = "array_set";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* OS built-ins */
    func.name = "getcwd";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "getenv";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* Advanced string operations */
    func.name = "char_at";
    func.params = NULL;
    func.param_count = 2;  /* string, index */
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "string_from_char";
    func.params = NULL;
    func.param_count = 1;  /* char code */
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* Character classification */
    func.name = "is_digit";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "is_alpha";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "is_alnum";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "is_whitespace";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "is_upper";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "is_lower";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* Type conversions */
    func.name = "int_to_string";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "string_to_int";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "digit_value";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "char_to_lower";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "char_to_upper";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* list_int operations */
    func.name = "list_int_new";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_LIST_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_with_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_LIST_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_push";
    func.params = NULL;
    func.param_count = 2;  /* list, value */
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_pop";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_get";
    func.params = NULL;
    func.param_count = 2;  /* list, index */
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_set";
    func.params = NULL;
    func.param_count = 3;  /* list, index, value */
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_insert";
    func.params = NULL;
    func.param_count = 3;  /* list, index, value */
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_remove";
    func.params = NULL;
    func.param_count = 2;  /* list, index */
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_is_empty";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_clear";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_int_free";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* list_string operations */
    func.name = "list_string_new";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_LIST_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_with_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_LIST_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_push";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_pop";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_get";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_set";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_insert";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_remove";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_is_empty";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_clear";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_string_free";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    /* list_token operations */
    func.name = "list_token_new";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_LIST_TOKEN;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_with_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_LIST_TOKEN;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_push";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_pop";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRUCT;  /* Returns Token struct */
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_get";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRUCT;  /* Returns Token struct */
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_set";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_insert";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_remove";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRUCT;  /* Returns Token struct */
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_capacity";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_is_empty";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_clear";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
    
    func.name = "list_token_free";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = false;
    env_define_function(env, func);
}

/* Compute Levenshtein distance between two strings */
static int levenshtein_distance(const char *s1, const char *s2) {
    /* Handle NULL pointers */
    if (!s1 || !s2) {
        /* If both are NULL, they're the same */
        if (!s1 && !s2) return 0;
        /* If one is NULL, return a large distance */
        /* Don't call strlen on NULL - just return a safe large value */
        return 999;
    }
    
    int len1 = strlen(s1);
    int len2 = strlen(s2);
    
    /* Create distance matrix */
    int **d = malloc((len1 + 1) * sizeof(int *));
    for (int i = 0; i <= len1; i++) {
        d[i] = malloc((len2 + 1) * sizeof(int));
    }
    
    /* Initialize first column and row */
    for (int i = 0; i <= len1; i++) {
        d[i][0] = i;
    }
    for (int j = 0; j <= len2; j++) {
        d[0][j] = j;
    }
    
    /* Compute distance */
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            
            int deletion = d[i-1][j] + 1;
            int insertion = d[i][j-1] + 1;
            int substitution = d[i-1][j-1] + cost;
            
            d[i][j] = deletion;
            if (insertion < d[i][j]) d[i][j] = insertion;
            if (substitution < d[i][j]) d[i][j] = substitution;
        }
    }
    
    int result = d[len1][len2];
    
    /* Free matrix */
    for (int i = 0; i <= len1; i++) {
        free(d[i]);
    }
    free(d);
    
    return result;
}

/* Check for similar function names and warn */
static void warn_similar_function_names(Environment *env) {
    for (int i = 0; i < env->function_count; i++) {
        for (int j = i + 1; j < env->function_count; j++) {
            /* Skip if either function doesn't have a body (shouldn't happen for user functions) */
            if (!env->functions[i].body || !env->functions[j].body) {
                continue;
            }
            
            /* Skip if function names are NULL */
            if (!env->functions[i].name || !env->functions[j].name) {
                continue;
            }
            
            int dist = levenshtein_distance(
                env->functions[i].name,
                env->functions[j].name
            );
            
            /* Warn if edit distance is small (1-2 characters) */
            if (dist > 0 && dist <= 2) {
                fprintf(stderr, "\nWarning: Function names '%s' and '%s' are very similar (edit distance: %d)\n",
                        env->functions[i].name,
                        env->functions[j].name,
                        dist);
                fprintf(stderr, "  '%s' defined at line %d, column %d\n",
                        env->functions[i].name,
                        env->functions[i].body->line,
                        env->functions[i].body->column);
                fprintf(stderr, "  '%s' defined at line %d, column %d\n",
                        env->functions[j].name,
                        env->functions[j].body->line,
                        env->functions[j].body->column);
                fprintf(stderr, "  Did you mean to define the same function twice?\n");
            }
        }
    }
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
            
            env_define_union(env, udef);
            
        } else if (item->type == AST_ENUM_DEF) {
            const char *enum_name = item->as.enum_def.name;
            
            if (!enum_name) {
                fprintf(stderr, "Error at line %d, column %d: Enum definition has NULL name\n",
                        item->line, item->column);
                tc.has_error = true;
                continue;
            }
            
            /* Check if enum already defined */
            if (env_get_enum(env, enum_name)) {
                fprintf(stderr, "Error at line %d, column %d: Enum '%s' is already defined\n",
                        item->line, item->column, enum_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register the enum */
            EnumDef edef;
            edef.name = strdup(enum_name);
            if (!edef.name) {
                fprintf(stderr, "Error: Failed to allocate memory for enum name\n");
                tc.has_error = true;
                continue;
            }
            edef.variant_count = item->as.enum_def.variant_count;
            if (edef.variant_count <= 0) {
                fprintf(stderr, "Error: Enum '%s' has invalid variant count: %d\n", enum_name, edef.variant_count);
                free(edef.name);
                tc.has_error = true;
                continue;
            }
            
            /* Check if variant_names array exists in AST */
            if (!item->as.enum_def.variant_names) {
                fprintf(stderr, "Error: Enum '%s' has NULL variant_names array\n", enum_name);
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
                if (j < item->as.enum_def.variant_count && item->as.enum_def.variant_names[j]) {
                    const char *src_name = item->as.enum_def.variant_names[j];
                    if (src_name) {
                        edef.variant_names[j] = strdup(src_name);
                        if (!edef.variant_names[j]) {
                            fprintf(stderr, "Error: Failed to duplicate variant name at index %d\n", j);
                            edef.variant_names[j] = NULL;
                        }
                    } else {
                        fprintf(stderr, "Error: Enum '%s' has NULL variant name at index %d\n", enum_name, j);
                        edef.variant_names[j] = NULL;
                    }
                } else {
                    fprintf(stderr, "Error: Enum '%s' has NULL variant name at index %d\n", enum_name, j);
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
            func.name = func_name;
            func.params = item->as.function.params;
            func.param_count = item->as.function.param_count;
            func.return_type = return_type;
            func.return_struct_type_name = item->as.function.return_struct_type_name;
            func.return_fn_sig = item->as.function.return_fn_sig;  /* Store function signature for TYPE_FUNCTION returns */
            func.body = item->as.function.body;
            func.shadow_test = NULL;
            func.is_extern = item->as.function.is_extern;

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
    
    /* Check for similar function names (warnings only) */
    warn_similar_function_names(env);

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

            /* Save current symbol count for scope restoration */
            int saved_symbol_count = env->symbol_count;

            /* Add parameters to environment (create a scope) */
            for (int j = 0; j < item->as.function.param_count; j++) {
                Value val = create_void();
                Type param_type = item->as.function.params[j].type;
                Type element_type = TYPE_UNKNOWN;
                
                /* For array parameters, set element type (currently only array<int> is supported) */
                if (param_type == TYPE_ARRAY) {
                    element_type = TYPE_INT;  /* array<int> has int elements */
                }
                
                env_define_var_with_element_type(env, item->as.function.params[j].name,
                             param_type, element_type, false, val);
                
                /* If parameter is a struct, store the struct type name */
                if (param_type == TYPE_STRUCT && 
                    item->as.function.params[j].struct_type_name) {
                    Symbol *param_sym = env_get_var(env, item->as.function.params[j].name);
                    if (param_sym) {
                        param_sym->struct_type_name = strdup(item->as.function.params[j].struct_type_name);
                    }
                }
            }

            /* Check function body */
            check_statement(&tc, item->as.function.body);

            /* Check for unused variables before leaving scope */
            check_unused_variables(&tc, saved_symbol_count);

            /* Restore environment (remove parameters) */
            env->symbol_count = saved_symbol_count;

            /* Verify function has shadow test (skip for extern functions and functions that use extern functions) */
            Function *func = env_get_function(env, item->as.function.name);
            if (!func->is_extern && !func->shadow_test) {
                /* Check if function body uses extern functions - if so, shadow test is optional */
                bool uses_extern = func->body && contains_extern_calls(func->body, env);
                if (!uses_extern) {
                    fprintf(stderr, "Error: Function '%s' is missing a shadow test\n",
                            item->as.function.name);
                    fprintf(stderr, "  Note: Extern functions and functions that use extern functions do not require shadow tests\n");
                    tc.has_error = true;
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
            
            env_define_union(env, udef);
            
        } else if (item->type == AST_ENUM_DEF) {
            const char *enum_name = item->as.enum_def.name;
            
            if (!enum_name) {
                fprintf(stderr, "Error at line %d, column %d: Enum definition has NULL name\n",
                        item->line, item->column);
                tc.has_error = true;
                continue;
            }
            
            /* Check if enum already defined */
            if (env_get_enum(env, enum_name)) {
                fprintf(stderr, "Error at line %d, column %d: Enum '%s' is already defined\n",
                        item->line, item->column, enum_name);
                tc.has_error = true;
                continue;
            }
            
            /* Register the enum */
            EnumDef edef;
            edef.name = strdup(enum_name);
            if (!edef.name) {
                fprintf(stderr, "Error: Failed to allocate memory for enum name\n");
                tc.has_error = true;
                continue;
            }
            edef.variant_count = item->as.enum_def.variant_count;
            if (edef.variant_count <= 0) {
                fprintf(stderr, "Error: Enum '%s' has invalid variant count: %d\n", enum_name, edef.variant_count);
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
                    edef.variant_names[j] = strdup(src_name);
                    if (!edef.variant_names[j]) {
                        fprintf(stderr, "Error: Failed to duplicate variant name at index %d\n", j);
                        edef.variant_names[j] = NULL;
                    }
                } else {
                    fprintf(stderr, "Error: Enum '%s' has NULL variant name at index %d\n", enum_name, j);
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
            f.body = item->as.function.body;
            f.shadow_test = NULL;  /* Will be linked in second pass */
            f.is_extern = item->as.function.is_extern;
            
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
            
            /* Add function parameters to environment */
            for (int j = 0; j < item->as.function.param_count; j++) {
                Type param_type = item->as.function.params[j].type;
                Value val;
                if (param_type == TYPE_INT) val = create_int(0);
                else if (param_type == TYPE_FLOAT) val = create_float(0.0);
                else if (param_type == TYPE_BOOL) val = create_bool(false);
                else if (param_type == TYPE_STRING) val = create_string("");
                else if (param_type == TYPE_ARRAY) {
                    val = create_array((ValueType)item->as.function.params[j].element_type, 0, 0);
                } else if (param_type == TYPE_STRUCT) {
                    val = create_struct(item->as.function.params[j].struct_type_name, NULL, NULL, 0);
                } else val = create_void();
                
                env_define_var(env, item->as.function.params[j].name, param_type, false, val);
                
                /* If parameter is a struct, store the struct type name */
                if (param_type == TYPE_STRUCT && 
                    item->as.function.params[j].struct_type_name) {
                    Symbol *param_sym = env_get_var(env, item->as.function.params[j].name);
                    if (param_sym) {
                        param_sym->struct_type_name = strdup(item->as.function.params[j].struct_type_name);
                    }
                }
            }

            /* Check function body */
            check_statement(&tc, item->as.function.body);

            /* Check for unused variables before leaving scope */
            check_unused_variables(&tc, saved_symbol_count);

            /* Restore environment (remove parameters) */
            env->symbol_count = saved_symbol_count;

            /* Verify function has shadow test (skip for extern functions and functions that use extern functions) */
            Function *func = env_get_function(env, item->as.function.name);
            if (!func->is_extern && !func->shadow_test) {
                /* Check if function body uses extern functions - if so, shadow test is optional */
                bool uses_extern = func->body && contains_extern_calls(func->body, env);
                if (!uses_extern) {
                    fprintf(stderr, "Error: Function '%s' is missing a shadow test\n",
                            item->as.function.name);
                    fprintf(stderr, "  Note: Extern functions and functions that use extern functions do not require shadow tests\n");
                    tc.has_error = true;
                }
            }
        }
    }

    /* Note: Modules don't require a main function */
    /* Main function check is skipped for modules */

    return !tc.has_error;
}