#include "nanolang.h"

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
        case TYPE_UNKNOWN: return "unknown";
        default: return "unknown";
    }
}

/* Forward declarations */
static Type check_statement(TypeChecker *tc, ASTNode *node);

/* Check if types are compatible */
static bool types_match(Type t1, Type t2) {
    return t1 == t2;
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
            if (!func) {
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
                    Type arg_type = check_expression(expr->as.call.args[i], env);
                    if (!types_match(arg_type, func->params[i].type)) {
                        fprintf(stderr, "Error at line %d, column %d: Argument %d type mismatch in call to '%s'\n",
                                expr->line, expr->column, i + 1, expr->as.call.name);
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
                            /* We need to track element types in variables */
                            /* For now, check if we stored it in the array literal */
                            /* This is a limitation - we'd need extended type info */
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
            Type value_type = check_expression(stmt->as.let.value, tc->env);
            if (!types_match(value_type, stmt->as.let.var_type)) {
                fprintf(stderr, "Error at line %d, column %d: Type mismatch in let statement\n", stmt->line, stmt->column);
                tc->has_error = true;
            }

            /* Add to environment */
            Value val = create_void(); /* Placeholder */
            env_define_var(tc->env, stmt->as.let.name, stmt->as.let.var_type, stmt->as.let.is_mut, val);
            
            /* Store definition location for unused variable warnings */
            Symbol *sym = env_get_var(tc->env, stmt->as.let.name);
            if (sym) {
                sym->def_line = stmt->line;
                sym->def_column = stmt->column;
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
    env_define_function(env, func);
    
    /* abs(x: int|float) -> int|float */
    func.name = "abs";
    func.params = NULL;  /* Accept int or float */
    func.param_count = 1;
    func.return_type = TYPE_INT;  /* Can also be float */
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* min(a: int|float, b: int|float) -> int|float */
    func.name = "min";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_INT;  /* Can also be float */
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* max(a: int|float, b: int|float) -> int|float */
    func.name = "max";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_INT;  /* Can also be float */
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* println(x: any) -> void */
    func.name = "println";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* Advanced math functions */
    /* sqrt(x: int|float) -> float */
    func.name = "sqrt";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* pow(base: int|float, exponent: int|float) -> float */
    func.name = "pow";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* floor(x: int|float) -> float */
    func.name = "floor";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* ceil(x: int|float) -> float */
    func.name = "ceil";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* round(x: int|float) -> float */
    func.name = "round";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* Trigonometric functions */
    /* sin(x: int|float) -> float */
    func.name = "sin";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* cos(x: int|float) -> float */
    func.name = "cos";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* tan(x: int|float) -> float */
    func.name = "tan";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_FLOAT;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* String operations */
    /* str_length(s: string) -> int */
    func.name = "str_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* str_concat(s1: string, s2: string) -> string */
    func.name = "str_concat";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* str_substring(s: string, start: int, length: int) -> string */
    func.name = "str_substring";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* str_contains(s: string, substr: string) -> bool */
    func.name = "str_contains";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* str_equals(s1: string, s2: string) -> bool */
    func.name = "str_equals";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_BOOL;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* Array operations */
    /* at(arr: array<T>, index: int) -> T */
    func.name = "at";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_UNKNOWN;  /* Will be determined by array element type */
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* array_length(arr: array<T>) -> int */
    func.name = "array_length";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_INT;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* array_new(size: int, default: T) -> array<T> */
    func.name = "array_new";
    func.params = NULL;
    func.param_count = 2;
    func.return_type = TYPE_ARRAY;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* array_set(arr: mut array<T>, index: int, value: T) -> void */
    func.name = "array_set";
    func.params = NULL;
    func.param_count = 3;
    func.return_type = TYPE_VOID;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    /* OS built-ins */
    func.name = "getcwd";
    func.params = NULL;
    func.param_count = 0;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
    env_define_function(env, func);
    
    func.name = "getenv";
    func.params = NULL;
    func.param_count = 1;
    func.return_type = TYPE_STRING;
    func.body = NULL;
    func.shadow_test = NULL;
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

    /* First pass: collect all function definitions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            Function func;
            func.name = item->as.function.name;
            func.params = item->as.function.params;
            func.param_count = item->as.function.param_count;
            func.return_type = item->as.function.return_type;
            func.body = item->as.function.body;
            func.shadow_test = NULL;

            env_define_function(env, func);
        }
    }

    /* Second pass: link shadow tests to functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_SHADOW) {
            Function *func = env_get_function(env, item->as.shadow.function_name);
            if (!func) {
                fprintf(stderr, "Error at line %d, column %d: Shadow test for undefined function '%s'\n",
                        item->line, item->column, item->as.shadow.function_name);
                tc.has_error = true;
            } else {
                func->shadow_test = item->as.shadow.body;
            }
        }
    }

    /* Third pass: type check all functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_FUNCTION) {
            tc.current_function_return_type = item->as.function.return_type;

            /* Save current symbol count for scope restoration */
            int saved_symbol_count = env->symbol_count;

            /* Add parameters to environment (create a scope) */
            for (int j = 0; j < item->as.function.param_count; j++) {
                Value val = create_void();
                env_define_var(env, item->as.function.params[j].name,
                             item->as.function.params[j].type, false, val);
            }

            /* Check function body */
            check_statement(&tc, item->as.function.body);

            /* Check for unused variables before leaving scope */
            check_unused_variables(&tc, saved_symbol_count);

            /* Restore environment (remove parameters) */
            env->symbol_count = saved_symbol_count;

            /* Verify function has shadow test */
            Function *func = env_get_function(env, item->as.function.name);
            if (!func->shadow_test) {
                fprintf(stderr, "Error: Function '%s' is missing a shadow test\n",
                        item->as.function.name);
                tc.has_error = true;
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