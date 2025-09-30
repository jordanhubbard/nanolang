#include "nanolang.h"

/* Type checking context */
typedef struct {
    Environment *env;
    Type current_function_return_type;
    bool has_error;
} TypeChecker;

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
                fprintf(stderr, "Error at line %d: Undefined variable '%s'\n",
                        expr->line, expr->as.identifier);
                return TYPE_UNKNOWN;
            }
            return sym->type;
        }

        case AST_PREFIX_OP: {
            TokenType op = expr->as.prefix_op.op;
            int arg_count = expr->as.prefix_op.arg_count;

            /* Arithmetic operators */
            if (op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR ||
                op == TOKEN_SLASH || op == TOKEN_PERCENT) {
                if (arg_count != 2) {
                    fprintf(stderr, "Error at line %d: Arithmetic operators require 2 arguments\n", expr->line);
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                if (left == TYPE_INT && right == TYPE_INT) return TYPE_INT;
                if (left == TYPE_FLOAT && right == TYPE_FLOAT) return TYPE_FLOAT;

                fprintf(stderr, "Error at line %d: Type mismatch in arithmetic operation\n", expr->line);
                return TYPE_UNKNOWN;
            }

            /* Comparison operators */
            if (op == TOKEN_LT || op == TOKEN_LE || op == TOKEN_GT || op == TOKEN_GE) {
                if (arg_count != 2) {
                    fprintf(stderr, "Error at line %d: Comparison operators require 2 arguments\n", expr->line);
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                if (!types_match(left, right)) {
                    fprintf(stderr, "Error at line %d: Type mismatch in comparison\n", expr->line);
                }
                return TYPE_BOOL;
            }

            /* Equality operators */
            if (op == TOKEN_EQ || op == TOKEN_NE) {
                if (arg_count != 2) {
                    fprintf(stderr, "Error at line %d: Equality operators require 2 arguments\n", expr->line);
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                if (!types_match(left, right)) {
                    fprintf(stderr, "Error at line %d: Type mismatch in equality check\n", expr->line);
                }
                return TYPE_BOOL;
            }

            /* Logical operators */
            if (op == TOKEN_AND || op == TOKEN_OR) {
                if (arg_count != 2) {
                    fprintf(stderr, "Error at line %d: Logical operators require 2 arguments\n", expr->line);
                    return TYPE_UNKNOWN;
                }
                Type left = check_expression(expr->as.prefix_op.args[0], env);
                Type right = check_expression(expr->as.prefix_op.args[1], env);

                if (left != TYPE_BOOL || right != TYPE_BOOL) {
                    fprintf(stderr, "Error at line %d: Logical operators require bool operands\n", expr->line);
                }
                return TYPE_BOOL;
            }

            if (op == TOKEN_NOT) {
                if (arg_count != 1) {
                    fprintf(stderr, "Error at line %d: 'not' requires 1 argument\n", expr->line);
                    return TYPE_UNKNOWN;
                }
                Type arg = check_expression(expr->as.prefix_op.args[0], env);
                if (arg != TYPE_BOOL) {
                    fprintf(stderr, "Error at line %d: 'not' requires bool operand\n", expr->line);
                }
                return TYPE_BOOL;
            }

            return TYPE_UNKNOWN;
        }

        case AST_CALL: {
            /* Check if function exists */
            Function *func = env_get_function(env, expr->as.call.name);
            if (!func) {
                fprintf(stderr, "Error at line %d: Undefined function '%s'\n",
                        expr->line, expr->as.call.name);
                return TYPE_UNKNOWN;
            }

            /* Check argument count */
            if (expr->as.call.arg_count != func->param_count) {
                fprintf(stderr, "Error at line %d: Function '%s' expects %d arguments, got %d\n",
                        expr->line, expr->as.call.name, func->param_count, expr->as.call.arg_count);
                return TYPE_UNKNOWN;
            }

            /* Check argument types */
            for (int i = 0; i < expr->as.call.arg_count; i++) {
                Type arg_type = check_expression(expr->as.call.args[i], env);
                if (!types_match(arg_type, func->params[i].type)) {
                    fprintf(stderr, "Error at line %d: Argument %d type mismatch in call to '%s'\n",
                            expr->line, i + 1, expr->as.call.name);
                }
            }

            return func->return_type;
        }

        case AST_IF: {
            Type cond_type = check_expression(expr->as.if_stmt.condition, env);
            if (cond_type != TYPE_BOOL) {
                fprintf(stderr, "Error at line %d: If condition must be bool\n", expr->line);
            }

            /* For if expressions, we need to infer the type from the blocks */
            /* This is simplified - just return UNKNOWN for now */
            /* A proper implementation would need to analyze the blocks */
            return TYPE_UNKNOWN;
        }

        default:
            fprintf(stderr, "Error at line %d: Invalid expression type\n", expr->line);
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
                fprintf(stderr, "Error at line %d: Type mismatch in let statement\n", stmt->line);
                tc->has_error = true;
            }

            /* Add to environment */
            Value val = create_void(); /* Placeholder */
            env_define_var(tc->env, stmt->as.let.name, stmt->as.let.var_type, stmt->as.let.is_mut, val);
            return TYPE_VOID;
        }

        case AST_SET: {
            Symbol *sym = env_get_var(tc->env, stmt->as.set.name);
            if (!sym) {
                fprintf(stderr, "Error at line %d: Undefined variable '%s'\n",
                        stmt->line, stmt->as.set.name);
                tc->has_error = true;
                return TYPE_VOID;
            }

            if (!sym->is_mut) {
                fprintf(stderr, "Error at line %d: Cannot assign to immutable variable '%s'\n",
                        stmt->line, stmt->as.set.name);
                tc->has_error = true;
            }

            Type value_type = check_expression(stmt->as.set.value, tc->env);
            if (!types_match(value_type, sym->type)) {
                fprintf(stderr, "Error at line %d: Type mismatch in assignment\n", stmt->line);
                tc->has_error = true;
            }

            return TYPE_VOID;
        }

        case AST_WHILE: {
            Type cond_type = check_expression(stmt->as.while_stmt.condition, tc->env);
            if (cond_type != TYPE_BOOL) {
                fprintf(stderr, "Error at line %d: While condition must be bool\n", stmt->line);
                tc->has_error = true;
            }

            check_statement(tc, stmt->as.while_stmt.body);
            return TYPE_VOID;
        }

        case AST_FOR: {
            /* For loop variable has type int */
            Value val = create_void();
            env_define_var(tc->env, stmt->as.for_stmt.var_name, TYPE_INT, false, val);

            /* Range expression should return a range (we'll treat as special) */
            check_expression(stmt->as.for_stmt.range_expr, tc->env);

            check_statement(tc, stmt->as.for_stmt.body);
            return TYPE_VOID;
        }

        case AST_RETURN: {
            if (stmt->as.return_stmt.value) {
                Type return_type = check_expression(stmt->as.return_stmt.value, tc->env);
                if (!types_match(return_type, tc->current_function_return_type)) {
                    fprintf(stderr, "Error at line %d: Return type mismatch\n", stmt->line);
                    tc->has_error = true;
                }
            } else {
                if (tc->current_function_return_type != TYPE_VOID) {
                    fprintf(stderr, "Error at line %d: Function must return a value\n", stmt->line);
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
                fprintf(stderr, "Error at line %d: Assert condition must be bool\n", stmt->line);
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
bool type_check(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program AST\n");
        return false;
    }

    TypeChecker tc;
    tc.env = env;
    tc.has_error = false;

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
                fprintf(stderr, "Error at line %d: Shadow test for undefined function '%s'\n",
                        item->line, item->as.shadow.function_name);
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