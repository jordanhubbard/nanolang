#include "nanolang.h"

/* Forward declarations */
static Value eval_expression(ASTNode *expr, Environment *env);
static Value eval_statement(ASTNode *stmt, Environment *env);

/* Print a value */
static void print_value(Value val) {
    switch (val.type) {
        case VAL_INT:
            printf("%lld", val.as.int_val);
            break;
        case VAL_FLOAT:
            printf("%g", val.as.float_val);
            break;
        case VAL_BOOL:
            printf("%s", val.as.bool_val ? "true" : "false");
            break;
        case VAL_STRING:
            printf("%s", val.as.string_val);
            break;
        case VAL_VOID:
            printf("void");
            break;
    }
}

/* Helper to convert value to boolean */
static bool is_truthy(Value val) {
    switch (val.type) {
        case VAL_BOOL:
            return val.as.bool_val;
        case VAL_INT:
            return val.as.int_val != 0;
        case VAL_FLOAT:
            return val.as.float_val != 0.0;
        case VAL_VOID:
            return false;
        default:
            return true; /* Strings are truthy if non-null */
    }
}

/* Evaluate prefix operation */
static Value eval_prefix_op(ASTNode *node, Environment *env) {
    TokenType op = node->as.prefix_op.op;
    int arg_count = node->as.prefix_op.arg_count;

    /* Arithmetic operators */
    if (op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR ||
        op == TOKEN_SLASH || op == TOKEN_PERCENT) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Arithmetic operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        Value right = eval_expression(node->as.prefix_op.args[1], env);

        if (left.type == VAL_INT && right.type == VAL_INT) {
            long long result;
            switch (op) {
                case TOKEN_PLUS: result = left.as.int_val + right.as.int_val; break;
                case TOKEN_MINUS: result = left.as.int_val - right.as.int_val; break;
                case TOKEN_STAR: result = left.as.int_val * right.as.int_val; break;
                case TOKEN_SLASH:
                    if (right.as.int_val == 0) {
                        fprintf(stderr, "Error: Division by zero\n");
                        return create_void();
                    }
                    result = left.as.int_val / right.as.int_val;
                    break;
                case TOKEN_PERCENT:
                    if (right.as.int_val == 0) {
                        fprintf(stderr, "Error: Modulo by zero\n");
                        return create_void();
                    }
                    result = left.as.int_val % right.as.int_val;
                    break;
                default: result = 0;
            }
            return create_int(result);
        } else if (left.type == VAL_FLOAT && right.type == VAL_FLOAT) {
            double result;
            switch (op) {
                case TOKEN_PLUS: result = left.as.float_val + right.as.float_val; break;
                case TOKEN_MINUS: result = left.as.float_val - right.as.float_val; break;
                case TOKEN_STAR: result = left.as.float_val * right.as.float_val; break;
                case TOKEN_SLASH:
                    if (right.as.float_val == 0.0) {
                        fprintf(stderr, "Error: Division by zero\n");
                        return create_void();
                    }
                    result = left.as.float_val / right.as.float_val;
                    break;
                default: result = 0.0;
            }
            return create_float(result);
        }
    }

    /* Comparison operators */
    if (op == TOKEN_LT || op == TOKEN_LE || op == TOKEN_GT || op == TOKEN_GE) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Comparison operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        Value right = eval_expression(node->as.prefix_op.args[1], env);

        if (left.type == VAL_INT && right.type == VAL_INT) {
            bool result;
            switch (op) {
                case TOKEN_LT: result = left.as.int_val < right.as.int_val; break;
                case TOKEN_LE: result = left.as.int_val <= right.as.int_val; break;
                case TOKEN_GT: result = left.as.int_val > right.as.int_val; break;
                case TOKEN_GE: result = left.as.int_val >= right.as.int_val; break;
                default: result = false;
            }
            return create_bool(result);
        }
    }

    /* Equality operators */
    if (op == TOKEN_EQ || op == TOKEN_NE) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Equality operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        Value right = eval_expression(node->as.prefix_op.args[1], env);

        bool equal = false;
        if (left.type == right.type) {
            switch (left.type) {
                case VAL_INT: equal = left.as.int_val == right.as.int_val; break;
                case VAL_FLOAT: equal = left.as.float_val == right.as.float_val; break;
                case VAL_BOOL: equal = left.as.bool_val == right.as.bool_val; break;
                case VAL_STRING: equal = strcmp(left.as.string_val, right.as.string_val) == 0; break;
                case VAL_VOID: equal = true; break;
            }
        }

        return create_bool(op == TOKEN_EQ ? equal : !equal);
    }

    /* Logical operators */
    if (op == TOKEN_AND || op == TOKEN_OR) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Logical operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);

        if (op == TOKEN_AND) {
            if (!is_truthy(left)) return create_bool(false);
            Value right = eval_expression(node->as.prefix_op.args[1], env);
            return create_bool(is_truthy(right));
        } else { /* OR */
            if (is_truthy(left)) return create_bool(true);
            Value right = eval_expression(node->as.prefix_op.args[1], env);
            return create_bool(is_truthy(right));
        }
    }

    if (op == TOKEN_NOT) {
        if (arg_count != 1) {
            fprintf(stderr, "Error: 'not' requires 1 argument\n");
            return create_void();
        }
        Value arg = eval_expression(node->as.prefix_op.args[0], env);
        return create_bool(!is_truthy(arg));
    }

    return create_void();
}

/* Evaluate function call */
static Value eval_call(ASTNode *node, Environment *env) {
    /* Special built-in: range */
    if (strcmp(node->as.call.name, "range") == 0) {
        /* This should not be called directly in shadow tests */
        /* Just return a placeholder */
        return create_void();
    }

    /* Get function */
    Function *func = env_get_function(env, node->as.call.name);
    if (!func) {
        fprintf(stderr, "Error: Undefined function '%s'\n", node->as.call.name);
        return create_void();
    }

    /* Create new environment for function */
    int old_symbol_count = env->symbol_count;

    /* Bind parameters */
    for (int i = 0; i < func->param_count; i++) {
        Value arg = eval_expression(node->as.call.args[i], env);
        env_define_var(env, func->params[i].name, func->params[i].type, false, arg);
    }

    /* Execute function body */
    Value result = create_void();
    for (int i = 0; i < func->body->as.block.count; i++) {
        ASTNode *stmt = func->body->as.block.statements[i];
        if (stmt->type == AST_RETURN) {
            if (stmt->as.return_stmt.value) {
                result = eval_expression(stmt->as.return_stmt.value, env);
            }
            break;
        }
        result = eval_statement(stmt, env);
    }

    /* Restore environment (pop function scope) */
    env->symbol_count = old_symbol_count;

    return result;
}

/* Evaluate expression */
static Value eval_expression(ASTNode *expr, Environment *env) {
    if (!expr) return create_void();

    switch (expr->type) {
        case AST_NUMBER:
            return create_int(expr->as.number);

        case AST_FLOAT:
            return create_float(expr->as.float_val);

        case AST_STRING:
            return create_string(expr->as.string_val);

        case AST_BOOL:
            return create_bool(expr->as.bool_val);

        case AST_IDENTIFIER: {
            Symbol *sym = env_get_var(env, expr->as.identifier);
            if (!sym) {
                fprintf(stderr, "Error: Undefined variable '%s'\n", expr->as.identifier);
                return create_void();
            }
            return sym->value;
        }

        case AST_PREFIX_OP:
            return eval_prefix_op(expr, env);

        case AST_CALL:
            return eval_call(expr, env);

        case AST_IF: {
            Value cond = eval_expression(expr->as.if_stmt.condition, env);
            if (is_truthy(cond)) {
                return eval_statement(expr->as.if_stmt.then_branch, env);
            } else {
                return eval_statement(expr->as.if_stmt.else_branch, env);
            }
        }

        default:
            return create_void();
    }
}

/* Evaluate statement */
static Value eval_statement(ASTNode *stmt, Environment *env) {
    if (!stmt) return create_void();

    switch (stmt->type) {
        case AST_LET: {
            Value value = eval_expression(stmt->as.let.value, env);
            env_define_var(env, stmt->as.let.name, stmt->as.let.var_type, stmt->as.let.is_mut, value);
            return create_void();
        }

        case AST_SET: {
            Value value = eval_expression(stmt->as.set.value, env);
            env_set_var(env, stmt->as.set.name, value);
            return create_void();
        }

        case AST_WHILE: {
            Value result = create_void();
            while (is_truthy(eval_expression(stmt->as.while_stmt.condition, env))) {
                result = eval_statement(stmt->as.while_stmt.body, env);
            }
            return result;
        }

        case AST_FOR: {
            /* Evaluate range */
            ASTNode *range_expr = stmt->as.for_stmt.range_expr;
            if (range_expr->type != AST_CALL || strcmp(range_expr->as.call.name, "range") != 0) {
                fprintf(stderr, "Error: for loop requires range expression\n");
                return create_void();
            }

            if (range_expr->as.call.arg_count != 2) {
                fprintf(stderr, "Error: range requires 2 arguments\n");
                return create_void();
            }

            Value start_val = eval_expression(range_expr->as.call.args[0], env);
            Value end_val = eval_expression(range_expr->as.call.args[1], env);

            if (start_val.type != VAL_INT || end_val.type != VAL_INT) {
                fprintf(stderr, "Error: range requires int arguments\n");
                return create_void();
            }

            long long start = start_val.as.int_val;
            long long end = end_val.as.int_val;

            /* Define loop variable before the loop */
            int loop_var_index = env->symbol_count;
            env_define_var(env, stmt->as.for_stmt.var_name, TYPE_INT, false, create_int(start));

            Value result = create_void();
            for (long long i = start; i < end; i++) {
                /* Update loop variable value */
                env->symbols[loop_var_index].value = create_int(i);

                /* Execute loop body */
                result = eval_statement(stmt->as.for_stmt.body, env);
            }

            /* Remove loop variable from scope */
            env->symbol_count = loop_var_index;

            return result;
        }

        case AST_RETURN:
            if (stmt->as.return_stmt.value) {
                return eval_expression(stmt->as.return_stmt.value, env);
            }
            return create_void();

        case AST_BLOCK: {
            Value result = create_void();
            for (int i = 0; i < stmt->as.block.count; i++) {
                result = eval_statement(stmt->as.block.statements[i], env);
            }
            return result;
        }

        case AST_PRINT: {
            Value value = eval_expression(stmt->as.print.expr, env);
            print_value(value);
            printf("\n");
            return create_void();
        }

        case AST_ASSERT: {
            Value cond = eval_expression(stmt->as.assert.condition, env);
            if (!is_truthy(cond)) {
                fprintf(stderr, "Assertion failed at line %d\n", stmt->line);
                exit(1);
            }
            return create_void();
        }

        default:
            /* Expression statements */
            return eval_expression(stmt, env);
    }
}

/* Run shadow tests */
bool run_shadow_tests(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program for shadow tests\n");
        return false;
    }

    printf("Running shadow tests...\n");

    bool all_passed = true;

    /* Run each shadow test */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        if (item->type == AST_SHADOW) {
            printf("Testing %s... ", item->as.shadow.function_name);
            fflush(stdout);

            /* Execute shadow test */
            eval_statement(item->as.shadow.body, env);

            printf("PASSED\n");
        }
    }

    if (all_passed) {
        printf("All shadow tests passed!\n");
    }

    return all_passed;
}