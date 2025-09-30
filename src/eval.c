#include "nanolang.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <libgen.h>

/* Forward declarations */
static Value eval_expression(ASTNode *expr, Environment *env);
static Value eval_statement(ASTNode *stmt, Environment *env);

/* ==========================================================================
 * Built-in OS Functions Implementation
 * ========================================================================== */

/* File Operations */
static Value builtin_file_read(Value *args) {
    const char *path = args[0].as.string_val;
    FILE *f = fopen(path, "r");
    if (!f) return create_string("");

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buffer = malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);

    Value result = create_string(buffer);
    free(buffer);
    return result;
}

static Value builtin_file_write(Value *args) {
    const char *path = args[0].as.string_val;
    const char *content = args[1].as.string_val;
    FILE *f = fopen(path, "w");
    if (!f) return create_int(-1);

    fputs(content, f);
    fclose(f);
    return create_int(0);
}

static Value builtin_file_append(Value *args) {
    const char *path = args[0].as.string_val;
    const char *content = args[1].as.string_val;
    FILE *f = fopen(path, "a");
    if (!f) return create_int(-1);

    fputs(content, f);
    fclose(f);
    return create_int(0);
}

static Value builtin_file_remove(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(remove(path) == 0 ? 0 : -1);
}

static Value builtin_file_rename(Value *args) {
    const char *old_path = args[0].as.string_val;
    const char *new_path = args[1].as.string_val;
    return create_int(rename(old_path, new_path) == 0 ? 0 : -1);
}

static Value builtin_file_exists(Value *args) {
    const char *path = args[0].as.string_val;
    return create_bool(access(path, F_OK) == 0);
}

static Value builtin_file_size(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_int(-1);
    return create_int(st.st_size);
}

/* Directory Operations */
static Value builtin_dir_create(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(mkdir(path, 0755) == 0 ? 0 : -1);
}

static Value builtin_dir_remove(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(rmdir(path) == 0 ? 0 : -1);
}

static Value builtin_dir_list(Value *args) {
    const char *path = args[0].as.string_val;
    DIR *dir = opendir(path);
    if (!dir) return create_string("");

    /* Build newline-separated list */
    char buffer[4096] = "";
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Skip . and .. */
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        strcat(buffer, entry->d_name);
        strcat(buffer, "\n");
    }
    closedir(dir);

    return create_string(buffer);
}

static Value builtin_dir_exists(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISDIR(st.st_mode));
}

static Value builtin_getcwd(Value *args) {
    (void)args;  /* Unused */
    char buffer[1024];
    if (getcwd(buffer, sizeof(buffer)) == NULL) {
        return create_string("");
    }
    return create_string(buffer);
}

static Value builtin_chdir(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(chdir(path) == 0 ? 0 : -1);
}

/* Path Operations */
static Value builtin_path_isfile(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISREG(st.st_mode));
}

static Value builtin_path_isdir(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISDIR(st.st_mode));
}

static Value builtin_path_join(Value *args) {
    const char *a = args[0].as.string_val;
    const char *b = args[1].as.string_val;
    char buffer[2048];

    /* Handle various cases */
    if (strlen(a) == 0) {
        snprintf(buffer, sizeof(buffer), "%s", b);
    } else if (a[strlen(a) - 1] == '/') {
        snprintf(buffer, sizeof(buffer), "%s%s", a, b);
    } else {
        snprintf(buffer, sizeof(buffer), "%s/%s", a, b);
    }

    return create_string(buffer);
}

static Value builtin_path_basename(Value *args) {
    const char *path = args[0].as.string_val;
    char *path_copy = strdup(path);
    char *base = basename(path_copy);
    Value result = create_string(base);
    free(path_copy);
    return result;
}

static Value builtin_path_dirname(Value *args) {
    const char *path = args[0].as.string_val;
    char *path_copy = strdup(path);
    char *dir = dirname(path_copy);
    Value result = create_string(dir);
    free(path_copy);
    return result;
}

/* Process Operations */
static Value builtin_system(Value *args) {
    const char *command = args[0].as.string_val;
    return create_int(system(command));
}

static Value builtin_exit(Value *args) {
    int code = (int)args[0].as.int_val;
    exit(code);
    return create_void();  /* Never reached */
}

static Value builtin_getenv(Value *args) {
    const char *name = args[0].as.string_val;
    const char *value = getenv(name);
    return create_string(value ? value : "");
}

/* ==========================================================================
 * End of Built-in OS Functions
 * ========================================================================== */

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
    const char *name = node->as.call.name;

    /* Special built-in: range (used in for loops only) */
    if (strcmp(name, "range") == 0) {
        /* This should not be called directly */
        return create_void();
    }

    /* Check for built-in OS functions */
    /* Evaluate arguments first */
    Value args[16];  /* Max args for function calls */
    for (int i = 0; i < node->as.call.arg_count; i++) {
        args[i] = eval_expression(node->as.call.args[i], env);
    }

    /* File operations */
    if (strcmp(name, "file_read") == 0) return builtin_file_read(args);
    if (strcmp(name, "file_write") == 0) return builtin_file_write(args);
    if (strcmp(name, "file_append") == 0) return builtin_file_append(args);
    if (strcmp(name, "file_remove") == 0) return builtin_file_remove(args);
    if (strcmp(name, "file_rename") == 0) return builtin_file_rename(args);
    if (strcmp(name, "file_exists") == 0) return builtin_file_exists(args);
    if (strcmp(name, "file_size") == 0) return builtin_file_size(args);

    /* Directory operations */
    if (strcmp(name, "dir_create") == 0) return builtin_dir_create(args);
    if (strcmp(name, "dir_remove") == 0) return builtin_dir_remove(args);
    if (strcmp(name, "dir_list") == 0) return builtin_dir_list(args);
    if (strcmp(name, "dir_exists") == 0) return builtin_dir_exists(args);
    if (strcmp(name, "getcwd") == 0) return builtin_getcwd(args);
    if (strcmp(name, "chdir") == 0) return builtin_chdir(args);

    /* Path operations */
    if (strcmp(name, "path_isfile") == 0) return builtin_path_isfile(args);
    if (strcmp(name, "path_isdir") == 0) return builtin_path_isdir(args);
    if (strcmp(name, "path_join") == 0) return builtin_path_join(args);
    if (strcmp(name, "path_basename") == 0) return builtin_path_basename(args);
    if (strcmp(name, "path_dirname") == 0) return builtin_path_dirname(args);

    /* Process operations */
    if (strcmp(name, "system") == 0) return builtin_system(args);
    if (strcmp(name, "exit") == 0) return builtin_exit(args);
    if (strcmp(name, "getenv") == 0) return builtin_getenv(args);

    /* Get user-defined function */
    Function *func = env_get_function(env, name);
    if (!func) {
        fprintf(stderr, "Error: Undefined function '%s'\n", name);
        return create_void();
    }

    /* If built-in with no body, already handled above */
    if (func->body == NULL) {
        fprintf(stderr, "Error: Built-in function '%s' not implemented in interpreter\n", name);
        return create_void();
    }

    /* Create new environment for function */
    int old_symbol_count = env->symbol_count;

    /* Bind parameters with copies of string values */
    for (int i = 0; i < func->param_count; i++) {
        Value param_value = args[i];

        /* Make a deep copy of string values to avoid memory corruption */
        if (param_value.type == VAL_STRING) {
            param_value = create_string(args[i].as.string_val);
        }

        env_define_var(env, func->params[i].name, func->params[i].type, false, param_value);
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

    /* Make a copy of the result if it's a string BEFORE cleaning up parameters */
    Value return_value = result;
    if (result.type == VAL_STRING) {
        return_value = create_string(result.as.string_val);
    }

    /* Clean up parameter strings and restore environment */
    for (int i = old_symbol_count; i < env->symbol_count; i++) {
        free(env->symbols[i].name);
        if (env->symbols[i].value.type == VAL_STRING) {
            free(env->symbols[i].value.as.string_val);
        }
    }
    env->symbol_count = old_symbol_count;

    return return_value;
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

/* Run the entire program (interpreter mode) */
bool run_program(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program\n");
        return false;
    }

    /* Execute all top-level items (functions, statements, etc.) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];

        /* Skip shadow tests in interpreter mode - they're for compiler validation */
        if (item->type == AST_SHADOW) {
            continue;
        }

        /* Execute the item */
        eval_statement(item, env);
    }

    return true;
}

/* Call a function by name with arguments */
Value call_function(const char *name, Value *args, int arg_count, Environment *env) {
    Function *func = env_get_function(env, name);
    if (!func) {
        fprintf(stderr, "Error: Function '%s' not found\n", name);
        return create_void();
    }

    /* Check argument count */
    if (arg_count != func->param_count) {
        fprintf(stderr, "Error: Function '%s' expects %d arguments, got %d\n",
                name, func->param_count, arg_count);
        return create_void();
    }

    /* Save original symbol count to restore environment after function call */
    int original_symbol_count = env->symbol_count;

    /* Add function parameters to environment with copies of string values */
    for (int i = 0; i < arg_count; i++) {
        Value param_value = args[i];

        /* Make a deep copy of string values to avoid memory corruption */
        if (param_value.type == VAL_STRING) {
            param_value = create_string(args[i].as.string_val);
        }

        env_define_var(env, func->params[i].name, func->params[i].type, false, param_value);
    }

    /* Execute the function body */
    Value result = eval_statement(func->body, env);

    /* Make a copy of the result if it's a string BEFORE cleaning up parameters */
    Value return_value = result;
    if (result.type == VAL_STRING) {
        return_value = create_string(result.as.string_val);
    }

    /* Clean up parameter strings and restore environment */
    for (int i = original_symbol_count; i < env->symbol_count; i++) {
        free(env->symbols[i].name);
        if (env->symbols[i].value.type == VAL_STRING) {
            free(env->symbols[i].value.as.string_val);
        }
    }
    env->symbol_count = original_symbol_count;

    return return_value;
}