#include "nanolang.h"

/* Create environment */
Environment *create_environment(void) {
    Environment *env = malloc(sizeof(Environment));
    env->capacity = 16;
    env->count = 0;
    env->symbols = malloc(sizeof(Symbol) * env->capacity);
    return env;
}

/* Free environment */
void free_environment(Environment *env) {
    for (int i = 0; i < env->count; i++) {
        free(env->symbols[i].name);
    }
    free(env->symbols);
    free(env);
}

/* Set variable in environment */
void env_set(Environment *env, const char *name, Value value) {
    /* Check if variable already exists */
    for (int i = 0; i < env->count; i++) {
        if (strcmp(env->symbols[i].name, name) == 0) {
            env->symbols[i].value = value;
            return;
        }
    }
    
    /* Add new variable */
    if (env->count >= env->capacity) {
        env->capacity *= 2;
        env->symbols = realloc(env->symbols, sizeof(Symbol) * env->capacity);
    }
    
    env->symbols[env->count].name = strdup(name);
    env->symbols[env->count].value = value;
    env->count++;
}

/* Get variable from environment */
Value env_get(Environment *env, const char *name, bool *found) {
    for (int i = 0; i < env->count; i++) {
        if (strcmp(env->symbols[i].name, name) == 0) {
            *found = true;
            return env->symbols[i].value;
        }
    }
    
    *found = false;
    Value null_val = {.type = VAL_NULL};
    return null_val;
}

/* Print a value */
void print_value(Value val) {
    switch (val.type) {
        case VAL_NUMBER:
            printf("%d", val.as.number);
            break;
        case VAL_BOOL:
            printf("%s", val.as.boolean ? "true" : "false");
            break;
        case VAL_NULL:
            printf("null");
            break;
    }
}

/* Helper to convert value to boolean */
static bool is_truthy(Value val) {
    switch (val.type) {
        case VAL_BOOL:
            return val.as.boolean;
        case VAL_NUMBER:
            return val.as.number != 0;
        case VAL_NULL:
            return false;
    }
    return false;
}

/* Evaluate AST node */
Value eval(ASTNode *node, Environment *env) {
    Value result = {.type = VAL_NULL};
    
    if (!node) return result;
    
    switch (node->type) {
        case AST_NUMBER:
            result.type = VAL_NUMBER;
            result.as.number = node->as.number;
            break;
            
        case AST_BOOL:
            result.type = VAL_BOOL;
            result.as.boolean = node->as.boolean;
            break;
            
        case AST_IDENTIFIER: {
            bool found = false;
            result = env_get(env, node->as.identifier, &found);
            if (!found) {
                fprintf(stderr, "Error: Undefined variable '%s'\n", node->as.identifier);
                result.type = VAL_NULL;
            }
            break;
        }
            
        case AST_BINARY_OP: {
            Value left = eval(node->as.binary.left, env);
            Value right = eval(node->as.binary.right, env);
            
            result.type = VAL_NUMBER;
            
            switch (node->as.binary.op) {
                case TOKEN_PLUS:
                    if (left.type == VAL_NUMBER && right.type == VAL_NUMBER) {
                        result.as.number = left.as.number + right.as.number;
                    }
                    break;
                case TOKEN_MINUS:
                    if (left.type == VAL_NUMBER && right.type == VAL_NUMBER) {
                        result.as.number = left.as.number - right.as.number;
                    }
                    break;
                case TOKEN_STAR:
                    if (left.type == VAL_NUMBER && right.type == VAL_NUMBER) {
                        result.as.number = left.as.number * right.as.number;
                    }
                    break;
                case TOKEN_SLASH:
                    if (left.type == VAL_NUMBER && right.type == VAL_NUMBER) {
                        if (right.as.number == 0) {
                            fprintf(stderr, "Error: Division by zero\n");
                            result.type = VAL_NULL;
                        } else {
                            result.as.number = left.as.number / right.as.number;
                        }
                    }
                    break;
                case TOKEN_EQ:
                    result.type = VAL_BOOL;
                    if (left.type == VAL_NUMBER && right.type == VAL_NUMBER) {
                        result.as.boolean = left.as.number == right.as.number;
                    } else if (left.type == VAL_BOOL && right.type == VAL_BOOL) {
                        result.as.boolean = left.as.boolean == right.as.boolean;
                    } else {
                        result.as.boolean = false;
                    }
                    break;
                case TOKEN_LT:
                    result.type = VAL_BOOL;
                    if (left.type == VAL_NUMBER && right.type == VAL_NUMBER) {
                        result.as.boolean = left.as.number < right.as.number;
                    }
                    break;
                case TOKEN_GT:
                    result.type = VAL_BOOL;
                    if (left.type == VAL_NUMBER && right.type == VAL_NUMBER) {
                        result.as.boolean = left.as.number > right.as.number;
                    }
                    break;
                default:
                    result.type = VAL_NULL;
                    break;
            }
            break;
        }
            
        case AST_ASSIGN: {
            Value value = eval(node->as.assign.value, env);
            env_set(env, node->as.assign.name, value);
            result = value;
            break;
        }
            
        case AST_PRINT: {
            Value value = eval(node->as.print.expr, env);
            print_value(value);
            printf("\n");
            result = value;
            break;
        }
            
        case AST_LET: {
            Value value = eval(node->as.let.value, env);
            env_set(env, node->as.let.name, value);
            result = value;
            break;
        }
            
        case AST_BLOCK: {
            for (int i = 0; i < node->as.block.count; i++) {
                result = eval(node->as.block.statements[i], env);
            }
            break;
        }
            
        case AST_IF: {
            Value condition = eval(node->as.if_stmt.condition, env);
            if (is_truthy(condition)) {
                result = eval(node->as.if_stmt.then_branch, env);
            } else if (node->as.if_stmt.else_branch) {
                result = eval(node->as.if_stmt.else_branch, env);
            }
            break;
        }
            
        case AST_WHILE: {
            Value condition = eval(node->as.while_stmt.condition, env);
            while (is_truthy(condition)) {
                result = eval(node->as.while_stmt.body, env);
                condition = eval(node->as.while_stmt.condition, env);
            }
            break;
        }
    }
    
    return result;
}
