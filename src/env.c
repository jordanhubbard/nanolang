#include "nanolang.h"

/* Create environment */
Environment *create_environment(void) {
    Environment *env = malloc(sizeof(Environment));
    env->symbols = malloc(sizeof(Symbol) * 8);
    env->symbol_count = 0;
    env->symbol_capacity = 8;
    env->functions = malloc(sizeof(Function) * 8);
    env->function_count = 0;
    env->function_capacity = 8;
    return env;
}

/* Free environment */
void free_environment(Environment *env) {
    for (int i = 0; i < env->symbol_count; i++) {
        free(env->symbols[i].name);
        if (env->symbols[i].value.type == VAL_STRING) {
            free(env->symbols[i].value.as.string_val);
        }
    }
    free(env->symbols);

    for (int i = 0; i < env->function_count; i++) {
        /* Note: function names are not owned by environment */
    }
    free(env->functions);

    free(env);
}

/* Define variable */
void env_define_var(Environment *env, const char *name, Type type, bool is_mut, Value value) {
    if (env->symbol_count >= env->symbol_capacity) {
        env->symbol_capacity *= 2;
        env->symbols = realloc(env->symbols, sizeof(Symbol) * env->symbol_capacity);
    }

    Symbol sym;
    sym.name = strdup(name);
    sym.type = type;
    sym.is_mut = is_mut;
    sym.value = value;

    env->symbols[env->symbol_count++] = sym;
}

/* Get variable */
Symbol *env_get_var(Environment *env, const char *name) {
    for (int i = env->symbol_count - 1; i >= 0; i--) {
        if (strcmp(env->symbols[i].name, name) == 0) {
            return &env->symbols[i];
        }
    }
    return NULL;
}

/* Set variable value */
void env_set_var(Environment *env, const char *name, Value value) {
    Symbol *sym = env_get_var(env, name);
    if (sym) {
        /* Free old string value if needed */
        if (sym->value.type == VAL_STRING) {
            free(sym->value.as.string_val);
        }
        sym->value = value;
    }
}

/* Define function */
void env_define_function(Environment *env, Function func) {
    if (env->function_count >= env->function_capacity) {
        env->function_capacity *= 2;
        env->functions = realloc(env->functions, sizeof(Function) * env->function_capacity);
    }

    env->functions[env->function_count++] = func;
}

/* Get function */
Function *env_get_function(Environment *env, const char *name) {
    if (!name) {
        return NULL;
    }
    /* Check built-in functions first */
    if (strcmp(name, "range") == 0) {
        static Function range_func = {
            .name = "range",
            .params = NULL,
            .param_count = 2,
            .return_type = TYPE_INT,  /* Special: returns iterator */
            .body = NULL,
            .shadow_test = NULL
        };
        return &range_func;
    }

    for (int i = 0; i < env->function_count; i++) {
        if (strcmp(env->functions[i].name, name) == 0) {
            return &env->functions[i];
        }
    }
    return NULL;
}

/* Value creation functions */
Value create_int(long long val) {
    Value v;
    v.type = VAL_INT;
    v.as.int_val = val;
    return v;
}

Value create_float(double val) {
    Value v;
    v.type = VAL_FLOAT;
    v.as.float_val = val;
    return v;
}

Value create_bool(bool val) {
    Value v;
    v.type = VAL_BOOL;
    v.as.bool_val = val;
    return v;
}

Value create_string(const char *val) {
    Value v;
    v.type = VAL_STRING;
    v.as.string_val = strdup(val);
    return v;
}

Value create_void(void) {
    Value v;
    v.type = VAL_VOID;
    return v;
}