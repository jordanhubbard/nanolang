#include "nanolang.h"

/* Built-in function metadata */
typedef struct {
    const char *name;
    int param_count;
    Type param_types[3];  /* Max 3 params for now */
    Type return_type;
} BuiltinFuncInfo;

/* Built-in function definitions */
static BuiltinFuncInfo builtin_functions[] = {
    /* Core functions */
    {"range", 2, {TYPE_INT, TYPE_INT, TYPE_UNKNOWN}, TYPE_INT},

    /* File operations */
    {"file_read", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_STRING},
    {"file_write", 2, {TYPE_STRING, TYPE_STRING, TYPE_UNKNOWN}, TYPE_INT},
    {"file_append", 2, {TYPE_STRING, TYPE_STRING, TYPE_UNKNOWN}, TYPE_INT},
    {"file_remove", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},
    {"file_rename", 2, {TYPE_STRING, TYPE_STRING, TYPE_UNKNOWN}, TYPE_INT},
    {"file_exists", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_BOOL},
    {"file_size", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},

    /* Directory operations */
    {"dir_create", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},
    {"dir_remove", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},
    {"dir_list", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_STRING},
    {"dir_exists", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_BOOL},
    {"getcwd", 0, {TYPE_UNKNOWN, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_STRING},
    {"chdir", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},

    /* Path operations */
    {"path_isfile", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_BOOL},
    {"path_isdir", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_BOOL},
    {"path_join", 2, {TYPE_STRING, TYPE_STRING, TYPE_UNKNOWN}, TYPE_STRING},
    {"path_basename", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_STRING},
    {"path_dirname", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_STRING},

    /* Process operations */
    {"system", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},
    {"exit", 1, {TYPE_INT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_VOID},
    {"getenv", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_STRING},
};

static const int builtin_function_count = sizeof(builtin_functions) / sizeof(BuiltinFuncInfo);

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
    sym.is_used = false;  /* Initialize as unused */
    sym.def_line = 0;     /* Will be set by type checker if needed */
    sym.def_column = 0;

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
    for (int i = 0; i < builtin_function_count; i++) {
        if (strcmp(builtin_functions[i].name, name) == 0) {
            /* Create static function objects for built-ins */
            static Function func_cache[64];  /* Should match builtin_function_count */
            static bool initialized[64] = {false};

            if (!initialized[i]) {
                func_cache[i].name = (char *)builtin_functions[i].name;
                func_cache[i].param_count = builtin_functions[i].param_count;
                func_cache[i].return_type = builtin_functions[i].return_type;
                func_cache[i].params = NULL;  /* Built-ins don't need param names */
                func_cache[i].body = NULL;
                func_cache[i].shadow_test = NULL;
                initialized[i] = true;
            }

            return &func_cache[i];
        }
    }

    /* Check user-defined functions */
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
    v.is_return = false;
    v.as.int_val = val;
    return v;
}

Value create_float(double val) {
    Value v;
    v.type = VAL_FLOAT;
    v.is_return = false;
    v.as.float_val = val;
    return v;
}

Value create_bool(bool val) {
    Value v;
    v.type = VAL_BOOL;
    v.is_return = false;
    v.as.bool_val = val;
    return v;
}

Value create_string(const char *val) {
    Value v;
    v.type = VAL_STRING;
    v.is_return = false;
    v.as.string_val = strdup(val);
    return v;
}

Value create_void(void) {
    Value v;
    v.type = VAL_VOID;
    v.is_return = false;
    return v;
}

Value create_array(ValueType elem_type, int length, int capacity) {
    Value v;
    v.type = VAL_ARRAY;
    v.is_return = false;
    v.as.array_val = malloc(sizeof(Array));
    v.as.array_val->element_type = elem_type;
    v.as.array_val->length = length;
    v.as.array_val->capacity = capacity > length ? capacity : length;
    
    /* Allocate data based on element type */
    size_t elem_size;
    switch (elem_type) {
        case VAL_INT:    elem_size = sizeof(long long); break;
        case VAL_FLOAT:  elem_size = sizeof(double); break;
        case VAL_BOOL:   elem_size = sizeof(bool); break;
        case VAL_STRING: elem_size = sizeof(char*); break;
        default:         elem_size = sizeof(void*); break;
    }
    v.as.array_val->data = calloc(v.as.array_val->capacity, elem_size);
    
    return v;
}