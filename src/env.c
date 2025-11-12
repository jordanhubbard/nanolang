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
    env->structs = malloc(sizeof(StructDef) * 8);
    env->struct_count = 0;
    env->struct_capacity = 8;
    env->enums = malloc(sizeof(EnumDef) * 8);
    env->enum_count = 0;
    env->enum_capacity = 8;
    return env;
}

/* Free environment */
void free_environment(Environment *env) {
    for (int i = 0; i < env->symbol_count; i++) {
        free(env->symbols[i].name);
        if (env->symbols[i].struct_type_name) {
            free(env->symbols[i].struct_type_name);
        }
        if (env->symbols[i].value.type == VAL_STRING) {
            free(env->symbols[i].value.as.string_val);
        }
        if (env->symbols[i].value.type == VAL_STRUCT) {
            StructValue *sv = env->symbols[i].value.as.struct_val;
            free(sv->struct_name);
            for (int j = 0; j < sv->field_count; j++) {
                free(sv->field_names[j]);
            }
            free(sv->field_names);
            free(sv->field_values);
            free(sv);
        }
    }
    free(env->symbols);

    for (int i = 0; i < env->function_count; i++) {
        /* Note: function names are not owned by environment */
    }
    free(env->functions);
    
    for (int i = 0; i < env->struct_count; i++) {
        free(env->structs[i].name);
        for (int j = 0; j < env->structs[i].field_count; j++) {
            free(env->structs[i].field_names[j]);
        }
        free(env->structs[i].field_names);
        free(env->structs[i].field_types);
    }
    free(env->structs);
    
    for (int i = 0; i < env->enum_count; i++) {
        free(env->enums[i].name);
        for (int j = 0; j < env->enums[i].variant_count; j++) {
            free(env->enums[i].variant_names[j]);
        }
        free(env->enums[i].variant_names);
        free(env->enums[i].variant_values);
    }
    free(env->enums);

    free(env);
}

/* Define variable */
void env_define_var(Environment *env, const char *name, Type type, bool is_mut, Value value) {
    env_define_var_with_element_type(env, name, type, TYPE_UNKNOWN, is_mut, value);
}

void env_define_var_with_element_type(Environment *env, const char *name, Type type, Type element_type, bool is_mut, Value value) {
    if (env->symbol_count >= env->symbol_capacity) {
        env->symbol_capacity *= 2;
        env->symbols = realloc(env->symbols, sizeof(Symbol) * env->symbol_capacity);
    }

    Symbol sym;
    sym.name = strdup(name);
    sym.type = type;
    sym.struct_type_name = NULL;  /* Initialize to NULL (set later for struct types) */
    sym.element_type = element_type;  /* Store element type for arrays */
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
        /* Skip symbols with NULL names */
        if (!env->symbols[i].name) {
            continue;
        }
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

/* Check if a name is a built-in function */
bool is_builtin_function(const char *name) {
    if (!name) {
        return false;
    }
    for (int i = 0; i < builtin_function_count; i++) {
        if (builtin_functions[i].name && strcmp(builtin_functions[i].name, name) == 0) {
            return true;
        }
    }
    return false;
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
        /* Skip functions with NULL names */
        if (!env->functions[i].name) {
            continue;
        }
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

Value create_struct(const char *struct_name, char **field_names, Value *field_values, int field_count) {
    Value v;
    v.type = VAL_STRUCT;
    v.is_return = false;
    v.as.struct_val = malloc(sizeof(StructValue));
    v.as.struct_val->struct_name = strdup(struct_name);
    v.as.struct_val->field_count = field_count;
    
    /* Allocate and copy field names */
    v.as.struct_val->field_names = malloc(sizeof(char*) * field_count);
    for (int i = 0; i < field_count; i++) {
        v.as.struct_val->field_names[i] = strdup(field_names[i]);
    }
    
    /* Allocate and copy field values */
    v.as.struct_val->field_values = malloc(sizeof(Value) * field_count);
    for (int i = 0; i < field_count; i++) {
        v.as.struct_val->field_values[i] = field_values[i];
    }
    
    return v;
}

/* Define struct */
void env_define_struct(Environment *env, StructDef struct_def) {
    if (env->struct_count >= env->struct_capacity) {
        env->struct_capacity *= 2;
        env->structs = realloc(env->structs, sizeof(StructDef) * env->struct_capacity);
    }
    env->structs[env->struct_count++] = struct_def;
}

/* Get struct definition */
StructDef *env_get_struct(Environment *env, const char *name) {
    for (int i = 0; i < env->struct_count; i++) {
        if (strcmp(env->structs[i].name, name) == 0) {
            return &env->structs[i];
        }
    }
    return NULL;
}

/* Define enum */
void env_define_enum(Environment *env, EnumDef enum_def) {
    if (env->enum_count >= env->enum_capacity) {
        env->enum_capacity *= 2;
        env->enums = realloc(env->enums, sizeof(EnumDef) * env->enum_capacity);
    }
    env->enums[env->enum_count++] = enum_def;
}

/* Get enum definition */
EnumDef *env_get_enum(Environment *env, const char *name) {
    for (int i = 0; i < env->enum_count; i++) {
        if (strcmp(env->enums[i].name, name) == 0) {
            return &env->enums[i];
        }
    }
    return NULL;
}

/* Get enum variant value */
int env_get_enum_variant(Environment *env, const char *variant_name) {
    for (int i = 0; i < env->enum_count; i++) {
        for (int j = 0; j < env->enums[i].variant_count; j++) {
            if (strcmp(env->enums[i].variant_names[j], variant_name) == 0) {
                return env->enums[i].variant_values[j];
            }
        }
    }
    return -1;  /* Not found */
}