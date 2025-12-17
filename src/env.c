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
    
    /* Math and utility functions */
    {"abs", 1, {TYPE_INT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},
    {"min", 2, {TYPE_INT, TYPE_INT, TYPE_UNKNOWN}, TYPE_INT},
    {"max", 2, {TYPE_INT, TYPE_INT, TYPE_UNKNOWN}, TYPE_INT},
    {"print", 1, {TYPE_INT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_VOID},
    {"println", 1, {TYPE_INT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_VOID},
    {"sqrt", 1, {TYPE_FLOAT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_FLOAT},
    {"pow", 2, {TYPE_FLOAT, TYPE_FLOAT, TYPE_UNKNOWN}, TYPE_FLOAT},
    {"floor", 1, {TYPE_FLOAT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_FLOAT},
    {"ceil", 1, {TYPE_FLOAT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_FLOAT},
    {"round", 1, {TYPE_FLOAT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_FLOAT},
    {"sin", 1, {TYPE_FLOAT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_FLOAT},
    {"cos", 1, {TYPE_FLOAT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_FLOAT},
    {"tan", 1, {TYPE_FLOAT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_FLOAT},
    {"atan2", 2, {TYPE_FLOAT, TYPE_FLOAT, TYPE_UNKNOWN}, TYPE_FLOAT},
    
    /* Type casting */
    {"cast_int", 1, {TYPE_UNKNOWN, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},
    {"cast_float", 1, {TYPE_UNKNOWN, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_FLOAT},
    {"cast_bool", 1, {TYPE_UNKNOWN, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_BOOL},
    {"cast_string", 1, {TYPE_UNKNOWN, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_STRING},
    
    /* String operations */
    {"str_length", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},
    {"str_concat", 2, {TYPE_STRING, TYPE_STRING, TYPE_UNKNOWN}, TYPE_STRING},
    {"str_substring", 3, {TYPE_STRING, TYPE_INT, TYPE_INT}, TYPE_STRING},
    {"str_contains", 2, {TYPE_STRING, TYPE_STRING, TYPE_UNKNOWN}, TYPE_BOOL},
    {"str_equals", 2, {TYPE_STRING, TYPE_STRING, TYPE_UNKNOWN}, TYPE_BOOL},
    
    /* Array operations */
    {"array_length", 1, {TYPE_ARRAY, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},
    {"array_new", 2, {TYPE_INT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_ARRAY},
    {"array_set", 3, {TYPE_ARRAY, TYPE_INT, TYPE_INT}, TYPE_VOID},
    {"at", 2, {TYPE_ARRAY, TYPE_INT, TYPE_UNKNOWN}, TYPE_INT},

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
    env->enums = calloc(8, sizeof(EnumDef));  /* Use calloc to zero-initialize */
    env->enum_count = 0;
    env->enum_capacity = 8;
    env->unions = malloc(sizeof(UnionDef) * 8);
    env->union_count = 0;
    env->union_capacity = 8;
    env->opaque_types = malloc(sizeof(OpaqueTypeDef) * 8);
    env->opaque_type_count = 0;
    env->opaque_type_capacity = 8;
    env->generic_instances = malloc(sizeof(GenericInstantiation) * 8);
    env->generic_instance_count = 0;
    env->generic_instance_capacity = 8;
    env->namespaces = malloc(sizeof(ModuleNamespace) * 8);
    env->namespace_count = 0;
    env->namespace_capacity = 8;
    env->current_module = NULL;  /* Start in global scope */
    
    /* Initialize import tracker */
    env->import_tracker = malloc(sizeof(ImportTracker));
    env->import_tracker->imports = malloc(sizeof(SelectiveImport) * 8);
    env->import_tracker->import_count = 0;
    env->import_tracker->import_capacity = 8;
    
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
        if (env->symbols[i].value.type == VAL_FUNCTION) {
            /* Free function value - strdup'd strings and signature */
            if (env->symbols[i].value.as.function_val.function_name) {
                free((char*)env->symbols[i].value.as.function_val.function_name);
            }
            if (env->symbols[i].value.as.function_val.signature) {
                free_function_signature(env->symbols[i].value.as.function_val.signature);
            }
        }
    }
    free(env->symbols);

    for (int i = 0; i < env->function_count; i++) {
        /* Note: function names are not owned by environment - they point to AST */
        /* Freeing them causes double-free crashes */
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
    
    /* Free unions */
    for (int i = 0; i < env->union_count; i++) {
        free(env->unions[i].name);
        for (int j = 0; j < env->unions[i].variant_count; j++) {
            free(env->unions[i].variant_names[j]);
            if (env->unions[i].variant_field_names && env->unions[i].variant_field_names[j]) {
                for (int k = 0; k < env->unions[i].variant_field_counts[j]; k++) {
                    free(env->unions[i].variant_field_names[j][k]);
                }
                free(env->unions[i].variant_field_names[j]);
            }
            if (env->unions[i].variant_field_types && env->unions[i].variant_field_types[j]) {
                free(env->unions[i].variant_field_types[j]);
            }
        }
        if (env->unions[i].variant_names) free(env->unions[i].variant_names);
        if (env->unions[i].variant_field_counts) free(env->unions[i].variant_field_counts);
        if (env->unions[i].variant_field_names) free(env->unions[i].variant_field_names);
        if (env->unions[i].variant_field_types) free(env->unions[i].variant_field_types);
    }
    free(env->unions);
    
    /* Free generic instantiations */
    for (int i = 0; i < env->generic_instance_count; i++) {
        free(env->generic_instances[i].generic_name);
        free(env->generic_instances[i].type_args);
        free(env->generic_instances[i].concrete_name);
        if (env->generic_instances[i].type_arg_names) {
            for (int j = 0; j < env->generic_instances[i].type_arg_count; j++) {
                if (env->generic_instances[i].type_arg_names[j]) {
                    free(env->generic_instances[i].type_arg_names[j]);
                }
            }
            free(env->generic_instances[i].type_arg_names);
        }
    }
    free(env->generic_instances);

    /* Free opaque types */
    for (int i = 0; i < env->opaque_type_count; i++) {
        free(env->opaque_types[i].name);
        free(env->opaque_types[i].c_type_name);
    }
    free(env->opaque_types);

    /* Free namespaces */
    for (int i = 0; i < env->namespace_count; i++) {
        free(env->namespaces[i].alias);
        for (int j = 0; j < env->namespaces[i].function_count; j++) {
            free(env->namespaces[i].function_names[j]);
        }
        free(env->namespaces[i].function_names);
        for (int j = 0; j < env->namespaces[i].struct_count; j++) {
            free(env->namespaces[i].struct_names[j]);
        }
        free(env->namespaces[i].struct_names);
        for (int j = 0; j < env->namespaces[i].enum_count; j++) {
            free(env->namespaces[i].enum_names[j]);
        }
        free(env->namespaces[i].enum_names);
        for (int j = 0; j < env->namespaces[i].union_count; j++) {
            free(env->namespaces[i].union_names[j]);
        }
        free(env->namespaces[i].union_names);
    }
    free(env->namespaces);

    free(env);
}

/* Define variable */
void env_define_var(Environment *env, const char *name, Type type, bool is_mut, Value value) {
    env_define_var_with_element_type(env, name, type, TYPE_UNKNOWN, is_mut, value);
}

void env_define_var_with_element_type(Environment *env, const char *name, Type type, Type element_type, bool is_mut, Value value) {
    env_define_var_with_type_info(env, name, type, element_type, NULL, is_mut, value);
}

void env_define_var_with_type_info(Environment *env, const char *name, Type type, Type element_type, TypeInfo *type_info, bool is_mut, Value value) {
    if (env->symbol_count >= env->symbol_capacity) {
        env->symbol_capacity *= 2;
        env->symbols = realloc(env->symbols, sizeof(Symbol) * env->symbol_capacity);
    }

    Symbol sym;
    sym.name = strdup(name);
    sym.type = type;
    sym.struct_type_name = NULL;  /* Initialize to NULL (set later for struct types) */
    sym.element_type = element_type;  /* Store element type for arrays */
    sym.type_info = type_info;  /* Store full type info for complex types (tuples, etc.) */
    sym.is_mut = is_mut;
    sym.value = value;
    sym.is_used = false;  /* Initialize as unused */
    sym.from_c_header = false;  /* Not from C header (normal nanolang variable) */
    sym.def_line = 0;     /* Will be set by type checker if needed */
    sym.def_column = 0;

    /* WORKAROUND: Check if symbol already exists and preserve/update metadata */
    /* This handles a bug where symbols are added multiple times during type-checking.
     * When a symbol is re-added, preserve or update struct_type_name to maintain type information. */
    Symbol *existing = env_get_var(env, name);
    if (existing) {
        /* If existing has struct_type_name but new one doesn't, preserve it */
        if (existing->struct_type_name && !sym.struct_type_name) {
            sym.struct_type_name = strdup(existing->struct_type_name);
        }
        /* If new one has struct_type_name but existing doesn't, update the existing symbol instead */
        else if (!existing->struct_type_name && sym.struct_type_name) {
            /* Update the existing symbol with the new metadata */
            existing->struct_type_name = strdup(sym.struct_type_name);
            existing->type = sym.type;
            existing->element_type = sym.element_type;
            existing->type_info = sym.type_info;
            existing->is_mut = sym.is_mut;
            existing->value = sym.value;
            /* Don't add a new symbol - we updated the existing one */
            return;
        }
    }
    
    env->symbols[env->symbol_count++] = sym;
}

/* Get variable */
Symbol *env_get_var(Environment *env, const char *name) {
    for (int i = env->symbol_count - 1; i >= 0; i--) {
        /* Skip symbols with NULL names */
        if (!env->symbols[i].name) {
            continue;
        }
        if (safe_strcmp(env->symbols[i].name, name) == 0) {
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
        /* Free old function value if needed */
        if (sym->value.type == VAL_FUNCTION) {
            if (sym->value.as.function_val.function_name) {
                free((char*)sym->value.as.function_val.function_name);
            }
            if (sym->value.as.function_val.signature) {
                free_function_signature(sym->value.as.function_val.signature);
            }
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
        if (safe_strcmp(builtin_functions[i].name, name) == 0) {
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
        if (!env->functions) {
            fprintf(stderr, "Error: Out of memory reallocating functions array\n");
            exit(1);
        }
    }

    env->functions[env->function_count++] = func;
}

/* Get function */
Function *env_get_function(Environment *env, const char *name) {
    if (!name) {
        return NULL;
    }

    /* Check for Module.function pattern */
    const char *dot = strchr(name, '.');
    if (dot) {
        char module_alias[256];
        size_t module_len = dot - name;
        if (module_len >= sizeof(module_alias)) {
            module_len = sizeof(module_alias) - 1;
        }
        strncpy(module_alias, name, module_len);
        module_alias[module_len] = '\0';
        const char *func_name = dot + 1;
        
        /* Find namespace */
        for (int i = 0; i < env->namespace_count; i++) {
            if (strcmp(env->namespaces[i].alias, module_alias) == 0) {
                /* Check if function is in this namespace */
                for (int j = 0; j < env->namespaces[i].function_count; j++) {
                    if (strcmp(env->namespaces[i].function_names[j], func_name) == 0) {
                        /* Look up the actual function by its original name */
                        return env_get_function(env, func_name);
                    }
                }
                /* Function not found in this module's namespace */
                return NULL;
            }
        }
        /* Module alias not found */
        return NULL;
    }

    /* Check built-in functions first */
    for (int i = 0; i < builtin_function_count; i++) {
        if (safe_strcmp(builtin_functions[i].name, name) == 0) {
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
        if (safe_strcmp(env->functions[i].name, name) == 0) {
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

Value create_union(const char *union_name, int variant_index, const char *variant_name, 
                   char **field_names, Value *field_values, int field_count) {
    Value v;
    v.type = VAL_UNION;
    v.is_return = false;
    v.as.union_val = malloc(sizeof(UnionValue));
    v.as.union_val->union_name = strdup(union_name);
    v.as.union_val->variant_index = variant_index;
    v.as.union_val->variant_name = strdup(variant_name);
    v.as.union_val->field_count = field_count;
    
    /* Allocate and copy field names */
    if (field_count > 0) {
        v.as.union_val->field_names = malloc(sizeof(char*) * field_count);
        for (int i = 0; i < field_count; i++) {
            v.as.union_val->field_names[i] = strdup(field_names[i]);
        }
        
        /* Allocate and copy field values */
        v.as.union_val->field_values = malloc(sizeof(Value) * field_count);
        for (int i = 0; i < field_count; i++) {
            v.as.union_val->field_values[i] = field_values[i];
        }
    } else {
        v.as.union_val->field_names = NULL;
        v.as.union_val->field_values = NULL;
    }
    
    return v;
}

/* Define struct */
void env_define_struct(Environment *env, StructDef struct_def) {
    /* Check if struct already exists - prevent duplicates */
    if (env_get_struct(env, struct_def.name) != NULL) {
        /* Struct already defined - skip duplicate registration */
        return;
    }
    
    if (env->struct_count >= env->struct_capacity) {
        env->struct_capacity *= 2;
        env->structs = realloc(env->structs, sizeof(StructDef) * env->struct_capacity);
    }
    env->structs[env->struct_count++] = struct_def;
}

/* Get struct definition */
StructDef *env_get_struct(Environment *env, const char *name) {
    /* Check for Module.Type pattern */
    const char *dot = strchr(name, '.');
    if (dot) {
        char module_alias[256];
        size_t module_len = dot - name;
        if (module_len >= sizeof(module_alias)) {
            module_len = sizeof(module_alias) - 1;
        }
        strncpy(module_alias, name, module_len);
        module_alias[module_len] = '\0';
        const char *type_name = dot + 1;
        
        /* Find namespace */
        for (int i = 0; i < env->namespace_count; i++) {
            if (strcmp(env->namespaces[i].alias, module_alias) == 0) {
                /* Check if struct is in this namespace */
                for (int j = 0; j < env->namespaces[i].struct_count; j++) {
                    if (strcmp(env->namespaces[i].struct_names[j], type_name) == 0) {
                        /* Look up the actual struct by its original name */
                        return env_get_struct(env, type_name);
                    }
                }
                return NULL;
            }
        }
        return NULL;
    }
    
    for (int i = 0; i < env->struct_count; i++) {
        if (safe_strcmp(env->structs[i].name, name) == 0) {
            return &env->structs[i];
        }
    }
    return NULL;
}

/* Define enum */
void env_define_enum(Environment *env, EnumDef enum_def) {
    if (!env || !enum_def.name) {
        return;  /* Invalid enum definition */
    }
    
    /* Check if enum already exists - prevent duplicates */
    if (env_get_enum(env, enum_def.name) != NULL) {
        /* Enum already defined - skip duplicate registration */
        return;
    }
    
    if (env->enum_count >= env->enum_capacity) {
        int old_capacity = env->enum_capacity;
        env->enum_capacity *= 2;
        EnumDef *new_enums = realloc(env->enums, sizeof(EnumDef) * env->enum_capacity);
        if (!new_enums) {
            fprintf(stderr, "Error: Failed to reallocate memory for enums\n");
            return;
        }
        env->enums = new_enums;
        /* Zero-initialize the newly allocated memory */
        memset(&env->enums[old_capacity], 0, sizeof(EnumDef) * (env->enum_capacity - old_capacity));
    }
    env->enums[env->enum_count++] = enum_def;
}

/* Get enum definition */
EnumDef *env_get_enum(Environment *env, const char *name) {
    if (!env || !name) return NULL;
    
    if (!env->enums) return NULL;
    
    /* Defensive check: ensure enum_count is valid */
    if (env->enum_count < 0 || env->enum_count > env->enum_capacity) {
        return NULL;
    }
    
    /* Check for Module.Type pattern */
    const char *dot = strchr(name, '.');
    if (dot) {
        char module_alias[256];
        size_t module_len = dot - name;
        if (module_len >= sizeof(module_alias)) {
            module_len = sizeof(module_alias) - 1;
        }
        strncpy(module_alias, name, module_len);
        module_alias[module_len] = '\0';
        const char *type_name = dot + 1;
        
        for (int i = 0; i < env->namespace_count; i++) {
            if (strcmp(env->namespaces[i].alias, module_alias) == 0) {
                for (int j = 0; j < env->namespaces[i].enum_count; j++) {
                    if (strcmp(env->namespaces[i].enum_names[j], type_name) == 0) {
                        return env_get_enum(env, type_name);
                    }
                }
                return NULL;
            }
        }
        return NULL;
    }
    
    for (int i = 0; i < env->enum_count; i++) {
        /* Use safe_strcmp which handles NULL pointers */
        if (safe_strcmp(env->enums[i].name, name) == 0) {
            return &env->enums[i];
        }
    }
    return NULL;
}

/* Get enum variant value */
int env_get_enum_variant(Environment *env, const char *variant_name) {
    if (!env || !variant_name) return -1;
    
    for (int i = 0; i < env->enum_count; i++) {
        if (!env->enums[i].variant_names) continue;
        for (int j = 0; j < env->enums[i].variant_count; j++) {
            if (safe_strcmp(env->enums[i].variant_names[j], variant_name) == 0) {
                return (env->enums[i].variant_values && j < env->enums[i].variant_count) ? 
                    env->enums[i].variant_values[j] : j;
            }
        }
    }
    return -1;  /* Not found */
}

/* Define union */
void env_define_union(Environment *env, UnionDef union_def) {
    /* Check if union already exists - prevent duplicates */
    if (env_get_union(env, union_def.name) != NULL) {
        /* Union already defined - skip duplicate registration */
        return;
    }
    
    if (env->union_count >= env->union_capacity) {
        env->union_capacity *= 2;
        env->unions = realloc(env->unions, sizeof(UnionDef) * env->union_capacity);
    }
    env->unions[env->union_count++] = union_def;
}

/* Get union definition */
UnionDef *env_get_union(Environment *env, const char *name) {
    /* Check for Module.Type pattern */
    const char *dot = strchr(name, '.');
    if (dot) {
        char module_alias[256];
        size_t module_len = dot - name;
        if (module_len >= sizeof(module_alias)) {
            module_len = sizeof(module_alias) - 1;
        }
        strncpy(module_alias, name, module_len);
        module_alias[module_len] = '\0';
        const char *type_name = dot + 1;
        
        for (int i = 0; i < env->namespace_count; i++) {
            if (strcmp(env->namespaces[i].alias, module_alias) == 0) {
                for (int j = 0; j < env->namespaces[i].union_count; j++) {
                    if (strcmp(env->namespaces[i].union_names[j], type_name) == 0) {
                        return env_get_union(env, type_name);
                    }
                }
                return NULL;
            }
        }
        return NULL;
    }
    
    for (int i = 0; i < env->union_count; i++) {
        if (safe_strcmp(env->unions[i].name, name) == 0) {
            return &env->unions[i];
        }
    }
    return NULL;
}

/* Get variant index in union (returns -1 if not found) */
int env_get_union_variant_index(Environment *env, const char *union_name, const char *variant_name) {
    UnionDef *udef = env_get_union(env, union_name);
    if (!udef) return -1;
    
    for (int i = 0; i < udef->variant_count; i++) {
        if (safe_strcmp(udef->variant_names[i], variant_name) == 0) {
            return i;
        }
    }
    return -1;
}

/* Define opaque type */
void env_define_opaque_type(Environment *env, const char *name) {
    /* Check if opaque type already exists - prevent duplicates */
    if (env_get_opaque_type(env, name) != NULL) {
        /* Opaque type already defined - skip duplicate registration */
        return;
    }
    
    if (env->opaque_type_count >= env->opaque_type_capacity) {
        env->opaque_type_capacity *= 2;
        env->opaque_types = realloc(env->opaque_types, sizeof(OpaqueTypeDef) * env->opaque_type_capacity);
    }
    
    OpaqueTypeDef opaque_type;
    opaque_type.name = strdup(name);
    
    /* Generate C type name by adding pointer: "GLFWwindow" -> "GLFWwindow*" */
    size_t len = strlen(name);
    opaque_type.c_type_name = malloc(len + 2);  /* +1 for '*', +1 for '\0' */
    strcpy(opaque_type.c_type_name, name);
    strcat(opaque_type.c_type_name, "*");
    
    env->opaque_types[env->opaque_type_count++] = opaque_type;
}

/* Get opaque type definition */
OpaqueTypeDef *env_get_opaque_type(Environment *env, const char *name) {
    if (!env || !name) return NULL;
    
    for (int i = 0; i < env->opaque_type_count; i++) {
        if (safe_strcmp(env->opaque_types[i].name, name) == 0) {
            return &env->opaque_types[i];
        }
    }
    return NULL;
}

/* Register a list instantiation for code generation */
void env_register_list_instantiation(Environment *env, const char *element_type) {
    /* Check if already registered */
    for (int i = 0; i < env->generic_instance_count; i++) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (safe_strcmp(inst->generic_name, "List") == 0 &&
            inst->type_arg_names && 
            safe_strcmp(inst->type_arg_names[0], element_type) == 0) {
            return;  /* Already registered */
        }
    }
    
    /* Add new instantiation */
    if (env->generic_instance_count >= env->generic_instance_capacity) {
        env->generic_instance_capacity *= 2;
        env->generic_instances = realloc(env->generic_instances,
            sizeof(GenericInstantiation) * env->generic_instance_capacity);
    }
    
    GenericInstantiation inst;
    inst.generic_name = strdup("List");
    inst.type_arg_count = 1;
    inst.type_args = malloc(sizeof(Type));
    inst.type_args[0] = TYPE_LIST_GENERIC;
    inst.type_arg_names = malloc(sizeof(char*));
    inst.type_arg_names[0] = strdup(element_type);
    
    /* Generate specialized name: List<Point> -> List_Point */
    char specialized[256];
    snprintf(specialized, sizeof(specialized), "List_%s", element_type);
    inst.concrete_name = strdup(specialized);
    
    env->generic_instances[env->generic_instance_count++] = inst;
    
    /* Register specialized functions in environment for type checking */
    char func_name[512];  /* Increased to handle long type names + suffixes */
    Function func;
    Parameter *params;
    
    /* List_T_new() -> List<T>* */
    snprintf(func_name, sizeof(func_name), "%s_new", specialized);
    func.name = strdup(func_name);
    func.param_count = 0;
    func.params = NULL;
    func.return_type = TYPE_LIST_GENERIC;
    func.return_struct_type_name = NULL;
    func.return_fn_sig = NULL;
    func.return_type_info = NULL;
    func.body = NULL;  /* Built-in */
    func.shadow_test = NULL;
    func.is_extern = true;
    func.is_pub = false;
    func.module_name = NULL;
    env_define_function(env, func);
    
    /* List_T_push(list: List<T>*, value: T) -> void */
    snprintf(func_name, sizeof(func_name), "%s_push", specialized);
    func.name = strdup(func_name);
    func.param_count = 2;
    params = malloc(sizeof(Parameter) * 2);
    params[0].name = strdup("list");
    params[0].type = TYPE_LIST_GENERIC;
    params[0].struct_type_name = NULL;
    params[0].element_type = TYPE_UNKNOWN;
    params[1].name = strdup("value");
    params[1].type = TYPE_STRUCT;
    params[1].struct_type_name = strdup(element_type);
    params[1].element_type = TYPE_UNKNOWN;
    func.params = params;
    func.return_type = TYPE_VOID;
    func.return_struct_type_name = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = true;
    env_define_function(env, func);
    
    /* List_T_get(list: List<T>*, index: int) -> T */
    snprintf(func_name, sizeof(func_name), "%s_get", specialized);
    func.name = strdup(func_name);
    func.param_count = 2;
    params = malloc(sizeof(Parameter) * 2);
    params[0].name = strdup("list");
    params[0].type = TYPE_LIST_GENERIC;
    params[0].struct_type_name = NULL;
    params[0].element_type = TYPE_UNKNOWN;
    params[1].name = strdup("index");
    params[1].type = TYPE_INT;
    params[1].struct_type_name = NULL;
    params[1].element_type = TYPE_UNKNOWN;
    func.params = params;
    func.return_type = TYPE_STRUCT;
    func.return_struct_type_name = strdup(element_type);
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = true;
    env_define_function(env, func);
    
    /* List_T_length(list: List<T>*) -> int */
    snprintf(func_name, sizeof(func_name), "%s_length", specialized);
    func.name = strdup(func_name);
    func.param_count = 1;
    params = malloc(sizeof(Parameter));
    params[0].name = strdup("list");
    params[0].type = TYPE_LIST_GENERIC;
    params[0].struct_type_name = NULL;
    params[0].element_type = TYPE_UNKNOWN;
    func.params = params;
    func.return_type = TYPE_INT;
    func.return_struct_type_name = NULL;
    func.return_fn_sig = NULL;
    func.return_type_info = NULL;
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = true;
    func.is_pub = false;
    func.module_name = NULL;
    env_define_function(env, func);
}

/* Register a generic union instantiation for code generation
 * Example: Result<int, string> -> Result_int_string
 */
void env_register_union_instantiation(Environment *env, const char *union_name, 
                                     const char **type_args, int type_arg_count) {
    if (!env || !union_name || !type_args || type_arg_count == 0) {
        return;
    }
    
    /* Check if already registered */
    for (int i = 0; i < env->generic_instance_count; i++) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (safe_strcmp(inst->generic_name, union_name) == 0 &&
            inst->type_arg_count == type_arg_count) {
            /* Check if all type args match */
            bool all_match = true;
            for (int j = 0; j < type_arg_count; j++) {
                if (safe_strcmp(inst->type_arg_names[j], type_args[j]) != 0) {
                    all_match = false;
                    break;
                }
            }
            if (all_match) {
                return;  /* Already registered */
            }
        }
    }
    
    /* Add new instantiation */
    if (env->generic_instance_count >= env->generic_instance_capacity) {
        env->generic_instance_capacity *= 2;
        env->generic_instances = realloc(env->generic_instances,
            sizeof(GenericInstantiation) * env->generic_instance_capacity);
    }
    
    GenericInstantiation inst;
    inst.generic_name = strdup(union_name);
    inst.type_arg_count = type_arg_count;
    inst.type_args = malloc(sizeof(Type) * type_arg_count);
    inst.type_arg_names = malloc(sizeof(char*) * type_arg_count);
    
    for (int i = 0; i < type_arg_count; i++) {
        inst.type_args[i] = TYPE_UNION;  /* Generic union type */
        inst.type_arg_names[i] = strdup(type_args[i]);
    }
    
    /* Generate specialized name: Result<int, string> -> Result_int_string */
    char specialized[512];
    int offset = snprintf(specialized, sizeof(specialized), "%s", union_name);
    for (int i = 0; i < type_arg_count && offset < (int)sizeof(specialized) - 20; i++) {
        offset += snprintf(specialized + offset, sizeof(specialized) - offset, 
                          "_%s", type_args[i]);
    }
    inst.concrete_name = strdup(specialized);
    
    env->generic_instances[env->generic_instance_count++] = inst;
}

/* ============================================================================
 * Function Signature Helpers (for first-class functions)
 * ============================================================================
 */

/* Create a function signature */
FunctionSignature *create_function_signature(Type *param_types, int param_count, Type return_type) {
    FunctionSignature *sig = malloc(sizeof(FunctionSignature));
    sig->param_count = param_count;
    sig->return_type = return_type;
    sig->return_struct_name = NULL;
    sig->return_fn_sig = NULL;  /* Initialize function return signature */
    
    if (param_count > 0) {
        sig->param_types = malloc(sizeof(Type) * param_count);
        sig->param_struct_names = malloc(sizeof(char*) * param_count);
        
        for (int i = 0; i < param_count; i++) {
            sig->param_types[i] = param_types[i];
            sig->param_struct_names[i] = NULL;
        }
    } else {
        sig->param_types = NULL;
        sig->param_struct_names = NULL;
    }
    
    return sig;
}

/* Free a function signature */
void free_function_signature(FunctionSignature *sig) {
    if (!sig) return;
    
    if (sig->param_types) {
        free(sig->param_types);
    }
    
    if (sig->param_struct_names) {
        for (int i = 0; i < sig->param_count; i++) {
            if (sig->param_struct_names[i]) {
                free(sig->param_struct_names[i]);
            }
        }
        free(sig->param_struct_names);
    }
    
    if (sig->return_struct_name) {
        free(sig->return_struct_name);
    }
    
    /* Free nested function signature if present */
    if (sig->return_fn_sig) {
        free_function_signature(sig->return_fn_sig);
    }
    
    free(sig);
}

/* Check if two function signatures are equal */
bool function_signatures_equal(FunctionSignature *sig1, FunctionSignature *sig2) {
    if (!sig1 || !sig2) return false;
    
    /* Check parameter count */
    if (sig1->param_count != sig2->param_count) return false;
    
    /* Check each parameter type */
    for (int i = 0; i < sig1->param_count; i++) {
        if (sig1->param_types[i] != sig2->param_types[i]) return false;
        
        /* For struct/enum parameters, check names match */
        if (sig1->param_types[i] == TYPE_STRUCT || 
            sig1->param_types[i] == TYPE_ENUM ||
            sig1->param_types[i] == TYPE_UNION) {
            
            const char *name1 = sig1->param_struct_names[i];
            const char *name2 = sig2->param_struct_names[i];
            
            if ((name1 == NULL) != (name2 == NULL)) return false;
            if (safe_strcmp(name1, name2) != 0) return false;
        }
    }
    
    /* Check return type */
    if (sig1->return_type != sig2->return_type) return false;
    
    /* For struct/enum returns, check names match */
    if (sig1->return_type == TYPE_STRUCT || 
        sig1->return_type == TYPE_ENUM ||
        sig1->return_type == TYPE_UNION) {
        
        const char *name1 = sig1->return_struct_name;
        const char *name2 = sig2->return_struct_name;
        
        if ((name1 == NULL) != (name2 == NULL)) return false;
        if (name1 && name2 && strcmp(name1, name2) != 0) return false;
    }
    
    /* For function returns, check function signatures match */
    if (sig1->return_type == TYPE_FUNCTION) {
        if (!function_signatures_equal(sig1->return_fn_sig, sig2->return_fn_sig)) {
            return false;
        }
    }
    
    return true;
}

/* Create a function value */
Value create_function(const char *function_name, FunctionSignature *signature) {
    Value val;
    val.type = VAL_FUNCTION;
    val.is_return = false;
    val.as.function_val.function_name = strdup(function_name);
    val.as.function_val.signature = signature;
    return val;
}

/* Create tuple value */
Value create_tuple(Value *elements, int element_count) {
    Value val;
    val.type = VAL_TUPLE;
    val.is_return = false;
    val.as.tuple_val = malloc(sizeof(TupleValue));
    val.as.tuple_val->element_count = element_count;
    
    /* Allocate and copy elements */
    if (element_count > 0) {
        val.as.tuple_val->elements = malloc(sizeof(Value) * element_count);
        for (int i = 0; i < element_count; i++) {
            val.as.tuple_val->elements[i] = elements[i];
            /* Deep copy strings */
            if (elements[i].type == VAL_STRING) {
                val.as.tuple_val->elements[i].as.string_val = strdup(elements[i].as.string_val);
            }
        }
    } else {
        val.as.tuple_val->elements = NULL;
    }
    
    return val;
}

/* Free tuple value */
void free_tuple(TupleValue *tuple) {
    if (!tuple) return;
    
    /* Free string elements */
    for (int i = 0; i < tuple->element_count; i++) {
        if (tuple->elements[i].type == VAL_STRING && tuple->elements[i].as.string_val) {
            free(tuple->elements[i].as.string_val);
        }
    }
    
    if (tuple->elements) {
        free(tuple->elements);
    }
    free(tuple);
}

/* Register a module namespace (for import aliases) */
void env_register_namespace(Environment *env, const char *alias, 
                            char **function_names, int function_count,
                            char **struct_names, int struct_count,
                            char **enum_names, int enum_count,
                            char **union_names, int union_count) {
    if (!env || !alias) {
        return;
    }
    
    /* Check if alias already exists */
    for (int i = 0; i < env->namespace_count; i++) {
        if (strcmp(env->namespaces[i].alias, alias) == 0) {
            /* Namespace already registered */
            return;
        }
    }
    
    /* Expand capacity if needed */
    if (env->namespace_count >= env->namespace_capacity) {
        env->namespace_capacity *= 2;
        env->namespaces = realloc(env->namespaces, sizeof(ModuleNamespace) * env->namespace_capacity);
    }
    
    /* Register the namespace */
    ModuleNamespace *ns = &env->namespaces[env->namespace_count++];
    ns->alias = strdup(alias);
    ns->function_names = function_names;
    ns->function_count = function_count;
    ns->struct_names = struct_names;
    ns->struct_count = struct_count;
    ns->enum_names = enum_names;
    ns->enum_count = enum_count;
    ns->union_names = union_names;
    ns->union_count = union_count;
}
