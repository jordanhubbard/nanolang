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
    
    /* String operations */
    {"str_length", 1, {TYPE_STRING, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},
    {"str_concat", 2, {TYPE_STRING, TYPE_STRING, TYPE_UNKNOWN}, TYPE_STRING},
    {"str_substring", 3, {TYPE_STRING, TYPE_INT, TYPE_INT}, TYPE_STRING},
    {"str_contains", 2, {TYPE_STRING, TYPE_STRING, TYPE_UNKNOWN}, TYPE_BOOL},
    {"str_equals", 2, {TYPE_STRING, TYPE_STRING, TYPE_UNKNOWN}, TYPE_BOOL},
    
    /* Array operations */
    {"array_length", 1, {TYPE_ARRAY, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_INT},
    {"array_new", 1, {TYPE_INT, TYPE_UNKNOWN, TYPE_UNKNOWN}, TYPE_ARRAY},
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
    env->enums = malloc(sizeof(EnumDef) * 8);
    env->enum_count = 0;
    env->enum_capacity = 8;
    env->unions = malloc(sizeof(UnionDef) * 8);
    env->union_count = 0;
    env->union_capacity = 8;
    env->generic_instances = malloc(sizeof(GenericInstantiation) * 8);
    env->generic_instance_count = 0;
    env->generic_instance_capacity = 8;
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
    for (int i = 0; i < env->struct_count; i++) {
        if (strcmp(env->structs[i].name, name) == 0) {
            return &env->structs[i];
        }
    }
    return NULL;
}

/* Define enum */
void env_define_enum(Environment *env, EnumDef enum_def) {
    /* Check if enum already exists - prevent duplicates */
    if (env_get_enum(env, enum_def.name) != NULL) {
        /* Enum already defined - skip duplicate registration */
        return;
    }
    
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
    for (int i = 0; i < env->union_count; i++) {
        if (strcmp(env->unions[i].name, name) == 0) {
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
        if (strcmp(udef->variant_names[i], variant_name) == 0) {
            return i;
        }
    }
    return -1;
}
/* Register a list instantiation for code generation */
void env_register_list_instantiation(Environment *env, const char *element_type) {
    /* Check if already registered */
    for (int i = 0; i < env->generic_instance_count; i++) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (strcmp(inst->generic_name, "List") == 0 &&
            inst->type_arg_names && 
            strcmp(inst->type_arg_names[0], element_type) == 0) {
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
    char func_name[256];
    Function func;
    Parameter *params;
    
    /* List_T_new() -> List<T>* */
    snprintf(func_name, sizeof(func_name), "%s_new", specialized);
    func.name = strdup(func_name);
    func.param_count = 0;
    func.params = NULL;
    func.return_type = TYPE_LIST_GENERIC;
    func.return_struct_type_name = NULL;
    func.body = NULL;  /* Built-in */
    func.shadow_test = NULL;
    func.is_extern = true;
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
    func.body = NULL;
    func.shadow_test = NULL;
    func.is_extern = true;
    env_define_function(env, func);
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
            if (name1 && name2 && strcmp(name1, name2) != 0) return false;
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
