#include "nanolang.h"
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

/* Forward declarations for recursive serialization */
static void serialize_function_signature(char **buffer_ptr, size_t *pos_ptr, size_t *capacity_ptr, 
                                         FunctionSignature *sig, int sig_idx);
static void serialize_type_info(char **buffer_ptr, size_t *pos_ptr, size_t *capacity_ptr,
                                TypeInfo *type_info, int info_idx);

/* Count total TypeInfo structures needed (for pre-allocation) */
static int count_type_infos(ModuleMetadata *meta) {
    int count = 0;
    for (int i = 0; i < meta->function_count; i++) {
        Function *f = &meta->functions[i];
        if (f->return_type_info) count++;
        /* TODO: Count nested TypeInfo in parameters and recursive structures */
    }
    return count;
}

/* Helper macro for appending to dynamic buffer */
#define APPEND_TO_BUFFER(buf_ptr, pos_ptr, cap_ptr, str) do { \
    size_t len = strlen(str); \
    if (*(pos_ptr) + len >= *(cap_ptr)) { \
        *(cap_ptr) *= 2; \
        *(buf_ptr) = realloc(*(buf_ptr), *(cap_ptr)); \
    } \
    safe_strncpy(*(buf_ptr) + *(pos_ptr), str, *(cap_ptr) - *(pos_ptr)); \
    *(pos_ptr) += len; \
} while(0)

/* Count total FunctionSignatures needed (for pre-allocation) */
static int count_function_signatures(ModuleMetadata *meta) {
    int count = 0;
    for (int i = 0; i < meta->function_count; i++) {
        Function *f = &meta->functions[i];
        if (f->return_fn_sig) count++;  /* Return type function sig */
        
        /* Count parameter function signatures */
        for (int j = 0; j < f->param_count; j++) {
            if (f->params[j].fn_sig) count++;
        }
    }
    return count;
}

/* Serialize a FunctionSignature to C initialization code */
static void serialize_function_signature(char **buffer_ptr, size_t *pos_ptr, size_t *capacity_ptr,
                                         FunctionSignature *sig, int sig_idx) {
    if (!sig) return;
    
    char temp[2048];
    
    /* Initialize param_types array */
    if (sig->param_count > 0 && sig->param_types) {
        snprintf(temp, sizeof(temp), "    static Type _fn_sig_%d_param_types[%d] = {", 
                 sig_idx, sig->param_count);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        
        for (int i = 0; i < sig->param_count; i++) {
            if (i > 0) APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, ", ");
            snprintf(temp, sizeof(temp), "%d", sig->param_types[i]);
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        }
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, "};\n");
        
        /* Initialize param_struct_names array if present */
        if (sig->param_struct_names) {
            snprintf(temp, sizeof(temp), "    static char* _fn_sig_%d_param_names[%d] = {",
                     sig_idx, sig->param_count);
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
            
            for (int i = 0; i < sig->param_count; i++) {
                if (i > 0) APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, ", ");
                if (sig->param_struct_names[i]) {
                    snprintf(temp, sizeof(temp), "\"%s\"", sig->param_struct_names[i]);
                    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
                } else {
                    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, "NULL");
                }
            }
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, "};\n");
        }
    }
    
    /* Initialize the FunctionSignature struct */
    snprintf(temp, sizeof(temp), "    _fn_signatures[%d].param_count = %d;\n", 
             sig_idx, sig->param_count);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    
    if (sig->param_count > 0) {
        snprintf(temp, sizeof(temp), "    _fn_signatures[%d].param_types = _fn_sig_%d_param_types;\n",
                 sig_idx, sig_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        
        if (sig->param_struct_names) {
            snprintf(temp, sizeof(temp), "    _fn_signatures[%d].param_struct_names = _fn_sig_%d_param_names;\n",
                     sig_idx, sig_idx);
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        } else {
            snprintf(temp, sizeof(temp), "    _fn_signatures[%d].param_struct_names = NULL;\n", sig_idx);
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        }
    } else {
        snprintf(temp, sizeof(temp), "    _fn_signatures[%d].param_types = NULL;\n", sig_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        snprintf(temp, sizeof(temp), "    _fn_signatures[%d].param_struct_names = NULL;\n", sig_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }
    
    snprintf(temp, sizeof(temp), "    _fn_signatures[%d].return_type = %d;\n",
             sig_idx, sig->return_type);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    
    if (sig->return_struct_name) {
        snprintf(temp, sizeof(temp), "    _fn_signatures[%d].return_struct_name = \"%s\";\n",
                 sig_idx, sig->return_struct_name);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    } else {
        snprintf(temp, sizeof(temp), "    _fn_signatures[%d].return_struct_name = NULL;\n", sig_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }
    
    /* TODO: Handle recursive return_fn_sig */
    snprintf(temp, sizeof(temp), "    _fn_signatures[%d].return_fn_sig = NULL;\n", sig_idx);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
}

/* Serialize a TypeInfo structure to C initialization code */
static void serialize_type_info(char **buffer_ptr, size_t *pos_ptr, size_t *capacity_ptr,
                                TypeInfo *type_info, int info_idx) {
    if (!type_info) return;
    
    char temp[2048];
    
    /* Base type */
    snprintf(temp, sizeof(temp), "    _type_infos[%d].base_type = %d;\n", info_idx, type_info->base_type);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    
    /* TODO: Handle recursive element_type */
    snprintf(temp, sizeof(temp), "    _type_infos[%d].element_type = NULL;\n", info_idx);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    
    /* Generic name */
    if (type_info->generic_name) {
        snprintf(temp, sizeof(temp), "    _type_infos[%d].generic_name = \"%s\";\n", 
                 info_idx, type_info->generic_name);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    } else {
        snprintf(temp, sizeof(temp), "    _type_infos[%d].generic_name = NULL;\n", info_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }
    
    /* TODO: Handle type_params array */
    snprintf(temp, sizeof(temp), "    _type_infos[%d].type_params = NULL;\n", info_idx);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    snprintf(temp, sizeof(temp), "    _type_infos[%d].type_param_count = 0;\n", info_idx);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    
    /* Tuple types */
    if (type_info->tuple_element_count > 0 && type_info->tuple_types) {
        snprintf(temp, sizeof(temp), "    static Type _type_info_%d_tuple_types[%d] = {",
                 info_idx, type_info->tuple_element_count);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        
        for (int i = 0; i < type_info->tuple_element_count; i++) {
            if (i > 0) APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, ", ");
            snprintf(temp, sizeof(temp), "%d", type_info->tuple_types[i]);
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        }
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, "};\n");
        
        snprintf(temp, sizeof(temp), "    _type_infos[%d].tuple_types = _type_info_%d_tuple_types;\n",
                 info_idx, info_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        
        /* Tuple type names */
        if (type_info->tuple_type_names) {
            snprintf(temp, sizeof(temp), "    static char* _type_info_%d_tuple_names[%d] = {",
                     info_idx, type_info->tuple_element_count);
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
            
            for (int i = 0; i < type_info->tuple_element_count; i++) {
                if (i > 0) APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, ", ");
                if (type_info->tuple_type_names[i]) {
                    snprintf(temp, sizeof(temp), "\"%s\"", type_info->tuple_type_names[i]);
                    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
                } else {
                    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, "NULL");
                }
            }
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, "};\n");
            
            snprintf(temp, sizeof(temp), "    _type_infos[%d].tuple_type_names = _type_info_%d_tuple_names;\n",
                     info_idx, info_idx);
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        } else {
            snprintf(temp, sizeof(temp), "    _type_infos[%d].tuple_type_names = NULL;\n", info_idx);
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        }
        
        snprintf(temp, sizeof(temp), "    _type_infos[%d].tuple_element_count = %d;\n",
                 info_idx, type_info->tuple_element_count);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    } else {
        snprintf(temp, sizeof(temp), "    _type_infos[%d].tuple_types = NULL;\n", info_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        snprintf(temp, sizeof(temp), "    _type_infos[%d].tuple_type_names = NULL;\n", info_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        snprintf(temp, sizeof(temp), "    _type_infos[%d].tuple_element_count = 0;\n", info_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }
    
    /* Opaque type name */
    if (type_info->opaque_type_name) {
        snprintf(temp, sizeof(temp), "    _type_infos[%d].opaque_type_name = \"%s\";\n",
                 info_idx, type_info->opaque_type_name);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    } else {
        snprintf(temp, sizeof(temp), "    _type_infos[%d].opaque_type_name = NULL;\n", info_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }
    
    /* TODO: Handle fn_sig */
    snprintf(temp, sizeof(temp), "    _type_infos[%d].fn_sig = NULL;\n", info_idx);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
}

/* Serialize module metadata to C code that can be embedded */
char *serialize_module_metadata_to_c(ModuleMetadata *meta) {
    if (!meta) return NULL;
    
    /* Estimate buffer size - start with 4KB */
    size_t capacity = 4096;
    char *buffer = malloc(capacity);
    size_t pos = 0;
    
    /* Helper to append string */
    #define APPEND(str) do { \
        size_t len = strlen(str); \
        if (pos + len >= capacity) { \
            capacity *= 2; \
            buffer = realloc(buffer, capacity); \
        } \
        safe_strncpy(buffer + pos, str, capacity - pos); \
        pos += len; \
    } while(0)
    
    
    APPEND("/* Module metadata - automatically generated */\n");
    char temp[2048];
    snprintf(temp, sizeof(temp), "/* Module: %s */\n\n", meta->module_name);
    APPEND(temp);
    APPEND("#include \"nanolang.h\"\n\n");
    
    /* Count and declare FunctionSignature arrays */
    int fn_sig_count = count_function_signatures(meta);
    if (fn_sig_count > 0) {
        snprintf(temp, sizeof(temp), "static FunctionSignature _fn_signatures[%d];\n", fn_sig_count);
        APPEND(temp);
    }
    
    /* Count and declare TypeInfo arrays */
    int type_info_count = count_type_infos(meta);
    if (type_info_count > 0) {
        snprintf(temp, sizeof(temp), "static TypeInfo _type_infos[%d];\n", type_info_count);
        APPEND(temp);
    }
    
    /* Serialize functions */
    snprintf(temp, sizeof(temp), "static Function _module_functions[%d];\n", meta->function_count);
    APPEND(temp);
    APPEND("static Parameter _module_params[");
    int total_params = 0;
    for (int i = 0; i < meta->function_count; i++) {
        total_params += meta->functions[i].param_count;
    }
    snprintf(temp, sizeof(temp), "%d];\n\n", total_params);
    APPEND(temp);
    
    /* Initialize functions */
    APPEND("static void _init_module_metadata(void) __attribute__((constructor));\n");
    APPEND("static void _init_module_metadata(void) {\n");
    APPEND("    int param_idx = 0;\n");
    
    /* Serialize FunctionSignatures first */
    if (fn_sig_count > 0) {
        APPEND("\n    /* Initialize FunctionSignatures */\n");
        int sig_idx = 0;
        
        /* Serialize return type function signatures */
        for (int i = 0; i < meta->function_count; i++) {
            Function *f = &meta->functions[i];
            if (f->return_fn_sig) {
                serialize_function_signature(&buffer, &pos, &capacity, f->return_fn_sig, sig_idx);
                sig_idx++;
            }
        }
        
        /* Serialize parameter function signatures */
        for (int i = 0; i < meta->function_count; i++) {
            Function *f = &meta->functions[i];
            for (int j = 0; j < f->param_count; j++) {
                if (f->params[j].fn_sig) {
                    serialize_function_signature(&buffer, &pos, &capacity, f->params[j].fn_sig, sig_idx);
                    sig_idx++;
                }
            }
        }
        
        APPEND("\n");
    }
    
    /* Serialize TypeInfos */
    if (type_info_count > 0) {
        APPEND("    /* Initialize TypeInfos */\n");
        int info_idx = 0;
        for (int i = 0; i < meta->function_count; i++) {
            Function *f = &meta->functions[i];
            if (f->return_type_info) {
                serialize_type_info(&buffer, &pos, &capacity, f->return_type_info, info_idx);
                info_idx++;
            }
        }
        APPEND("\n");
    }
    
    int param_idx = 0;
    int sig_idx = 0;  /* Track which FunctionSignature index to reference */
    
    /* Calculate starting index for parameter signatures (after return signatures) */
    int param_sig_start_idx = 0;
    for (int i = 0; i < meta->function_count; i++) {
        if (meta->functions[i].return_fn_sig) param_sig_start_idx++;
    }
    int param_sig_idx = param_sig_start_idx;  /* Track parameter fn_sig indices */
    
    for (int i = 0; i < meta->function_count; i++) {
        Function *f = &meta->functions[i];
        char temp[2048];
        snprintf(temp, sizeof(temp), "    /* Function: %s */\n", f->name);
        APPEND(temp);
        snprintf(temp, sizeof(temp), "    _module_functions[%d].name = \"%s\";\n", i, f->name);
        APPEND(temp);
        snprintf(temp, sizeof(temp), "    _module_functions[%d].param_count = %d;\n", i, f->param_count);
        APPEND(temp);
        snprintf(temp, sizeof(temp), "    _module_functions[%d].return_type = %d;\n", i, f->return_type);
        APPEND(temp);
        if (f->return_struct_type_name) {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].return_struct_type_name = \"%s\";\n", i, f->return_struct_type_name);
            APPEND(temp);
        } else {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].return_struct_type_name = NULL;\n", i);
            APPEND(temp);
        }
        /* Link to FunctionSignature if present */
        if (f->return_fn_sig) {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].return_fn_sig = &_fn_signatures[%d];\n", i, sig_idx);
            APPEND(temp);
            sig_idx++;
        } else {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].return_fn_sig = NULL;\n", i);
            APPEND(temp);
        }
        /* Link to TypeInfo if present */
        if (f->return_type_info) {
            /* Find the TypeInfo index for this function's return type */
            int type_info_idx = 0;
            for (int k = 0; k < i; k++) {
                if (meta->functions[k].return_type_info) type_info_idx++;
            }
            snprintf(temp, sizeof(temp), "    _module_functions[%d].return_type_info = &_type_infos[%d];\n", i, type_info_idx);
            APPEND(temp);
        } else {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].return_type_info = NULL;\n", i);
            APPEND(temp);
        }
        snprintf(temp, sizeof(temp), "    _module_functions[%d].body = NULL;\n", i);
        APPEND(temp);
        snprintf(temp, sizeof(temp), "    _module_functions[%d].shadow_test = NULL;\n", i);
        APPEND(temp);
        snprintf(temp, sizeof(temp), "    _module_functions[%d].is_extern = %s;\n", i, f->is_extern ? "true" : "false");
        APPEND(temp);
        if (f->param_count > 0) {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].params = &_module_params[%d];\n", i, param_idx);
            APPEND(temp);
            for (int j = 0; j < f->param_count; j++) {
                Parameter *p = &f->params[j];
                snprintf(temp, sizeof(temp), "    _module_params[%d].name = \"%s\";\n", param_idx, p->name ? p->name : "");
                APPEND(temp);
                snprintf(temp, sizeof(temp), "    _module_params[%d].type = %d;\n", param_idx, p->type);
                APPEND(temp);
                if (p->struct_type_name) {
                    snprintf(temp, sizeof(temp), "    _module_params[%d].struct_type_name = \"%s\";\n", param_idx, p->struct_type_name);
                    APPEND(temp);
                } else {
                    snprintf(temp, sizeof(temp), "    _module_params[%d].struct_type_name = NULL;\n", param_idx);
                    APPEND(temp);
                }
                snprintf(temp, sizeof(temp), "    _module_params[%d].element_type = %d;\n", param_idx, p->element_type);
                APPEND(temp);
                
                /* Link to parameter's function signature if present */
                if (p->fn_sig) {
                    snprintf(temp, sizeof(temp), "    _module_params[%d].fn_sig = &_fn_signatures[%d];\n", param_idx, param_sig_idx);
                    APPEND(temp);
                    param_sig_idx++;
                } else {
                    snprintf(temp, sizeof(temp), "    _module_params[%d].fn_sig = NULL;\n", param_idx);
                    APPEND(temp);
                }
                
                param_idx++;
            }
        } else {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].params = NULL;\n", i);
            APPEND(temp);
        }
        APPEND("\n");
    }
    
    APPEND("}\n\n");
    
    /* Export metadata accessor */
    snprintf(temp, sizeof(temp), "ModuleMetadata _module_metadata = {\n");
    APPEND(temp);
    snprintf(temp, sizeof(temp), "    .module_name = \"%s\",\n", meta->module_name);
    APPEND(temp);
    snprintf(temp, sizeof(temp), "    .function_count = %d,\n", meta->function_count);
    APPEND(temp);
    APPEND("    .functions = _module_functions,\n");
    snprintf(temp, sizeof(temp), "    .struct_count = %d,\n", meta->struct_count);
    APPEND(temp);
    APPEND("    .structs = NULL,  /* TODO: serialize structs */\n");
    snprintf(temp, sizeof(temp), "    .enum_count = %d,\n", meta->enum_count);
    APPEND(temp);
    APPEND("    .enums = NULL,  /* TODO: serialize enums */\n");
    snprintf(temp, sizeof(temp), "    .union_count = %d,\n", meta->union_count);
    APPEND(temp);
    APPEND("    .unions = NULL  /* TODO: serialize unions */\n");
    APPEND("};\n\n");
    
    /* Serialize constants */
    if (meta->constant_count > 0) {
        snprintf(temp, sizeof(temp), "/* Constants from module */\n");
        APPEND(temp);
        for (int i = 0; i < meta->constant_count; i++) {
            ConstantDef *c = &meta->constants[i];
            if (c->type == TYPE_INT) {
                snprintf(temp, sizeof(temp), "static const int64_t %s = %lldLL;\n", 
                         c->name, (long long)c->value);
                APPEND(temp);
            } else if (c->type == TYPE_FLOAT) {
                /* Reconstruct float from int64 bit pattern */
                union { double d; int64_t i; } u;
                u.i = c->value;
                snprintf(temp, sizeof(temp), "static const double %s = %g;\n", 
                         c->name, u.d);
                APPEND(temp);
            }
        }
        APPEND("\n");
    }
    
    #undef APPEND
    
    buffer[pos] = '\0';
    return buffer;
}

/* Embed metadata in module C code */
bool embed_metadata_in_module_c(char *c_code, ModuleMetadata *meta, size_t buffer_size) {
    if (!c_code || !meta) return false;
    
    char *metadata_c = serialize_module_metadata_to_c(meta);
    if (!metadata_c) return false;
    
    /* Find insertion point - before the last closing brace or at end */
    assert(c_code != NULL);
    assert(metadata_c != NULL);
    size_t code_len = safe_strlen(c_code);
    size_t meta_len = safe_strlen(metadata_c);
    
    if (code_len + meta_len + 100 >= buffer_size) {
        free(metadata_c);
        return false;  /* Not enough space */
    }
    
    /* Insert metadata before any main() function or at the end */
    char *insert_pos = strstr(c_code, "int main()");
    if (!insert_pos) {
        insert_pos = c_code + code_len;
    }
    
    /* Make room and insert */
    memmove(insert_pos + meta_len, insert_pos, code_len - (insert_pos - c_code) + 1);
    memcpy(insert_pos, metadata_c, meta_len);
    
    free(metadata_c);
    return true;
}

/* Deserialize metadata from C code (simplified - would need full C parser) */
bool deserialize_module_metadata_from_c(const char *c_code, ModuleMetadata **meta_out) {
    (void)c_code;  /* Unused parameter - stub function */
    /* TODO: Implement C code parsing to extract metadata */
    /* For now, this is a placeholder */
    *meta_out = NULL;
    return false;
}

