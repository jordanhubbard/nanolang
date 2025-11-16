#include "nanolang.h"
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

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
    
    int param_idx = 0;
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
        }
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
                }
                snprintf(temp, sizeof(temp), "    _module_params[%d].element_type = %d;\n", param_idx, p->element_type);
                APPEND(temp);
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
    APPEND("};\n");
    
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
    /* TODO: Implement C code parsing to extract metadata */
    /* For now, this is a placeholder */
    *meta_out = NULL;
    return false;
}

