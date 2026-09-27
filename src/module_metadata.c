#include "nanolang.h"
#include "module_symbol.h"
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>

/* I index the graph before emission so aliases and recursive edges retain
 * identity without recursive traversal or dependency-order assumptions. */
typedef struct {
    const void **items;
    int count;
    int capacity;
} MetadataPointers;

typedef struct {
    MetadataPointers types;
    MetadataPointers signatures;
} MetadataGraph;

static int metadata_index(const MetadataPointers *table, const void *pointer) {
    for (int i = 0; pointer && i < table->count; i++)
        if (table->items[i] == pointer) return i;
    return -1;
}

static bool metadata_add(MetadataPointers *table, const void *pointer) {
    if (!pointer || metadata_index(table, pointer) >= 0) return true;
    if (table->count == table->capacity) {
        if (table->capacity > INT_MAX / 2) return false;
        int capacity = table->capacity ? table->capacity * 2 : 16;
        if ((size_t)capacity > SIZE_MAX / sizeof(*table->items)) return false;
        const void **items = realloc(table->items, (size_t)capacity * sizeof(*items));
        if (!items) return false;
        table->items = items;
        table->capacity = capacity;
    }
    table->items[table->count++] = pointer;
    return true;
}

static void metadata_graph_free(MetadataGraph *graph) {
    free(graph->types.items);
    free(graph->signatures.items);
}

static bool metadata_graph_collect(MetadataGraph *graph, const ModuleMetadata *meta) {
    if (meta->function_count < 0 || (meta->function_count && !meta->functions)) return false;
    for (int i = 0; i < meta->function_count; i++) {
        const Function *f = &meta->functions[i];
        if (f->param_count < 0 || (f->param_count && !f->params)) return false;
        if (!metadata_add(&graph->types, f->return_type_info) ||
            !metadata_add(&graph->signatures, f->return_fn_sig)) return false;
        for (int j = 0; j < f->param_count; j++) {
            if (!metadata_add(&graph->types, f->params[j].type_info) ||
                !metadata_add(&graph->signatures, f->params[j].fn_sig)) return false;
        }
    }
    int ti = 0, si = 0;
    while (ti < graph->types.count || si < graph->signatures.count) {
        while (ti < graph->types.count) {
            const TypeInfo *t = graph->types.items[ti++];
            if (t->type_param_count < 0 || (t->type_param_count && !t->type_params) ||
                t->tuple_element_count < 0 || (t->tuple_element_count && !t->tuple_types) ||
                t->row_field_count < 0 || (t->row_field_count && !t->row_field_types) ||
                t->type_var_count < 0 || (t->type_var_count && !t->type_var_names)) return false;
            if (t->base_type == TYPE_TUPLE && !type_info_tuple_valid(t)) return false;
            if (!metadata_add(&graph->types, t->element_type) ||
                !metadata_add(&graph->signatures, t->fn_sig)) return false;
            for (int j = 0; j < t->type_param_count; j++)
                if (!metadata_add(&graph->types, t->type_params[j])) return false;
        }
        while (si < graph->signatures.count) {
            const FunctionSignature *sig = graph->signatures.items[si++];
            if (sig->param_count < 0 || (sig->param_count && !sig->param_types)) return false;
            if (!metadata_add(&graph->types, sig->return_type_info) ||
                !metadata_add(&graph->signatures, sig->return_fn_sig)) return false;
            for (int j = 0; sig->param_type_info && j < sig->param_count; j++)
                if (!metadata_add(&graph->types, sig->param_type_info[j])) return false;
        }
    }
    return true;
}

/* Helper macro for appending to dynamic buffer */
#define APPEND_TO_BUFFER(buf_ptr, pos_ptr, cap_ptr, str) do { \
    size_t len = strlen(str); \
    if (*(pos_ptr) + len + 1 >= *(cap_ptr)) { \
        while (*(pos_ptr) + len + 1 >= *(cap_ptr)) *(cap_ptr) *= 2; \
        *(buf_ptr) = realloc(*(buf_ptr), *(cap_ptr)); \
    } \
    memcpy(*(buf_ptr) + *(pos_ptr), str, len); \
    (*(buf_ptr))[*(pos_ptr) + len] = '\0'; \
    *(pos_ptr) += len; \
} while(0)

/* I append string literals separately so long annotations cannot be truncated
 * by the small formatting buffer, and source escapes retain their bytes. */
static void serialize_string_field(char **buffer_ptr, size_t *pos_ptr, size_t *capacity_ptr,
                                   const char *object, int index, const char *field,
                                   const char *value) {
    char temp[128];
    snprintf(temp, sizeof(temp), "    %s[%d].%s = ", object, index, field);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, value ? module_c_literal(value) : "NULL");
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, ";\n");
}

static void serialize_type_array(char **buffer_ptr, size_t *pos_ptr, size_t *capacity_ptr,
                                 int index, const char *field, int count,
                                 const Type *types, char *const *names) {
    if (!count || (!types && !names)) return;
    char temp[192];
    snprintf(temp, sizeof(temp), "    static %s _type_info_%d_%s[%d] = {", types ? "Type" : "char*", index, field, count);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    for (int i = 0; i < count; i++) {
        if (i) APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, ", ");
        if (types) {
            snprintf(temp, sizeof(temp), "%d", types[i]);
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        } else {
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, names[i] ? module_c_literal(names[i]) : "NULL");
        }
    }
    snprintf(temp, sizeof(temp), "};\n    _type_infos[%d].%s = _type_info_%d_%s;\n", index, field, index, field);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
}

/* Serialize a FunctionSignature to C initialization code */
static void serialize_function_signature(char **buffer_ptr, size_t *pos_ptr, size_t *capacity_ptr,
                                         const MetadataGraph *graph, const FunctionSignature *sig, int sig_idx) {
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
                    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, module_c_literal(sig->param_struct_names[i]));
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
    
    serialize_string_field(buffer_ptr, pos_ptr, capacity_ptr, "_fn_signatures", sig_idx,
                           "return_struct_name", sig->return_struct_name);

    if (sig->param_count && sig->param_type_info) {
        snprintf(temp, sizeof(temp), "    static TypeInfo* _fn_sig_%d_param_info[%d];\n", sig_idx, sig->param_count);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        for (int i = 0; i < sig->param_count; i++) {
            if (!sig->param_type_info[i]) continue;
            snprintf(temp, sizeof(temp), "    _fn_sig_%d_param_info[%d] = &_type_infos[%d];\n",
                     sig_idx, i, metadata_index(&graph->types, sig->param_type_info[i]));
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        }
        snprintf(temp, sizeof(temp), "    _fn_signatures[%d].param_type_info = _fn_sig_%d_param_info;\n", sig_idx, sig_idx);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }
    if (sig->return_type_info) {
        snprintf(temp, sizeof(temp), "    _fn_signatures[%d].return_type_info = &_type_infos[%d];\n",
                 sig_idx, metadata_index(&graph->types, sig->return_type_info));
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }
    if (sig->return_fn_sig) {
        snprintf(temp, sizeof(temp), "    _fn_signatures[%d].return_fn_sig = &_fn_signatures[%d];\n",
                 sig_idx, metadata_index(&graph->signatures, sig->return_fn_sig));
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }

}

/* Serialize a TypeInfo structure to C initialization code */
static void serialize_type_info(char **buffer_ptr, size_t *pos_ptr, size_t *capacity_ptr,
                                const MetadataGraph *graph, const TypeInfo *type_info, int info_idx) {
    if (!type_info) return;
    
    char temp[2048];
    
    /* Base type */
    snprintf(temp, sizeof(temp), "    _type_infos[%d].base_type = %d;\n", info_idx, type_info->base_type);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    
    if (type_info->element_type) {
        snprintf(temp, sizeof(temp), "    _type_infos[%d].element_type = &_type_infos[%d];\n",
                 info_idx, metadata_index(&graph->types, type_info->element_type));
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }

    serialize_string_field(buffer_ptr, pos_ptr, capacity_ptr, "_type_infos", info_idx,
                           "generic_name", type_info->generic_name);

    if (type_info->type_param_count) {
        snprintf(temp, sizeof(temp), "    static TypeInfo* _type_info_%d_params[%d];\n", info_idx, type_info->type_param_count);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        for (int i = 0; i < type_info->type_param_count; i++) {
            if (!type_info->type_params[i]) continue;
            snprintf(temp, sizeof(temp), "    _type_info_%d_params[%d] = &_type_infos[%d];\n",
                     info_idx, i, metadata_index(&graph->types, type_info->type_params[i]));
            APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
        }
        snprintf(temp, sizeof(temp), "    _type_infos[%d].type_params = _type_info_%d_params;\n    _type_infos[%d].type_param_count = %d;\n",
                 info_idx, info_idx, info_idx, type_info->type_param_count);
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }

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
                    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, module_c_literal(type_info->tuple_type_names[i]));
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
    
    serialize_string_field(buffer_ptr, pos_ptr, capacity_ptr, "_type_infos", info_idx,
                           "opaque_type_name", type_info->opaque_type_name);
    serialize_string_field(buffer_ptr, pos_ptr, capacity_ptr, "_type_infos", info_idx,
                           "row_var_name", type_info->row_var_name);
    serialize_type_array(buffer_ptr, pos_ptr, capacity_ptr, info_idx, "row_field_names",
                         type_info->row_field_count, NULL, type_info->row_field_names);
    serialize_type_array(buffer_ptr, pos_ptr, capacity_ptr, info_idx, "row_field_types",
                         type_info->row_field_count, type_info->row_field_types, NULL);
    serialize_type_array(buffer_ptr, pos_ptr, capacity_ptr, info_idx, "row_field_type_names",
                         type_info->row_field_count, NULL, type_info->row_field_type_names);
    serialize_type_array(buffer_ptr, pos_ptr, capacity_ptr, info_idx, "type_var_names",
                         type_info->type_var_count, NULL, type_info->type_var_names);
    snprintf(temp, sizeof(temp), "    _type_infos[%d].row_field_count = %d;\n    _type_infos[%d].is_open_row = %s;\n    _type_infos[%d].type_var_count = %d;\n",
             info_idx, type_info->row_field_count, info_idx, type_info->is_open_row ? "true" : "false", info_idx, type_info->type_var_count);
    APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    if (type_info->fn_sig) {
        snprintf(temp, sizeof(temp), "    _type_infos[%d].fn_sig = &_fn_signatures[%d];\n",
                 info_idx, metadata_index(&graph->signatures, type_info->fn_sig));
        APPEND_TO_BUFFER(buffer_ptr, pos_ptr, capacity_ptr, temp);
    }

}

/* Serialize module metadata to C code that can be embedded */
char *serialize_module_metadata_to_c(ModuleMetadata *meta) {
    if (!meta) return NULL;
    MetadataGraph graph = {0};
    if (!metadata_graph_collect(&graph, meta)) {
        metadata_graph_free(&graph);
        return NULL;
    }
    
    /* Estimate buffer size - start with 4KB */
    size_t capacity = 4096;
    char *buffer = malloc(capacity);
    if (!buffer) { metadata_graph_free(&graph); return NULL; }
    size_t pos = 0;
    
    /* Helper to append string */
    #define APPEND(str) do { \
        size_t len = strlen(str); \
        if (pos + len + 1 >= capacity) { \
            while (pos + len + 1 >= capacity) capacity *= 2; \
            buffer = realloc(buffer, capacity); \
        } \
        memcpy(buffer + pos, str, len); \
        buffer[pos + len] = '\0'; \
        pos += len; \
    } while(0)
    
    
    APPEND("/* Module metadata - automatically generated */\n");
    char temp[2048];
    APPEND("#include \"nanolang.h\"\n\n");

    /* Derive a C-safe, per-module identifier suffix so the exported metadata
     * symbol does not collide when several module objects are linked into a
     * single binary (previously every module exported `_module_metadata`,
     * causing multiple-definition link errors). */
    const char *module_ident = module_symbol_suffix(meta->module_name ? meta->module_name : "unknown");
    
    /* Count and declare FunctionSignature arrays */
    int fn_sig_count = graph.signatures.count;
    if (fn_sig_count > 0) {
        snprintf(temp, sizeof(temp), "static FunctionSignature _fn_signatures[%d];\n", fn_sig_count);
        APPEND(temp);
    }
    
    /* Count and declare TypeInfo arrays */
    int type_info_count = graph.types.count;
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
    
    for (int i = 0; i < graph.signatures.count; i++)
        serialize_function_signature(&buffer, &pos, &capacity, &graph, graph.signatures.items[i], i);
    for (int i = 0; i < graph.types.count; i++)
        serialize_type_info(&buffer, &pos, &capacity, &graph, graph.types.items[i], i);

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
        } else {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].return_struct_type_name = NULL;\n", i);
            APPEND(temp);
        }
        /* Link to FunctionSignature if present */
        if (f->return_fn_sig) {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].return_fn_sig = &_fn_signatures[%d];\n", i, metadata_index(&graph.signatures, f->return_fn_sig));
            APPEND(temp);
        } else {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].return_fn_sig = NULL;\n", i);
            APPEND(temp);
        }
        /* Link to TypeInfo if present */
        if (f->return_type_info) {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].return_type_info = &_type_infos[%d];\n", i, metadata_index(&graph.types, f->return_type_info));
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

        /* Memory semantics annotations */
        snprintf(temp, sizeof(temp), "    _module_functions[%d].returns_gc_managed = %s;\n", i, f->returns_gc_managed ? "true" : "false");
        APPEND(temp);
        snprintf(temp, sizeof(temp), "    _module_functions[%d].requires_manual_free = %s;\n", i, f->requires_manual_free ? "true" : "false");
        APPEND(temp);
        snprintf(temp, sizeof(temp), "    _module_functions[%d].returns_borrowed = %s;\n", i, f->returns_borrowed ? "true" : "false");
        APPEND(temp);
        if (f->cleanup_function) {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].cleanup_function = \"%s\";\n", i, f->cleanup_function);
            APPEND(temp);
        } else {
            snprintf(temp, sizeof(temp), "    _module_functions[%d].cleanup_function = NULL;\n", i);
            APPEND(temp);
        }

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
                    snprintf(temp, sizeof(temp), "    _module_params[%d].fn_sig = &_fn_signatures[%d];\n", param_idx, metadata_index(&graph.signatures, p->fn_sig));
                    APPEND(temp);
                } else {
                    snprintf(temp, sizeof(temp), "    _module_params[%d].fn_sig = NULL;\n", param_idx);
                    APPEND(temp);
                }
                
                if (p->type_info) {
                    snprintf(temp, sizeof(temp), "    _module_params[%d].type_info = &_type_infos[%d];\n", param_idx, metadata_index(&graph.types, p->type_info));
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
    
    /* Export metadata accessor (per-module symbol name to avoid link clashes) */
    APPEND("ModuleMetadata _module_metadata_");
    APPEND(module_ident);
    APPEND(" = {\n");
    APPEND("    .module_name = ");
    APPEND(module_c_literal(meta->module_name));
    APPEND(",\n");
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
    
    /* Serialize constants.
     *
     * These are file-local records of the module's exported constant values.
     * They are prefixed with the per-module identifier so they never collide
     * with the module's own top-level `static const` definitions in the same
     * translation unit (an un-prefixed name such as `MAX` produced a
     * `redefinition` compile error and broke module object generation). */
    if (meta->constant_count > 0) {
        snprintf(temp, sizeof(temp), "/* Constants from module */\n");
        APPEND(temp);
        for (int i = 0; i < meta->constant_count; i++) {
            ConstantDef *c = &meta->constants[i];
            const char *cname = c->name ? c->name : "unnamed";
            if (c->type == TYPE_INT) {
                APPEND("static const int64_t _module_const_");
                APPEND(module_ident);
                APPEND("_");
                APPEND(cname);
                snprintf(temp, sizeof(temp), " = %lldLL;\n", (long long)c->value);
                APPEND(temp);
            } else if (c->type == TYPE_FLOAT) {
                /* Reconstruct float from int64 bit pattern */
                union { double d; int64_t i; } u;
                u.i = c->value;
                APPEND("static const double _module_const_");
                APPEND(module_ident);
                APPEND("_");
                APPEND(cname);
                snprintf(temp, sizeof(temp), " = %g;\n", u.d);
                APPEND(temp);
            }
        }
        APPEND("\n");
    }
    
    #undef APPEND
    
    buffer[pos] = '\0';
    metadata_graph_free(&graph);
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
