/* I poison env.c malloc results to check generated annotation initialization. */
#include "../src/nanolang.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int g_argc;
char **g_argv;

void *nano_metadata_poison_malloc(size_t size) {
    void *memory = malloc(size);
    if (memory) memset(memory, 0xa5, size);
    return memory;
}

int main(void) {
    Environment *env = create_environment();
    assert(env);
    assert(!env_register_list_instantiation(env, "MetadataItem"));
    StructDef item = {0}; item.name = strdup("MetadataItem"); assert(item.name);
    env_define_struct(env, item);
    assert(env_register_list_instantiation(env, "MetadataItem"));
    assert(env->generic_instance_count == 1);
    assert(env->generic_instances[0].list_element.kind == TYPE_STRUCT);
    assert(env->generic_instances[0].list_element.ordinal == 1);
    const char *names[] = {"List_MetadataItem_new", "List_MetadataItem_push",
                          "List_MetadataItem_get", "List_MetadataItem_length"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        Function *function = env_get_function(env, names[i]);
        assert(function);
        assert(function->return_fn_sig == NULL);
        assert(function->return_type_info == NULL);
        for (int j = 0; j < function->param_count; ++j) {
            Parameter *parameter = &function->params[j];
            assert(parameter->fn_sig == NULL);
            assert(parameter->type_info == NULL);
        }
    }
    ModuleMetadata *metadata = extract_module_metadata(env, "generated_lists");
    assert(metadata);
    char *source = serialize_module_metadata_to_c(metadata);
    assert(source);
    assert(strstr(source, "List_MetadataItem_push"));
    free(source);
    free_module_metadata(metadata);
    free_environment(env);
    return 0;
}
