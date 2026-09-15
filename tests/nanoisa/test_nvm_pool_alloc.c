/* I inject each pool-growth failure, preserve old entries, and retry. */
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static int fail_at;
static int fail_calloc_at;
static void *pool_malloc(size_t size) {
    if (fail_at && --fail_at == 0) return NULL;
    return malloc(size);
}
static void *pool_calloc(size_t count, size_t size) {
    if (fail_calloc_at && --fail_calloc_at == 0) return NULL;
    return calloc(count, size);
}
#define malloc pool_malloc
#define calloc pool_calloc
#include "../../src/nanoisa/nvm_format.c"
#undef malloc
#undef calloc

int main(void) {
    for (int failure = 1; failure <= 3; failure++) {
        NvmModule *m = nvm_module_new();
        assert(m);
        char value[32];
        uint32_t count = m->string_capacity;
        for (uint32_t i = 0; i < count; i++) {
            snprintf(value, sizeof value, "entry%u", i);
            assert(nvm_add_string(m, value, (uint32_t)strlen(value)) == i);
        }
        fail_at = failure;
        assert(nvm_add_string(m, "next", 4) == UINT32_MAX);
        fail_at = 0;
        assert(m->string_count == count);
        for (uint32_t i = 0; i < count; i++) {
            snprintf(value, sizeof value, "entry%u", i);
            assert(strcmp(nvm_get_string(m, i), value) == 0);
        }
        assert(nvm_add_string(m, "next", 4) == count);
        assert(nvm_add_string(m, NULL, 1) == UINT32_MAX);
        assert(nvm_add_string(m, NULL, 0) != UINT32_MAX);
        nvm_module_free(m);

        m = nvm_module_new();
        assert(m);
        count = m->import_capacity;
        uint8_t tags[] = {1, 2};
        for (uint32_t i = 0; i < count; i++)
            assert(nvm_add_import(m, i, i + 1, 2, 1, tags) == i);
        fail_at = failure;
        assert(nvm_add_import(m, 0, 1, 2, 1, tags) == UINT32_MAX);
        fail_at = 0;
        assert(m->import_count == count);
        for (uint32_t i = 0; i < count; i++) {
            assert(m->imports[i].module_name_idx == i);
            assert(memcmp(m->import_param_types[i], tags, 2) == 0);
        }
        assert(nvm_add_import(m, 0, 1, 2, 1, tags) == count);
        nvm_module_free(m);
    }
    for (int failure = 1; failure <= 2; failure++) {
        NvmModule *m = nvm_module_new();
        assert(m);
        NvmFunctionEntry fn = {.arity = 2, .local_count = 2};
        uint8_t tags[] = {TAG_INT, TAG_FLOAT};
        uint32_t count = m->function_capacity;
        for (uint32_t i = 0; i < count; i++) {
            assert(nvm_add_function(m, &fn) == i);
            assert(nvm_set_function_param_types(m, i, tags, 2));
        }
        NvmFunctionEntry *old_functions = m->functions;
        uint8_t **old_types = m->function_param_types;
        fail_calloc_at = failure;
        assert(nvm_add_function(m, &fn) == UINT32_MAX);
        fail_calloc_at = 0;
        assert(m->function_count == count && m->function_capacity == count);
        assert(m->functions == old_functions && m->function_param_types == old_types);
        for (uint32_t i = 0; i < count; i++)
            assert(memcmp(m->function_param_types[i], tags, 2) == 0);
        assert(nvm_add_function(m, &m->functions[0]) == count);
        uint8_t replacement[] = {TAG_BOOL, TAG_INT};
        fail_at = 1;
        assert(!nvm_set_function_param_types(m, 0, replacement, 2));
        fail_at = 0;
        assert(memcmp(m->function_param_types[0], tags, 2) == 0);
        assert(nvm_set_function_param_types(m, 0, replacement, 2));
        assert(memcmp(m->function_param_types[0], replacement, 2) == 0);
        nvm_module_free(m);
    }
    puts("I passed pool and function-signature allocation failure/recovery scenarios.");
    return 0;
}
