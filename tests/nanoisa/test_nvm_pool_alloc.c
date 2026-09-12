/* I inject each pool-growth failure, preserve old entries, and retry. */
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static int fail_at;
static void *pool_malloc(size_t size) {
    if (fail_at && --fail_at == 0) return NULL;
    return malloc(size);
}
#define malloc pool_malloc
#include "../../src/nanoisa/nvm_format.c"
#undef malloc

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
    puts("I passed six pool allocation failure/recovery scenarios.");
    return 0;
}
