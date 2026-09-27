/* I borrow extern lookup spelling from my immutable module string entries. */
#include "../../src/nanovirt/codegen.c"
#include <assert.h>

int g_argc;
char **g_argv;

int main(void) {
    for (int repeat = 0; repeat < 8; ++repeat) {
        CG *cg = calloc(1, sizeof *cg);
        assert(cg);
        cg->module = nvm_module_new();
        assert(cg->module);
        const uint8_t parameter = TAG_INT;
        for (int i = 0; i < 80; ++i) {
            char name[32], owner[32];
            assert(snprintf(name, sizeof name, "owned_extern_%d", i) > 0);
            assert(snprintf(owner, sizeof owner, "owned_module_%d", i) > 0);
            register_extern(cg, name, owner, 1, TAG_INT, &parameter);
            assert(!cg->had_error && cg->extern_count == i + 1);
            memset(name, '?', sizeof name);
            memset(owner, '?', sizeof owner);
        }
        assert(cg->module->string_count == 160);
        assert(cg->module->string_capacity >= 160);
        CG *snapshot = malloc(sizeof *snapshot);
        assert(snapshot);
        memcpy(snapshot, cg, sizeof *snapshot);
        for (int i = 0; i < 80; ++i) {
            char expected[32];
            assert(snprintf(expected, sizeof expected, "owned_extern_%d", i) > 0);
            ExternFn *entry = &snapshot->externs[i];
            NvmImportEntry *import = &cg->module->imports[entry->import_idx];
            assert(!strcmp(entry->name, expected));
            assert(entry->name == nvm_get_string(cg->module, import->function_name_idx));
            assert(entry->module_name == nvm_get_string(cg->module, import->module_name_idx));
            assert(extern_find(snapshot, expected) == i);
        }
        register_extern(cg, "refused", "refused_owner", NANO_MAX_FFI_ARGS + 1,
                        TAG_INT, &parameter);
        assert(cg->had_error && cg->extern_count == 80);
        assert(cg->module->import_count == 80 && cg->module->string_count == 160);
        free(snapshot);
        nvm_module_free(cg->module);
        free(cg);
    }
    return 0;
}
