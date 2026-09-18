/* I observe repeated execution on one VM instance, then a fresh instance. */
#include "../../modules/nanoisa/nanoisa.h"
#include "../../src/nanoisa/verifier.h"
#include "../../src/nanovm/vm.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
int g_argc = 0;
char **g_argv = NULL;
int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) return 2;
    NanoisaErr error;
    NvmModule *m = nanoisa_load_file(argv[1], &error);
    if (!m) return 2;
    if (!nvm_verify_profile(m, NVM_PROFILE_CLOSED_SCALAR).ok) {
        nvm_module_free(m); return 2;
    }
    if (argc == 3) {
        /* My text assembler requires unique symbols. The module API permits
         * repeated display names; I test vm_execute's first-name selection. */
        if (m->function_count != 3 ||
            strcmp(nvm_get_string(m, m->functions[1].name_idx), "__init__") ||
            strcmp(nvm_get_string(m, m->functions[2].name_idx), "later")) {
            nvm_module_free(m); return 2;
        }
        m->functions[2].name_idx = m->functions[1].name_idx;
        int ok = nvm_verify_profile(m, NVM_PROFILE_CLOSED_SCALAR).ok &&
                 nanoisa_save_file(m, argv[2], &error) == NANOISA_OK;
        nvm_module_free(m);
        return ok ? 0 : 1;
    }
    VmState vm;
    for (int instance = 0; instance < 2; instance++) {
        vm_init(&vm, m);
        for (int call = 0; call < (instance ? 1 : 2); call++) {
            if (vm_execute(&vm) != VM_OK) { vm_destroy(&vm); nvm_module_free(m); return 1; }
            NanoValue value = vm_get_result(&vm);
            if (value.tag != TAG_INT) { vm_destroy(&vm); nvm_module_free(m); return 1; }
            printf("%" PRId64 "\n", value.as.i64);
        }
        vm_destroy(&vm);
    }
    nvm_module_free(m);
    return 0;
}
