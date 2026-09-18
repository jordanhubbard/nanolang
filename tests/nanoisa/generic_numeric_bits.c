/* I observe an ordinary float helper result without changing its instructions. */
#include "../../modules/nanoisa/nanoisa.h"
#include "../../src/nanoisa/verifier.h"
#include "../../src/nanovm/vm.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
int g_argc = 0;
char **g_argv = NULL;
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    NanoisaErr error;
    NvmModule *m = nanoisa_load_file(argv[1], &error);
    if (!m) return 2;
    if (!nvm_verify_profile(m, NVM_PROFILE_CLOSED_SCALAR).ok) {
        nvm_module_free(m); return 2;
    }
    VmState vm;
    vm_init(&vm, m);
    NanoValue value = val_void();
    VmResult result = vm_invoke(&vm, 1, NULL, 0, &value);
    int ok = result == VM_OK && value.tag == TAG_FLOAT;
    if (ok) {
        uint64_t bits;
        memcpy(&bits, &value.as.f64, sizeof bits);
        printf("%016" PRIx64 "\n", bits);
    }
    vm_destroy(&vm);
    nvm_module_free(m);
    return ok ? 0 : 1;
}
