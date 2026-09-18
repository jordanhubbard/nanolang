/* I exercise equal bytes at distinct pool indices through the public module
 * API. File loading normally deduplicates them before lowering. */
#include "../../modules/nanoisa/nanoisa.h"
#include "../../src/nanoisa/verifier.h"
#include "../../src/nanoisa/nvm2llvm.h"
#include "../../src/nanovm/vm.h"
#include <stdio.h>
#include <string.h>
int g_argc = 0;
char **g_argv = NULL;
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    NanoisaErr error;
    NvmModule *m = nanoisa_load_file(argv[1], &error);
    if (!m) return 2;
    unsigned original = 0, changed = 0;
    for (uint32_t i = 0; i < m->string_count; i++) {
        if (nvm_get_string_len(m, i) != 3) continue;
        if (!memcmp(m->strings[i], "a\0z", 3)) original++;
        if (!memcmp(m->strings[i], "a\0x", 3)) {
            m->strings[i][2] = 'z';
            changed++;
        }
    }
    if (original != 1 || changed != 1 ||
        !nvm_verify_profile(m, NVM_PROFILE_CLOSED_LITERAL_STRINGS).ok) {
        nvm_module_free(m); return 2;
    }
    VmState vm;
    vm_init(&vm, m);
    int ok = vm_execute(&vm) == VM_OK;
    vm_destroy(&vm);
    FILE *out = fopen(argv[2], "w");
    if (!out) { nvm_module_free(m); return 2; }
    char message[512];
    ok = ok && nvm2llvm_emit_entry(m, out, message, sizeof message, "nano_entry");
    ok = fclose(out) == 0 && ok;
    nvm_module_free(m);
    return ok ? 0 : 1;
}
