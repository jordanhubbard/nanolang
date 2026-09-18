/* I compare caller-selected policies with ordinary verification and lowering. */
#include "../../modules/nanoisa/nanoisa.h"
#include "../../src/nanoisa/verifier.h"
#include "../../src/nanoisa/nvm2llvm.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    /* I retain ordinary verifier rejection for an absent input module. */
    NvmVerifyResult absent = nvm_verify(NULL);
    NvmVerifyResult absent_general = nvm_verify_profile(NULL, NVM_PROFILE_GENERAL);
    NvmVerifyResult absent_scalar = nvm_verify_profile(NULL, NVM_PROFILE_CLOSED_SCALAR);
    if (absent.ok || absent_general.ok || absent_scalar.ok ||
        strcmp(absent.error_msg, absent_general.error_msg) ||
        strcmp(absent.error_msg, absent_scalar.error_msg)) return 1;
    NanoisaErr error;
    NvmModule *m = nanoisa_load_file(argv[1], &error);
    if (!m) { fprintf(stderr, "%s\n", error.message); return 2; }
    NvmVerifyResult ordinary = nvm_verify(m);
    NvmVerifyResult general = nvm_verify_profile(m, NVM_PROFILE_GENERAL);
    NvmVerifyResult scalar = nvm_verify_profile(m, NVM_PROFILE_CLOSED_SCALAR);
    NvmVerifyResult managed = nvm_verify_profile(m, NVM_PROFILE_CLOSED_MANAGED_STRINGS);
    NvmVerifyResult literal = nvm_verify_profile(m, NVM_PROFILE_CLOSED_LITERAL_STRINGS);
    NvmVerifyResult unknown = nvm_verify_profile(m, (NvmVerifyProfile)99);
    FILE *out = tmpfile();
    if (!out) { nvm_module_free(m); return 2; }
    char diagnostic[512] = {0};
    int translated = nvm2llvm_emit(m, out, diagnostic, sizeof diagnostic);
    long length = ftell(out);
    fclose(out);
    const char *reserved[] = {"nano_try_entry", "nano_dispose", "nano_runtime_hidden"};
    for (unsigned i = 0; i < 4; i++) {
        FILE *preserved = tmpfile();
        if (!preserved) { nvm_module_free(m); return 2; }
        fputs("prior", preserved);
        long before = ftell(preserved);
        int accepted = nvm2llvm_emit_target(m, preserved, diagnostic, sizeof diagnostic,
            i < 3 ? reserved[i] : "main", i < 3 ? NVM_LLVM_NATIVE : (NvmLlvmTarget)99);
        int unchanged = ftell(preserved) == before;
        fclose(preserved);
        if (accepted || !unchanged) { nvm_module_free(m); return 1; }
    }
    int same = ordinary.ok == general.ok && !strcmp(ordinary.error_msg, general.error_msg);
    int refusal_untouched = translated || length == 0;
    printf("%d %d %d %d %d %d %d %d\n", general.ok, scalar.ok, translated,
           same, !unknown.ok, refusal_untouched, literal.ok, managed.ok);
    if (!scalar.ok) puts(scalar.error_msg);
    nvm_module_free(m);
    return same && !unknown.ok && refusal_untouched && managed.ok == (translated != 0) && (!literal.ok || managed.ok) ? 0 : 1;
}
