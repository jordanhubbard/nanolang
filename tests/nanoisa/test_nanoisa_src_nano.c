/*
 * Cut A: src_nano NanoISA lowering vs the C seed on a pinned i64 program.
 *
 * argv[1] = C-seed .nvm (nano_virt --emit-nvm --strip-debug)
 * argv[2] = src_nano .nasm (bin/nanoisa_emit)
 *
 * I compare function bytecode, not string-pool extras. Debug is not the claim.
 */

#include "assembler.h"
#include "nanoisa.h"
#include "nvm_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass;
static int g_fail;

#define CHECK(cond, what) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  FAIL: %s  (%s:%d)\n", (what), __FILE__, __LINE__); } \
} while (0)

static const NvmFunctionEntry *fn_by_name(const NvmModule *m, const char *name) {
    uint32_t i;
    for (i = 0; i < m->function_count; i++) {
        const char *n = nvm_get_string(m, m->functions[i].name_idx);
        if (n && strcmp(n, name) == 0) return &m->functions[i];
    }
    return NULL;
}

static int code_equal(const NvmModule *a, const NvmFunctionEntry *fa,
                      const NvmModule *b, const NvmFunctionEntry *fb) {
    if (!fa || !fb) return 0;
    if (fa->code_length != fb->code_length) return 0;
    if (fa->arity != fb->arity) return 0;
    if (fa->local_count != fb->local_count) return 0;
    if (fa->code_length == 0) return 1;
    return memcmp(a->code + fa->code_offset, b->code + fb->code_offset,
                  fa->code_length) == 0;
}

int main(int argc, char **argv) {
    NanoisaErr err;
    AsmResult asm_err;
    NvmModule *c_mod;
    NvmModule *s_mod;
    const NvmFunctionEntry *c_add;
    const NvmFunctionEntry *s_add;
    const NvmFunctionEntry *c_main;
    const NvmFunctionEntry *s_main;

    printf("\n[nanoisa src_nano] Cut A pinned i64 subset...\n\n");
    if (argc < 3) {
        printf("  FAIL: usage: test_nanoisa_src_nano <c.nvm> <src.nasm>\n");
        return 1;
    }

    memset(&err, 0, sizeof err);
    c_mod = nanoisa_load_file(argv[1], &err);
    CHECK(c_mod != NULL, "C-seed module loads");
    if (!c_mod) {
        printf("    load: %s\n", err.message);
        return 1;
    }
    nvm_strip_debug_info(c_mod);

    memset(&asm_err, 0, sizeof asm_err);
    s_mod = asm_assemble_file(argv[2], &asm_err);
    CHECK(s_mod != NULL, "src_nano nasm assembles and verifies");
    if (!s_mod) {
        printf("    assemble: %s (line %u)\n", asm_err.message, asm_err.line);
        nvm_module_free(c_mod);
        return 1;
    }

    CHECK(c_mod->function_count >= 2, "C seed emitted add and main");
    CHECK(s_mod->function_count >= 2, "src_nano emitted add and main");

    c_add = fn_by_name(c_mod, "add");
    s_add = fn_by_name(s_mod, "add");
    c_main = fn_by_name(c_mod, "main");
    s_main = fn_by_name(s_mod, "main");
    CHECK(c_add != NULL && s_add != NULL, "both modules have add");
    CHECK(c_main != NULL && s_main != NULL, "both modules have main");
    CHECK(code_equal(c_mod, c_add, s_mod, s_add),
          "add bytecode matches C seed");
    CHECK(code_equal(c_mod, c_main, s_mod, s_main),
          "main bytecode matches C seed");
    CHECK((c_mod->header.flags & NVM_FLAG_HAS_MAIN) != 0, "C seed has_main");
    CHECK((s_mod->header.flags & NVM_FLAG_HAS_MAIN) != 0, "src_nano has_main");

    if (g_fail) {
        printf("    C add locals=%u len=%u  src add locals=%u len=%u\n",
               c_add ? c_add->local_count : 0, c_add ? c_add->code_length : 0,
               s_add ? s_add->local_count : 0, s_add ? s_add->code_length : 0);
        printf("    C main locals=%u len=%u  src main locals=%u len=%u\n",
               c_main ? c_main->local_count : 0, c_main ? c_main->code_length : 0,
               s_main ? s_main->local_count : 0, s_main ? s_main->code_length : 0);
    }

    nvm_module_free(c_mod);
    nvm_module_free(s_mod);
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
