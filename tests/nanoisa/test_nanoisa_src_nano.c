/*
 * Cut A: src_nano NanoISA lowering vs the C seed on a pinned subset.
 *
 * argv[1] = C-seed .nvm (nano_virt --emit-nvm --strip-debug)
 * argv[2] = src_nano .nasm (bin/nanoisa_emit)
 *
 * I compare function bytecode. PUSH_STR operands are resolved through the
 * string pool so intern order is not the claim. Debug is not the claim.
 */

#include "assembler.h"
#include "isa.h"
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
    size_t pa;
    size_t pb;
    if (!fa || !fb) return 0;
    if (fa->arity != fb->arity) return 0;
    if (fa->local_count != fb->local_count) return 0;
    if (fa->result_tag != fb->result_tag) return 0;
    if (fa->result_count != fb->result_count) return 0;
    pa = 0;
    pb = 0;
    while (pa < fa->code_length && pb < fb->code_length) {
        DecodedInstruction ia;
        DecodedInstruction ib;
        uint32_t na = isa_decode(a->code + fa->code_offset + pa,
                                 fa->code_length - pa, &ia);
        uint32_t nb = isa_decode(b->code + fb->code_offset + pb,
                                 fb->code_length - pb, &ib);
        if (na == 0 || nb == 0 || ia.opcode != ib.opcode) return 0;
        if (ia.opcode == OP_PUSH_STR) {
            const char *sa = nvm_get_string(a, ia.operands[0].u32);
            const char *sb = nvm_get_string(b, ib.operands[0].u32);
            if (!sa || !sb || strcmp(sa, sb) != 0) return 0;
        } else if (na != nb ||
                   memcmp(a->code + fa->code_offset + pa,
                          b->code + fb->code_offset + pb, na) != 0) {
            return 0;
        }
        pa += na;
        pb += nb;
    }
    return pa == fa->code_length && pb == fb->code_length;
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
    const NvmFunctionEntry *c_choose;
    const NvmFunctionEntry *s_choose;
    const NvmFunctionEntry *c_loop;
    const NvmFunctionEntry *s_loop;
    const NvmFunctionEntry *c_greet;
    const NvmFunctionEntry *s_greet;
    const NvmFunctionEntry *c_glue;
    const NvmFunctionEntry *s_glue;
    const NvmFunctionEntry *c_len3;
    const NvmFunctionEntry *s_len3;
    const NvmFunctionEntry *c_first;
    const NvmFunctionEntry *s_first;
    const NvmFunctionEntry *c_getx;
    const NvmFunctionEntry *s_getx;
    const NvmFunctionEntry *c_pos;
    const NvmFunctionEntry *s_pos;
    const NvmFunctionEntry *c_yes;
    const NvmFunctionEntry *s_yes;
    const NvmFunctionEntry *c_no;
    const NvmFunctionEntry *s_no;
    const NvmFunctionEntry *c_inv;
    const NvmFunctionEntry *s_inv;
    const NvmFunctionEntry *c_both;
    const NvmFunctionEntry *s_both;
    const NvmFunctionEntry *c_either;
    const NvmFunctionEntry *s_either;

    printf("\n[nanoisa src_nano] Cut A pinned subset...\n\n");
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

    CHECK(c_mod->function_count >= 15, "C seed emitted add through either");
    CHECK(s_mod->function_count >= 15, "src_nano emitted add through either");

    c_add = fn_by_name(c_mod, "add");
    s_add = fn_by_name(s_mod, "add");
    c_main = fn_by_name(c_mod, "main");
    s_main = fn_by_name(s_mod, "main");
    c_choose = fn_by_name(c_mod, "choose");
    s_choose = fn_by_name(s_mod, "choose");
    c_loop = fn_by_name(c_mod, "loop_sum");
    s_loop = fn_by_name(s_mod, "loop_sum");
    c_greet = fn_by_name(c_mod, "greeting");
    s_greet = fn_by_name(s_mod, "greeting");
    c_glue = fn_by_name(c_mod, "glue");
    s_glue = fn_by_name(s_mod, "glue");
    c_len3 = fn_by_name(c_mod, "len3");
    s_len3 = fn_by_name(s_mod, "len3");
    c_first = fn_by_name(c_mod, "first");
    s_first = fn_by_name(s_mod, "first");
    c_getx = fn_by_name(c_mod, "getx");
    s_getx = fn_by_name(s_mod, "getx");
    c_pos = fn_by_name(c_mod, "is_pos");
    s_pos = fn_by_name(s_mod, "is_pos");
    c_yes = fn_by_name(c_mod, "yes");
    s_yes = fn_by_name(s_mod, "yes");
    c_no = fn_by_name(c_mod, "no");
    s_no = fn_by_name(s_mod, "no");
    c_inv = fn_by_name(c_mod, "invert");
    s_inv = fn_by_name(s_mod, "invert");
    c_both = fn_by_name(c_mod, "both");
    s_both = fn_by_name(s_mod, "both");
    c_either = fn_by_name(c_mod, "either");
    s_either = fn_by_name(s_mod, "either");
    CHECK(c_add != NULL && s_add != NULL, "both modules have add");
    CHECK(c_main != NULL && s_main != NULL, "both modules have main");
    CHECK(c_choose != NULL && s_choose != NULL, "both modules have choose");
    CHECK(c_loop != NULL && s_loop != NULL, "both modules have loop_sum");
    CHECK(c_greet != NULL && s_greet != NULL, "both modules have greeting");
    CHECK(c_glue != NULL && s_glue != NULL, "both modules have glue");
    CHECK(c_len3 != NULL && s_len3 != NULL, "both modules have len3");
    CHECK(c_first != NULL && s_first != NULL, "both modules have first");
    CHECK(c_getx != NULL && s_getx != NULL, "both modules have getx");
    CHECK(c_pos != NULL && s_pos != NULL, "both modules have is_pos");
    CHECK(c_yes != NULL && s_yes != NULL, "both modules have yes");
    CHECK(c_no != NULL && s_no != NULL, "both modules have no");
    CHECK(c_inv != NULL && s_inv != NULL, "both modules have invert");
    CHECK(c_both != NULL && s_both != NULL, "both modules have both");
    CHECK(c_either != NULL && s_either != NULL, "both modules have either");
    CHECK(code_equal(c_mod, c_add, s_mod, s_add),
          "add bytecode matches C seed");
    CHECK(code_equal(c_mod, c_main, s_mod, s_main),
          "main bytecode matches C seed");
    CHECK(code_equal(c_mod, c_choose, s_mod, s_choose),
          "choose bytecode matches C seed");
    CHECK(code_equal(c_mod, c_loop, s_mod, s_loop),
          "loop_sum bytecode matches C seed");
    CHECK(code_equal(c_mod, c_greet, s_mod, s_greet),
          "greeting bytecode matches C seed");
    CHECK(code_equal(c_mod, c_glue, s_mod, s_glue),
          "glue bytecode matches C seed");
    CHECK(code_equal(c_mod, c_len3, s_mod, s_len3),
          "len3 bytecode matches C seed");
    CHECK(code_equal(c_mod, c_first, s_mod, s_first),
          "first bytecode matches C seed");
    CHECK(code_equal(c_mod, c_getx, s_mod, s_getx),
          "getx bytecode matches C seed");
    CHECK(code_equal(c_mod, c_pos, s_mod, s_pos),
          "is_pos bytecode matches C seed");
    CHECK(code_equal(c_mod, c_yes, s_mod, s_yes),
          "yes bytecode matches C seed");
    CHECK(code_equal(c_mod, c_no, s_mod, s_no),
          "no bytecode matches C seed");
    CHECK(code_equal(c_mod, c_inv, s_mod, s_inv),
          "invert bytecode matches C seed");
    CHECK(code_equal(c_mod, c_both, s_mod, s_both),
          "both bytecode matches C seed");
    CHECK(code_equal(c_mod, c_either, s_mod, s_either),
          "either bytecode matches C seed");
    CHECK((c_mod->header.flags & NVM_FLAG_HAS_MAIN) != 0, "C seed has_main");
    CHECK((s_mod->header.flags & NVM_FLAG_HAS_MAIN) != 0, "src_nano has_main");

    if (g_fail) {
        printf("    C add locals=%u len=%u  src add locals=%u len=%u\n",
               c_add ? c_add->local_count : 0, c_add ? c_add->code_length : 0,
               s_add ? s_add->local_count : 0, s_add ? s_add->code_length : 0);
        printf("    C main locals=%u len=%u  src main locals=%u len=%u\n",
               c_main ? c_main->local_count : 0, c_main ? c_main->code_length : 0,
               s_main ? s_main->local_count : 0, s_main ? s_main->code_length : 0);
        printf("    C choose locals=%u len=%u  src choose locals=%u len=%u\n",
               c_choose ? c_choose->local_count : 0, c_choose ? c_choose->code_length : 0,
               s_choose ? s_choose->local_count : 0, s_choose ? s_choose->code_length : 0);
        printf("    C loop_sum locals=%u len=%u  src loop_sum locals=%u len=%u\n",
               c_loop ? c_loop->local_count : 0, c_loop ? c_loop->code_length : 0,
               s_loop ? s_loop->local_count : 0, s_loop ? s_loop->code_length : 0);
        printf("    C greeting locals=%u len=%u  src greeting locals=%u len=%u\n",
               c_greet ? c_greet->local_count : 0, c_greet ? c_greet->code_length : 0,
               s_greet ? s_greet->local_count : 0, s_greet ? s_greet->code_length : 0);
        printf("    C glue locals=%u len=%u  src glue locals=%u len=%u\n",
               c_glue ? c_glue->local_count : 0, c_glue ? c_glue->code_length : 0,
               s_glue ? s_glue->local_count : 0, s_glue ? s_glue->code_length : 0);
        printf("    C len3 locals=%u len=%u  src len3 locals=%u len=%u\n",
               c_len3 ? c_len3->local_count : 0, c_len3 ? c_len3->code_length : 0,
               s_len3 ? s_len3->local_count : 0, s_len3 ? s_len3->code_length : 0);
        printf("    C first locals=%u len=%u  src first locals=%u len=%u\n",
               c_first ? c_first->local_count : 0, c_first ? c_first->code_length : 0,
               s_first ? s_first->local_count : 0, s_first ? s_first->code_length : 0);
        printf("    C getx locals=%u len=%u  src getx locals=%u len=%u\n",
               c_getx ? c_getx->local_count : 0, c_getx ? c_getx->code_length : 0,
               s_getx ? s_getx->local_count : 0, s_getx ? s_getx->code_length : 0);
        printf("    C is_pos locals=%u len=%u  src is_pos locals=%u len=%u\n",
               c_pos ? c_pos->local_count : 0, c_pos ? c_pos->code_length : 0,
               s_pos ? s_pos->local_count : 0, s_pos ? s_pos->code_length : 0);
        printf("    C yes locals=%u len=%u  src yes locals=%u len=%u\n",
               c_yes ? c_yes->local_count : 0, c_yes ? c_yes->code_length : 0,
               s_yes ? s_yes->local_count : 0, s_yes ? s_yes->code_length : 0);
        printf("    C invert locals=%u len=%u  src invert locals=%u len=%u\n",
               c_inv ? c_inv->local_count : 0, c_inv ? c_inv->code_length : 0,
               s_inv ? s_inv->local_count : 0, s_inv ? s_inv->code_length : 0);
        printf("    C both locals=%u len=%u  src both locals=%u len=%u\n",
               c_both ? c_both->local_count : 0, c_both ? c_both->code_length : 0,
               s_both ? s_both->local_count : 0, s_both ? s_both->code_length : 0);
        printf("    C either locals=%u len=%u  src either locals=%u len=%u\n",
               c_either ? c_either->local_count : 0, c_either ? c_either->code_length : 0,
               s_either ? s_either->local_count : 0, s_either ? s_either->code_length : 0);
    }

    nvm_module_free(c_mod);
    nvm_module_free(s_mod);
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
