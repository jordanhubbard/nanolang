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
    const NvmFunctionEntry *c_pick;
    const NvmFunctionEntry *s_pick;
    const NvmFunctionEntry *c_say;
    const NvmFunctionEntry *s_say;
    const NvmFunctionEntry *c_shout;
    const NvmFunctionEntry *s_shout;
    const NvmFunctionEntry *c_mutter;
    const NvmFunctionEntry *s_mutter;
    const NvmFunctionEntry *c_prove;
    const NvmFunctionEntry *s_prove;
    const NvmFunctionEntry *c_grow;
    const NvmFunctionEntry *s_grow;
    const NvmFunctionEntry *c_has;
    const NvmFunctionEntry *s_has;
    const NvmFunctionEntry *c_digits;
    const NvmFunctionEntry *s_digits;
    const NvmFunctionEntry *c_names;
    const NvmFunctionEntry *s_names;
    const NvmFunctionEntry *c_head;
    const NvmFunctionEntry *s_head;
    const NvmFunctionEntry *c_same;
    const NvmFunctionEntry *s_same;
    const NvmFunctionEntry *c_diff;
    const NvmFunctionEntry *s_diff;
    const NvmFunctionEntry *c_at;
    const NvmFunctionEntry *s_at;
    const NvmFunctionEntry *c_slen;
    const NvmFunctionEntry *s_slen;
    const NvmFunctionEntry *c_slice;
    const NvmFunctionEntry *s_slice;
    const NvmFunctionEntry *c_blank;
    const NvmFunctionEntry *s_blank;
    const NvmFunctionEntry *c_grow_l;
    const NvmFunctionEntry *s_grow_l;
    const NvmFunctionEntry *c_ch;
    const NvmFunctionEntry *s_ch;
    const NvmFunctionEntry *c_blank_s;
    const NvmFunctionEntry *s_blank_s;
    const NvmFunctionEntry *c_grow_s;
    const NvmFunctionEntry *s_grow_s;
    const NvmFunctionEntry *c_get_s;
    const NvmFunctionEntry *s_get_s;
    const NvmFunctionEntry *c_blank_t;
    const NvmFunctionEntry *s_blank_t;
    const NvmFunctionEntry *c_grow_t;
    const NvmFunctionEntry *s_grow_t;
    const NvmFunctionEntry *c_get_v;
    const NvmFunctionEntry *s_get_v;
    const NvmFunctionEntry *c_grow_lex;
    const NvmFunctionEntry *s_grow_lex;
    const NvmFunctionEntry *c_has_pre;
    const NvmFunctionEntry *s_has_pre;
    const NvmFunctionEntry *c_has_suf;
    const NvmFunctionEntry *s_has_suf;
    const NvmFunctionEntry *c_put_l;
    const NvmFunctionEntry *s_put_l;
    const NvmFunctionEntry *c_put_t;
    const NvmFunctionEntry *s_put_t;
    const NvmFunctionEntry *c_put_s;
    const NvmFunctionEntry *s_put_s;
    const NvmFunctionEntry *c_upto;
    const NvmFunctionEntry *s_upto;
    const NvmFunctionEntry *c_quiet;
    const NvmFunctionEntry *s_quiet;
    const NvmFunctionEntry *c_via_quiet;
    const NvmFunctionEntry *s_via_quiet;
    const NvmFunctionEntry *c_origin;
    const NvmFunctionEntry *s_origin;
    const NvmFunctionEntry *c_via_o;
    const NvmFunctionEntry *s_via_o;

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

    CHECK(c_mod->function_count >= 50, "C seed emitted add through via_o");
    CHECK(s_mod->function_count >= 50, "src_nano emitted add through via_o");

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
    c_pick = fn_by_name(c_mod, "pick");
    s_pick = fn_by_name(s_mod, "pick");
    c_say = fn_by_name(c_mod, "say");
    s_say = fn_by_name(s_mod, "say");
    c_shout = fn_by_name(c_mod, "shout");
    s_shout = fn_by_name(s_mod, "shout");
    c_mutter = fn_by_name(c_mod, "mutter");
    s_mutter = fn_by_name(s_mod, "mutter");
    c_prove = fn_by_name(c_mod, "prove");
    s_prove = fn_by_name(s_mod, "prove");
    c_grow = fn_by_name(c_mod, "grow");
    s_grow = fn_by_name(s_mod, "grow");
    c_has = fn_by_name(c_mod, "has_hi");
    s_has = fn_by_name(s_mod, "has_hi");
    c_digits = fn_by_name(c_mod, "digits");
    s_digits = fn_by_name(s_mod, "digits");
    c_names = fn_by_name(c_mod, "names");
    s_names = fn_by_name(s_mod, "names");
    c_head = fn_by_name(c_mod, "head_s");
    s_head = fn_by_name(s_mod, "head_s");
    c_same = fn_by_name(c_mod, "same");
    s_same = fn_by_name(s_mod, "same");
    c_diff = fn_by_name(c_mod, "diff");
    s_diff = fn_by_name(s_mod, "diff");
    c_at = fn_by_name(c_mod, "via_at");
    s_at = fn_by_name(s_mod, "via_at");
    c_slen = fn_by_name(c_mod, "slen");
    s_slen = fn_by_name(s_mod, "slen");
    c_slice = fn_by_name(c_mod, "slice");
    s_slice = fn_by_name(s_mod, "slice");
    c_blank = fn_by_name(c_mod, "blank_l");
    s_blank = fn_by_name(s_mod, "blank_l");
    c_grow_l = fn_by_name(c_mod, "grow_l");
    s_grow_l = fn_by_name(s_mod, "grow_l");
    c_ch = fn_by_name(c_mod, "ch");
    s_ch = fn_by_name(s_mod, "ch");
    c_blank_s = fn_by_name(c_mod, "blank_s");
    s_blank_s = fn_by_name(s_mod, "blank_s");
    c_grow_s = fn_by_name(c_mod, "grow_s");
    s_grow_s = fn_by_name(s_mod, "grow_s");
    c_get_s = fn_by_name(c_mod, "get_s");
    s_get_s = fn_by_name(s_mod, "get_s");
    c_blank_t = fn_by_name(c_mod, "blank_t");
    s_blank_t = fn_by_name(s_mod, "blank_t");
    c_grow_t = fn_by_name(c_mod, "grow_t");
    s_grow_t = fn_by_name(s_mod, "grow_t");
    c_get_v = fn_by_name(c_mod, "get_v");
    s_get_v = fn_by_name(s_mod, "get_v");
    c_grow_lex = fn_by_name(c_mod, "grow_lex");
    s_grow_lex = fn_by_name(s_mod, "grow_lex");
    c_has_pre = fn_by_name(c_mod, "has_pre");
    s_has_pre = fn_by_name(s_mod, "has_pre");
    c_has_suf = fn_by_name(c_mod, "has_suf");
    s_has_suf = fn_by_name(s_mod, "has_suf");
    c_put_l = fn_by_name(c_mod, "put_l");
    s_put_l = fn_by_name(s_mod, "put_l");
    c_put_t = fn_by_name(c_mod, "put_t");
    s_put_t = fn_by_name(s_mod, "put_t");
    c_put_s = fn_by_name(c_mod, "put_s");
    s_put_s = fn_by_name(s_mod, "put_s");
    c_upto = fn_by_name(c_mod, "upto");
    s_upto = fn_by_name(s_mod, "upto");
    c_quiet = fn_by_name(c_mod, "quiet");
    s_quiet = fn_by_name(s_mod, "quiet");
    c_via_quiet = fn_by_name(c_mod, "via_quiet");
    s_via_quiet = fn_by_name(s_mod, "via_quiet");
    c_origin = fn_by_name(c_mod, "origin");
    s_origin = fn_by_name(s_mod, "origin");
    c_via_o = fn_by_name(c_mod, "via_o");
    s_via_o = fn_by_name(s_mod, "via_o");
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
    CHECK(c_pick != NULL && s_pick != NULL, "both modules have pick");
    CHECK(c_say != NULL && s_say != NULL, "both modules have say");
    CHECK(c_shout != NULL && s_shout != NULL, "both modules have shout");
    CHECK(c_mutter != NULL && s_mutter != NULL, "both modules have mutter");
    CHECK(c_prove != NULL && s_prove != NULL, "both modules have prove");
    CHECK(c_grow != NULL && s_grow != NULL, "both modules have grow");
    CHECK(c_has != NULL && s_has != NULL, "both modules have has_hi");
    CHECK(c_digits != NULL && s_digits != NULL, "both modules have digits");
    CHECK(c_names != NULL && s_names != NULL, "both modules have names");
    CHECK(c_head != NULL && s_head != NULL, "both modules have head_s");
    CHECK(c_same != NULL && s_same != NULL, "both modules have same");
    CHECK(c_diff != NULL && s_diff != NULL, "both modules have diff");
    CHECK(c_at != NULL && s_at != NULL, "both modules have via_at");
    CHECK(c_slen != NULL && s_slen != NULL, "both modules have slen");
    CHECK(c_slice != NULL && s_slice != NULL, "both modules have slice");
    CHECK(c_blank != NULL && s_blank != NULL, "both modules have blank_l");
    CHECK(c_grow_l != NULL && s_grow_l != NULL, "both modules have grow_l");
    CHECK(c_ch != NULL && s_ch != NULL, "both modules have ch");
    CHECK(c_blank_s != NULL && s_blank_s != NULL, "both modules have blank_s");
    CHECK(c_grow_s != NULL && s_grow_s != NULL, "both modules have grow_s");
    CHECK(c_get_s != NULL && s_get_s != NULL, "both modules have get_s");
    CHECK(c_blank_t != NULL && s_blank_t != NULL, "both modules have blank_t");
    CHECK(c_grow_t != NULL && s_grow_t != NULL, "both modules have grow_t");
    CHECK(c_get_v != NULL && s_get_v != NULL, "both modules have get_v");
    CHECK(c_grow_lex != NULL && s_grow_lex != NULL, "both modules have grow_lex");
    CHECK(c_has_pre != NULL && s_has_pre != NULL, "both modules have has_pre");
    CHECK(c_has_suf != NULL && s_has_suf != NULL, "both modules have has_suf");
    CHECK(c_put_l != NULL && s_put_l != NULL, "both modules have put_l");
    CHECK(c_put_t != NULL && s_put_t != NULL, "both modules have put_t");
    CHECK(c_put_s != NULL && s_put_s != NULL, "both modules have put_s");
    CHECK(c_upto != NULL && s_upto != NULL, "both modules have upto");
    CHECK(c_quiet != NULL && s_quiet != NULL, "both modules have quiet");
    CHECK(c_via_quiet != NULL && s_via_quiet != NULL, "both modules have via_quiet");
    CHECK(c_origin != NULL && s_origin != NULL, "both modules have origin");
    CHECK(c_via_o != NULL && s_via_o != NULL, "both modules have via_o");
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
    CHECK(code_equal(c_mod, c_pick, s_mod, s_pick),
          "pick bytecode matches C seed");
    CHECK(code_equal(c_mod, c_say, s_mod, s_say),
          "say bytecode matches C seed");
    CHECK(code_equal(c_mod, c_shout, s_mod, s_shout),
          "shout bytecode matches C seed");
    CHECK(code_equal(c_mod, c_mutter, s_mod, s_mutter),
          "mutter bytecode matches C seed");
    CHECK(code_equal(c_mod, c_prove, s_mod, s_prove),
          "prove bytecode matches C seed");
    CHECK(code_equal(c_mod, c_grow, s_mod, s_grow),
          "grow bytecode matches C seed");
    CHECK(code_equal(c_mod, c_has, s_mod, s_has),
          "has_hi bytecode matches C seed");
    CHECK(code_equal(c_mod, c_digits, s_mod, s_digits),
          "digits bytecode matches C seed");
    CHECK(code_equal(c_mod, c_names, s_mod, s_names),
          "names bytecode matches C seed");
    CHECK(code_equal(c_mod, c_head, s_mod, s_head),
          "head_s bytecode matches C seed");
    CHECK(code_equal(c_mod, c_same, s_mod, s_same),
          "same bytecode matches C seed");
    CHECK(code_equal(c_mod, c_diff, s_mod, s_diff),
          "diff bytecode matches C seed");
    CHECK(code_equal(c_mod, c_at, s_mod, s_at),
          "via_at bytecode matches C seed");
    CHECK(code_equal(c_mod, c_slen, s_mod, s_slen),
          "slen bytecode matches C seed");
    CHECK(code_equal(c_mod, c_slice, s_mod, s_slice),
          "slice bytecode matches C seed");
    CHECK(code_equal(c_mod, c_blank, s_mod, s_blank),
          "blank_l bytecode matches C seed");
    CHECK(code_equal(c_mod, c_grow_l, s_mod, s_grow_l),
          "grow_l bytecode matches C seed");
    CHECK(code_equal(c_mod, c_ch, s_mod, s_ch),
          "ch bytecode matches C seed");
    CHECK(code_equal(c_mod, c_blank_s, s_mod, s_blank_s),
          "blank_s bytecode matches C seed");
    CHECK(code_equal(c_mod, c_grow_s, s_mod, s_grow_s),
          "grow_s bytecode matches C seed");
    CHECK(code_equal(c_mod, c_get_s, s_mod, s_get_s),
          "get_s bytecode matches C seed");
    CHECK(code_equal(c_mod, c_blank_t, s_mod, s_blank_t),
          "blank_t bytecode matches C seed");
    CHECK(code_equal(c_mod, c_grow_t, s_mod, s_grow_t),
          "grow_t bytecode matches C seed");
    CHECK(code_equal(c_mod, c_get_v, s_mod, s_get_v),
          "get_v bytecode matches C seed");
    CHECK(code_equal(c_mod, c_grow_lex, s_mod, s_grow_lex),
          "grow_lex bytecode matches C seed");
    CHECK(code_equal(c_mod, c_has_pre, s_mod, s_has_pre),
          "has_pre bytecode matches C seed");
    CHECK(code_equal(c_mod, c_has_suf, s_mod, s_has_suf),
          "has_suf bytecode matches C seed");
    CHECK(code_equal(c_mod, c_put_l, s_mod, s_put_l),
          "put_l bytecode matches C seed");
    CHECK(code_equal(c_mod, c_put_t, s_mod, s_put_t),
          "put_t bytecode matches C seed");
    CHECK(code_equal(c_mod, c_put_s, s_mod, s_put_s),
          "put_s bytecode matches C seed");
    CHECK(code_equal(c_mod, c_upto, s_mod, s_upto),
          "upto bytecode matches C seed");
    CHECK(code_equal(c_mod, c_quiet, s_mod, s_quiet),
          "quiet bytecode matches C seed");
    CHECK(code_equal(c_mod, c_via_quiet, s_mod, s_via_quiet),
          "via_quiet bytecode matches C seed");
    CHECK(code_equal(c_mod, c_origin, s_mod, s_origin),
          "origin bytecode matches C seed");
    CHECK(code_equal(c_mod, c_via_o, s_mod, s_via_o),
          "via_o bytecode matches C seed");
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
               s_either ? s_either->local_count : 0, s_either ? s_either->local_count : 0);
        printf("    C pick locals=%u len=%u  src pick locals=%u len=%u\n",
               c_pick ? c_pick->local_count : 0, c_pick ? c_pick->code_length : 0,
               s_pick ? s_pick->local_count : 0, s_pick ? s_pick->code_length : 0);
        printf("    C say locals=%u len=%u  src say locals=%u len=%u\n",
               c_say ? c_say->local_count : 0, c_say ? c_say->code_length : 0,
               s_say ? s_say->local_count : 0, s_say ? s_say->code_length : 0);
        printf("    C shout locals=%u len=%u  src shout locals=%u len=%u\n",
               c_shout ? c_shout->local_count : 0, c_shout ? c_shout->code_length : 0,
               s_shout ? s_shout->local_count : 0, s_shout ? s_shout->code_length : 0);
        printf("    C mutter locals=%u len=%u  src mutter locals=%u len=%u\n",
               c_mutter ? c_mutter->local_count : 0, c_mutter ? c_mutter->code_length : 0,
               s_mutter ? s_mutter->local_count : 0, s_mutter ? s_mutter->code_length : 0);
        printf("    C prove locals=%u len=%u  src prove locals=%u len=%u\n",
               c_prove ? c_prove->local_count : 0, c_prove ? c_prove->code_length : 0,
               s_prove ? s_prove->local_count : 0, s_prove ? s_prove->code_length : 0);
        printf("    C grow locals=%u len=%u  src grow locals=%u len=%u\n",
               c_grow ? c_grow->local_count : 0, c_grow ? c_grow->code_length : 0,
               s_grow ? s_grow->local_count : 0, s_grow ? s_grow->code_length : 0);
        printf("    C has_hi locals=%u len=%u  src has_hi locals=%u len=%u\n",
               c_has ? c_has->local_count : 0, c_has ? c_has->code_length : 0,
               s_has ? s_has->local_count : 0, s_has ? s_has->code_length : 0);
        printf("    C digits locals=%u len=%u  src digits locals=%u len=%u\n",
               c_digits ? c_digits->local_count : 0, c_digits ? c_digits->code_length : 0,
               s_digits ? s_digits->local_count : 0, s_digits ? s_digits->code_length : 0);
        printf("    C names locals=%u len=%u  src names locals=%u len=%u\n",
               c_names ? c_names->local_count : 0, c_names ? c_names->code_length : 0,
               s_names ? s_names->local_count : 0, s_names ? s_names->code_length : 0);
        printf("    C head_s locals=%u len=%u  src head_s locals=%u len=%u\n",
               c_head ? c_head->local_count : 0, c_head ? c_head->code_length : 0,
               s_head ? s_head->local_count : 0, s_head ? s_head->code_length : 0);
        printf("    C same locals=%u len=%u  src same locals=%u len=%u\n",
               c_same ? c_same->local_count : 0, c_same ? c_same->code_length : 0,
               s_same ? s_same->local_count : 0, s_same ? s_same->code_length : 0);
        printf("    C diff locals=%u len=%u  src diff locals=%u len=%u\n",
               c_diff ? c_diff->local_count : 0, c_diff ? c_diff->code_length : 0,
               s_diff ? s_diff->local_count : 0, s_diff ? s_diff->code_length : 0);
        printf("    C via_at locals=%u len=%u  src via_at locals=%u len=%u\n",
               c_at ? c_at->local_count : 0, c_at ? c_at->code_length : 0,
               s_at ? s_at->local_count : 0, s_at ? s_at->code_length : 0);
        printf("    C slen locals=%u len=%u  src slen locals=%u len=%u\n",
               c_slen ? c_slen->local_count : 0, c_slen ? c_slen->code_length : 0,
               s_slen ? s_slen->local_count : 0, s_slen ? s_slen->code_length : 0);
        printf("    C slice locals=%u len=%u  src slice locals=%u len=%u\n",
               c_slice ? c_slice->local_count : 0, c_slice ? c_slice->code_length : 0,
               s_slice ? s_slice->local_count : 0, s_slice ? s_slice->code_length : 0);
        printf("    C blank_l locals=%u len=%u  src blank_l locals=%u len=%u\n",
               c_blank ? c_blank->local_count : 0, c_blank ? c_blank->code_length : 0,
               s_blank ? s_blank->local_count : 0, s_blank ? s_blank->code_length : 0);
        printf("    C grow_l locals=%u len=%u  src grow_l locals=%u len=%u\n",
               c_grow_l ? c_grow_l->local_count : 0, c_grow_l ? c_grow_l->code_length : 0,
               s_grow_l ? s_grow_l->local_count : 0, s_grow_l ? s_grow_l->code_length : 0);
        printf("    C ch locals=%u len=%u  src ch locals=%u len=%u\n",
               c_ch ? c_ch->local_count : 0, c_ch ? c_ch->code_length : 0,
               s_ch ? s_ch->local_count : 0, s_ch ? s_ch->code_length : 0);
        printf("    C blank_s locals=%u len=%u  src blank_s locals=%u len=%u\n",
               c_blank_s ? c_blank_s->local_count : 0, c_blank_s ? c_blank_s->code_length : 0,
               s_blank_s ? s_blank_s->local_count : 0, s_blank_s ? s_blank_s->code_length : 0);
        printf("    C grow_s locals=%u len=%u  src grow_s locals=%u len=%u\n",
               c_grow_s ? c_grow_s->local_count : 0, c_grow_s ? c_grow_s->code_length : 0,
               s_grow_s ? s_grow_s->local_count : 0, s_grow_s ? s_grow_s->code_length : 0);
        printf("    C get_s locals=%u len=%u  src get_s locals=%u len=%u\n",
               c_get_s ? c_get_s->local_count : 0, c_get_s ? c_get_s->code_length : 0,
               s_get_s ? s_get_s->local_count : 0, s_get_s ? s_get_s->code_length : 0);
        printf("    C blank_t locals=%u len=%u  src blank_t locals=%u len=%u\n",
               c_blank_t ? c_blank_t->local_count : 0, c_blank_t ? c_blank_t->code_length : 0,
               s_blank_t ? s_blank_t->local_count : 0, s_blank_t ? s_blank_t->code_length : 0);
        printf("    C grow_t locals=%u len=%u  src grow_t locals=%u len=%u\n",
               c_grow_t ? c_grow_t->local_count : 0, c_grow_t ? c_grow_t->code_length : 0,
               s_grow_t ? s_grow_t->local_count : 0, s_grow_t ? s_grow_t->code_length : 0);
        printf("    C get_v locals=%u len=%u  src get_v locals=%u len=%u\n",
               c_get_v ? c_get_v->local_count : 0, c_get_v ? c_get_v->code_length : 0,
               s_get_v ? s_get_v->local_count : 0, s_get_v ? s_get_v->code_length : 0);
        printf("    C grow_lex locals=%u len=%u  src grow_lex locals=%u len=%u\n",
               c_grow_lex ? c_grow_lex->local_count : 0, c_grow_lex ? c_grow_lex->code_length : 0,
               s_grow_lex ? s_grow_lex->local_count : 0, s_grow_lex ? s_grow_lex->code_length : 0);
        printf("    C has_pre locals=%u len=%u  src has_pre locals=%u len=%u\n",
               c_has_pre ? c_has_pre->local_count : 0, c_has_pre ? c_has_pre->code_length : 0,
               s_has_pre ? s_has_pre->local_count : 0, s_has_pre ? s_has_pre->code_length : 0);
        printf("    C has_suf locals=%u len=%u  src has_suf locals=%u len=%u\n",
               c_has_suf ? c_has_suf->local_count : 0, c_has_suf ? c_has_suf->code_length : 0,
               s_has_suf ? s_has_suf->local_count : 0, s_has_suf ? s_has_suf->code_length : 0);
        printf("    C put_l locals=%u len=%u  src put_l locals=%u len=%u\n",
               c_put_l ? c_put_l->local_count : 0, c_put_l ? c_put_l->code_length : 0,
               s_put_l ? s_put_l->local_count : 0, s_put_l ? s_put_l->code_length : 0);
        printf("    C put_t locals=%u len=%u  src put_t locals=%u len=%u\n",
               c_put_t ? c_put_t->local_count : 0, c_put_t ? c_put_t->code_length : 0,
               s_put_t ? s_put_t->local_count : 0, s_put_t ? s_put_t->code_length : 0);
        printf("    C put_s locals=%u len=%u  src put_s locals=%u len=%u\n",
               c_put_s ? c_put_s->local_count : 0, c_put_s ? c_put_s->code_length : 0,
               s_put_s ? s_put_s->local_count : 0, s_put_s ? s_put_s->code_length : 0);
        printf("    C upto locals=%u len=%u  src upto locals=%u len=%u\n",
               c_upto ? c_upto->local_count : 0, c_upto ? c_upto->code_length : 0,
               s_upto ? s_upto->local_count : 0, s_upto ? s_upto->code_length : 0);
        printf("    C quiet locals=%u len=%u  src quiet locals=%u len=%u\n",
               c_quiet ? c_quiet->local_count : 0, c_quiet ? c_quiet->code_length : 0,
               s_quiet ? s_quiet->local_count : 0, s_quiet ? s_quiet->code_length : 0);
        printf("    C via_quiet locals=%u len=%u  src via_quiet locals=%u len=%u\n",
               c_via_quiet ? c_via_quiet->local_count : 0, c_via_quiet ? c_via_quiet->code_length : 0,
               s_via_quiet ? s_via_quiet->local_count : 0, s_via_quiet ? s_via_quiet->code_length : 0);
        printf("    C origin locals=%u len=%u  src origin locals=%u len=%u\n",
               c_origin ? c_origin->local_count : 0, c_origin ? c_origin->code_length : 0,
               s_origin ? s_origin->local_count : 0, s_origin ? s_origin->code_length : 0);
        printf("    C via_o locals=%u len=%u  src via_o locals=%u len=%u\n",
               c_via_o ? c_via_o->local_count : 0, c_via_o ? c_via_o->code_length : 0,
               s_via_o ? s_via_o->local_count : 0, s_via_o ? s_via_o->code_length : 0);
    }

    nvm_module_free(c_mod);
    nvm_module_free(s_mod);
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
