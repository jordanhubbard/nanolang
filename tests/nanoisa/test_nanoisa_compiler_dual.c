/*
 * Compiler-subset dual: every function in a C-seed .nvm must exist in the
 * src_nano .nasm with matching bytecode. Same comparison rules as the Cut A
 * pin (string / call / extern names, not pool or slot order). Layout and
 * enum def indices are intern order, the way string-pool indices are.
 * LOAD_LOCAL slots are the claim. Debug is not the claim.
 *
 * argv[1] = C-seed .nvm (nano_virt --emit-nvm --strip-debug)
 * argv[2] = src_nano .nasm (bin/nanoisa_emit)
 */

#include "assembler.h"
#include "isa.h"
#include "nanoisa.h"
#include "nvm_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const NvmFunctionEntry *fn_by_name(const NvmModule *m, const char *name) {
    uint32_t i;
    for (i = 0; i < m->function_count; i++) {
        const char *n = nvm_get_string(m, m->functions[i].name_idx);
        if (n && strcmp(n, name) == 0) return &m->functions[i];
    }
    return NULL;
}

static const char *fn_name(const NvmModule *m, const NvmFunctionEntry *fn) {
    const char *n;
    if (!m || !fn) return "?";
    n = nvm_get_string(m, fn->name_idx);
    return n ? n : "?";
}

static const char *op_name(uint8_t opcode) {
    const InstructionInfo *info = isa_get_info(opcode);
    return info && info->name ? info->name : "?";
}

static int call_names_equal(const NvmModule *a, uint32_t ia_idx,
                            const NvmModule *b, uint32_t ib_idx) {
    const char *ca;
    const char *cb;
    if (ia_idx >= a->function_count || ib_idx >= b->function_count) return 0;
    ca = nvm_get_string(a, a->functions[ia_idx].name_idx);
    cb = nvm_get_string(b, b->functions[ib_idx].name_idx);
    if (!ca || !cb || strcmp(ca, cb) != 0) return 0;
    return 1;
}

static int instr_equal(const NvmModule *a, const DecodedInstruction *ia, uint32_t na,
                       const NvmModule *b, const DecodedInstruction *ib, uint32_t nb,
                       const uint8_t *bytes_a, const uint8_t *bytes_b) {
    if (ia->opcode != ib->opcode) return 0;
    if (ia->opcode == OP_PUSH_STR) {
        const char *sa = nvm_get_string(a, ia->operands[0].u32);
        const char *sb = nvm_get_string(b, ib->operands[0].u32);
        if (!sa || !sb || strcmp(sa, sb) != 0) return 0;
        return 1;
    }
    if (ia->opcode == OP_CALL || ia->opcode == OP_TAIL_CALL ||
        ia->opcode == OP_FUNCREF) {
        return call_names_equal(a, ia->operands[0].u32, b, ib->operands[0].u32);
    }
    if (ia->opcode == OP_CALL_EXTERN) {
        uint32_t ia_idx = ia->operands[0].u32;
        uint32_t ib_idx = ib->operands[0].u32;
        const char *ca;
        const char *cb;
        if (ia_idx >= a->import_count || ib_idx >= b->import_count) return 0;
        ca = nvm_get_string(a, a->imports[ia_idx].function_name_idx);
        cb = nvm_get_string(b, b->imports[ib_idx].function_name_idx);
        if (!ca || !cb || strcmp(ca, cb) != 0) return 0;
        return 1;
    }
    if (ia->opcode == OP_AGG_PACK) {
        if (ia->operands[0].u8 != ib->operands[0].u8) return 0;
        if (ia->operands[2].u16 != ib->operands[2].u16) return 0;
        if (ia->operands[3].u16 != ib->operands[3].u16) return 0;
        return 1;
    }
    if (ia->opcode == OP_ENUM_VAL) {
        if (ia->operands[1].u16 != ib->operands[1].u16) return 0;
        return 1;
    }
    if (ia->opcode == OP_STRUCT_NEW) {
        return 1;
    }
    if (ia->opcode == OP_STRUCT_LITERAL) {
        if (ia->operands[1].u16 != ib->operands[1].u16) return 0;
        return 1;
    }
    if (ia->opcode == OP_UNION_CONSTRUCT) {
        if (ia->operands[1].u16 != ib->operands[1].u16) return 0;
        if (ia->operands[2].u16 != ib->operands[2].u16) return 0;
        return 1;
    }
    if (na != nb || memcmp(bytes_a, bytes_b, na) != 0) return 0;
    return 1;
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
        if (na == 0 || nb == 0) return 0;
        if (!instr_equal(a, &ia, na, b, &ib, nb,
                         a->code + fa->code_offset + pa,
                         b->code + fb->code_offset + pb)) {
            return 0;
        }
        pa += na;
        pb += nb;
    }
    return pa == fa->code_length && pb == fb->code_length;
}

static void print_decoded(const char *side, size_t off, const DecodedInstruction *d) {
    uint8_t i;
    printf("    %s @%u %s", side, (unsigned)off, op_name(d->opcode));
    for (i = 0; i < d->operand_count; i++) {
        switch (d->operand_types[i]) {
        case OPERAND_U8:
            printf(" %u", (unsigned)d->operands[i].u8);
            break;
        case OPERAND_U16:
            printf(" %u", (unsigned)d->operands[i].u16);
            break;
        case OPERAND_U32:
            printf(" %u", (unsigned)d->operands[i].u32);
            break;
        case OPERAND_I32:
            printf(" %d", (int)d->operands[i].i32);
            break;
        case OPERAND_I64:
            printf(" %lld", (long long)d->operands[i].i64);
            break;
        case OPERAND_F64:
            printf(" %g", d->operands[i].f64);
            break;
        default:
            printf(" %u", (unsigned)d->operands[i].u32);
            break;
        }
    }
    printf("\n");
}

static void describe_mismatch(const NvmModule *a, const NvmFunctionEntry *fa,
                              const NvmModule *b, const NvmFunctionEntry *fb) {
    size_t pa = 0;
    size_t pb = 0;
    printf("    %s: C arity=%u locals=%u tag=%u len=%u  src arity=%u locals=%u tag=%u len=%u\n",
           fn_name(a, fa),
           fa ? fa->arity : 0, fa ? fa->local_count : 0,
           fa ? fa->result_tag : 0, fa ? fa->code_length : 0,
           fb ? fb->arity : 0, fb ? fb->local_count : 0,
           fb ? fb->result_tag : 0, fb ? fb->code_length : 0);
    if (!fa || !fb) return;
    if (fa->arity != fb->arity || fa->local_count != fb->local_count ||
        fa->result_tag != fb->result_tag || fa->result_count != fb->result_count) {
        printf("    header mismatch\n");
        return;
    }
    while (pa < fa->code_length && pb < fb->code_length) {
        DecodedInstruction ia;
        DecodedInstruction ib;
        uint32_t na = isa_decode(a->code + fa->code_offset + pa,
                                 fa->code_length - pa, &ia);
        uint32_t nb = isa_decode(b->code + fb->code_offset + pb,
                                 fb->code_length - pb, &ib);
        if (na == 0 || nb == 0) {
            printf("    decode failed C @%u src @%u\n", (unsigned)pa, (unsigned)pb);
            return;
        }
        if (!instr_equal(a, &ia, na, b, &ib, nb,
                         a->code + fa->code_offset + pa,
                         b->code + fb->code_offset + pb)) {
            print_decoded("C", pa, &ia);
            print_decoded("src", pb, &ib);
            return;
        }
        pa += na;
        pb += nb;
    }
    if (pa != fa->code_length || pb != fb->code_length) {
        printf("    length mismatch after matching instructions C@%u/%u src@%u/%u\n",
               (unsigned)pa, fa->code_length, (unsigned)pb, fb->code_length);
    } else {
        printf("    header or named-operand mismatch (same instructions)\n");
    }
}

int main(int argc, char **argv) {
    NanoisaErr err;
    AsmResult asm_err;
    NvmModule *c_mod;
    NvmModule *s_mod;
    uint32_t i;
    int fail = 0;
    int matched = 0;

    if (argc < 3) {
        fprintf(stderr, "usage: test_nanoisa_compiler_dual <c.nvm> <src.nasm>\n");
        return 1;
    }

    memset(&err, 0, sizeof err);
    c_mod = nanoisa_load_file(argv[1], &err);
    if (!c_mod) {
        printf("FAIL: C-seed module loads (%s)\n", err.message);
        return 1;
    }
    nvm_strip_debug_info(c_mod);

    memset(&asm_err, 0, sizeof asm_err);
    s_mod = asm_assemble_file(argv[2], &asm_err);
    if (!s_mod) {
        printf("FAIL: src_nano nasm assembles (%s line %u)\n",
               asm_err.message, asm_err.line);
        nvm_module_free(c_mod);
        return 1;
    }

    if (c_mod->function_count != s_mod->function_count) {
        printf("FAIL: function_count C=%u src=%u\n",
               c_mod->function_count, s_mod->function_count);
        fail++;
    }

    for (i = 0; i < c_mod->function_count; i++) {
        const char *name = nvm_get_string(c_mod, c_mod->functions[i].name_idx);
        const NvmFunctionEntry *sf;
        if (!name) {
            printf("FAIL: C function %u has no name\n", i);
            fail++;
            continue;
        }
        sf = fn_by_name(s_mod, name);
        if (!sf) {
            printf("FAIL: src missing %s\n", name);
            fail++;
            continue;
        }
        if (!code_equal(c_mod, &c_mod->functions[i], s_mod, sf)) {
            printf("FAIL: %s bytecode matches C seed\n", name);
            describe_mismatch(c_mod, &c_mod->functions[i], s_mod, sf);
            fail++;
        } else {
            matched++;
        }
    }

    for (i = 0; i < s_mod->function_count; i++) {
        const char *name = nvm_get_string(s_mod, s_mod->functions[i].name_idx);
        if (!name) continue;
        if (!fn_by_name(c_mod, name)) {
            printf("FAIL: C missing %s\n", name);
            fail++;
        }
    }

    printf("compiler dual: %d matched, %d failed (C %u fn, src %u fn)\n",
           matched, fail, c_mod->function_count, s_mod->function_count);
    nvm_module_free(c_mod);
    nvm_module_free(s_mod);
    return fail == 0 ? 0 : 1;
}
