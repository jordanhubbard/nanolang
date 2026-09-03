/*
 * verifier_types.c — abstract type interpretation over the operand stack
 *
 * The height walk proves how many values are on the stack at every reachable
 * instruction. This proves what some of them are.
 *
 * The lattice is deliberately shallow: a slot holds either a known NanoValue
 * tag or TYPE_UNKNOWN, and a merge of two different known tags widens to
 * TYPE_UNKNOWN rather than failing. Widening rather than rejecting is what
 * makes this safe to turn on: codegen legitimately produces joins where one
 * path pushed a real value and another pushed void -- a `match` arm does
 * exactly that -- so demanding equality at every merge would reject working
 * programs. What is rejected is a *definite* contradiction: an instruction
 * that requires an integer operand, given a slot known to hold a string.
 *
 * That asymmetry is the whole design. Unknown never fails, so imprecision
 * costs nothing but missed diagnostics; a known mismatch always fails, so
 * every rejection names a real error rather than a limitation of the analysis.
 *
 * Only the typed instruction families carry signatures, and every rule here
 * restates a check the VM already performs: I64_ADD traps with "requires two
 * integers", F64_ADD with "requires two floats", BOOL_AND with "requires two
 * booleans", MEM_STORE with "requires integer address and value". Proving them
 * statically turns those traps into rejections, which is the whole point --
 * but it also means a rule must reflect what the VM requires rather than what
 * looks tidy. I first gave JMP_TRUE and JMP_FALSE a boolean condition and had
 * to take it back out: the VM branches on val_truthy, so any value is a legal
 * condition, and the rule was rejecting correct programs. A rule that is not
 * true of the VM is not a type rule, it is a new language restriction smuggled
 * in through the verifier.
 *
 * Anything that loads from a local, a global or an upvalue yields
 * TYPE_UNKNOWN, because a v1 module records no type for those.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "verifier.h"
#include "isa.h"
#include "../nanovm/vm_decode.h"

/* One past the real tags, so it cannot collide with a NanoValueTag. */
#define TYPE_UNKNOWN 0xFFu

/* What an instruction requires of its operands and leaves behind. A count of
 * zero for `args` means "no constraint", which is not the same as "takes no
 * arguments" -- the height walk owns arity, this owns types. */
typedef struct {
    uint8_t args[3];      /* expected tags, topmost operand last */
    uint8_t arg_count;
    uint8_t result;       /* tag pushed, or TYPE_UNKNOWN */
    uint8_t result_count; /* 0 or 1; wider shapes are left unknown */
} TypeRule;

static bool type_rule_for(uint8_t opcode, TypeRule *out) {
    memset(out, 0, sizeof *out);
    out->result = TYPE_UNKNOWN;
    out->result_count = 0;

#define RULE0(tag)            do { out->result = (tag); out->result_count = 1; return true; } while (0)
#define RULE1(a, tag)         do { out->args[0] = (a); out->arg_count = 1; \
                                   out->result = (tag); out->result_count = 1; return true; } while (0)
#define RULE2(a, b, tag)      do { out->args[0] = (a); out->args[1] = (b); out->arg_count = 2; \
                                   out->result = (tag); out->result_count = 1; return true; } while (0)

    switch (opcode) {
    /* Constants: the only place a type enters from nothing. */
    case OP_PUSH_I64:  RULE0(TAG_INT);
    case OP_PUSH_F64:  RULE0(TAG_FLOAT);
    case OP_PUSH_BOOL: RULE0(TAG_BOOL);
    case OP_PUSH_STR:  RULE0(TAG_STRING);
    case OP_PUSH_U8:   RULE0(TAG_U8);
    case OP_PUSH_VOID: RULE0(TAG_VOID);

    /* Integer arithmetic and bitwise: integers in, integer out. */
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL:
    case OP_I64_DIV_S: case OP_I64_REM_S:
    case OP_I64_DIV_U: case OP_I64_REM_U:
    case OP_I64_AND: case OP_I64_OR: case OP_I64_XOR:
    case OP_I64_SHL: case OP_I64_SHR_S: case OP_I64_SHR_U:
        RULE2(TAG_INT, TAG_INT, TAG_INT);
    case OP_I64_NEG: case OP_I64_INVERT:
        RULE1(TAG_INT, TAG_INT);

    /* Float arithmetic. */
    case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV:
        RULE2(TAG_FLOAT, TAG_FLOAT, TAG_FLOAT);
    case OP_F64_NEG:
        RULE1(TAG_FLOAT, TAG_FLOAT);

    /* Comparisons: typed operands, boolean result. */
    case OP_I64_EQ: case OP_I64_NE:
    case OP_I64_LT_S: case OP_I64_LE_S: case OP_I64_GT_S: case OP_I64_GE_S:
    case OP_I64_LT_U: case OP_I64_LE_U: case OP_I64_GT_U: case OP_I64_GE_U:
        RULE2(TAG_INT, TAG_INT, TAG_BOOL);
    case OP_F64_EQ: case OP_F64_NE:
    case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
        RULE2(TAG_FLOAT, TAG_FLOAT, TAG_BOOL);

    /* Booleans. */
    case OP_BOOL_AND: case OP_BOOL_OR:
        RULE2(TAG_BOOL, TAG_BOOL, TAG_BOOL);
    case OP_BOOL_NOT:
        RULE1(TAG_BOOL, TAG_BOOL);

    /* Byte-addressed memory: integer address, integer value. */
    case OP_MEM_LOAD8: case OP_MEM_LOAD16:
    case OP_MEM_LOAD32: case OP_MEM_LOAD64:
        RULE1(TAG_INT, TAG_INT);
    case OP_MEM_STORE8: case OP_MEM_STORE16:
    case OP_MEM_STORE32: case OP_MEM_STORE64:
        out->args[0] = TAG_INT; out->args[1] = TAG_INT; out->arg_count = 2;
        out->result_count = 0;
        return true;

    /* Constructors whose result tag is fixed regardless of operands. */
    case OP_ARR_NEW: case OP_ARR_LITERAL:  RULE0(TAG_ARRAY);
    case OP_HM_NEW:                        RULE0(TAG_HASHMAP);
    case OP_STRUCT_NEW: case OP_STRUCT_LITERAL: RULE0(TAG_STRUCT);
    case OP_UNION_CONSTRUCT:               RULE0(TAG_UNION);
    case OP_TUPLE_NEW:                     RULE0(TAG_TUPLE);
    case OP_CLOSURE_NEW:                   RULE0(TAG_CLOSURE);
    case OP_FUNCREF:                       RULE0(TAG_FUNCTION);
    case OP_OPAQUE_NULL:                   RULE0(TAG_OPAQUE);

    /* String producers. A string operand would be checkable too, but the
     * legacy string opcodes accept more than one shape in practice, so only
     * the result is claimed. */
    case OP_STR_CONCAT: case OP_STR_SUBSTR: case OP_STR_TRIM:
    case OP_STR_TO_LOWER: case OP_STR_TO_UPPER: case OP_STR_REPLACE:
    case OP_STR_FROM_INT: case OP_STR_FROM_FLOAT:
        RULE0(TAG_STRING);
    case OP_STR_LEN: case OP_STR_CHAR_AT:
        RULE0(TAG_INT);
    case OP_STR_EQ: case OP_STR_CONTAINS:
    case OP_STR_STARTS_WITH: case OP_STR_ENDS_WITH:
        RULE0(TAG_BOOL);
    case OP_STR_SPLIT:
        RULE0(TAG_ARRAY);

    /* Casts state their own result. */
    case OP_CAST_INT:    RULE0(TAG_INT);
    case OP_CAST_FLOAT:  RULE0(TAG_FLOAT);
    case OP_CAST_BOOL:   RULE0(TAG_BOOL);
    case OP_CAST_STRING: RULE0(TAG_STRING);
    case OP_TYPE_CHECK:  RULE0(TAG_BOOL);
    case OP_OPAQUE_VALID: RULE0(TAG_BOOL);

    case OP_ARR_LEN: case OP_HM_LEN: case OP_AGG_TAG: case OP_UNION_TAG:
    case OP_ENUM_VAL:
        RULE0(TAG_INT);
    case OP_HM_HAS:
        RULE0(TAG_BOOL);
    case OP_HM_KEYS: case OP_HM_VALUES: case OP_ARR_SLICE:
        RULE0(TAG_ARRAY);

    default:
        return false;
    }
#undef RULE0
#undef RULE1
#undef RULE2
}

static uint8_t join(uint8_t a, uint8_t b) {
    return a == b ? a : TYPE_UNKNOWN;
}

NvmVerifyResult nvm_verify_function_types(const NvmModule *mod, uint32_t fn_idx,
                                          const VmDecodedFunction *decoded,
                                          uint16_t max_depth,
                                          char *error, size_t error_size) {
    NvmVerifyResult ok;
    ok.ok = true;
    ok.error_msg[0] = '\0';

    /* A deep stack would need a large snapshot per instruction. The analysis
     * is an optimisation over the height proof, not a soundness requirement,
     * so an unusually deep function simply keeps the height guarantee. */
    if (max_depth == 0 || max_depth > 256 || decoded->instruction_count == 0)
        return ok;

    const uint32_t slots = (uint32_t)max_depth;
    const uint32_t n = decoded->instruction_count;

    uint8_t *state = malloc((size_t)(n + 1) * slots);
    uint16_t *depth = malloc((size_t)(n + 1) * sizeof(*depth));
    bool *seen = calloc((size_t)(n + 1), sizeof(*seen));
    uint32_t *work = malloc((size_t)(n + 1) * sizeof(*work));
    if (!state || !depth || !seen || !work) {
        free(state); free(depth); free(seen); free(work);
        return ok;   /* height verification already passed; this is extra */
    }
    memset(state, TYPE_UNKNOWN, (size_t)(n + 1) * slots);
    memset(depth, 0, (size_t)(n + 1) * sizeof(*depth));

    uint32_t head = 0, tail = 0;
    seen[0] = true;
    work[tail++] = 0;

    NvmVerifyResult result = ok;

    while (head < tail) {
        uint32_t index = work[head++];
        if (index >= n) continue;

        const VmDecodedInstruction *di = &decoded->instructions[index];
        const DecodedInstruction *instr = &di->instruction;
        const InstructionInfo *info = isa_get_info(instr->opcode);
        if (!info) continue;

        uint8_t *in = state + (size_t)index * slots;
        uint16_t d = depth[index];

        TypeRule rule;
        bool typed = type_rule_for(instr->opcode, &rule);

        /* Check what the instruction requires of the values it consumes. */
        if (typed && rule.arg_count > 0) {
            for (uint8_t k = 0; k < rule.arg_count; k++) {
                /* args[0] is the deepest operand; the topmost is last. */
                uint32_t from_top = rule.arg_count - 1 - k;
                if (from_top >= d) break;          /* height walk owns arity */
                uint8_t have = in[d - 1 - from_top];
                uint8_t want = rule.args[k];
                if (have != TYPE_UNKNOWN && want != TYPE_UNKNOWN && have != want) {
                    result.ok = false;
                    snprintf(result.error_msg, sizeof(result.error_msg),
                             "function[%u] %s at offset %u expects %s but the "
                             "operand is %s",
                             fn_idx, info->name, di->byte_offset,
                             isa_tag_name(want), isa_tag_name(have));
                    goto done;
                }
            }
        }

        /* Model the effect on the type stack. Anything without a rule leaves
         * unknowns behind rather than a guess. */
        uint8_t next[256];
        memcpy(next, in, slots);
        uint16_t nd = d;

        int32_t pops = info->pop_count;
        int32_t pushes = info->push_count;
        if (typed && rule.arg_count > 0 && pops < 0) pops = rule.arg_count;
        if (pops < 0 || pushes < 0) {
            /* Operand-dependent effect: the height walk resolved it, but this
             * pass does not model it, so everything becomes unknown. */
            memset(next, TYPE_UNKNOWN, slots);
            nd = 0;
        } else {
            if ((int32_t)nd >= pops) nd = (uint16_t)(nd - pops);
            else nd = 0;
            for (int32_t k = 0; k < pushes && nd < slots; k++) {
                next[nd++] = (typed && rule.result_count == 1 && pushes == 1)
                               ? rule.result : TYPE_UNKNOWN;
            }
        }

        /* Successors, mirroring the height walk. */
        uint32_t successors[2];
        uint32_t successor_count = 0;
        uint8_t opcode = instr->opcode;
        if (opcode == OP_JMP || opcode == OP_JMP_TRUE || opcode == OP_JMP_FALSE
                || opcode == OP_MATCH_TAG) {
            uint32_t base = mod->functions[fn_idx].code_offset;
            uint32_t target = di->resolved_target;
            if (target >= base) {
                uint32_t rel = target - base;
                if (rel == decoded->code_size) {
                    /* function end: no successor to type */
                } else {
                    const VmDecodedInstruction *t =
                        vm_decoded_function_at(decoded, rel);
                    if (t) successors[successor_count++] =
                        (uint32_t)(t - decoded->instructions);
                }
            }
        }
        if (opcode != OP_JMP && opcode != OP_RET && opcode != OP_TAIL_CALL
                && opcode != OP_HALT && index + 1 < n) {
            successors[successor_count++] = index + 1;
        }

        for (uint32_t i = 0; i < successor_count; i++) {
            uint32_t s = successors[i];
            uint8_t *dst = state + (size_t)s * slots;
            if (!seen[s]) {
                memcpy(dst, next, slots);
                depth[s] = nd;
                seen[s] = true;
                work[tail++] = s;
                continue;
            }
            /* Merge. A disagreement widens rather than failing: two paths
             * reaching the same point with different known tags is ordinary
             * in generated code, and the value simply is not statically
             * known there. */
            bool changed = false;
            uint16_t common = depth[s] < nd ? depth[s] : nd;
            for (uint16_t k = 0; k < common; k++) {
                uint8_t merged = join(dst[k], next[k]);
                if (merged != dst[k]) { dst[k] = merged; changed = true; }
            }
            for (uint16_t k = common; k < depth[s] && k < slots; k++) {
                if (dst[k] != TYPE_UNKNOWN) { dst[k] = TYPE_UNKNOWN; changed = true; }
            }
            if (changed && tail <= n) work[tail++] = s;
        }
    }

done:
    free(state); free(depth); free(seen); free(work);
    if (!result.ok && error && error_size > 0)
        snprintf(error, error_size, "%s", result.error_msg);
    return result;
}
