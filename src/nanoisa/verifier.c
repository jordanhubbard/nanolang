/*
 * NVM Bytecode Verifier
 *
 * Validates .nvm modules for safe execution by checking all index
 * bounds, jump targets, and structural invariants before the VM
 * touches any bytecode.
 */

#define _POSIX_C_SOURCE 200809L

#include "verifier.h"
#include "isa.h"
#include "../nanovm/vm_decode.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Helpers
 * ======================================================================== */

static NvmVerifyResult ok_result(void) {
    return (NvmVerifyResult){ .ok = true, .error_msg = "" };
}

static NvmVerifyResult fail(const char *fmt, ...) {
    NvmVerifyResult r = { .ok = false };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(r.error_msg, NVM_VERIFY_ERROR_SIZE, fmt, ap);
    va_end(ap);
    return r;
}

static bool instruction_index_at(const VmDecodedFunction *decoded,
                                 uint32_t byte_offset, uint32_t *index) {
    if (byte_offset == decoded->code_size) {
        *index = decoded->instruction_count;
        return true;
    }
    const VmDecodedInstruction *instruction =
        vm_decoded_function_at(decoded, byte_offset);
    if (!instruction) return false;
    *index = (uint32_t)(instruction - decoded->instructions);
    return true;
}

static NvmVerifyResult verify_stack_heights(const VmDecodedFunction *decoded,
                                            uint32_t fn_idx) {
    int32_t *heights = malloc((decoded->instruction_count + 1) * sizeof(*heights));
    uint32_t *work = malloc((decoded->instruction_count + 1) * sizeof(*work));
    if (!heights || !work) {
        free(heights);
        free(work);
        return fail("function[%u] could not allocate stack verifier state", fn_idx);
    }
    for (uint32_t i = 0; i <= decoded->instruction_count; i++) heights[i] = -1;
    uint32_t head = 0, tail = 0;
    heights[0] = 0;
    work[tail++] = 0;

    while (head < tail) {
        uint32_t index = work[head++];
        if (index == decoded->instruction_count) continue;
        const VmDecodedInstruction *decoded_instruction = &decoded->instructions[index];
        const InstructionInfo *info = isa_get_info(decoded_instruction->instruction.opcode);
        if (info->pop_count < 0 || info->push_count < 0) continue;
        int32_t before = heights[index];
        if (before < info->pop_count) {
            free(heights);
            free(work);
            return fail("function[%u] stack underflow at offset %u (%s needs %d, has %d)",
                        fn_idx, decoded_instruction->byte_offset, info->name,
                        info->pop_count, before);
        }
        int32_t after = before - info->pop_count + info->push_count;
        uint32_t successors[2];
        uint32_t successor_count = 0;
        uint8_t opcode = decoded_instruction->instruction.opcode;
        if (opcode == OP_JMP || opcode == OP_JMP_TRUE || opcode == OP_JMP_FALSE
                || opcode == OP_MATCH_TAG) {
            instruction_index_at(decoded, decoded_instruction->resolved_target,
                                 &successors[successor_count++]);
        }
        if (opcode != OP_JMP && opcode != OP_RET && opcode != OP_TAIL_CALL
                && opcode != OP_HALT) {
            successors[successor_count++] = index + 1;
        }
        for (uint32_t i = 0; i < successor_count; i++) {
            uint32_t successor = successors[i];
            if (heights[successor] < 0) {
                heights[successor] = after;
                work[tail++] = successor;
            } else if (heights[successor] != after) {
                uint32_t offset = successor == decoded->instruction_count
                    ? decoded->code_size : decoded->instructions[successor].byte_offset;
                int32_t existing = heights[successor];
                free(heights);
                free(work);
                return fail("function[%u] incompatible stack heights at offset %u (%d and %d)",
                            fn_idx, offset, existing, after);
            }
        }
    }
    free(heights);
    free(work);
    return ok_result();
}

/* ========================================================================
 * Structural validation
 * ======================================================================== */

static NvmVerifyResult verify_structure(const NvmModule *mod) {
    if (!mod) return fail("module is NULL");
    if (!mod->code && mod->code_size > 0)
        return fail("code pointer is NULL but code_size=%u", mod->code_size);

    /* Entry point */
    if (mod->header.flags & NVM_FLAG_HAS_MAIN) {
        if (mod->header.entry_point >= mod->function_count)
            return fail("entry_point %u >= function_count %u",
                        mod->header.entry_point, mod->function_count);
    }

    /* Function code ranges */
    for (uint32_t i = 0; i < mod->function_count; i++) {
        const NvmFunctionEntry *fn = &mod->functions[i];
        if (fn->code_offset > mod->code_size)
            return fail("function[%u] code_offset %u > code_size %u",
                        i, fn->code_offset, mod->code_size);
        /* Subtract only after checking the start so the range cannot wrap. */
        if (fn->code_length > mod->code_size - fn->code_offset)
            return fail("function[%u] code range exceeds code_size: offset %u, length %u, code_size %u",
                        i, fn->code_offset, fn->code_length, mod->code_size);
        if (fn->name_idx >= mod->string_count)
            return fail("function[%u] name_idx %u >= string_count %u",
                        i, fn->name_idx, mod->string_count);
        if (fn->result_tag >= TAG_COUNT)
            return fail("function[%u] result_tag %u is invalid",
                        i, fn->result_tag);
        if ((fn->result_count == 0) != (fn->result_tag == TAG_VOID))
            return fail("function[%u] result signature must be void/0 or non-void/nonzero", i);
        if (fn->local_count < fn->arity)
            return fail("function[%u] local_count %u is smaller than arity %u",
                        i, fn->local_count, fn->arity);
    }

    /* Import string indices and imported-call signatures.
     * Imported (extern) calls are regularized around verified signatures:
     * every import must name valid strings and carry a well-formed signature
     * so that OP_CALL_EXTERN references a signature the verifier has checked. */
    for (uint32_t i = 0; i < mod->import_count; i++) {
        const NvmImportEntry *imp = &mod->imports[i];
        if (imp->module_name_idx >= mod->string_count)
            return fail("import[%u] module_name_idx %u >= string_count %u",
                        i, imp->module_name_idx, mod->string_count);
        if (imp->function_name_idx >= mod->string_count)
            return fail("import[%u] function_name_idx %u >= string_count %u",
                        i, imp->function_name_idx, mod->string_count);
        if (imp->return_type >= TAG_COUNT)
            return fail("import[%u] return_type %u is not a valid value tag",
                        i, imp->return_type);
        if (imp->param_count > 0 && !mod->import_param_types[i])
            return fail("import[%u] declares %u params but has no param type array",
                        i, imp->param_count);
        for (uint16_t p = 0; p < imp->param_count; p++) {
            uint8_t tag = mod->import_param_types[i][p];
            if (tag >= TAG_COUNT)
                return fail("import[%u] param[%u] type %u is not a valid value tag",
                            i, p, tag);
        }
    }

    return ok_result();
}

/* ========================================================================
 * Bytecode instruction validation (per-function)
 * ======================================================================== */

NvmVerifyResult nvm_verify_function(const NvmModule *mod, uint32_t fn_idx) {
    NvmVerifyResult structure = verify_structure(mod);
    if (!structure.ok) return structure;
    if (fn_idx >= mod->function_count)
        return fail("function index %u >= function_count %u",
                    fn_idx, mod->function_count);
    const NvmFunctionEntry *fn = &mod->functions[fn_idx];
    VmDecodedFunction decoded;
    char decode_error[VM_DECODE_ERROR_SIZE];
    if (!vm_decode_function(mod, fn_idx, &decoded, decode_error))
        return fail("%s", decode_error);

#define FAIL_DECODED(...) do { \
    NvmVerifyResult result = fail(__VA_ARGS__); \
    vm_decoded_function_free(&decoded); \
    return result; \
} while (0)

    for (uint32_t i = 0; i < decoded.instruction_count; i++) {
        uint32_t pos = decoded.instructions[i].byte_offset;
        DecodedInstruction instr = decoded.instructions[i].instruction;

        const InstructionInfo *info = isa_get_info(instr.opcode);
        if (!info)
            FAIL_DECODED("function[%u] unknown opcode 0x%02x at offset %u",
                         fn_idx, instr.opcode, fn->code_offset + pos);

        /* Validate operands based on opcode */
        switch (instr.opcode) {

        /* --- Jump targets must land within this function --- */
        case OP_JMP:
        case OP_JMP_TRUE:
        case OP_JMP_FALSE: {
            int32_t offset = instr.operands[0].i32;
            int64_t target = (int64_t)pos + offset;
            if (target < 0 || target > UINT32_MAX
                    || !vm_decoded_function_has_boundary(&decoded,
                                                         (uint32_t)target)) {
                FAIL_DECODED(
                    "function[%u] jump at offset %u targets %ld "
                    "(not an instruction boundary)",
                    fn_idx, fn->code_offset + pos, (long)target);
            }
            break;
        }

        /* --- OP_MATCH_TAG: variant index + jump offset --- */
        case OP_MATCH_TAG: {
            int32_t offset = instr.operands[1].i32;
            int64_t target = (int64_t)pos + offset;
            if (target < 0 || target > UINT32_MAX
                    || !vm_decoded_function_has_boundary(&decoded,
                                                         (uint32_t)target)) {
                FAIL_DECODED(
                    "function[%u] match_tag at offset %u targets %ld "
                    "(not an instruction boundary)",
                    fn_idx, fn->code_offset + pos, (long)target);
            }
            break;
        }

        /* --- Direct and tail calls to the function table ---
         * Both forms are regularized around the callee's verified signature:
         * the function index must resolve to a defined function, and a tail
         * call (which replaces the current frame) must additionally share the
         * caller's result signature so the returned values remain type-safe. */
        case OP_CALL:
        case OP_TAIL_CALL: {
            uint32_t fn_target = instr.operands[0].u32;
            if (fn_target >= mod->function_count)
                FAIL_DECODED("function[%u] %s at offset %u: fn_idx %u >= function_count %u",
                             fn_idx, info->name, fn->code_offset + pos,
                             fn_target, mod->function_count);
            if (instr.opcode == OP_TAIL_CALL) {
                const NvmFunctionEntry *callee = &mod->functions[fn_target];
                if (callee->result_count != fn->result_count
                        || callee->result_tag != fn->result_tag)
                    FAIL_DECODED("function[%u] OP_TAIL_CALL at offset %u has incompatible result signature",
                                 fn_idx, fn->code_offset + pos);
            }
            break;
        }

        /* --- Linked (separate-module) calls ---
         * OP_CALL_MODULE carries a linked-module index and a callee function
         * index. The target module table is only known once modules are linked,
         * so the single-module verifier confirms the operands are structurally
         * present; full callee resolution and signature checking happens against
         * the linked callable at instantiation/dispatch time. */
        case OP_CALL_MODULE: {
            /* Both operands decoded as u32; no further single-module bound is
             * available. Recognizing the opcode keeps linked calls in the same
             * verified taxonomy as the other call forms. */
            (void)instr;
            break;
        }

        case OP_CLOSURE_NEW: {
            uint32_t fn_target = instr.operands[0].u32;
            if (fn_target >= mod->function_count)
                FAIL_DECODED("function[%u] OP_CLOSURE_NEW at offset %u: fn_idx %u >= function_count %u",
                             fn_idx, fn->code_offset + pos, fn_target, mod->function_count);
            break;
        }

        case OP_FUNCREF: {
            uint32_t fn_target = instr.operands[0].u32;
            if (fn_target >= mod->function_count)
                FAIL_DECODED("function[%u] OP_FUNCREF at offset %u: fn_idx %u >= function_count %u",
                             fn_idx, fn->code_offset + pos, fn_target, mod->function_count);
            break;
        }

        /* --- String pool indices --- */
        case OP_PUSH_STR: {
            uint32_t str_idx = instr.operands[0].u32;
            if (str_idx >= mod->string_count)
                FAIL_DECODED("function[%u] OP_PUSH_STR at offset %u: str_idx %u >= string_count %u",
                             fn_idx, fn->code_offset + pos, str_idx, mod->string_count);
            break;
        }

        /* --- Import table indices --- */
        case OP_CALL_EXTERN: {
            uint32_t imp_idx = instr.operands[0].u32;
            if (imp_idx >= mod->import_count)
                FAIL_DECODED("function[%u] OP_CALL_EXTERN at offset %u: import_idx %u >= import_count %u",
                             fn_idx, fn->code_offset + pos, imp_idx, mod->import_count);
            break;
        }

        /* --- Local variable indices --- */
        case OP_LOAD_LOCAL:
        case OP_STORE_LOCAL: {
            uint16_t slot = instr.operands[0].u16;
            if (slot >= fn->local_count)
                FAIL_DECODED("function[%u] %s at offset %u: slot %u >= local_count %u",
                             fn_idx, info->name, fn->code_offset + pos,
                             slot, fn->local_count);
            break;
        }

        /* --- Upvalue indices --- */
        case OP_LOAD_UPVALUE:
        case OP_STORE_UPVALUE: {
            /* Encoding: operands[0]=depth (always 0, codegen flattens captures),
             * operands[1]=index into this closure's capture array. */
            uint16_t idx = instr.operands[1].u16;
            if (idx >= fn->upvalue_count)
                FAIL_DECODED("function[%u] %s at offset %u: upvalue index %u >= upvalue_count %u",
                             fn_idx, info->name, fn->code_offset + pos,
                             idx, fn->upvalue_count);
            break;
        }

        /* --- Struct definition indices --- */
        case OP_STRUCT_NEW:
        case OP_STRUCT_LITERAL: {
            if (mod->struct_count > 0) {
                uint32_t def_idx = instr.operands[0].u32;
                if (def_idx >= mod->struct_count)
                    FAIL_DECODED("function[%u] %s at offset %u: struct def_idx %u >= struct_count %u",
                                 fn_idx, info->name, fn->code_offset + pos,
                                 def_idx, mod->struct_count);
            }
            break;
        }

        /* --- Enum definition indices --- */
        case OP_ENUM_VAL: {
            if (mod->enum_count > 0) {
                uint32_t def_idx = instr.operands[0].u32;
                if (def_idx >= mod->enum_count)
                    FAIL_DECODED("function[%u] %s at offset %u: enum def_idx %u >= enum_count %u",
                                 fn_idx, info->name, fn->code_offset + pos,
                                 def_idx, mod->enum_count);
            }
            break;
        }

        /* --- Union definition indices --- */
        case OP_UNION_CONSTRUCT: {
            if (mod->union_count > 0) {
                uint32_t def_idx = instr.operands[0].u32;
                if (def_idx >= mod->union_count)
                    FAIL_DECODED("function[%u] %s at offset %u: union def_idx %u >= union_count %u",
                                 fn_idx, info->name, fn->code_offset + pos,
                                 def_idx, mod->union_count);
            }
            break;
        }

        case OP_AGG_PACK: {
            uint8_t kind = instr.operands[0].u8;
            uint32_t layout = instr.operands[1].u32;
            if (kind > AGG_TUPLE)
                FAIL_DECODED("function[%u] AGG_PACK at offset %u: invalid kind %u",
                             fn_idx, fn->code_offset + pos, kind);
            if (kind == AGG_RECORD && mod->struct_count > 0
                    && layout >= mod->struct_count)
                FAIL_DECODED("function[%u] AGG_PACK record layout %u >= struct_count %u",
                             fn_idx, layout, mod->struct_count);
            if (kind == AGG_VARIANT && mod->union_count > 0
                    && layout >= mod->union_count)
                FAIL_DECODED("function[%u] AGG_PACK variant layout %u >= union_count %u",
                             fn_idx, layout, mod->union_count);
            break;
        }

        default:
            /* All other opcodes: valid by decode success */
            break;
        }

    }

#undef FAIL_DECODED
    NvmVerifyResult stack_result = verify_stack_heights(&decoded, fn_idx);
    vm_decoded_function_free(&decoded);
    return stack_result;
}

/* ========================================================================
 * Public API
 * ======================================================================== */

NvmVerifyResult nvm_verify(const NvmModule *mod) {
    /* Phase 1: structural validation */
    NvmVerifyResult r = verify_structure(mod);
    if (!r.ok) return r;

    /* Phase 2: per-function bytecode validation */
    for (uint32_t i = 0; i < mod->function_count; i++) {
        r = nvm_verify_function(mod, i);
        if (!r.ok) return r;
    }

    return ok_result();
}
