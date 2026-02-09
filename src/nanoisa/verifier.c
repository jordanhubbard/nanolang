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
#include <stdio.h>
#include <stdarg.h>
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
        if (fn->code_offset + fn->code_length > mod->code_size)
            return fail("function[%u] code_offset+length %u > code_size %u",
                        i, fn->code_offset + fn->code_length, mod->code_size);
        if (fn->name_idx >= mod->string_count)
            return fail("function[%u] name_idx %u >= string_count %u",
                        i, fn->name_idx, mod->string_count);
    }

    /* Import string indices */
    for (uint32_t i = 0; i < mod->import_count; i++) {
        const NvmImportEntry *imp = &mod->imports[i];
        if (imp->module_name_idx >= mod->string_count)
            return fail("import[%u] module_name_idx %u >= string_count %u",
                        i, imp->module_name_idx, mod->string_count);
        if (imp->function_name_idx >= mod->string_count)
            return fail("import[%u] function_name_idx %u >= string_count %u",
                        i, imp->function_name_idx, mod->string_count);
    }

    return ok_result();
}

/* ========================================================================
 * Bytecode instruction validation (per-function)
 * ======================================================================== */

static NvmVerifyResult verify_function(const NvmModule *mod, uint32_t fn_idx) {
    const NvmFunctionEntry *fn = &mod->functions[fn_idx];
    const uint8_t *code = mod->code + fn->code_offset;
    uint32_t code_end = fn->code_length;
    uint32_t pos = 0;

    while (pos < code_end) {
        DecodedInstruction instr;
        uint32_t instr_size = isa_decode(code + pos, code_end - pos, &instr);
        if (instr_size == 0)
            return fail("function[%u] invalid instruction at offset %u",
                        fn_idx, fn->code_offset + pos);

        const InstructionInfo *info = isa_get_info(instr.opcode);
        if (!info)
            return fail("function[%u] unknown opcode 0x%02x at offset %u",
                        fn_idx, instr.opcode, fn->code_offset + pos);

        /* Validate operands based on opcode */
        switch (instr.opcode) {

        /* --- Jump targets must land within this function --- */
        case OP_JMP:
        case OP_JMP_TRUE:
        case OP_JMP_FALSE: {
            int32_t offset = instr.operands[0].i32;
            int64_t target = (int64_t)pos + offset;
            if (target < 0 || (uint32_t)target > code_end)
                return fail("function[%u] jump at offset %u targets %ld "
                            "(outside function range 0..%u)",
                            fn_idx, fn->code_offset + pos, (long)target, code_end);
            break;
        }

        /* --- OP_MATCH_TAG: variant index + jump offset --- */
        case OP_MATCH_TAG: {
            int32_t offset = instr.operands[1].i32;
            int64_t target = (int64_t)pos + offset;
            if (target < 0 || (uint32_t)target > code_end)
                return fail("function[%u] match_tag at offset %u targets %ld "
                            "(outside function range 0..%u)",
                            fn_idx, fn->code_offset + pos, (long)target, code_end);
            break;
        }

        /* --- Function table indices --- */
        case OP_CALL: {
            uint32_t fn_target = instr.operands[0].u32;
            if (fn_target >= mod->function_count)
                return fail("function[%u] OP_CALL at offset %u: fn_idx %u >= function_count %u",
                            fn_idx, fn->code_offset + pos, fn_target, mod->function_count);
            break;
        }

        case OP_CLOSURE_NEW: {
            uint32_t fn_target = instr.operands[0].u32;
            if (fn_target >= mod->function_count)
                return fail("function[%u] OP_CLOSURE_NEW at offset %u: fn_idx %u >= function_count %u",
                            fn_idx, fn->code_offset + pos, fn_target, mod->function_count);
            break;
        }

        /* --- String pool indices --- */
        case OP_PUSH_STR: {
            uint32_t str_idx = instr.operands[0].u32;
            if (str_idx >= mod->string_count)
                return fail("function[%u] OP_PUSH_STR at offset %u: str_idx %u >= string_count %u",
                            fn_idx, fn->code_offset + pos, str_idx, mod->string_count);
            break;
        }

        /* --- Import table indices --- */
        case OP_CALL_EXTERN: {
            uint32_t imp_idx = instr.operands[0].u32;
            if (imp_idx >= mod->import_count)
                return fail("function[%u] OP_CALL_EXTERN at offset %u: import_idx %u >= import_count %u",
                            fn_idx, fn->code_offset + pos, imp_idx, mod->import_count);
            break;
        }

        /* --- Local variable indices --- */
        case OP_LOAD_LOCAL:
        case OP_STORE_LOCAL: {
            uint16_t slot = instr.operands[0].u16;
            if (slot >= fn->local_count)
                return fail("function[%u] %s at offset %u: slot %u >= local_count %u",
                            fn_idx, info->name, fn->code_offset + pos,
                            slot, fn->local_count);
            break;
        }

        /* --- Upvalue indices --- */
        case OP_LOAD_UPVALUE:
        case OP_STORE_UPVALUE: {
            uint16_t slot = instr.operands[0].u16;
            if (slot >= fn->upvalue_count)
                return fail("function[%u] %s at offset %u: slot %u >= upvalue_count %u",
                            fn_idx, info->name, fn->code_offset + pos,
                            slot, fn->upvalue_count);
            break;
        }

        /* --- Struct definition indices --- */
        case OP_STRUCT_NEW:
        case OP_STRUCT_LITERAL: {
            /* Struct indices are not tracked in module — skip for now.
             * TODO: add struct_count to NvmModule and validate here. */
            break;
        }

        default:
            /* All other opcodes: valid by decode success */
            break;
        }

        pos += instr_size;
    }

    return ok_result();
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
        r = verify_function(mod, i);
        if (!r.ok) return r;
    }

    return ok_result();
}
