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
#include "../nanovm/vm.h"
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

/* Walks every reachable instruction, checking that the stack height agrees at
 * every join and never underflows. `out_max_stack`, when given, also receives
 * the deepest height reached -- the value a v2 module declares and a loader
 * confirms. It is only written on success: there is no honest maximum for code
 * that does not verify. */
static NvmVerifyResult verify_stack_heights(const NvmModule *mod,
                                            const VmDecodedFunction *decoded,
                                            uint32_t fn_idx,
                                            uint16_t *out_max_stack) {
    int32_t *heights = malloc((decoded->instruction_count + 1) * sizeof(*heights));
    uint32_t *work = malloc((decoded->instruction_count + 1) * sizeof(*work));
    if (!heights || !work) {
        free(heights);
        free(work);
        return fail("function[%u] could not allocate stack verifier state", fn_idx);
    }
    int32_t max_depth = 0;
    for (uint32_t i = 0; i <= decoded->instruction_count; i++) heights[i] = -1;
    uint32_t head = 0, tail = 0;
    heights[0] = 0;
    work[tail++] = 0;

    while (head < tail) {
        uint32_t index = work[head++];
        if (index == decoded->instruction_count) continue;
        const VmDecodedInstruction *decoded_instruction = &decoded->instructions[index];
        const DecodedInstruction *instruction = &decoded_instruction->instruction;
        const InstructionInfo *info = isa_get_info(instruction->opcode);
        int32_t pop_count = info->pop_count;
        int32_t push_count = info->push_count;
        switch (instruction->opcode) {
        case OP_CALL:
        case OP_TAIL_CALL: {
            const NvmFunctionEntry *callee =
                &mod->functions[instruction->operands[0].u32];
            pop_count = callee->arity;
            push_count = callee->result_count;
            break;
        }
        case OP_CALL_EXTERN: {
            const NvmImportEntry *import =
                &mod->imports[instruction->operands[0].u32];
            pop_count = import->param_count;
            push_count = import->return_type == TAG_VOID ? 0 : 1;
            break;
        }
        case OP_CLOSURE_NEW:
            pop_count = instruction->operands[1].u16;
            push_count = 1;
            break;
        case OP_STRUCT_LITERAL:
            pop_count = instruction->operands[1].u16;
            push_count = 1;
            break;
        case OP_UNION_CONSTRUCT:
            pop_count = instruction->operands[2].u16;
            push_count = 1;
            break;
        case OP_TUPLE_NEW:
            pop_count = instruction->operands[0].u16;
            push_count = 1;
            break;
        case OP_AGG_PACK:
            pop_count = instruction->operands[3].u16;
            push_count = 1;
            break;
        case OP_ARR_LITERAL:
            pop_count = instruction->operands[1].u16;
            push_count = 1;
            break;
        case OP_CALL_INDIRECT:
            /* The callee is only known at run time, so its shape is encoded:
             * the arguments plus the callable itself come off, the declared
             * results go on. The VM checks the callee against this. */
            pop_count = (int32_t)instruction->operands[0].u16 + 1;
            push_count = instruction->operands[1].u16;
            break;
        case OP_CALL_MODULE:
            /* The callee lives in another module, so its shape is encoded too.
             * That makes a module's stack discipline provable before linking,
             * and gives nvm_verify_linked a declared shape to check the real
             * callee against -- a link-time signature mismatch that used to be
             * invisible. */
            pop_count = instruction->operands[2].u16;
            push_count = instruction->operands[3].u16;
            break;
        case OP_RET:
            /* A return consumes this function's declared results and leaves
             * nothing: the frame goes away with it. */
            pop_count = mod->functions[fn_idx].result_count;
            push_count = 0;
            break;
        /* PICK and ROLL neither add nor remove anything below the depth they
         * name, but they require it to exist. Charging the depth to both sides
         * expresses that with the machinery already here: the underflow check
         * sees the real requirement and the net effect stays correct. */
        case OP_PICK:
            pop_count = instruction->operands[0].u16 + 1;
            push_count = instruction->operands[0].u16 + 2;
            break;
        case OP_ROLL:
            pop_count = instruction->operands[0].u16 + 1;
            push_count = instruction->operands[0].u16 + 1;
            break;
        default:
            break;
        }

        /* Fail closed. An instruction whose stack effect is unknown used to be
         * skipped, which also skipped enqueueing its successors -- so the walk
         * stopped there and every instruction after it went unverified while
         * nvm_verify still returned ok. Absence of data must not read as
         * proof. See issue #212. */
        if (pop_count < 0 || push_count < 0)
            return fail("function[%u] %s at offset %u has no known stack effect",
                        fn_idx, info->name, decoded_instruction->byte_offset);
        int32_t before = heights[index];
        if (before < pop_count) {
            free(heights);
            free(work);
            return fail("function[%u] stack underflow at offset %u (%s needs %d, has %d)",
                        fn_idx, decoded_instruction->byte_offset, info->name,
                        pop_count, before);
        }
        int32_t after = before - pop_count + push_count;
        /* The deepest point is often neither the first height nor the last --
         * three values live before a binary op leaves two -- so both sides of
         * every instruction count. */
        if (before > max_depth) max_depth = before;
        if (after > max_depth) max_depth = after;
        uint32_t successors[2];
        uint32_t successor_count = 0;
        uint8_t opcode = instruction->opcode;
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
    if (out_max_stack) {
        if (max_depth > UINT16_MAX)
            return fail("function[%u] maximum operand depth %d exceeds %u",
                        fn_idx, max_depth, (unsigned)UINT16_MAX);
        *out_max_stack = (uint16_t)max_depth;
    }
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

        /* Empty functions own no bytes. Non-empty function ranges must be
         * disjoint, but they may be adjacent or appear out of table order. */
        if (fn->code_length != 0) {
            uint32_t fn_end = fn->code_offset + fn->code_length;
            for (uint32_t j = 0; j < i; j++) {
                const NvmFunctionEntry *other = &mod->functions[j];
                if (other->code_length == 0) continue;

                /* The earlier iteration proved this range cannot wrap. */
                uint32_t other_end = other->code_offset + other->code_length;
                if (fn->code_offset < other_end && other->code_offset < fn_end)
                    return fail("function[%u] code range overlaps function[%u]", i, j);
            }
        }
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
        if (imp->param_count > NANO_MAX_FFI_ARGS)
            return fail("import[%u] param_count %u exceeds the foreign-call "
                        "argument limit of %u",
                        i, imp->param_count, (unsigned)NANO_MAX_FFI_ARGS);
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

static NvmVerifyResult verify_function_impl(const NvmModule *mod, uint32_t fn_idx,
                                           const NvmModule *const *linked_modules,
                                           uint32_t linked_count,
                                           uint16_t *out_max_stack) {
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
         * index. The target module table is only known once modules are linked.
         * When a linked-module table is supplied (nvm_verify_linked) the operand
         * pair is fully resolved: the module index must be in range, the linked
         * module must be present, and the callee function index must be within
         * that module. Without a table (bare nvm_verify) the operands are
         * accepted structurally and resolved at instantiation/dispatch time. */
        case OP_CALL_MODULE: {
            uint32_t mod_target = instr.operands[0].u32;
            uint32_t fn_target  = instr.operands[1].u32;
            if (linked_count > 0) {
                if (mod_target >= linked_count)
                    FAIL_DECODED("function[%u] OP_CALL_MODULE at offset %u: module_idx %u >= linked_count %u",
                                 fn_idx, fn->code_offset + pos, mod_target, linked_count);
                const NvmModule *target = linked_modules[mod_target];
                if (!target)
                    FAIL_DECODED("function[%u] OP_CALL_MODULE at offset %u: linked module %u is unresolved",
                                 fn_idx, fn->code_offset + pos, mod_target);
                if (fn_target >= target->function_count)
                    FAIL_DECODED("function[%u] OP_CALL_MODULE at offset %u: fn_idx %u >= linked function_count %u",
                                 fn_idx, fn->code_offset + pos, fn_target, target->function_count);
                /* The encoded shape is what this module's stack discipline was
                 * proven against. If the real callee disagrees, linking has
                 * silently changed the meaning of the call. */
                const NvmFunctionEntry *callee = &target->functions[fn_target];
                if (instr.operands[2].u16 != callee->arity)
                    FAIL_DECODED("function[%u] OP_CALL_MODULE at offset %u declares arity %u but linked callee takes %u",
                                 fn_idx, fn->code_offset + pos,
                                 instr.operands[2].u16, callee->arity);
                if (instr.operands[3].u16 != callee->result_count)
                    FAIL_DECODED("function[%u] OP_CALL_MODULE at offset %u declares %u results but linked callee returns %u",
                                 fn_idx, fn->code_offset + pos,
                                 instr.operands[3].u16, callee->result_count);
            }
            break;
        }

        case OP_CLOSURE_NEW: {
            uint32_t fn_target = instr.operands[0].u32;
            if (fn_target >= mod->function_count)
                FAIL_DECODED("function[%u] OP_CLOSURE_NEW at offset %u: fn_idx %u >= function_count %u",
                             fn_idx, fn->code_offset + pos, fn_target, mod->function_count);
            if (instr.operands[1].u16 != mod->functions[fn_target].upvalue_count)
                FAIL_DECODED("function[%u] OP_CLOSURE_NEW at offset %u: capture_count %u does not match upvalue_count %u",
                             fn_idx, fn->code_offset + pos, instr.operands[1].u16,
                             mod->functions[fn_target].upvalue_count);
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

        /* --- Global indices use the VM's fixed global table --- */
        case OP_LOAD_GLOBAL:
        case OP_STORE_GLOBAL: {
            uint32_t idx = instr.operands[0].u32;
            if (idx >= VM_MAX_GLOBALS)
                FAIL_DECODED("function[%u] %s at offset %u: global index %u >= limit %u",
                             fn_idx, info->name, fn->code_offset + pos,
                             idx, VM_MAX_GLOBALS);
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
            if (instr.operands[0].u16 != 0)
                FAIL_DECODED("function[%u] %s at offset %u: upvalue depth must be zero",
                             fn_idx, info->name, fn->code_offset + pos);
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

        /* --- Type-tag operands (element/key/value/expected tags) ---
         * ARR_NEW, HM_NEW, and TYPE_CHECK each carry NanoValueTag byte(s) that
         * name a runtime value type. A tag outside [0, TAG_COUNT) would let the
         * VM index type tables out of range, so every family member is checked
         * rather than trusting decode success. */
        case OP_ARR_NEW:
        case OP_TYPE_CHECK: {
            uint8_t tag = instr.operands[0].u8;
            if (tag >= TAG_COUNT)
                FAIL_DECODED("function[%u] %s at offset %u: type tag %u >= TAG_COUNT %u",
                             fn_idx, info->name, fn->code_offset + pos,
                             tag, (unsigned)TAG_COUNT);            break;
        }

        case OP_HM_NEW: {
            uint8_t key_tag = instr.operands[0].u8;
            uint8_t val_tag = instr.operands[1].u8;
            if (key_tag >= TAG_COUNT)
                FAIL_DECODED("function[%u] OP_HM_NEW at offset %u: key tag %u >= TAG_COUNT %u",
                             fn_idx, fn->code_offset + pos, key_tag, (unsigned)TAG_COUNT);
            if (val_tag >= TAG_COUNT)
                FAIL_DECODED("function[%u] OP_HM_NEW at offset %u: value tag %u >= TAG_COUNT %u",
                             fn_idx, fn->code_offset + pos, val_tag, (unsigned)TAG_COUNT);
            break;
        }

        case OP_ARR_LITERAL: {
            uint8_t tag = instr.operands[0].u8;
            if (tag >= TAG_COUNT)
                FAIL_DECODED("function[%u] OP_ARR_LITERAL at offset %u: element tag %u >= TAG_COUNT %u",
                             fn_idx, fn->code_offset + pos, tag, (unsigned)TAG_COUNT);
            break;
        }

        default: {
            /* Exhaustive opcode-family closure.
             *
             * Every opcode that carries an operand referencing a table, layout,
             * type tag, or branch target is validated by an explicit case above.
             * The families that remain reach this point and are safe once the
             * instruction has decoded, because their operands are either:
             *
             *   - self-describing immediates whose full value range is legal
             *     (PUSH_I64/PUSH_F64/PUSH_BOOL/PUSH_U8, DEBUG_LINE), or
             *   - fixed stack-machine operations with no table operand
             *     (arithmetic, comparison, logic, casts, string/array/hashmap/
             *     tuple algorithms, memory loads/stores, GC scopes, RET/HALT/
             *     PRINT/ASSERT), or
             *   - purely stack-relative depth operands (PICK, ROLL) and
             *     field/index accessors (STRUCT_GET/SET, UNION_FIELD, TUPLE_GET,
             *     AGG_GET/AGG_SET) whose bound is the runtime aggregate rather
             *     than a module table, so they are enforced dynamically, or
             *   - LOAD_GLOBAL/STORE_GLOBAL, whose slot count is derived from the
             *     declarations the module itself references (globals are sized
             *     dynamically and carry no separate declared bound in a single
             *     module), so no static ceiling exists to compare against.
             *
             * A default failure would be wrong for these, but a silent
             * accept-all would also be wrong. The guard below keeps the family
             * closure honest: only opcodes within the primary plane may reach
             * here, so a value at or above the plane limit (the extension-prefix
             * escape byte) is rejected rather than accepted unchecked. */
            if (instr.opcode >= NANOISA_PRIMARY_OPCODE_LIMIT)
                FAIL_DECODED("function[%u] opcode 0x%02x at offset %u is outside the primary plane",
                             fn_idx, instr.opcode, fn->code_offset + pos);
            break;
        }
        }

    }

#undef FAIL_DECODED
    NvmVerifyResult stack_result =
        verify_stack_heights(mod, &decoded, fn_idx, out_max_stack);
    vm_decoded_function_free(&decoded);
    return stack_result;
}

NvmVerifyResult nvm_verify_function(const NvmModule *mod, uint32_t fn_idx) {
    return verify_function_impl(mod, fn_idx, NULL, 0, NULL);
}

NvmVerifyResult nvm_verify_function_max_stack(const NvmModule *mod,
                                              uint32_t fn_idx,
                                              uint16_t *out_max_stack) {
    return verify_function_impl(mod, fn_idx, NULL, 0, out_max_stack);
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
        r = verify_function_impl(mod, i, NULL, 0, NULL);
        if (!r.ok) return r;
    }

    return ok_result();
}

NvmVerifyResult nvm_verify_linked(const NvmModule *mod,
                                  const NvmModule *const *linked_modules,
                                  uint32_t linked_count) {
    if (linked_count > 0 && !linked_modules)
        return fail("linked_count %u but linked_modules table is NULL", linked_count);

    /* Phase 1: structural validation */
    NvmVerifyResult r = verify_structure(mod);
    if (!r.ok) return r;

    /* Phase 2: per-function validation, resolving OP_CALL_MODULE against the
     * supplied linked-module table so cross-module call operands are bounded. */
    for (uint32_t i = 0; i < mod->function_count; i++) {
        r = verify_function_impl(mod, i, linked_modules, linked_count, NULL);
        if (!r.ok) return r;
    }

    return ok_result();
}
