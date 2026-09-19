/*
 * NVM Bytecode Verifier
 *
 * Validates .nvm modules for safe execution by checking all index
 * bounds, jump targets, and structural invariants before the VM
 * touches any bytecode.
 */

#define _POSIX_C_SOURCE 200809L

#include "service_bindings_module.h"
#include "verifier.h"
#include "managed_array_shapes.h"
#include "managed_record_shapes.h"
#include "passive.h"
#include "retained_layouts.h"
#include "ownership_contracts.h"
#include "affine_bytecode.h"
#include "affine_state.h"
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
    /* Ownership balance alongside stack height: GC_RETAIN and GC_RELEASE
     * adjust an object's reference count without touching the operand stack,
     * so height says nothing about whether they pair up. An unbalanced pair
     * is a leak or a premature free -- both invisible until much later -- and
     * both are decidable here for the same cost as the height walk. Nothing
     * emits these today; the assembler accepts them, which is exactly the
     * path that has no other check. */
    int32_t *owed = malloc((decoded->instruction_count + 1) * sizeof(*owed));
    if (!owed) {
        free(heights);
        free(work);
        return fail("function[%u] could not allocate stack verifier state", fn_idx);
    }
    for (uint32_t i = 0; i <= decoded->instruction_count; i++) owed[i] = -1;
    owed[0] = 0;

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
        case OP_PERFORM:
            pop_count = instruction->operands[1].u16;
            push_count = 1;
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
        if (pop_count < 0 || push_count < 0) {
            free(heights);
            free(work);
            free(owed);
            return fail("function[%u] %s at offset %u has no known stack effect",
                        fn_idx, info->name, decoded_instruction->byte_offset);
        }
        int32_t before = heights[index];

        /* Return shape: a return must leave exactly the results the function
         * declares. Merely having enough is not the same thing -- a function
         * declaring one result and returning with two leaves a value on the
         * caller's stack that the caller was never verified against. */
        if (instruction->opcode == OP_RET
                && before != (int32_t)mod->functions[fn_idx].result_count) {
            free(heights);
            free(work);
            free(owed);
            return fail("function[%u] returns with %d values at offset %u but declares %u",
                        fn_idx, before, decoded_instruction->byte_offset,
                        mod->functions[fn_idx].result_count);
        }

        if (instruction->opcode == OP_HANDLER_PUSH && owed[index] != 0) {
            free(heights); free(work); free(owed);
            return fail("I cannot install a nonlocal-return handler across outstanding explicit retains.");
        }
        if (instruction->opcode == OP_EFFECT_RESUME && (before != 1 || owed[index] != 0)) {
            free(heights); free(work); free(owed);
            return fail("I require one result and balanced ownership when resuming an effect.");
        }
        if (before < pop_count) {
            free(heights);
            free(work);
            free(owed);
            return fail("function[%u] stack underflow at offset %u (%s needs %d, has %d)",
                        fn_idx, decoded_instruction->byte_offset, info->name,
                        pop_count, before);
        }
        int32_t after = before - pop_count + push_count;

        /* Ownership balance across this instruction. */
        int32_t owed_before = owed[index];
        int32_t owed_after = owed_before;
        if (instruction->opcode == OP_GC_RETAIN) owed_after++;
        else if (instruction->opcode == OP_GC_RELEASE) owed_after--;
        if (owed_after < 0) {
            free(heights);
            free(work);
            free(owed);
            return fail("function[%u] releases a reference it does not hold at offset %u",
                        fn_idx, decoded_instruction->byte_offset);
        }
        if (instruction->opcode == OP_RET && owed_before != 0) {
            free(heights);
            free(work);
            free(owed);
            return fail("function[%u] returns at offset %u still holding %d retained reference(s)",
                        fn_idx, decoded_instruction->byte_offset, owed_before);
        }
        /* The deepest point is often neither the first height nor the last --
         * three values live before a binary op leaves two -- so both sides of
         * every instruction count. */
        if (before > max_depth) max_depth = before;
        if (after > max_depth) max_depth = after;
        uint32_t successors[2];
        uint32_t successor_count = 0;
        uint8_t opcode = instruction->opcode;
        if (opcode == OP_JMP || opcode == OP_JMP_TRUE || opcode == OP_JMP_FALSE
                || opcode == OP_MATCH_TAG || opcode == OP_HANDLER_PUSH) {
            /* resolved_target is an offset into the whole CODE section, while
             * the instruction-index table is per function, so the function's
             * base has to come off first. Without that, every branch in a
             * function at a nonzero code offset looked up an out-of-range
             * offset -- and because the old code incremented successor_count
             * whether or not the lookup succeeded, it then walked an
             * uninitialized successor. Function 0 starts at offset 0, so it
             * was the only one where the two agreed, which is why this stayed
             * hidden until stack heights actually propagated past the first
             * branch. */
            uint32_t base = mod->functions[fn_idx].code_offset;
            uint32_t target = decoded_instruction->resolved_target;
            uint32_t target_index;
            if (target < base
                    || !instruction_index_at(decoded, target - base, &target_index)) {
                free(heights);
                free(work);
                free(owed);
                return fail("function[%u] branch at offset %u targets %u, which is "
                            "not an instruction boundary in this function",
                            fn_idx, decoded_instruction->byte_offset, target);
            }
            successors[successor_count++] = target_index;
        }
        if (opcode != OP_JMP && opcode != OP_RET && opcode != OP_TAIL_CALL
                && opcode != OP_HALT && opcode != OP_EFFECT_RESUME) {
            successors[successor_count++] = index + 1;
        }
        for (uint32_t i = 0; i < successor_count; i++) {
            uint32_t successor = successors[i];
            int32_t successor_height = opcode == OP_HANDLER_PUSH && i == 0 ? 0 : after;
            int32_t successor_owed = opcode == OP_HANDLER_PUSH && i == 0 ? 0 : owed_after;
            /* Reaching the end of a function's code is an implicit return,
             * not a fall-through into whatever follows: the VM checks the
             * result count and tags there exactly as OP_RET does. So the rule
             * is not "a path must end in RET" -- it is that every way out
             * leaves what the function declares. Checking it here is what
             * makes the implicit exit as verified as the explicit one. */
            if (successor == decoded->instruction_count) {
                if (owed_after != 0) {
                    free(heights);
                    free(work);
                    free(owed);
                    return fail("function[%u] reaches its end after offset %u still holding "
                                "%d retained reference(s)",
                                fn_idx, decoded_instruction->byte_offset, owed_after);
                }
                if (after != (int32_t)mod->functions[fn_idx].result_count) {
                    free(heights);
                    free(work);
                    free(owed);
                    return fail("function[%u] reaches its end after offset %u with %d values "
                                "but declares %u",
                                fn_idx, decoded_instruction->byte_offset, after,
                                mod->functions[fn_idx].result_count);
                }
                continue;
            }
            if (heights[successor] < 0) {
                heights[successor] = successor_height;
                owed[successor] = successor_owed;
                work[tail++] = successor;
            } else if (owed[successor] != successor_owed) {
                uint32_t offset = decoded->instructions[successor].byte_offset;
                int32_t existing = owed[successor];
                free(heights);
                free(work);
                free(owed);
                return fail("function[%u] incompatible ownership balance at offset %u (%d and %d)",
                            fn_idx, offset, existing, successor_owed);
            } else if (heights[successor] != successor_height) {
                uint32_t offset = successor == decoded->instruction_count
                    ? decoded->code_size : decoded->instructions[successor].byte_offset;
                int32_t existing = heights[successor];
                free(heights);
                free(work);
                free(owed);
                return fail("function[%u] incompatible stack heights at offset %u (%d and %d)",
                            fn_idx, offset, existing, successor_height);
            }
        }
    }
    free(heights);
    free(work);
    free(owed);
    /* Frame depth: a call reserves this function's locals plus the operand
     * depth just proven. Both are u16 individually, so only their sum can
     * exceed what a frame can address -- and a frame that wrapped would
     * overlap its caller's. */
    if ((uint32_t)mod->functions[fn_idx].local_count + (uint32_t)max_depth > UINT16_MAX)
        return fail("function[%u] frame needs %u slots (%u locals + operand depth %d), "
                    "more than a frame can address",
                    fn_idx,
                    (uint32_t)mod->functions[fn_idx].local_count + (uint32_t)max_depth,
                    mod->functions[fn_idx].local_count, max_depth);

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

static NvmVerifyResult verify_structure_checked(const NvmModule *mod, bool affine_only,
                                        bool *owned_admitted, bool mixed_composed) {
    if (owned_admitted) *owned_admitted=false;
    bool admitted=false;
    if (!mod) return fail("module is NULL");
    if (nvm_service_bindings_present(mod))
        return fail("I require reviewed service lifetime and dispatch admission before execution");
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
        if (mod->function_param_types && mod->function_param_types[i]) {
            for (uint16_t p = 0; p < fn->arity; p++) {
                if (mod->function_param_types[i][p] >= TAG_COUNT)
                    return fail("I found an invalid parameter tag in function[%u] parameter[%u]", i, p);
            }
        }
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

    bool needs_ownership = false;
    if (!mixed_composed && nvm_ownership_contracts_validate(mod, &needs_ownership) != NVM_V2_OK)
        return fail("I found invalid ownership declarations");
    if (needs_ownership && !affine_only) {
        for (uint32_t i=0;i<mod->function_count;i++) {
            NvmAffineAnalysis analysis=nvm_affine_analyze_function(mod,i);
            if (!analysis.ok) return fail("I refuse reference lifetime and ownership instruction dataflow in function[%u] at %u: %s",
                                          i,analysis.byte_offset,analysis.message);
        }
        NvmVerifyResult admission = nvm_verify_owned_module(mod);
        if (!admission.ok) return admission;
        admitted=true;
    }

    if (!nvm_retained_layouts_valid(mod))
        return fail("I found invalid retained layout metadata");

    if (!nvm_passive_valid(mod))
        return fail("I found invalid passive eligibility metadata");

    if (!nvm_callback_contracts_valid(mod))
        return fail("I found an invalid retained callback import contract");

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
        if (imp->kind > NVM_IMPORT_ARTIFACT)
            return fail("import[%u] has unknown kind %u", i, imp->kind);
        if (imp->kind == NVM_IMPORT_ARTIFACT) {
            const char *path = nvm_get_string(mod, imp->module_name_idx);
            if (!path || path[0] != '/' || strlen(path) != nvm_get_string_len(mod, imp->module_name_idx))
                return fail("import[%u] artifact path must be absolute and contain no NUL", i);
        }
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

    if (owned_admitted) *owned_admitted=admitted;
    return ok_result();
}

/* All existing public routes keep the original ownership validation. The only
 * true delegation caller is private preparation after fresh complete composition. */
static NvmVerifyResult verify_structure(const NvmModule *mod, bool affine_only,
                                        bool *owned_admitted) {
    return verify_structure_checked(mod,affine_only,owned_admitted,false);
}
#include "mixed_samples_prepare.inc"
/* The private complete query remains descriptive; the separate public
 * wrapper below owns candidate routing and executable admission policy. */
#include "owned_array_prepare.inc"
#include "owned_array_admit.inc"

/* ========================================================================
 * Bytecode instruction validation (per-function)
 * ======================================================================== */

static NvmVerifyResult verify_function_impl(const NvmModule *mod, uint32_t fn_idx,
                                           const NvmModule *const *linked_modules,
                                           uint32_t linked_count,
                                           uint16_t *out_max_stack) {
    if(nvm_service_bindings_present(mod))
        return fail("I refuse service contracts before mixed execution selection");
    if(nvm_owned_array_route(mod)!=NVM_OWNER_ARRAY_NOT_SELECTED) {
        if(linked_count)return fail("I refuse linked owner ARRAY execution contracts");
        return verify_owned_arrays(mod,fn_idx,out_max_stack);
    }
    if(nvm_mixed_samples_candidate(mod)) {
        if(linked_count)return fail("I refuse linked mixed ownership execution contracts");
        return verify_mixed_samples(mod,fn_idx,out_max_stack);
    }
    bool owned_admitted=false;
    NvmVerifyResult structure = verify_structure(mod, false, &owned_admitted);
    if (!structure.ok) return structure;
    if (fn_idx >= mod->function_count)
        return fail("function index %u >= function_count %u",
                    fn_idx, mod->function_count);
    if (mod->ownership_size && (owned_admitted || nvm_verify_owned_module(mod).ok)) {
        if (linked_count) return fail("I refuse linked ownership execution contracts");
        if (out_max_stack) *out_max_stack = NVM_AFFINE_MAX_STACK;
        return ok_result();
    }
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

        case OP_BORROW_PATH_SHARED: case OP_BORROW_PATH_EXCLUSIVE:
        case OP_REBORROW_SHARED: case OP_REBORROW_EXCLUSIVE:
        case OP_REGION_BEGIN: case OP_REGION_END:
        case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE: case OP_REF_GET: case OP_REF_SET:
        case OP_OWN_MOVE_LOCAL: case OP_OWN_STORE_LOCAL:
        case OP_OWN_PACK: case OP_OWN_UNPACK_LOCAL: {
            NvmAffineAnalysis analysis=nvm_affine_analyze_function(mod,fn_idx);
            if (!analysis.ok) FAIL_DECODED("I refuse reference lifetime and ownership instruction dataflow: %s",analysis.message);
            FAIL_DECODED("I require owned-transfer execution semantics before execution");
        }

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

        case OP_HANDLER_PUSH:
            if (instr.operands[0].u32 >= mod->string_count ||
                (uint32_t)instr.operands[2].u16 + instr.operands[3].u16 > fn->local_count)
                FAIL_DECODED("I require valid handler names and parameter slots.");
            if (decoded.instructions[i].resolved_target >= fn->code_offset + fn->code_length)
                FAIL_DECODED("I require an executable handler arm.");
            break;
        case OP_PERFORM:
            if (instr.operands[0].u32 >= mod->string_count)
                FAIL_DECODED("I require a valid effect operation name.");
            break;
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
    uint16_t proven_depth = 0;
    NvmVerifyResult stack_result =
        verify_stack_heights(mod, &decoded, fn_idx, &proven_depth);
    if (out_max_stack) *out_max_stack = proven_depth;

    /* Types only once the shape is proven: the type pass indexes slots the
     * height walk guarantees exist. */
    if (stack_result.ok)
        stack_result = nvm_verify_function_types(mod, fn_idx, &decoded,
                                                 proven_depth, NULL, 0);
    vm_decoded_function_free(&decoded);
    return stack_result;
}

NvmVerifyResult nvm_verify_affine_function(const NvmModule *mod, uint32_t fn_idx) {
    NvmVerifyResult structure=verify_structure(mod,true,NULL);
    if (!structure.ok) return structure;
    NvmAffineAnalysis analysis=nvm_affine_analyze_function(mod,fn_idx);
    if (!analysis.ok) return fail("I refuse reference lifetime and ownership instruction dataflow at %u: %s",
                                  analysis.byte_offset,analysis.message);
    return ok_result();
}

/* I refuse transfer instructions even without their required declarations. */
bool nvm_uses_owned_transfers(const NvmModule *mod) {
    if (!mod) return true;
    if (mod->function_count && !mod->functions) return true;
    for (uint32_t f=0;f<mod->function_count;f++) {
        const NvmFunctionEntry *fn=&mod->functions[f];
        if (fn->code_offset>mod->code_size || fn->code_length>mod->code_size-fn->code_offset ||
            (fn->code_length && !mod->code)) return true;
        uint32_t offset=0;
        while (offset<fn->code_length) {
            DecodedInstruction instruction;
            uint32_t count=isa_decode(mod->code+fn->code_offset+offset,fn->code_length-offset,&instruction);
            if (!count) break;
            if ((instruction.opcode>=OP_OWN_MOVE_LOCAL && instruction.opcode<=OP_CALL_REF) ||
                (instruction.opcode>=OP_REGION_BEGIN && instruction.opcode<=OP_REBORROW_EXCLUSIVE)) return true;
            offset+=count;
        }
    }
    return false;
}

/* I keep runtime admission closed even if affine analysis grows new operations. */
static bool owned_runtime_opcode(uint8_t op,bool value_graph) {
    switch (op) {
    case OP_CALL: case OP_CALL_REF:
    case OP_BORROW_PATH_SHARED: case OP_BORROW_PATH_EXCLUSIVE:
    case OP_REBORROW_SHARED: case OP_REBORROW_EXCLUSIVE:
    case OP_REGION_BEGIN: case OP_REGION_END:
    case OP_BORROW_LOCAL_SHARED: case OP_BORROW_LOCAL_EXCLUSIVE: case OP_REF_GET: case OP_REF_SET:
    case OP_OWN_MOVE_LOCAL: case OP_OWN_STORE_LOCAL: case OP_OWN_PACK: case OP_OWN_UNPACK_LOCAL:
    case OP_NOP: case OP_PUSH_I64: case OP_PUSH_U8: case OP_PUSH_BOOL: case OP_PUSH_F64:
    case OP_DUP: case OP_POP: case OP_SWAP: case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
    case OP_AGG_GET: case OP_STRUCT_GET: case OP_ADD: case OP_SUB: case OP_MUL:
    case OP_DIV: case OP_MOD: case OP_NEG: case OP_EQ: case OP_NE: case OP_LT:
    case OP_LE: case OP_GT: case OP_GE: case OP_AND: case OP_OR: case OP_NOT:
    case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV: case OP_F64_NEG:
    case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT: case OP_F64_LE: case OP_F64_GT: case OP_F64_GE:
    case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE: case OP_RET: case OP_ASSERT:
        return true;
    case OP_PUSH_STR: case OP_PRINT: case OP_PRINTLN:
        return value_graph;
    default: return false;
    }
}

NvmVerifyResult nvm_verify_owned_module(const NvmModule *mod) {
    NvmVerifyResult structure = verify_structure(mod, true,NULL);
    if (!structure.ok) return structure;
    if (!mod->ownership_size || (!mod->function_count || mod->function_count>NVM_OWNED_MAX_FUNCTIONS) || mod->header.entry_point != 0 ||
        mod->import_count || mod->module_ref_count || mod->callback_contract_count || mod->passive_size)
        return fail("I require standalone ownership instruction execution semantics without linked contracts");
    bool value_graph=nvm_affine_value_call_graph(mod);
    if (!value_graph && mod->function_count>2)
        return fail("I require a bounded acyclic value graph or my separate borrowed helper");
    for(uint32_t function=0;function<mod->function_count;function++) {
        const NvmFunctionEntry *fn=&mod->functions[function];
        const char *name=nvm_get_string(mod,fn->name_idx);
        if ((function ? ((!value_graph && !fn->arity) || fn->arity>NVM_AFFINE_MAX_PARAMETERS) : fn->arity!=0) || fn->upvalue_count ||
            ((!function || !value_graph) && (fn->result_count!=1 ||
             (fn->result_tag!=TAG_INT && fn->result_tag!=TAG_BOOL && fn->result_tag!=TAG_U8))) ||
            fn->local_count>NVM_AFFINE_MAX_LOCALS || (name && !strcmp(name,"__init__")))
            return fail("I require a scalar entry and exact bounded value-result helper signatures");
        NvmAffineState *state=nvm_affine_state_create(mod,function,fn->local_count);
        if(!state) return fail("I require complete ownership local declarations");
        NvmAffineType result;uint16_t fields=0;
        bool valid=nvm_affine_value_result(state,&result,&fields);
        for(uint16_t i=0;i<fn->local_count;i++) {
            NvmAffineType type;NvmReferenceMode mode;
            if(function && !value_graph && i<fn->arity) {
                if(!nvm_affine_parameter_at(state,i,&type,&mode)) valid=false;
            } else if(!nvm_affine_local_type(state,i,&type) ||
                (type.tag!=TAG_INT && type.tag!=TAG_BOOL && type.tag!=TAG_U8 &&
                 type.tag!=TAG_STRUCT && !(i>=fn->arity && type.tag==TAG_FLOAT &&
                                           type.layout==NVM_V2_NO_INDEX) &&
                 !(value_graph && type.tag==TAG_STRING &&
                                           type.layout==NVM_V2_NO_INDEX))) valid=false;
        }
        nvm_affine_state_free(state);
        if(!valid) return fail("I require value entry locals and exact borrowed or consuming value helper parameters");
    }
    NvmV2Layouts layouts = {0};
    if (nvm_v2_layouts_decode(mod->layout_data, mod->layout_size, &layouts) != NVM_V2_OK)
        return fail("I require complete owned record layouts");
    bool supported = true;
    for (uint32_t i=0; i<layouts.count; i++) {
        const NvmV2Layout *layout = &layouts.items[i];
        if (!(mod->ownership_data[8+i] & NVM_LAYOUT_COMPLETE) ||
            layout->kind != NVM_V2_LAYOUT_STRUCT || layout->field_count > NVM_AFFINE_MAX_STACK)
            supported = false;
        for (uint16_t f=0; f<layout->field_count; f++) {
            uint8_t tag=layout->fields[f].type_tag;
            /* My owned execution profile retains its prior-only graph. */
            uint32_t child=layout->fields[f].nested_idx;
            if(child!=NVM_V2_NO_INDEX && child>=i) supported=false;
            if (tag!=TAG_INT && tag!=TAG_BOOL && tag!=TAG_U8 && tag!=TAG_STRUCT &&
                !(value_graph && tag==TAG_STRING && child==NVM_V2_NO_INDEX)) supported=false;
        }
    }
    nvm_v2_layouts_free(&layouts);
    if (!supported) return fail("I require exact scalar, retained STRING or owned-child record fields before execution");
    bool transfer=false;
    for(uint32_t function=0;function<mod->function_count;function++) {
        VmDecodedFunction decoded;char error[VM_DECODE_ERROR_SIZE];
        if(!vm_decode_function(mod,function,&decoded,error)) return fail("%s",error);
        for(uint32_t i=0;i<decoded.instruction_count;i++) {
            const DecodedInstruction *in=&decoded.instructions[i].instruction;
            uint8_t op=in->opcode;
            if(op>=OP_OWN_MOVE_LOCAL && op<=OP_OWN_UNPACK_LOCAL) transfer=true;
            if(!owned_runtime_opcode(op,value_graph)) supported=false;
            if(op==OP_PUSH_STR) {
                uint32_t index=in->operands[0].u32;
                if(index>=mod->string_count || !mod->strings || !mod->string_lengths ||
                   !mod->strings[index] ||
                   memchr(mod->strings[index],'\0',mod->string_lengths[index])) supported=false;
            }
            if(op==OP_CALL_REF && (function || value_graph || mod->function_count!=2 || in->operands[0].u32!=1)) supported=false;
            if(op==OP_CALL && (!value_graph || !in->operands[0].u32 || in->operands[0].u32>=mod->function_count)) supported=false;
            if(function && !value_graph && (op==OP_AGG_GET || op==OP_STRUCT_GET || (
                ((op==OP_LOAD_LOCAL || op==OP_STORE_LOCAL || op==OP_OWN_MOVE_LOCAL ||
                  op==OP_OWN_STORE_LOCAL || op==OP_OWN_UNPACK_LOCAL) && in->operands[0].u16<mod->functions[function].arity) ||
                ((op==OP_BORROW_LOCAL_SHARED || op==OP_BORROW_LOCAL_EXCLUSIVE ||
                  op==OP_BORROW_PATH_SHARED || op==OP_BORROW_PATH_EXCLUSIVE) &&
                 in->operands[1].u16<mod->functions[function].arity)))) supported=false;
        }
        vm_decoded_function_free(&decoded);
        if(function && !nvm_affine_analyze_function(mod,function).ok) supported=false;
    }
    if(!supported || !transfer) return fail("I require explicit owned entry execution and a non-escaping scalar helper");
    return nvm_verify_affine_function(mod, 0);
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
    if(nvm_service_bindings_present(mod))
        return fail("I refuse service contracts before mixed execution selection");
    if(nvm_owned_array_route(mod)!=NVM_OWNER_ARRAY_NOT_SELECTED)return verify_owned_arrays(mod,0,NULL);
    if(nvm_mixed_samples_candidate(mod))return verify_mixed_samples(mod,0,NULL);
    /* I reuse only this invocation's completed full owned-module proof. */
    bool owned_admitted=false;
    NvmVerifyResult r = verify_structure(mod, false, &owned_admitted);
    if (!r.ok || owned_admitted) return r;

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

    for (uint32_t i=0; i<linked_count; i++)
        if (nvm_service_bindings_present(linked_modules[i]))
            return fail("I refuse linked service contracts before reviewed dispatch admission");
    if(nvm_service_bindings_present(mod))
        return fail("I refuse service contracts before mixed execution selection");
    for(uint32_t i=0;i<linked_count;i++)
        if(nvm_owned_array_route(linked_modules[i])!=NVM_OWNER_ARRAY_NOT_SELECTED)
            return fail("I refuse an owner ARRAY candidate or invalid descriptor in a linked graph");
    if(nvm_owned_array_route(mod)!=NVM_OWNER_ARRAY_NOT_SELECTED) {
        if(linked_count)return fail("I refuse linked owner ARRAY execution contracts");
        return verify_owned_arrays(mod,0,NULL);
    }
    if(nvm_mixed_samples_candidate(mod)) {
        if(linked_count)return fail("I refuse linked mixed ownership execution contracts");
        return verify_mixed_samples(mod,0,NULL);
    }
    if (linked_count) {
        for(uint32_t i=0;i<linked_count;i++)if(nvm_mixed_samples_candidate(linked_modules[i]))
            return fail("I refuse a mixed module in a linked graph");
        bool needs = false;
        if (mod && ((nvm_ownership_contracts_validate(mod, &needs)==NVM_V2_OK && needs) || nvm_uses_owned_transfers(mod)))
            return fail("I refuse linked ownership execution contracts");
        for (uint32_t i=0; i<linked_count; i++) {
            needs=false;
            if (linked_modules[i] && ((nvm_ownership_contracts_validate(linked_modules[i], &needs)==NVM_V2_OK && needs) || nvm_uses_owned_transfers(linked_modules[i])))
                return fail("I refuse linked ownership execution contracts");
        }
    }
    /* Zero linked modules retain the same invocation-local owned proof. */
    bool owned_admitted=false;
    NvmVerifyResult r = verify_structure(mod, false, &owned_admitted);
    if (!r.ok || (!linked_count && owned_admitted)) return r;

    /* Phase 2: per-function validation, resolving OP_CALL_MODULE against the
     * supplied linked-module table so cross-module call operands are bounded. */
    for (uint32_t i = 0; i < mod->function_count; i++) {
        r = verify_function_impl(mod, i, linked_modules, linked_count, NULL);
        if (!r.ok) return r;
    }

    return ok_result();
}

/* I preserve the original scalar translator eligibility as one shared policy. */
static int profile_scalar(uint8_t tag) { return tag == TAG_INT || tag == TAG_U8 || tag == TAG_BOOL || tag == TAG_VOID || tag == TAG_FLOAT || tag == TAG_ENUM; }
static int profile_supported(uint8_t op) {
    switch (op) {
    case OP_ENUM_VAL: case OP_LOAD_GLOBAL: case OP_STORE_GLOBAL:
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: case OP_NEG:
    case OP_F64_ADD: case OP_F64_SUB: case OP_F64_MUL: case OP_F64_DIV:
    case OP_F64_NEG: case OP_F64_EQ: case OP_F64_NE: case OP_F64_LT:
    case OP_F64_LE: case OP_F64_GT: case OP_F64_GE: case OP_PUSH_F64:
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
    case OP_CAST_BOOL: case OP_AND: case OP_OR: case OP_NOT:
    case OP_F64_FROM_BITS: case OP_F64_TO_BITS:
    case OP_CAST_INT: case OP_CAST_FLOAT:
    case OP_NOP: case OP_PUSH_U8: case OP_PUSH_I64: case OP_PUSH_BOOL: case OP_PUSH_VOID:
    case OP_DUP: case OP_POP: case OP_SWAP: case OP_LOAD_LOCAL: case OP_STORE_LOCAL:
    case OP_I64_ADD: case OP_I64_SUB: case OP_I64_MUL: case OP_I64_DIV_S: case OP_I64_REM_S:
    case OP_I64_NEG: case OP_I64_EQ: case OP_I64_NE: case OP_I64_LT_S: case OP_I64_LE_S:
    case OP_I64_GT_S: case OP_I64_GE_S: case OP_BOOL_AND: case OP_BOOL_OR: case OP_BOOL_NOT:
    case OP_JMP: case OP_JMP_TRUE: case OP_JMP_FALSE: case OP_CALL: case OP_RET:
    case OP_ASSERT: case OP_TYPE_CHECK: return 1;
    default: return 0;
    }
}
NvmVerifyResult nvm_verify_profile(const NvmModule *m, NvmVerifyProfile profile) {
    if (profile != NVM_PROFILE_GENERAL && profile != NVM_PROFILE_CLOSED_SCALAR &&
        profile != NVM_PROFILE_CLOSED_LITERAL_STRINGS &&
        profile != NVM_PROFILE_CLOSED_MANAGED_STRINGS)
        return fail("I do not recognize verifier profile %d", (int)profile);
    NvmVerifyResult verified = nvm_verify(m);
    if (!verified.ok || profile == NVM_PROFILE_GENERAL) return verified;
    if(nvm_owned_array_route(m)!=NVM_OWNER_ARRAY_NOT_SELECTED)return fail("I keep owner ARRAY candidates outside closed backend profiles");
    if(nvm_mixed_samples_candidate(m))return fail("I keep mixed ownership outside closed backend profiles");
    const bool record_profile = profile == NVM_PROFILE_CLOSED_MANAGED_STRINGS &&
        (m->struct_count || m->layout_size || m->ownership_size);
    if (m->import_count || m->module_ref_count || m->union_count || m->passive_size ||
        (!record_profile && (m->struct_count || m->ownership_size || m->layout_size)))
        return fail("I support only closed scalar modules without imports, nominal layouts or ownership/passive contracts");
    if (!(m->header.flags & NVM_FLAG_HAS_MAIN))
        return fail("I require an explicit executable entry point");
    if (m->functions[m->header.entry_point].result_count != 1 ||
        (m->functions[m->header.entry_point].result_tag != TAG_INT &&
         m->functions[m->header.entry_point].result_tag != TAG_BOOL))
        return fail("I require an integer/bool executable entry result");
    if (m->functions[m->header.entry_point].arity)
        return fail("I require a zero-argument scalar entry point");
    const bool managed_profile = profile == NVM_PROFILE_CLOSED_MANAGED_STRINGS;
    const bool literal_profile = profile == NVM_PROFILE_CLOSED_LITERAL_STRINGS || managed_profile;
    bool has_strings = false;
    bool mutable_arrays = false;
    bool needs_string_runtime = false;
    bool initializer_seen = false;
    for (uint32_t i = 0; i < m->function_count; ++i) {
        const NvmFunctionEntry *f = &m->functions[i];
        const char *name = nvm_get_string(m, f->name_idx);
        if (!initializer_seen && name && !strcmp(name, "__init__")) {
            initializer_seen = true;
            if (f->arity)
                return fail("I require a zero-argument scalar module initializer");
        }
        if (f->upvalue_count || !((f->result_count == 0 && f->result_tag == TAG_VOID) ||
            (f->result_count == 1 && (f->result_tag == TAG_INT || f->result_tag == TAG_U8 || f->result_tag == TAG_ENUM || f->result_tag == TAG_BOOL || f->result_tag == TAG_FLOAT ||
             (literal_profile && f->result_tag == TAG_STRING) || (managed_profile && f->result_tag == TAG_ARRAY) || (record_profile && f->result_tag == TAG_STRUCT)))))
            return fail("I require zero void results or one admitted closed-profile result and no captures in function %u", i);
        has_strings |= f->result_count && f->result_tag == TAG_STRING;
        for (uint16_t p = 0; p < f->arity; ++p) {
            if (!m->function_param_types || !m->function_param_types[i]) continue;
            uint8_t tag = m->function_param_types[i][p];
            has_strings |= tag == TAG_STRING;
            if (!profile_scalar(tag) && !(literal_profile && tag == TAG_STRING) && !(managed_profile && tag == TAG_ARRAY) && !(record_profile && tag == TAG_STRUCT))
                return fail("I require admitted closed-profile parameters in function %u", i);
        }
        for (uint32_t pc = 0; pc < f->code_length;) {
            DecodedInstruction ins = {0};
            uint32_t width = isa_decode(m->code + f->code_offset + pc, f->code_length - pc, &ins);
            mutable_arrays |= ins.opcode == OP_ARR_NEW || ins.opcode == OP_ARR_PUSH ||
                              ins.opcode == OP_ARR_SET || ins.opcode == OP_ARR_POP ||
                              ins.opcode == OP_ARR_LITERAL || ins.opcode == OP_ARR_SLICE;
            has_strings |= ins.opcode == OP_PUSH_STR || ins.opcode == OP_STR_CONCAT || ins.opcode == OP_STR_SUBSTR || ins.opcode == OP_CAST_STRING;
            needs_string_runtime |= !managed_profile && (ins.opcode == OP_ADD || ins.opcode == OP_CAST_INT || ins.opcode == OP_CAST_FLOAT);
            bool literal_op = literal_profile && (ins.opcode == OP_PUSH_STR ||
                              ins.opcode == OP_STR_LEN || ins.opcode == OP_STR_EQ ||
                              (managed_profile && (ins.opcode == OP_ARR_LITERAL || ins.opcode == OP_ARR_SLICE || ins.opcode == OP_ARR_NEW || ins.opcode == OP_ARR_PUSH || ins.opcode == OP_ARR_SET || ins.opcode == OP_ARR_POP || ins.opcode == OP_STR_SPLIT || ins.opcode == OP_ARR_GET || ins.opcode == OP_ARR_LEN || ins.opcode == OP_STR_REPLACE || ins.opcode == OP_STR_FROM_INT || ins.opcode == OP_STR_FROM_FLOAT || ins.opcode == OP_STR_TO_LOWER || ins.opcode == OP_STR_TO_UPPER || ins.opcode == OP_STR_CHAR_AT || ins.opcode == OP_STR_TRIM || ins.opcode == OP_STR_CONCAT || ins.opcode == OP_STR_SUBSTR || ins.opcode == OP_CAST_STRING ||
                               ins.opcode == OP_STR_CONTAINS || ins.opcode == OP_STR_STARTS_WITH || ins.opcode == OP_STR_ENDS_WITH)));
            bool record_op = record_profile && (ins.opcode == OP_STRUCT_NEW ||
                ins.opcode == OP_STRUCT_LITERAL || ins.opcode == OP_STRUCT_GET ||
                ins.opcode == OP_STRUCT_SET || ins.opcode == OP_AGG_PACK ||
                ins.opcode == OP_AGG_GET || ins.opcode == OP_AGG_SET);
            if (!width || (!profile_supported(ins.opcode) && !literal_op && !record_op)) return fail("I do not support opcode 0x%02x at function %u offset %u in my scalar LLVM profile", ins.opcode, i, pc);
            pc += width;
        }
    }
    if (mutable_arrays || record_profile) {
        NvmManagedHeapPlan *plan = NULL;
        NvmArrayEligibilityResult arrays = nvm_select_managed_heap(m, mutable_arrays, &plan);
        nvm_managed_heap_plan_free(plan);
        if (arrays.status != NVM_ARRAY_ELIGIBLE)
            return fail("I cannot establish mutable array eligibility (status %u) at function %u offset %u: %s",
                        (unsigned)arrays.status, arrays.function, arrays.pc, arrays.message);
    }
    if (has_strings && needs_string_runtime)
        return fail("I refuse ADD/CAST_INT/CAST_FLOAT in the literal-string profile");
    return ok_result();
}
