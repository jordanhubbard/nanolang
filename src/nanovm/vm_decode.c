#include "vm_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool decode_fail(char error[VM_DECODE_ERROR_SIZE], const char *message,
                        uint32_t function_index, uint32_t offset) {
    if (error) {
        snprintf(error, VM_DECODE_ERROR_SIZE, message, function_index, offset);
    }
    return false;
}

void vm_decoded_function_free(VmDecodedFunction *function) {
    if (!function) return;
    free(function->instructions);
    free(function->boundaries);
    free(function->instruction_indices);
    memset(function, 0, sizeof(*function));
}

void vm_decoded_module_free(VmDecodedModule *module) {
    if (!module) return;
    for (uint32_t i = 0; i < module->function_count; i++) {
        vm_decoded_function_free(&module->functions[i]);
    }
    free(module->functions);
    memset(module, 0, sizeof(*module));
}

bool vm_decoded_function_has_boundary(const VmDecodedFunction *function,
                                      uint32_t byte_offset) {
    return function && byte_offset <= function->code_size
        && function->boundaries && function->boundaries[byte_offset] != 0;
}

bool vm_decode_function(const NvmModule *module, uint32_t function_index,
                        VmDecodedFunction *out, char error[VM_DECODE_ERROR_SIZE]) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!module || function_index >= module->function_count) {
        return decode_fail(error, "function index %u is invalid at offset %u",
                           function_index, 0);
    }

    const NvmFunctionEntry *entry = &module->functions[function_index];
    if (entry->code_offset > module->code_size
            || entry->code_length > module->code_size - entry->code_offset
            || (!module->code && entry->code_length > 0)) {
        return decode_fail(error, "function[%u] has an invalid code range at offset %u",
                           function_index, entry->code_offset);
    }

    out->code_size = entry->code_length;
    out->code_offset = entry->code_offset;
    out->boundaries = calloc((size_t)entry->code_length + 1, 1);
    out->instruction_indices = calloc((size_t)entry->code_length + 1,
                                      sizeof(*out->instruction_indices));
    if (!out->boundaries || !out->instruction_indices) {
        vm_decoded_function_free(out);
        return decode_fail(error, "function[%u] allocation failed at offset %u",
                           function_index, entry->code_offset);
    }

    uint32_t capacity = 0;
    uint32_t position = 0;
    const uint8_t *code = entry->code_length > 0
        ? module->code + entry->code_offset
        : NULL;
    while (position < entry->code_length) {
        if (out->instruction_count == capacity) {
            uint32_t next_capacity = capacity ? capacity * 2 : 16;
            VmDecodedInstruction *next = realloc(
                out->instructions, (size_t)next_capacity * sizeof(*next));
            if (!next) {
                vm_decoded_function_free(out);
                return decode_fail(error, "function[%u] allocation failed at offset %u",
                                   function_index, entry->code_offset + position);
            }
            out->instructions = next;
            capacity = next_capacity;
        }

        VmDecodedInstruction *decoded =
            &out->instructions[out->instruction_count];
        uint32_t size = isa_decode(code + position,
                                   entry->code_length - position,
                                   &decoded->instruction);
        if (size == 0) {
            vm_decoded_function_free(out);
            return decode_fail(error, "function[%u] invalid instruction at offset %u",
                               function_index, entry->code_offset + position);
        }
        decoded->byte_offset = position;
        decoded->next_byte_offset = position + size;
        decoded->resolved_target = UINT32_MAX;
        out->boundaries[position] = 1;
        out->instruction_indices[position] = out->instruction_count + 1;
        out->instruction_count++;
        position += size;
    }
    out->boundaries[entry->code_length] = 1;

    for (uint32_t i = 0; i < out->instruction_count; i++) {
        VmDecodedInstruction *decoded = &out->instructions[i];
        uint8_t opcode = decoded->instruction.opcode;
        if (opcode == OP_JMP || opcode == OP_JMP_TRUE
                || opcode == OP_JMP_FALSE || opcode == OP_MATCH_TAG) {
            int32_t relative = opcode == OP_MATCH_TAG
                ? decoded->instruction.operands[1].i32
                : decoded->instruction.operands[0].i32;
            int64_t target = (int64_t)decoded->byte_offset + relative;
            if (target < 0 || target > UINT32_MAX
                    || !vm_decoded_function_has_boundary(out, (uint32_t)target)) {
                uint32_t bad_offset = entry->code_offset + decoded->byte_offset;
                vm_decoded_function_free(out);
                return decode_fail(error,
                    "function[%u] branch targets a non-instruction boundary at offset %u",
                    function_index, bad_offset);
            }
            decoded->resolved_target = entry->code_offset + (uint32_t)target;
        } else if (opcode == OP_CALL || opcode == OP_TAIL_CALL) {
            uint32_t target = decoded->instruction.operands[0].u32;
            if (target >= module->function_count) {
                uint32_t bad_offset = entry->code_offset + decoded->byte_offset;
                vm_decoded_function_free(out);
                return decode_fail(error,
                    "function[%u] direct call has an invalid target at offset %u",
                    function_index, bad_offset);
            }
            decoded->resolved_target = target;
        }
    }
    if (error) error[0] = '\0';
    return true;
}

const VmDecodedInstruction *vm_decoded_function_at(
        const VmDecodedFunction *function, uint32_t byte_offset) {
    if (!function || byte_offset >= function->code_size
            || !function->instruction_indices) return NULL;
    uint32_t encoded_index = function->instruction_indices[byte_offset];
    if (encoded_index == 0) return NULL;
    return &function->instructions[encoded_index - 1];
}

bool vm_decode_module(const NvmModule *module, VmDecodedModule *out,
                      char error[VM_DECODE_ERROR_SIZE]) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!module) {
        if (error) snprintf(error, VM_DECODE_ERROR_SIZE, "module is NULL");
        return false;
    }
    if (module->function_count == 0) {
        if (error) error[0] = '\0';
        return true;
    }

    out->functions = calloc(module->function_count, sizeof(*out->functions));
    if (!out->functions) {
        if (error) snprintf(error, VM_DECODE_ERROR_SIZE, "module allocation failed");
        return false;
    }
    out->function_count = module->function_count;
    for (uint32_t i = 0; i < module->function_count; i++) {
        if (!vm_decode_function(module, i, &out->functions[i], error)) {
            vm_decoded_module_free(out);
            return false;
        }
    }
    if (error) error[0] = '\0';
    return true;
}

/* ========================================================================
 * Optimized dispatch IR
 *
 * Projects the verified IR into a flat program whose successors and
 * in-function targets are already instruction indices.  The verified IR is
 * authoritative for correctness; this pass only rearranges proven data for
 * fast execution and never widens what a branch or call may reach.
 * ======================================================================== */

static bool dispatch_fail(char error[VM_DECODE_ERROR_SIZE], const char *message) {
    if (error) {
        snprintf(error, VM_DECODE_ERROR_SIZE, "%s", message);
    }
    return false;
}

void vm_dispatch_function_free(VmDispatchFunction *function) {
    if (!function) return;
    free(function->instructions);
    free(function->offset_to_index);
    memset(function, 0, sizeof(*function));
}

void vm_dispatch_module_free(VmDispatchModule *module) {
    if (!module) return;
    for (uint32_t i = 0; i < module->function_count; i++) {
        vm_dispatch_function_free(&module->functions[i]);
    }
    free(module->functions);
    memset(module, 0, sizeof(*module));
}

uint32_t vm_dispatch_function_index_at(const VmDispatchFunction *function,
                                       uint32_t byte_offset) {
    if (!function || !function->offset_to_index
            || byte_offset > function->code_size) {
        return VM_DISPATCH_NO_INDEX;
    }
    return function->offset_to_index[byte_offset];
}

bool vm_dispatch_function_build(const VmDecodedFunction *decoded,
                                VmDispatchFunction *out,
                                char error[VM_DECODE_ERROR_SIZE]) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!decoded) {
        return dispatch_fail(error, "decoded function is NULL");
    }

    out->code_size = decoded->code_size;
    out->instruction_count = decoded->instruction_count;
    out->offset_to_index = malloc(((size_t)decoded->code_size + 1)
                                  * sizeof(*out->offset_to_index));
    if (!out->offset_to_index) {
        return dispatch_fail(error, "dispatch function allocation failed");
    }
    for (uint32_t i = 0; i <= decoded->code_size; i++) {
        out->offset_to_index[i] = VM_DISPATCH_NO_INDEX;
    }
    if (decoded->instruction_count == 0) {
        if (error) error[0] = '\0';
        return true;
    }

    out->instructions = calloc(decoded->instruction_count,
                               sizeof(*out->instructions));
    if (!out->instructions) {
        vm_dispatch_function_free(out);
        return dispatch_fail(error, "dispatch function allocation failed");
    }

    for (uint32_t i = 0; i < decoded->instruction_count; i++) {
        const VmDecodedInstruction *src = &decoded->instructions[i];
        if (src->byte_offset <= decoded->code_size) {
            out->offset_to_index[src->byte_offset] = i;
        }
    }

    for (uint32_t i = 0; i < decoded->instruction_count; i++) {
        const VmDecodedInstruction *src = &decoded->instructions[i];
        VmDispatchInstruction *slot = &out->instructions[i];
        slot->instruction = src->instruction;
        slot->start_offset = src->byte_offset;
        slot->next_offset = src->next_byte_offset;
        slot->next_index = (i + 1 < decoded->instruction_count)
            ? i + 1 : VM_DISPATCH_NO_INDEX;
        slot->target_index = VM_DISPATCH_NO_INDEX;
        slot->target_function = VM_DISPATCH_NO_INDEX;
        slot->target_byte_offset = VM_DISPATCH_NO_INDEX;

        if (src->resolved_target == UINT32_MAX) {
            continue;
        }

        uint8_t opcode = src->instruction.opcode;
        if (opcode == OP_CALL || opcode == OP_TAIL_CALL) {
            /* resolved_target is a callee function index. */
            slot->target_function = src->resolved_target;
        } else {
            /* resolved_target is a module-absolute byte offset of an
             * in-function branch target.  Convert it to a function-relative
             * offset; the verified IR has already proven it is an
             * instruction boundary or the function-end sentinel. */
            if (src->resolved_target < decoded->code_offset) {
                vm_dispatch_function_free(out);
                return dispatch_fail(error,
                    "dispatch branch target precedes the function");
            }
            uint32_t local_offset = src->resolved_target - decoded->code_offset;
            if (local_offset > decoded->code_size) {
                vm_dispatch_function_free(out);
                return dispatch_fail(error,
                    "dispatch branch target is outside the function");
            }
            uint32_t target_index = out->offset_to_index[local_offset];
            if (target_index == VM_DISPATCH_NO_INDEX
                    && local_offset != decoded->code_size) {
                vm_dispatch_function_free(out);
                return dispatch_fail(error,
                    "dispatch branch target is not an instruction boundary");
            }
            slot->target_index = target_index;
            slot->target_byte_offset = local_offset;
        }
    }

    if (error) error[0] = '\0';
    return true;
}

bool vm_dispatch_module_build(const VmDecodedModule *decoded,
                              VmDispatchModule *out,
                              char error[VM_DECODE_ERROR_SIZE]) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!decoded) {
        return dispatch_fail(error, "decoded module is NULL");
    }
    if (decoded->function_count == 0) {
        if (error) error[0] = '\0';
        return true;
    }

    out->functions = calloc(decoded->function_count, sizeof(*out->functions));
    if (!out->functions) {
        return dispatch_fail(error, "dispatch module allocation failed");
    }
    out->function_count = decoded->function_count;
    for (uint32_t i = 0; i < decoded->function_count; i++) {
        if (!vm_dispatch_function_build(&decoded->functions[i],
                                        &out->functions[i], error)) {
            vm_dispatch_module_free(out);
            return false;
        }
    }
    if (error) error[0] = '\0';
    return true;
}
