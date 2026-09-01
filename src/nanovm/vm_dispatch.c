#include "vm_dispatch.h"

#include "../nanoisa/isa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool dispatch_fail(char error[VM_DISPATCH_ERROR_SIZE],
                          const char *message, uint32_t value) {
    if (error) {
        snprintf(error, VM_DISPATCH_ERROR_SIZE, message, value);
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

static bool dispatch_is_branch(uint8_t opcode) {
    return opcode == OP_JMP || opcode == OP_JMP_TRUE
        || opcode == OP_JMP_FALSE || opcode == OP_MATCH_TAG;
}

static bool dispatch_is_direct_call(uint8_t opcode) {
    return opcode == OP_CALL || opcode == OP_TAIL_CALL;
}

/*
 * Look up the dispatch index of the instruction that begins at
 * `function_offset`.  Returns VM_DISPATCH_NO_INDEX when the offset is not an
 * instruction boundary within this function.
 */
static uint32_t dispatch_index_at(const VmDispatchFunction *function,
                                  uint32_t function_offset) {
    if (!function || function_offset >= function->code_size
            || !function->offset_to_index) {
        return VM_DISPATCH_NO_INDEX;
    }
    uint32_t encoded = function->offset_to_index[function_offset];
    if (encoded == 0 || encoded > function->instruction_count) {
        return VM_DISPATCH_NO_INDEX;
    }
    return encoded - 1;
}

bool vm_dispatch_build_function(const VmDecodedFunction *decoded,
                                VmDispatchFunction *out,
                                char error[VM_DISPATCH_ERROR_SIZE]) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!decoded) {
        return dispatch_fail(error, "verified IR is NULL", 0);
    }

    out->code_size = decoded->code_size;
    out->instruction_count = decoded->instruction_count;
    out->offset_to_index = calloc((size_t)decoded->code_size + 1,
                                  sizeof(*out->offset_to_index));
    if (!out->offset_to_index) {
        vm_dispatch_function_free(out);
        return dispatch_fail(error, "dispatch offset map allocation failed", 0);
    }
    if (decoded->instruction_count > 0) {
        out->instructions = calloc(decoded->instruction_count,
                                   sizeof(*out->instructions));
        if (!out->instructions) {
            vm_dispatch_function_free(out);
            return dispatch_fail(error, "dispatch instruction allocation failed", 0);
        }
    }

    /* First pass: copy the verified instructions and record boundaries. */
    for (uint32_t i = 0; i < decoded->instruction_count; i++) {
        const VmDecodedInstruction *src = &decoded->instructions[i];
        VmDispatchInstruction *dst = &out->instructions[i];
        dst->instruction = src->instruction;
        dst->call_handle = src->call_handle;
        dst->byte_offset = src->byte_offset;
        dst->next_byte_offset = src->next_byte_offset;
        dst->branch_target = VM_DISPATCH_NO_INDEX;
        dst->branch_target_offset = VM_DISPATCH_NO_INDEX;
        dst->call_target = VM_DISPATCH_NO_INDEX;
        if (src->byte_offset >= decoded->code_size) {
            vm_dispatch_function_free(out);
            return dispatch_fail(error,
                "dispatch instruction lies outside code at offset %u",
                src->byte_offset);
        }
        out->offset_to_index[src->byte_offset] = i + 1;
    }

    /* Second pass: precompute successors and resolved control-flow targets.
     * The verified IR stores branch targets as module-absolute offsets
     * (function code_offset + relative position); recover the function-local
     * offset from the boundary that the verified IR already validated. */
    for (uint32_t i = 0; i < decoded->instruction_count; i++) {
        const VmDecodedInstruction *src = &decoded->instructions[i];
        VmDispatchInstruction *dst = &out->instructions[i];
        uint8_t opcode = src->instruction.opcode;

        dst->next_index = dispatch_index_at(out, src->next_byte_offset);

        if (dispatch_is_branch(opcode)) {
            /* resolved_target = code_offset + local_offset, and byte_offset is
             * the local offset of this instruction, so:
             *   local_target = resolved_target - (this_absolute - byte_offset)
             * We do not know code_offset directly, but the verified IR keeps
             * every branch target on a boundary within [0, code_size], so map
             * through the boundary table using the local delta. */
            int64_t relative = opcode == OP_MATCH_TAG
                ? src->instruction.operands[1].i32
                : src->instruction.operands[0].i32;
            int64_t local_target = (int64_t)src->byte_offset + relative;
            if (local_target < 0 || local_target > (int64_t)decoded->code_size) {
                vm_dispatch_function_free(out);
                return dispatch_fail(error,
                    "dispatch branch target out of range at offset %u",
                    src->byte_offset);
            }
            uint32_t target_index = dispatch_index_at(out, (uint32_t)local_target);
            if (target_index == VM_DISPATCH_NO_INDEX
                    && (uint32_t)local_target != decoded->code_size) {
                vm_dispatch_function_free(out);
                return dispatch_fail(error,
                    "dispatch branch misses an instruction boundary at offset %u",
                    src->byte_offset);
            }
            dst->branch_target = target_index;
            dst->branch_target_offset = src->resolved_target;
        } else if (dispatch_is_direct_call(opcode)) {
            dst->call_target = src->resolved_target;
        }
    }

    if (error) error[0] = '\0';
    return true;
}

bool vm_dispatch_build_module(const VmDecodedModule *decoded,
                              VmDispatchModule *out,
                              char error[VM_DISPATCH_ERROR_SIZE]) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!decoded) {
        return dispatch_fail(error, "verified module is NULL", 0);
    }
    if (decoded->function_count == 0) {
        if (error) error[0] = '\0';
        return true;
    }

    out->functions = calloc(decoded->function_count, sizeof(*out->functions));
    if (!out->functions) {
        return dispatch_fail(error, "dispatch module allocation failed", 0);
    }
    out->function_count = decoded->function_count;
    for (uint32_t i = 0; i < decoded->function_count; i++) {
        if (!vm_dispatch_build_function(&decoded->functions[i],
                                        &out->functions[i], error)) {
            vm_dispatch_module_free(out);
            return false;
        }
    }
    if (error) error[0] = '\0';
    return true;
}

bool vm_dispatch_seek(VmDispatchCursor *cursor,
                      const VmDispatchFunction *function,
                      uint32_t byte_offset) {
    if (!cursor || !function) return false;
    uint32_t index = dispatch_index_at(function, byte_offset);
    if (index == VM_DISPATCH_NO_INDEX) return false;
    cursor->function = function;
    cursor->index = index;
    return true;
}

const VmDispatchInstruction *vm_dispatch_current(const VmDispatchCursor *cursor) {
    if (!cursor || !cursor->function
            || cursor->index >= cursor->function->instruction_count) {
        return NULL;
    }
    return &cursor->function->instructions[cursor->index];
}

const VmDispatchInstruction *vm_dispatch_advance(VmDispatchCursor *cursor) {
    const VmDispatchInstruction *current = vm_dispatch_current(cursor);
    if (!current) return NULL;
    uint32_t next = current->next_index;
    if (next == VM_DISPATCH_NO_INDEX
            || next >= cursor->function->instruction_count) {
        cursor->index = cursor->function->instruction_count;
        return NULL;
    }
    cursor->index = next;
    return &cursor->function->instructions[next];
}
