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
    out->boundaries = calloc((size_t)entry->code_length + 1, 1);
    if (!out->boundaries) {
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
        out->boundaries[position] = 1;
        out->instruction_count++;
        position += size;
    }
    out->boundaries[entry->code_length] = 1;
    if (error) error[0] = '\0';
    return true;
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
