#ifndef NANOVM_VM_DECODE_H
#define NANOVM_VM_DECODE_H

#include "../nanoisa/isa.h"
#include "../nanoisa/nvm_format.h"

#include <stdbool.h>
#include <stdint.h>

#define VM_DECODE_ERROR_SIZE 256

/* Callable handle for cross-module (OP_CALL_MODULE) calls.
 * Resolved once during linking so dispatch never carries a
 * module/function index pair or repeats per-call bounds checks. */
typedef struct {
    const NvmModule *module;                 /* Target module, NULL if unresolved */
    const NvmFunctionEntry *function; /* Target function entry */
    uint32_t function_index;                 /* Function index within the target module */
    bool resolved;                           /* True once linking bound the handle */
} VmCallHandle;

typedef struct {
    uint32_t byte_offset;
    uint32_t next_byte_offset;
    uint32_t resolved_target;
    VmCallHandle call_handle;   /* Valid for OP_CALL_MODULE, resolved at link time */
    DecodedInstruction instruction;
} VmDecodedInstruction;

typedef struct {
    VmDecodedInstruction *instructions;
    uint32_t instruction_count;
    uint32_t code_size;
    uint8_t *boundaries;
    uint32_t *instruction_indices;
} VmDecodedFunction;

typedef struct {
    VmDecodedFunction *functions;
    uint32_t function_count;
} VmDecodedModule;

bool vm_decode_function(const NvmModule *module, uint32_t function_index,
                        VmDecodedFunction *out, char error[VM_DECODE_ERROR_SIZE]);
bool vm_decode_module(const NvmModule *module, VmDecodedModule *out,
                      char error[VM_DECODE_ERROR_SIZE]);
bool vm_decoded_function_has_boundary(const VmDecodedFunction *function,
                                      uint32_t byte_offset);
const VmDecodedInstruction *vm_decoded_function_at(
    const VmDecodedFunction *function, uint32_t byte_offset);
void vm_decoded_function_free(VmDecodedFunction *function);
void vm_decoded_module_free(VmDecodedModule *module);

#endif /* NANOVM_VM_DECODE_H */
