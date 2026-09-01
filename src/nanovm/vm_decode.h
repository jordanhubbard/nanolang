#ifndef NANOVM_VM_DECODE_H
#define NANOVM_VM_DECODE_H

#include "../nanoisa/isa.h"
#include "../nanoisa/nvm_format.h"

#include <stdbool.h>
#include <stdint.h>

#define VM_DECODE_ERROR_SIZE 256

/* ========================================================================
 * Execution architecture: three separate representations
 *
 *   1. Compact serialized bytecode      -> NvmModule (nvm_format).
 *   2. Verified instruction IR          -> VmDecodedFunction below.
 *   3. Optimized dispatch IR            -> VmDispatchFunction below.
 *
 * The verified IR decodes each function once, records instruction
 * boundaries, and proves that every branch and direct/tail call targets an
 * instruction boundary or the function-end sentinel.  It keeps byte-offset
 * indexed side tables so the verifier and diagnostics can reason about the
 * serialized form.
 *
 * The dispatch IR is a flat, execution-oriented projection of the verified
 * IR.  It carries no byte maps on the hot path: every slot already knows the
 * index of its fall-through successor and, for control transfers, the index
 * of its resolved in-function target.  The runtime advances an instruction
 * index instead of re-deriving one from a byte offset on every retired
 * instruction.
 * ======================================================================== */

typedef struct {
    uint32_t byte_offset;
    uint32_t next_byte_offset;
    uint32_t resolved_target;
    DecodedInstruction instruction;
} VmDecodedInstruction;

typedef struct {
    VmDecodedInstruction *instructions;
    uint32_t instruction_count;
    uint32_t code_size;
    uint32_t code_offset;
    uint8_t *boundaries;
    uint32_t *instruction_indices;
} VmDecodedFunction;

typedef struct {
    VmDecodedFunction *functions;
    uint32_t function_count;
} VmDecodedModule;

/* One slot of optimized dispatch IR.
 *
 *  - instruction / start_offset / next_offset mirror the verified slot but
 *    are copied so the hot loop never dereferences the verified IR.
 *  - next_index is the fall-through successor index, or VM_DISPATCH_NO_INDEX
 *    for the final slot of a function.
 *  - target_index is the resolved index of an in-function branch target, or
 *    VM_DISPATCH_NO_INDEX when the instruction has no in-function target
 *    (calls keep their callee function index in target_function).
 *  - target_function is the resolved callee function index for direct and
 *    tail calls, or VM_DISPATCH_NO_INDEX otherwise.
 *  - target_byte_offset is the code_offset-relative byte offset of an
 *    in-function target, kept for diagnostics and the byte-addressed IP. */
#define VM_DISPATCH_NO_INDEX UINT32_MAX

typedef struct {
    DecodedInstruction instruction;
    uint32_t start_offset;
    uint32_t next_offset;
    uint32_t next_index;
    uint32_t target_index;
    uint32_t target_function;
    uint32_t target_byte_offset;
} VmDispatchInstruction;

typedef struct {
    VmDispatchInstruction *instructions;
    uint32_t instruction_count;
    uint32_t code_size;
    uint32_t *offset_to_index; /* [code_size+1]; VM_DISPATCH_NO_INDEX if none */
} VmDispatchFunction;

typedef struct {
    VmDispatchFunction *functions;
    uint32_t function_count;
} VmDispatchModule;

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

/* Build the optimized dispatch IR from the verified IR.  The verified IR
 * must already be validated by vm_decode_function/vm_decode_module. */
bool vm_dispatch_function_build(const VmDecodedFunction *decoded,
                                VmDispatchFunction *out,
                                char error[VM_DECODE_ERROR_SIZE]);
bool vm_dispatch_module_build(const VmDecodedModule *decoded,
                              VmDispatchModule *out,
                              char error[VM_DECODE_ERROR_SIZE]);
/* Resolve a code_offset-relative byte offset to a dispatch instruction index.
 * Returns VM_DISPATCH_NO_INDEX when the offset is not an instruction start. */
uint32_t vm_dispatch_function_index_at(const VmDispatchFunction *function,
                                       uint32_t byte_offset);
void vm_dispatch_function_free(VmDispatchFunction *function);
void vm_dispatch_module_free(VmDispatchModule *module);

#endif /* NANOVM_VM_DECODE_H */
