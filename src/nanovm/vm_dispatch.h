#ifndef NANOVM_VM_DISPATCH_H
#define NANOVM_VM_DISPATCH_H

/*
 * NanoVM - Optimized Dispatch IR
 *
 * The VM keeps three separate representations of a program:
 *
 *   1. Compact serialized bytecode   (NvmModule / nvm_format)
 *      - the on-disk / on-wire form; variable-length, byte addressed.
 *   2. Verified instruction IR       (VmDecodedModule / vm_decode)
 *      - one decode pass per function that establishes instruction
 *        boundaries and resolves every branch and direct call against a
 *        verified boundary map.  This is the representation the verifier
 *        reasons about; it is byte-offset addressed.
 *   3. Optimized dispatch IR         (VmDispatchModule / this file)
 *      - a projection of the verified IR that is shaped for the hot fetch
 *        loop.  Instructions are stored in a flat, instruction-indexed
 *        array so the linear path advances by an instruction index with no
 *        per-step byte-map lookup, and branch/call targets are precomputed
 *        as dispatch indices.  It is derived from (and validated against)
 *        the verified IR; it never re-derives program structure itself.
 *
 * This file owns only representation (3).  It depends on representation (2)
 * and must be rebuilt whenever the verified IR is rebuilt.
 */

#include "vm_decode.h"

#include <stdbool.h>
#include <stdint.h>

#define VM_DISPATCH_ERROR_SIZE 256

/*
 * A single dispatch instruction.  It carries everything the hot loop needs
 * without touching the verified IR again:
 *   - opcode + operands, copied from the verified instruction;
 *   - byte_offset / next_byte_offset preserve the byte-addressed `ip`
 *     contract the rest of the VM relies on (frames, returns, traps);
 *   - next_index is the dispatch index of the fall-through successor,
 *     precomputed so the linear path is a single array step;
 *   - branch_target is the dispatch index of a taken branch (or
 *     VM_DISPATCH_NO_INDEX when the instruction is not a branch);
 *   - branch_target_offset is the same target as a byte offset, so branch
 *     handlers can keep updating `ip` directly;
 *   - call_target is the resolved callee function index for direct and tail
 *     calls (or VM_DISPATCH_NO_INDEX otherwise).
 */
#define VM_DISPATCH_NO_INDEX ((uint32_t)0xFFFFFFFFu)

typedef struct {
    DecodedInstruction instruction;
    VmCallHandle call_handle;
    uint32_t byte_offset;
    uint32_t next_byte_offset;
    uint32_t next_index;
    uint32_t branch_target;
    uint32_t branch_target_offset;
    uint32_t call_target;
} VmDispatchInstruction;

typedef struct {
    VmDispatchInstruction *instructions;
    uint32_t instruction_count;
    uint32_t code_size;
    /* byte_offset -> dispatch index + 1 (0 means "not an instruction start") */
    uint32_t *offset_to_index;
} VmDispatchFunction;

typedef struct {
    VmDispatchFunction *functions;
    uint32_t function_count;
} VmDispatchModule;

/*
 * A cursor over the optimized dispatch IR.  The linear path uses
 * vm_dispatch_advance(); control transfers use vm_dispatch_seek() to
 * re-enter the stream at an instruction boundary.
 */
typedef struct {
    const VmDispatchFunction *function;
    uint32_t index;
} VmDispatchCursor;

/* Project the verified IR of one function into the optimized dispatch IR. */
bool vm_dispatch_build_function(const VmDecodedFunction *decoded,
                                VmDispatchFunction *out,
                                char error[VM_DISPATCH_ERROR_SIZE]);

/* Project every function of a verified module into optimized dispatch IR. */
bool vm_dispatch_build_module(const VmDecodedModule *decoded,
                              VmDispatchModule *out,
                              char error[VM_DISPATCH_ERROR_SIZE]);

void vm_dispatch_function_free(VmDispatchFunction *function);
void vm_dispatch_module_free(VmDispatchModule *module);

/* Position `cursor` at the instruction that begins at `byte_offset`.
 * Returns false when the offset is not an instruction boundary. */
bool vm_dispatch_seek(VmDispatchCursor *cursor,
                      const VmDispatchFunction *function,
                      uint32_t byte_offset);

/* The instruction the cursor currently points at, or NULL when exhausted. */
const VmDispatchInstruction *vm_dispatch_current(const VmDispatchCursor *cursor);

/* Advance the cursor to the fall-through successor and return it (or NULL). */
const VmDispatchInstruction *vm_dispatch_advance(VmDispatchCursor *cursor);

#endif /* NANOVM_VM_DISPATCH_H */
