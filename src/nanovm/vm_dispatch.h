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
 *
 * Private superinstructions live only inside representation (3).  A
 * profile-selected fusion pass may collapse a short run of verified
 * instructions into a single dispatch step whose `super_op` names a private
 * handler.  These handlers are an internal execution optimization: they are
 * never portable opcodes, never appear in the serialized bytecode
 * (representation 1) or the verified IR (representation 2), and never leak
 * frontend bookkeeping into the ISA.  The verified IR still owns program
 * meaning; a superinstruction only fuses steps the verifier already proved
 * safe, and it preserves the byte-addressed `ip` contract so frames, returns,
 * and traps observe exactly the same behavior as the unfused stream.
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
/*
 * Private superinstruction opcodes.  These identifiers live in a namespace
 * that is disjoint from the portable NanoISA opcode plane (0x00..0xFF): they
 * are 16-bit values that never round-trip through serialization, so a private
 * fusion can never be confused with a portable instruction.  VM_SUPER_NONE
 * marks an ordinary (unfused) dispatch instruction that dispatches on its
 * portable opcode.
 */
typedef enum {
    VM_SUPER_NONE = 0,
    /* LOAD_LOCAL idx ; AGG_GET field  ->  load a local aggregate field. */
    VM_SUPER_LOAD_LOCAL_FIELD,
    VM_SUPER__COUNT
} VmSuperOp;

/*
 * A dispatch fusion profile.  It is the only thing that decides whether a
 * private superinstruction is selected, so fusion stays a measured, opt-in
 * policy rather than a property of the portable program.  Each flag enables
 * one candidate fusion; every flag defaults off so an unconfigured build
 * executes the plain verified stream.  The optimization policy in
 * docs/NANOISA_OPTIMIZATION_POLICY.md governs when a flag may ship on.
 */
typedef struct {
    bool fuse_load_local_field;
} VmDispatchProfile;

/* A profile with every fusion disabled (the conservative default). */
VmDispatchProfile vm_dispatch_profile_none(void);

/* A profile that selects every implemented candidate fusion.  Intended for
 * measurement and tests, not as a shipping default. */
VmDispatchProfile vm_dispatch_profile_all(void);

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
    /* VM_SUPER_NONE for an ordinary instruction, otherwise the private
     * superinstruction to run.  `super_operand` carries the fused second
     * operand (e.g. the AGG_GET field index) so the primary operand still
     * lives in `instruction.operands`. */
    uint16_t super_op;
    uint16_t super_operand;
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

/* Project the verified IR of one function into the optimized dispatch IR,
 * applying the profile-selected private fusions.  Pass a
 * vm_dispatch_profile_none() profile to build the plain (unfused) stream. */
bool vm_dispatch_build_function(const VmDecodedFunction *decoded,
                                VmDispatchProfile profile,
                                VmDispatchFunction *out,
                                char error[VM_DISPATCH_ERROR_SIZE]);

/* Project every function of a verified module into optimized dispatch IR,
 * applying `profile` to every function. */
bool vm_dispatch_build_module(const VmDecodedModule *decoded,
                              VmDispatchProfile profile,
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
