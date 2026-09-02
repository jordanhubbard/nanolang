/*
 * NanoVM - Virtual Machine Execution Engine
 *
 * Loads an NvmModule and executes its bytecode via switch dispatch.
 */

#ifndef NANOVM_VM_H
#define NANOVM_VM_H

#include "value.h"
#include "heap.h"
#include "vm_decode.h"
#include "vm_dispatch.h"
#include "../nanoisa/isa.h"
#include "../nanoisa/nvm_format.h"

/* ========================================================================
 * VM Configuration
 * ======================================================================== */

#define VM_STACK_INITIAL    4096
#define VM_MAX_FRAMES       1024
#define VM_MAX_GLOBALS      4096
#define VM_PROFILE_TRIPLES  4096

typedef struct {
    uint32_t key;
    uint64_t count;
    bool used;
} VmProfileTriple;

typedef struct {
    bool enabled;
    bool has_previous;
    bool has_previous_pair;
    uint8_t previous;
    uint16_t previous_pair;
    uint64_t retired;
    uint64_t opcode_counts[256];
    uint64_t pair_counts[256 * 256];
    VmProfileTriple triples[VM_PROFILE_TRIPLES];
    uint64_t branches;
    uint64_t branches_taken;
    uint64_t direct_calls;
    uint64_t indirect_calls;
    uint64_t extern_calls;
    uint64_t module_calls;
    uint64_t traps;
    uint64_t ffi_request_bytes;
    uint64_t ffi_response_bytes;
    uint64_t ffi_elapsed_ns;
    uint64_t ffi_failures;
    uint32_t max_stack_depth;
    uint32_t max_frame_depth;
} VmProfile;

typedef struct {
    VmString **strings;
    uint32_t count;
} VmModuleConstants;

/* ========================================================================
 * Call Frame
 * ======================================================================== */

typedef struct {
    uint32_t fn_idx;          /* Function table index */
    uint32_t return_ip;       /* Instruction pointer to return to */
    uint32_t stack_base;      /* Stack index where this frame's locals begin */
    uint16_t local_count;     /* Number of locals (including params) */
    VmClosure *closure;       /* Non-NULL if this is a closure call */
    const NvmModule *module;  /* Module this frame belongs to (for cross-module calls) */
    uint32_t current_line;    /* Most recently seen OP_DEBUG_LINE value (0 = unknown) */
    uint32_t current_col;     /* Column from most recent debug entry (0 = unknown) */
} VmCallFrame;

/* ========================================================================
 * VM Execution Result
 * ======================================================================== */

typedef enum {
    VM_OK = 0,
    VM_ERR_STACK_OVERFLOW,
    VM_ERR_STACK_UNDERFLOW,
    VM_ERR_CALL_DEPTH,
    VM_ERR_INVALID_OPCODE,
    VM_ERR_TYPE_ERROR,
    VM_ERR_OUT_OF_BOUNDS,
    VM_ERR_DIV_ZERO,         /* Not used (div by zero = 0) but reserved */
    VM_ERR_ASSERT_FAILED,
    VM_ERR_UNDEFINED_GLOBAL,
    VM_ERR_UNDEFINED_FUNCTION,
    VM_ERR_NOT_IMPLEMENTED,
    VM_ERR_MEMORY,
    VM_ERR_DECODE
} VmResult;

/* ========================================================================
 * VM State
 * ======================================================================== */

typedef struct VmState {
    /* Module being executed */
    const NvmModule *module;
    const NvmModule *root_module;
    VmDecodedModule decoded_module;
    bool decoded_module_valid;
    /* True once cross-module OP_CALL_MODULE handles have been resolved
     * against the linked-module table. Reset when linking changes. */
    bool module_calls_resolved;
    VmDispatchModule dispatch_module;
    bool dispatch_module_valid;
    /* True once every module the VM will execute (root plus all linked
     * modules) has passed nvm_verify(). This is the safety proof that
     * lets the hot path use the unchecked private stack handlers: the
     * verifier has already established stack depth and index bounds for
     * every reachable instruction, so re-checking them at dispatch time
     * is redundant. Cleared conservatively whenever a module changes or a
     * new, unverified module is linked. */
    bool verified;    VmModuleConstants module_constants;

    /* Operand stack */
    NanoValue *stack;
    uint32_t stack_size;
    uint32_t stack_capacity;

    /* Call stack */
    VmCallFrame frames[VM_MAX_FRAMES];
    uint32_t frame_count;

    /* Current execution state */
    uint32_t ip;              /* Instruction pointer (byte offset in code) */
    uint32_t current_fn;      /* Current function index */

    /* Global variables.
     * Dynamically sized from the declared/used global slots of the root and
     * linked modules instead of embedding VM_MAX_GLOBALS values in every VM.
     * global_capacity is the number of allocated slots (0 => unallocated);
     * global_count is the high-water mark of initialized slots. */
    NanoValue *globals;
    uint32_t global_capacity;
    uint32_t global_count;

    /* Byte-addressed linear memory for portable loads, stores, and Forth. */
    uint8_t *memory;
    uint64_t memory_size;

    /* GC Heap */
    VmHeap heap;

    /* Linked modules for cross-module calls */
    const NvmModule **linked_modules;
    VmDecodedModule *decoded_linked_modules;
    bool *decoded_linked_modules_valid;
    VmDispatchModule *dispatch_linked_modules;
    bool *dispatch_linked_modules_valid;
    VmModuleConstants *linked_module_constants;
    uint32_t linked_module_count;
    uint32_t linked_module_capacity;

    /* Output capture (NULL = stdout) */
    FILE *output;

    /* FFI isolation: if true, use co-process for extern calls */
    bool isolate_ffi;

    /* Original pipe fds (used for INIT/SHUTDOWN and large-payload fallback) */
    int cop_in_fd;            /* Pipe to co-process stdin (-1 if none) */
    int cop_out_fd;           /* Pipe from co-process stdout (-1 if none) */
    int cop_pid;              /* Co-process PID (-1 if none) */

    /* Shared-memory mailbox (fast path for small FFI calls, e.g. pixels) */
    struct CopMailbox *cop_mailbox;   /* mmap'd before fork, inherited by child */
    size_t cop_mailbox_size;
    int    cop_sig_send_fd;   /* Parent writes 1 byte to wake child (-1 if none) */
    int    cop_sig_recv_fd;   /* Parent reads 1 byte when child done (-1 if none) */
    int    cop_timeout_ms;    /* Per-call timeout in ms; -1 = unlimited */

    /* Error info */
    VmResult last_error;
    char error_msg[256];

    /* Debug mode: emit stack trace on any runtime error.
     * Enabled via --debug flag or DEBUG env var. */
    bool debug_mode;

    /* Optional per-instruction diagnostics. Configured once in vm_init(). */
    bool opcode_trace;

    /* Optional low-overhead instruction and control-flow counters. */
    VmProfile profile;

    /* Profile that selects which private dispatch superinstructions the
     * optimized IR fuses.  Defaults to none, so an unconfigured VM runs the
     * plain verified stream; configure it before loading a module. */
    VmDispatchProfile dispatch_profile;
} VmState;

/* ========================================================================
 * Co-Processor Trap Model
 *
 * The NanoISA core (vm_core_execute) handles all pure computation.
 * When it encounters an external operation — I/O, FFI, or halt — it
 * returns a VmTrap descriptor.  The runtime harness (vm_execute /
 * vm_call_function) handles the trap and resumes the core.
 *
 * This separation defines the hardware interface contract: the 83+
 * pure-compute opcodes run on the FPGA; the 5 trap types are bus
 * transactions between the FPGA and the host co-processor.
 * ======================================================================== */

typedef enum {
    TRAP_NONE = 0,          /* Normal completion (RET from top frame) */
    TRAP_EXTERN_CALL,       /* OP_CALL_EXTERN — FFI request */
    TRAP_PRINT,             /* OP_PRINT — stdout output */
    TRAP_ASSERT,            /* OP_ASSERT — assertion check */
    TRAP_HALT,              /* OP_HALT — explicit stop */
    TRAP_ERROR              /* Runtime error */
} VmTrapType;

typedef struct {
    VmTrapType type;
    union {
        struct { uint32_t import_idx; NanoValue args[16]; int argc; } extern_call;
        struct { NanoValue value; bool newline; } print;
        struct { NanoValue condition; } assert_check;
        struct { VmResult code; } error;
    } data;
} VmTrap;

/* ========================================================================
 * VM API
 * ======================================================================== */

/* Initialize VM state for a module */
void vm_init(VmState *vm, const NvmModule *module);

/* Destroy VM state (free stack, heap, etc.) */
void vm_destroy(VmState *vm);

/* Execute from the module's entry point. Returns VM_OK on success.
 * This is the runtime harness that calls vm_core_execute() in a loop
 * and handles each trap. */
VmResult vm_execute(VmState *vm);

/* Execute a specific function by index. Returns VM_OK on success. */
VmResult vm_call_function(VmState *vm, uint32_t fn_idx, NanoValue *args, uint16_t arg_count);

/* Invoke one function as an isolated host call on a persistent VM.
 * Exact arity is required. On success, ownership of the returned value moves
 * to out_result; pass NULL to discard it. On failure, temporary operand-stack
 * values and call frames are removed while globals and heap state remain. */
VmResult vm_invoke(VmState *vm, uint32_t fn_idx, const NanoValue *args,
                   uint16_t arg_count, NanoValue *out_result);

/* Run pure NanoISA instructions until a trap occurs.
 * This is the "processor" — no I/O, no dlopen, no stdout.
 * On an FPGA, this would be implemented in RTL. */
VmTrap vm_core_execute(VmState *vm);

/* Get the return value (top of stack after execution) */
NanoValue vm_get_result(VmState *vm);

/* Reset and enable or disable execution profiling. */
void vm_profile_enable(VmState *vm, bool enabled);

/* Select which private dispatch superinstructions are fused when the VM
 * projects a module's optimized dispatch IR.  Rebuilds are required to take
 * effect, so call this before the module is loaded (or invalidate and rebuild
 * afterwards). */
void vm_set_dispatch_profile(VmState *vm, VmDispatchProfile profile);

/* Write deterministic JSON containing execution counters. */
bool vm_profile_write_json(const VmState *vm, FILE *out);

/* Get error message string */
const char *vm_error_string(VmResult result);

/* Link a module for legacy roots without MODULE_REFS (OP_CALL_MODULE).
 * Returns the module index, or (uint32_t)-1 on error. */
uint32_t vm_link_module(VmState *vm, const NvmModule *mod);

/* Link the next dependency declared by the root module's MODULE_REFS section.
 * The name and declaration order define the OP_CALL_MODULE index. */
uint32_t vm_link_named_module(VmState *vm, const char *name,
                              const NvmModule *mod);

/* Ensure the dynamically-sized globals array can hold at least `count` slots.
 * Grows (and zero-initializes new slots) up to VM_MAX_GLOBALS. Returns true on
 * success, false if `count` exceeds VM_MAX_GLOBALS or allocation fails. */
bool vm_ensure_globals(VmState *vm, uint32_t count);

/* Resolve every cross-module OP_CALL_MODULE operand pair into a direct
 * callable handle against the linked-module table. Runs once after
 * linking; dispatch then follows the handle instead of re-indexing the
 * module/function tables on every call. Returns false on a decode error. */
bool vm_resolve_module_calls(VmState *vm);

/* Mark one mutable module's cached instructions stale before changing it. */
void vm_invalidate_module(VmState *vm, const NvmModule *module);
void vm_invalidate_decoded_module(VmState *vm, const NvmModule *module);

/* Atomically decode a module again after mutation. Returns false on malformed code. */
bool vm_rebuild_module(VmState *vm, const NvmModule *module);
VmResult vm_rebuild_decoded_module(VmState *vm, const NvmModule *module);

/* Resize linear memory, preserving existing bytes and zeroing new storage. */
bool vm_memory_resize(VmState *vm, uint64_t size);

/* ========================================================================
 * Debug / Stack Trace
 * ======================================================================== */

/* Print a source-mapped stack trace for the current VM state to `out`.
 * Each frame is printed as:
 *   #N  <function_name>  line <line>  (module: <module_name>)
 * Frames without debug info show "line ?" or "??" for unknown fields.
 * Pass stderr or any FILE* for `out`. */
void vm_stack_trace(const VmState *vm, FILE *out);

#endif /* NANOVM_VM_H */
