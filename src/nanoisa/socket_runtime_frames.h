#ifndef NANOISA_SOCKET_RUNTIME_FRAMES_H
#define NANOISA_SOCKET_RUNTIME_FRAMES_H
#include "socket_runtime.h"

/* Private checked arena/call transfers, not an interpreter, emitter or public
 * admission API. External serialization and output-disjointness requirements
 * from socket_runtime.h apply. Getters preserve outputs on failure. The eventual
 * matched dispatcher must evaluate ordinary instructions and branch predicates;
 * selecting a decoded successor here is not a proof of that evaluation. */
typedef struct {
    NvmSocketRuntimeMode mode;
    uint32_t function, instruction, byte_offset;
    uint16_t depth, locals, staging_slots, operand_peak;
    uint32_t locals_base, staging_base, stack_base, stack_count;
    uint32_t reference_base, region_floor;
} NvmSocketRuntimeFrameView;
/* Start only the currently prepared root, with no residual values/references. */
NvmSocketRuntimeStatus nvm_socket_runtime_frame_start(NvmSocketRuntime *);
bool nvm_socket_runtime_frame_view(const NvmSocketRuntime *,NvmSocketRuntimeFrameView *);
bool nvm_socket_runtime_frame_local(const NvmSocketRuntime *,uint16_t,uint32_t *);
bool nvm_socket_runtime_frame_operand(const NvmSocketRuntime *,uint16_t,uint32_t *);
/* Empty output slot within the prepared operand peak; does not change count. */
bool nvm_socket_runtime_frame_reserve(const NvmSocketRuntime *,uint16_t,uint32_t *);
/* Empty last staging root for a matched private handler; no allocation. */
bool nvm_socket_runtime_frame_scratch(const NvmSocketRuntime *,uint32_t *);
bool nvm_socket_runtime_frame_reference(const NvmSocketRuntime *,uint16_t,uint32_t *);
/* Exact current STORE_LOCAL/OWN_STORE_LOCAL. Other ordinary handlers use the
 * private carrier primitives, then next checks the completed physical stack. */
NvmSocketRuntimeStatus nvm_socket_runtime_frame_store(NvmSocketRuntime *);
NvmSocketRuntimeStatus nvm_socket_runtime_frame_next(NvmSocketRuntime *,uint8_t successor);
NvmSocketRuntimeStatus nvm_socket_runtime_frame_call(NvmSocketRuntime *);
NvmSocketRuntimeStatus nvm_socket_runtime_frame_return(NvmSocketRuntime *);
/* Exact current region/borrow opcodes; local origins cannot cross the frame's
 * region floor or end a borrowed formal. Complete with frame_next afterwards. */
NvmSocketRuntimeStatus nvm_socket_runtime_frame_region_begin(NvmSocketRuntime *);
NvmSocketRuntimeStatus nvm_socket_runtime_frame_region_end(NvmSocketRuntime *);
NvmSocketRuntimeStatus nvm_socket_runtime_frame_borrow(NvmSocketRuntime *);
NvmSocketRuntimeStatus nvm_socket_runtime_frame_end_reference(NvmSocketRuntime *);
#endif
