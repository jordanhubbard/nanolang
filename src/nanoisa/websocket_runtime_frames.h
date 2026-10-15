#ifndef NANOISA_WEBSOCKET_RUNTIME_FRAMES_H
#define NANOISA_WEBSOCKET_RUNTIME_FRAMES_H
#include "websocket_runtime.h"

/* Private checked arena/call transfers, not an interpreter, emitter or public
 * admission API. External serialization and output-disjointness requirements
 * from websocket_runtime.h apply. Getters preserve outputs on failure. The eventual
 * matched dispatcher must evaluate ordinary instructions and branch predicates;
 * selecting a decoded successor here is not a proof of that evaluation. */
typedef struct {
    NvmWebSocketRuntimeMode mode;
    uint32_t function, instruction, byte_offset;
    uint16_t depth, locals, staging_slots, operand_peak;
    uint32_t locals_base, staging_base, stack_base, stack_count;
    uint32_t reference_base, region_floor;
} NvmWebSocketRuntimeFrameView;
/* Start only the currently prepared root, with no residual values/references. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_frame_start(NvmWebSocketRuntime *);
bool nvm_websocket_runtime_frame_view(const NvmWebSocketRuntime *,NvmWebSocketRuntimeFrameView *);
bool nvm_websocket_runtime_frame_local(const NvmWebSocketRuntime *,uint16_t,uint32_t *);
bool nvm_websocket_runtime_frame_operand(const NvmWebSocketRuntime *,uint16_t,uint32_t *);
/* Empty output slot within the prepared operand peak; does not change count. */
bool nvm_websocket_runtime_frame_reserve(const NvmWebSocketRuntime *,uint16_t,uint32_t *);
/* Empty last staging root for a matched private handler; no allocation. */
bool nvm_websocket_runtime_frame_scratch(const NvmWebSocketRuntime *,uint32_t *);
bool nvm_websocket_runtime_frame_reference(const NvmWebSocketRuntime *,uint16_t,uint32_t *);
/* Exact current STORE_LOCAL/OWN_STORE_LOCAL. Other ordinary handlers use the
 * private carrier primitives, then next checks the completed physical stack. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_frame_store(NvmWebSocketRuntime *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_frame_next(NvmWebSocketRuntime *,uint8_t successor);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_frame_call(NvmWebSocketRuntime *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_frame_return(NvmWebSocketRuntime *);
/* Exact current region/borrow opcodes; local origins cannot cross the frame's
 * region floor or end a borrowed formal. Complete with frame_next afterwards. */
NvmWebSocketRuntimeStatus nvm_websocket_runtime_frame_region_begin(NvmWebSocketRuntime *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_frame_region_end(NvmWebSocketRuntime *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_frame_borrow(NvmWebSocketRuntime *);
NvmWebSocketRuntimeStatus nvm_websocket_runtime_frame_end_reference(NvmWebSocketRuntime *);
#endif
