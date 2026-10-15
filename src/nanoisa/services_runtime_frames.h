#ifndef NANOISA_SERVICES_RUNTIME_FRAMES_H
#define NANOISA_SERVICES_RUNTIME_FRAMES_H
#include "services_runtime.h"

/* Private checked arena/call transfers, not an interpreter, emitter or public
 * admission API. External serialization and output-disjointness requirements
 * from services_runtime.h apply. Getters preserve outputs on failure. The eventual
 * matched dispatcher must evaluate ordinary instructions and branch predicates;
 * selecting a decoded successor here is not a proof of that evaluation. */
typedef struct {
    NvmServicesRuntimeMode mode;
    uint32_t function, instruction, byte_offset;
    uint16_t depth, locals, staging_slots, operand_peak;
    uint32_t locals_base, staging_base, stack_base, stack_count;
    uint32_t reference_base, region_floor;
} NvmServicesRuntimeFrameView;
/* Start only the currently prepared root, with no residual values/references. */
NvmServicesRuntimeStatus nvm_services_runtime_frame_start(NvmServicesRuntime *);
bool nvm_services_runtime_frame_view(const NvmServicesRuntime *,NvmServicesRuntimeFrameView *);
bool nvm_services_runtime_frame_local(const NvmServicesRuntime *,uint16_t,uint32_t *);
bool nvm_services_runtime_frame_operand(const NvmServicesRuntime *,uint16_t,uint32_t *);
/* Empty output slot within the prepared operand peak; does not change count. */
bool nvm_services_runtime_frame_reserve(const NvmServicesRuntime *,uint16_t,uint32_t *);
/* Empty last staging root for a matched private handler; no allocation. */
bool nvm_services_runtime_frame_scratch(const NvmServicesRuntime *,uint32_t *);
bool nvm_services_runtime_frame_reference(const NvmServicesRuntime *,uint16_t,uint32_t *);
/* Exact current STORE_LOCAL/OWN_STORE_LOCAL. Other ordinary handlers use the
 * private carrier primitives, then next checks the completed physical stack. */
NvmServicesRuntimeStatus nvm_services_runtime_frame_store(NvmServicesRuntime *);
NvmServicesRuntimeStatus nvm_services_runtime_frame_next(NvmServicesRuntime *,uint8_t successor);
NvmServicesRuntimeStatus nvm_services_runtime_frame_call(NvmServicesRuntime *);
NvmServicesRuntimeStatus nvm_services_runtime_frame_return(NvmServicesRuntime *);
/* Exact current region/borrow opcodes; local origins cannot cross the frame's
 * region floor or end a borrowed formal. Complete with frame_next afterwards. */
NvmServicesRuntimeStatus nvm_services_runtime_frame_region_begin(NvmServicesRuntime *);
NvmServicesRuntimeStatus nvm_services_runtime_frame_region_end(NvmServicesRuntime *);
NvmServicesRuntimeStatus nvm_services_runtime_frame_borrow(NvmServicesRuntime *);
NvmServicesRuntimeStatus nvm_services_runtime_frame_end_reference(NvmServicesRuntime *);
#endif
