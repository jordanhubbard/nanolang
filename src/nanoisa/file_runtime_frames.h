#ifndef NANOISA_FILE_RUNTIME_FRAMES_H
#define NANOISA_FILE_RUNTIME_FRAMES_H
#include "file_runtime.h"

/* Private checked arena/call transfers, not an interpreter, emitter or public
 * admission API. External serialization and output-disjointness requirements
 * from file_runtime.h apply. Getters preserve outputs on failure. The eventual
 * matched dispatcher must evaluate ordinary instructions and branch predicates;
 * selecting a decoded successor here is not a proof of that evaluation. */
typedef struct {
    NvmFileRuntimeMode mode;
    uint32_t function, instruction, byte_offset;
    uint16_t depth, locals, staging_slots, operand_peak;
    uint32_t locals_base, staging_base, stack_base, stack_count;
    uint32_t reference_base, region_floor;
} NvmFileRuntimeFrameView;
/* Start only the currently prepared root, with no residual values/references. */
NvmFileRuntimeStatus nvm_file_runtime_frame_start(NvmFileRuntime *);
bool nvm_file_runtime_frame_view(const NvmFileRuntime *,NvmFileRuntimeFrameView *);
bool nvm_file_runtime_frame_local(const NvmFileRuntime *,uint16_t,uint32_t *);
bool nvm_file_runtime_frame_operand(const NvmFileRuntime *,uint16_t,uint32_t *);
/* Empty output slot within the prepared operand peak; does not change count. */
bool nvm_file_runtime_frame_reserve(const NvmFileRuntime *,uint16_t,uint32_t *);
/* Empty last staging root for a matched private handler; no allocation. */
bool nvm_file_runtime_frame_scratch(const NvmFileRuntime *,uint32_t *);
bool nvm_file_runtime_frame_reference(const NvmFileRuntime *,uint16_t,uint32_t *);
/* Exact current STORE_LOCAL/OWN_STORE_LOCAL. Other ordinary handlers use the
 * private carrier primitives, then next checks the completed physical stack. */
NvmFileRuntimeStatus nvm_file_runtime_frame_store(NvmFileRuntime *);
NvmFileRuntimeStatus nvm_file_runtime_frame_next(NvmFileRuntime *,uint8_t successor);
NvmFileRuntimeStatus nvm_file_runtime_frame_call(NvmFileRuntime *);
NvmFileRuntimeStatus nvm_file_runtime_frame_return(NvmFileRuntime *);
/* Exact current region/borrow opcodes; local origins cannot cross the frame's
 * region floor or end a borrowed formal. Complete with frame_next afterwards. */
NvmFileRuntimeStatus nvm_file_runtime_frame_region_begin(NvmFileRuntime *);
NvmFileRuntimeStatus nvm_file_runtime_frame_region_end(NvmFileRuntime *);
NvmFileRuntimeStatus nvm_file_runtime_frame_borrow(NvmFileRuntime *);
NvmFileRuntimeStatus nvm_file_runtime_frame_end_reference(NvmFileRuntime *);
#endif
