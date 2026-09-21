/* I decode the required capture payload before granting execution authority.
 * This codec alone does not admit the feature or validate instruction edges. */
#ifndef NANOISA_CAPTURE_BINDINGS_H
#define NANOISA_CAPTURE_BINDINGS_H

#include "nvm_format.h"

#define NVM_CAPTURE_BINDINGS_VERSION 1u
#define NVM_CAPTURE_VALUE 0u
#define NVM_CAPTURE_SHARED 1u
#define NVM_CAPTURE_LOCAL 0u
#define NVM_CAPTURE_UPVALUE 1u

typedef struct {
    uint16_t local_count, upvalue_count;
    const uint8_t *local_modes, *upvalue_modes;
} NvmCaptureFunction;

typedef struct {
    uint32_t owner, instruction_offset, target;
    uint16_t capture_count;
    const uint8_t *sources; /* capture_count canonical four-byte records */
} NvmCaptureSite;

typedef struct {
    NvmCaptureFunction *functions;
    NvmCaptureSite *sites;
    uint32_t function_count, site_count;
    size_t allocation_bytes;
} NvmCaptureBindings;

typedef enum {
    NVM_CAPTURE_OK = 0,
    NVM_CAPTURE_INVALID,
    NVM_CAPTURE_LIMIT,
    NVM_CAPTURE_MEMORY
} NvmCaptureResult;

/* I borrow immutable data and own only the two index tables. data must remain
 * unchanged and live until nvm_capture_bindings_free. module's function table
 * must already be structurally decoded. limit bounds the combined table bytes.
 * On failure I preserve *out exactly; on success it must not replace live owned
 * tables. A successful payload decode is NOT an instruction/verifier proof. */
NvmCaptureResult nvm_capture_bindings_decode(const uint8_t *data, size_t size,
    const NvmModule *module, size_t limit, NvmCaptureBindings *out);
void nvm_capture_bindings_free(NvmCaptureBindings *bindings);

/* I return a decoded source only for an in-range descriptor. */
bool nvm_capture_source(const NvmCaptureSite *site, uint16_t index,
    uint8_t *kind, uint8_t *mode, uint16_t *slot);

#endif
