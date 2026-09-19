#ifndef NL_NSI_GPU_H
#define NL_NSI_GPU_H
#include "nsi_cap.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* I expose a private Linux LP64 CUDA adapter, not source/NSI/VM admission.
 * All calls, across contexts too, must be serialized by the caller. */
#define NL_GPU_CONTEXT_LIMIT 8u
#define NL_GPU_BUFFER_LIMIT 64u
#define NL_GPU_BYTE_LIMIT ((size_t)1048576)
typedef struct NlGpuService NlGpuService;
typedef struct { uint64_t context_id; NlCap cap; } NlGpuToken;
typedef enum {
    NL_GPU_OK, NL_GPU_ARGUMENT, NL_GPU_UNAVAILABLE, NL_GPU_MEMORY,
    NL_GPU_CAPACITY, NL_GPU_LIMIT, NL_GPU_TOKEN, NL_GPU_RIGHTS,
    NL_GPU_DISPOSED, NL_GPU_TERMINAL, NL_GPU_DRIVER
} NlGpuStatus;
typedef struct {
    NlGpuStatus status;
    int driver_error, cleanup_error;
    uint64_t record_id;
    size_t bytes;
    unsigned free_attempts, freed_count, context_destroy_attempts;
    bool consumed, cleanup_failed, release_unknown, context_restore_unknown;
    bool context_destroyed, context_unknown, library_retained;
} NlGpuResult;
typedef struct {
    int ordinal, driver_version;
    char name[256];
    unsigned char uuid[16];
    bool cuda_gpu;
} NlGpuDevice;
typedef struct {
    uint64_t record_id;
    unsigned tracked_allocations, free_attempts, freed_count;
    bool quarantined, context_retained, library_retained;
    bool release_unknown, context_restore_unknown, context_unknown;
    int first_error;
} NlGpuLifetime;
typedef struct {
    bool acquisition_latched;
    unsigned records, quarantined;
    NlGpuLifetime lifetimes[NL_GPU_CONTEXT_LIMIT];
} NlGpuDiagnostics;

/* I publish outputs only on success. Inputs/output must be valid C storage;
 * read output cannot overlap its token. Transfer supports out==token.
 * Writing a token output must not discard an unrelated live owner. */
NlGpuResult nl_gpu_service_create(int ordinal, NlGpuService **out);
NlGpuResult nl_gpu_service_device(const NlGpuService *, NlGpuDevice *out);
NlGpuResult nl_gpu_buffer_allocate(NlGpuService *, size_t bytes, uint32_t rights, NlGpuToken *out);
NlGpuResult nl_gpu_buffer_write(NlGpuService *, const NlGpuToken *, size_t offset,
                                const void *bytes, size_t count);
NlGpuResult nl_gpu_buffer_read(NlGpuService *, const NlGpuToken *, size_t offset,
                               void *bytes, size_t count);
NlGpuResult nl_gpu_buffer_transfer(NlGpuService *, const NlGpuToken *, NlGpuToken *out);
NlGpuResult nl_gpu_buffer_close(NlGpuService *, const NlGpuToken *);
NlGpuResult nl_gpu_service_dispose(NlGpuService *);
/* I free wrapper storage, never a quarantined raw context/library record.
 * No call using that wrapper is valid after destroy. */
NlGpuResult nl_gpu_service_destroy(NlGpuService *);
NlGpuDiagnostics nl_gpu_service_diagnostics(void);
#endif
