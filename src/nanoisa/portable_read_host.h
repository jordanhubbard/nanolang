#ifndef NANOISA_PORTABLE_READ_HOST_H
#define NANOISA_PORTABLE_READ_HOST_H
#include <stdint.h>

/* My private host API grants no NanoISA profile or execution admission. */
#define NPR_PATH_LIMIT 4096u
#define NPR_TEXT_LIMIT 1048576u
#define NPR_ALLOWLIST_LIMIT 64u
typedef enum {
    NPR_OK = 0, NPR_DENIED = 1, NPR_LIMIT = 2,
    NPR_MEMORY = 3, NPR_INVALID = 4
} NprStatus;
typedef struct { const uint8_t *data; uint32_t length; } NprPath;
typedef struct NprFileHost NprFileHost;
typedef int32_t (*NprReadCallback)(void *context,
    const uint8_t *path, uint32_t path_length,
    uint8_t *destination, uint32_t capacity, uint32_t *length_out);
typedef struct { NprReadCallback read; void *context; } NprHostBinding;

/* I copy every path. Failure preserves *out. Empty allowlists deny all reads.
 * These APIs are serialized. The context stays alive until the call returns;
 * no callback may retain buffers or reenter/mutate the calling runtime.
 * C storage validity is the caller's responsibility, not a sandbox guarantee. */
NprStatus npr_file_host_create(const NprPath *, uint32_t, NprFileHost **);
NprStatus npr_file_host_destroy(NprFileHost *);
/* I borrow buffers. On error, destination is unpublished scratch and length
 * is unchanged. I reject overlap with each other and my entire context. */
int32_t npr_file_read(void *, const uint8_t *, uint32_t,
                      uint8_t *, uint32_t, uint32_t *);
#endif
