#ifndef NL_NSI_SHM_H
#define NL_NSI_SHM_H

#include "nsi_cap.h"

#include <stddef.h>
#include <stdint.h>

#define NL_SHM_OK 0
#define NL_SHM_ERR 1
#define NL_SHM_ERR_RIGHTS 2
#define NL_SHM_ERR_RANGE 3
#define NL_SHM_ERR_ALIGN 4
#define NL_SHM_ERR_OWNED 5
#define NL_SHM_ERR_SEALED 6
#define NL_SHM_ERR_DIR 7

#define NL_SHM_DIR_READ 1
#define NL_SHM_DIR_WRITE 2

typedef enum {
    NL_SHM_AUDIO = 0,
    NL_SHM_GRAPHICS,
    NL_SHM_NET,
    NL_SHM_FILE,
    NL_SHM_GPU
} NlShmKind;

typedef struct NlShm NlShm;

typedef struct {
    double control_ns;
    double copy_ns;
    double map_ns;
    size_t payload;
    int copies;
    int mappings;
    int used_copy_fallback;
} NlShmBench;

NlShm *nl_shm_create(NlCapTable *t, const NlCap *cap, size_t bytes, NlShmKind kind,
                     int force_copy);
void nl_shm_destroy(NlShm *r);

int nl_shm_map(NlShm *r, size_t off, size_t len, size_t align, int dir);
int nl_shm_read(NlShm *r, size_t off, size_t len, void *dst);
int nl_shm_write(NlShm *r, size_t off, size_t len, const void *src);
int nl_shm_seal(NlShm *r);
int nl_shm_transfer(NlShm *r);
int nl_shm_borrow(NlShm *r);
int nl_shm_return(NlShm *r);
int nl_shm_revoke(NlShm *r);

int nl_shm_copy_fallback(const NlShm *r);
int nl_shm_service_owns(const NlShm *r);
NlShmKind nl_shm_kind(const NlShm *r);
size_t nl_shm_size(const NlShm *r);

int nl_shm_bench(size_t payload, NlShmBench *out);

#endif
