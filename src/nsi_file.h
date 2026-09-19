#ifndef NL_NSI_FILE_H
#define NL_NSI_FILE_H

#include "nsi_cap.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Private native adapter only: no source, NanoISA import or NSI dispatch uses
 * this interface. Context creation/use is serialized by the caller. */
typedef struct NlFileService NlFileService;
/* I report service plus capability storage without allocating. Failure preserves *out. */
bool nl_file_service_storage_bound(size_t *out);

typedef struct {
    uint64_t context_id;
    NlCap cap;
} NlFileToken;

typedef enum {
    NL_FILE_OK = 0,
    NL_FILE_ARGUMENT,
    NL_FILE_DISPOSED,
    NL_FILE_TOKEN,
    NL_FILE_RIGHTS,
    NL_FILE_CAPACITY,
    NL_FILE_LIMIT,
    NL_FILE_MEMORY,
    NL_FILE_IO,
    NL_FILE_DIRECTION
} NlFileStatus;

typedef struct {
    NlFileStatus status;
    int host_errno;       /* Saved immediately; zero may mean unavailable. */
    int cleanup_errno;    /* Separate rollback close error; zero is possible. */
    size_t bytes;         /* Exact progress, including a partial I/O failure. */
    bool eof;
    bool consumed;        /* True after an accepted close/disposal/transfer. */
    bool cleanup_failed;
} NlFileResult;

/* Failure leaves *out unchanged. The private context owns every acquired file. */
NlFileResult nl_file_service_create(NlFileService **out);
NlFileResult nl_file_acquire_temp(NlFileService *, uint32_t rights, NlFileToken *out);
/* These operations retain the token. Invalid arguments/rights/direction do not
 * touch the stream or buffer. Actual read/write errors retain partial progress.
 * Both changes of direction require a successful explicit rewind. */
NlFileResult nl_file_read(NlFileService *, const NlFileToken *, void *, size_t);
NlFileResult nl_file_write(NlFileService *, const NlFileToken *, const void *, size_t);
NlFileResult nl_file_rewind(NlFileService *, const NlFileToken *);
/* Transfer can alias out/token. Failure preserves both; success invalidates every
 * old token copy while retaining this file and its stream direction. */
NlFileResult nl_file_transfer(NlFileService *, const NlFileToken *, NlFileToken *out);
/* A valid close consumes even on host I/O error. A rejected token never closes. */
NlFileResult nl_file_consume_close(NlFileService *, const NlFileToken *);
/* Disposal is terminal and idempotent; it attempts every remaining close and
 * retains the first error. Destruction also releases C storage; no later pointer
 * use is valid. Neither function executes a caller-supplied cleanup callback. */
NlFileResult nl_file_service_dispose(NlFileService *);
NlFileResult nl_file_service_destroy(NlFileService *);

#endif
