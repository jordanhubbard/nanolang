#ifndef NANOISA_FILE_PUBLIC_H
#define NANOISA_FILE_PUBLIC_H
#include "file_host_grant.h"
#include "file_runtime.h"

/* I publish only a clean scalar after terminal cleanup. No owner or context
 * escapes. Valid input/output storage is disjoint for the complete call. */
typedef struct { uint8_t tag; int64_t value; } NvmFileScalar;
#ifdef __cplusplus
extern "C" {
#endif
NvmFileRuntimeReport nvm_file_execute_bytes(NvmFileHostGrant *,
    const uint8_t *, size_t, NvmFileScalar *out);
/* Nonexecuting emission, serialized by the owning query gate. The identifier
 * has 1..63 ASCII letters/digits/underscores, starting with a letter. Success
 * publishes malloc-owned C99 source, exporting nvm_file_program_<identifier>.
 * Failure preserves *out; diagnostic may be updated if its size is nonzero. */
NvmFileRuntimeStatus nvm2c_emit_file_bytes(const uint8_t *, size_t,
    const char *entry_identifier, char **out, char *diagnostic, size_t);
#ifdef __cplusplus
}
#endif
#endif
