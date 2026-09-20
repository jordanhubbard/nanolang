#ifndef NANOISA_FILE_CYCLIC_PUBLIC_H
#define NANOISA_FILE_CYCLIC_PUBLIC_H
#include "file_public.h"
#include "file_cyclic_report.h"
/* I require explicit revision 1 options, including a zero limit. No default is
 * substituted. Input/options/output storage is valid, immutable and disjoint
 * throughout the invocation. I publish a scalar only after clean destruction. */
#ifdef __cplusplus
extern "C" {
#endif
NvmFileCyclicExecutionReport nvm_file_execute_cyclic_bytes(NvmFileHostGrant *,
    const uint8_t *, size_t, const NvmFileCyclicOptions *, NvmFileScalar *);
/* Serialized nonexecuting emission. Identifier/output rules match file_public;
 * generated nvm_file_cyclic_program_<identifier> takes grant, options, scalar. */
NvmFileRuntimeStatus nvm2c_emit_file_cyclic_bytes(const uint8_t *, size_t,
    const char *entry_identifier, char **out, char *diagnostic, size_t);
#ifdef __cplusplus
}
#endif
#endif
