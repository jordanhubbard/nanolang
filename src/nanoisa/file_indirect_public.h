#ifndef NANOISA_FILE_INDIRECT_PUBLIC_H
#define NANOISA_FILE_INDIRECT_PUBLIC_H
#include "file_public.h"
#include "file_indirect_report.h"
/* I require explicit revision 1 options, including a zero limit. No default is
 * substituted. Input/options/output storage is valid, immutable and disjoint
 * throughout the invocation. I publish a scalar only after clean destruction. */
#ifdef __cplusplus
extern "C" {
#endif
NvmFileIndirectExecutionReport nvm_file_execute_indirect_bytes(NvmFileHostGrant *,
    const uint8_t *, size_t, const NvmFileIndirectOptions *, NvmFileScalar *);
/* Serialized nonexecuting emission. Identifier/output rules match file_public;
 * generated nvm_file_indirect_program_<identifier> takes grant, options, scalar. */
NvmFileRuntimeStatus nvm2c_emit_file_indirect_bytes(const uint8_t *, size_t,
    const char *entry_identifier, char **out, char *diagnostic, size_t);
#ifdef __cplusplus
}
#endif
#endif
