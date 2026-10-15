#ifndef NANOISA_SERVICES_INDIRECT_PUBLIC_H
#define NANOISA_SERVICES_INDIRECT_PUBLIC_H
#include "services_public.h"
#include "services_indirect_report.h"
/* I require explicit revision 1 options, including a zero limit. No default is
 * substituted. Input/options/output storage is valid, immutable and disjoint
 * throughout the invocation. I publish a scalar only after clean destruction. */
#ifdef __cplusplus
extern "C" {
#endif
NvmServicesIndirectExecutionReport nvm_services_execute_indirect_bytes(NvmServicesHostGrant *,
    const uint8_t *, size_t, const NvmServicesIndirectOptions *, NvmServicesScalar *);
/* Serialized nonexecuting emission. My identifier is 1..63 ASCII letters/digits/underscores, starting with a letter;
 * output and diagnostics are disjoint; success publishes malloc-owned C99. The
 * generated nvm_services_indirect_program_<identifier> takes grant, options, scalar. */
NvmServicesRuntimeStatus nvm2c_emit_services_indirect_bytes(const uint8_t *, size_t,
    const char *entry_identifier, char **out, char *diagnostic, size_t);
#ifdef __cplusplus
}
#endif
#endif
