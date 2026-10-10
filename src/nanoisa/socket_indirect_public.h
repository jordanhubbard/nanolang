#ifndef NANOISA_SOCKET_INDIRECT_PUBLIC_H
#define NANOISA_SOCKET_INDIRECT_PUBLIC_H
#include "socket_public.h"
#include "socket_indirect_report.h"
/* I require explicit revision 1 options, including a zero limit. No default is
 * substituted. Input/options/output storage is valid, immutable and disjoint
 * throughout the invocation. I publish a scalar only after clean destruction. */
#ifdef __cplusplus
extern "C" {
#endif
NvmSocketIndirectExecutionReport nvm_socket_execute_indirect_bytes(NvmSocketHostGrant *,
    const uint8_t *, size_t, const NvmSocketIndirectOptions *, NvmSocketScalar *);
/* Serialized nonexecuting emission. My identifier is 1..63 ASCII letters/digits/underscores, starting with a letter;
 * output and diagnostics are disjoint; success publishes malloc-owned C99. The
 * generated nvm_socket_indirect_program_<identifier> takes grant, options, scalar. */
NvmSocketRuntimeStatus nvm2c_emit_socket_indirect_bytes(const uint8_t *, size_t,
    const char *entry_identifier, char **out, char *diagnostic, size_t);
#ifdef __cplusplus
}
#endif
#endif
