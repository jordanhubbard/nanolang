#ifndef NL_NSI_SOCKET_BINDING_H
#define NL_NSI_SOCKET_BINDING_H
#include <stdbool.h>
#include <stddef.h>

#include "nsi_binding.h"

typedef struct NlSocketBindingPlan NlSocketBindingPlan;
typedef NlBindingStatus NlSocketBindingStatus;
#define NL_SOCKET_BINDING_MAX_BYTES NL_BINDING_MAX_BYTES
#define NL_SOCKET_BINDING_MAX_TOKENS NL_BINDING_MAX_TOKENS
#define NL_SOCKET_BINDING_MAX_DEPTH NL_BINDING_MAX_DEPTH
#define NL_SOCKET_BINDING_MAX_OBJECTS NL_BINDING_MAX_OBJECTS
#define NL_SOCKET_BINDING_MAX_MEMBERS NL_BINDING_MAX_MEMBERS
#define NL_SOCKET_BINDING_MAX_ELEMENTS NL_BINDING_MAX_ELEMENTS
#define NL_SOCKET_BINDING_MAX_LEXEME NL_BINDING_MAX_LEXEME
#define NL_SOCKET_BINDING_MAX_ALLOCATION NL_BINDING_MAX_ALLOCATION
#define NL_SOCKET_BINDING_OK NL_BINDING_OK
#define NL_SOCKET_BINDING_INVALID NL_BINDING_INVALID
#define NL_SOCKET_BINDING_LIMIT NL_BINDING_LIMIT
#define NL_SOCKET_BINDING_MEMORY NL_BINDING_MEMORY
#define NL_SOCKET_BINDING_UNRESOLVED NL_BINDING_UNRESOLVED
#define NL_SOCKET_BINDING_IO NL_BINDING_IO
#define NL_SOCKET_BINDING_EXISTS NL_BINDING_EXISTS
#define NL_SOCKET_BINDING_UNSUPPORTED NL_BINDING_UNSUPPORTED

/* I validate one complete immutable byte span and render two owned outputs.
 * No path, publication, compiler or Socket service operation occurs. The source
 * declares the checked TCP catalog without synthetic network shadows. Paired
 * compiler admission and actual selected shadows remain separate requirements.
 * Input and output storage are disjoint and readable/writable for the call.
 * Failure preserves *out. Success owns a plan independent of input lifetime.
 * cJSON hooks must remain stable; no cJSON thread-safety claim follows. */
NlSocketBindingStatus nl_socket_binding_prepare(const unsigned char *bytes,
                                          size_t size, NlSocketBindingPlan **out);
void nl_socket_binding_free(NlSocketBindingPlan *plan);
/* Views borrow until free. NULL plan/size returns NULL without writing size. */
const unsigned char *nl_socket_binding_interface_bytes(const NlSocketBindingPlan *, size_t *size);
const unsigned char *nl_socket_binding_source_bytes(const NlSocketBindingPlan *, size_t *size);
size_t nl_socket_binding_storage_size(const NlSocketBindingPlan *);
size_t nl_socket_binding_peak_bound(const NlSocketBindingPlan *);
/* I report a conservative project-requested heap bound before any allocation.
 * Caller storage, stack, allocator overhead and libc internals are excluded.
 * Failure preserves *out; this query never allocates. */
bool nl_socket_binding_allocation_bound(size_t *out);
#endif
