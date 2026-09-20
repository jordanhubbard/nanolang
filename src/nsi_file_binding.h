#ifndef NL_NSI_FILE_BINDING_H
#define NL_NSI_FILE_BINDING_H
#include <stdbool.h>
#include <stddef.h>

#define NL_FILE_BINDING_MAX_BYTES 1048576u
#define NL_FILE_BINDING_MAX_TOKENS 8192u
#define NL_FILE_BINDING_MAX_DEPTH 64u
#define NL_FILE_BINDING_MAX_OBJECTS 256u
#define NL_FILE_BINDING_MAX_MEMBERS 64u
#define NL_FILE_BINDING_MAX_ELEMENTS 256u
#define NL_FILE_BINDING_MAX_LEXEME 4096u
#define NL_FILE_BINDING_MAX_ALLOCATION 16777216u

typedef struct NlFileBindingPlan NlFileBindingPlan;
typedef enum {
    NL_FILE_BINDING_OK, NL_FILE_BINDING_INVALID, NL_FILE_BINDING_LIMIT,
    NL_FILE_BINDING_MEMORY, NL_FILE_BINDING_UNRESOLVED,
    NL_FILE_BINDING_IO, NL_FILE_BINDING_EXISTS, NL_FILE_BINDING_UNSUPPORTED
} NlFileBindingStatus;

/* I validate one complete immutable byte span and render two owned outputs.
 * No path, publication, compiler or File service operation occurs. The source
 * uses the proposed service declaration; existing compilers still refuse it.
 * Input and output storage are disjoint and readable/writable for the call.
 * Failure preserves *out. Success owns a plan independent of input lifetime.
 * cJSON hooks must remain stable; no cJSON thread-safety claim follows. */
NlFileBindingStatus nl_file_binding_prepare(const unsigned char *bytes,
                                          size_t size, NlFileBindingPlan **out);
void nl_file_binding_free(NlFileBindingPlan *plan);
/* Views borrow until free. NULL plan/size returns NULL without writing size. */
const unsigned char *nl_file_binding_interface_bytes(const NlFileBindingPlan *, size_t *size);
const unsigned char *nl_file_binding_source_bytes(const NlFileBindingPlan *, size_t *size);
size_t nl_file_binding_storage_size(const NlFileBindingPlan *);
size_t nl_file_binding_peak_bound(const NlFileBindingPlan *);
/* I report a conservative project-requested heap bound before any allocation.
 * Caller storage, stack, allocator overhead and libc internals are excluded.
 * Failure preserves *out; this query never allocates. */
bool nl_file_binding_allocation_bound(size_t *out);
#endif
