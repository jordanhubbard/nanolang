#ifndef NL_FILE_PRODUCT_BRIDGE_H
#define NL_FILE_PRODUCT_BRIDGE_H
#include <stdint.h>
typedef struct NlFileProduct NlFileProduct;
/* I own copied paths, bytes and labels. Flags: 1 bytecode output, 2 explicit
 * File shadow grants, 4 explicit TCP shadow grants. New returns NULL on invalid input/allocation failure; free
 * accepts NULL. Callers protect source/companion aliases before publication.
 * I accept no AST or source-resolution facts. Calls are serialized. */
NlFileProduct *nl_file_product_new(const char *,const char *,int64_t);
int64_t nl_file_product_valid(NlFileProduct *);
/* I append 1..4096 bytes as lowercase hex, bounded to 64 MiB across all modules.
 * Seal first records main (empty name), then at most 64 named shadows. Paths
 * and names are at most 4096 bytes. Failure poisons the context; no partial
 * transaction can publish. These calls allocate but perform no filesystem I/O. */
int64_t nl_file_product_append(NlFileProduct *,const char *);
int64_t nl_file_product_seal(NlFileProduct *,const char *,const char *);
/* I consume publication eligibility once and return zero only on successful
 * staged publication. The caller frees the context after success or failure. */
int64_t nl_file_product_publish(NlFileProduct *);
void nl_file_product_free(NlFileProduct *);
#endif
