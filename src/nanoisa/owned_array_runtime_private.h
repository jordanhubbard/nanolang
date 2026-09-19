#ifndef NANOISA_OWNED_ARRAY_RUNTIME_PRIVATE_H
#define NANOISA_OWNED_ARRAY_RUNTIME_PRIVATE_H
/* I expose fixture adapters only in an explicit private test build. Normal
 * objects and public selectors do not admit this pending execution profile. */
#ifdef NANO_OWNED_ARRAY_PRIVATE_RUNTIME
#include "nvm_format.h"
#include <stdbool.h>
#include <stddef.h>
/* I prepare fresh authority internally; failure preserves *out. The successful
 * caller owns the allocated C source. No supplied certificate is accepted. */
bool nvm2c_emit_owned_array_private(const NvmModule *,char **out,char *err,size_t err_len);
#endif
#endif
