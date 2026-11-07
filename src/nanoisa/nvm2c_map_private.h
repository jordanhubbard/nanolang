#ifndef NANOISA_NVM2C_MAP_PRIVATE_H
#define NANOISA_NVM2C_MAP_PRIVATE_H
#include "nvm_format.h"
#include <stddef.h>
#ifdef NANO_RECORD_MAP_PRIVATE_NATIVE
/* A module is independently proved and copied internally; no trusted plan input.
 * Source remains immutable for this synchronous call. Output is malloc-owned. */
char *nvm2c_emit_record_map_private(const NvmModule *, char *, size_t);
#endif
#endif
