#ifndef NANOISA_NVM2C_FILE_CYCLIC_PRIVATE_H
#define NANOISA_NVM2C_FILE_CYCLIC_PRIVATE_H
#include "file_cyclic_runtime.h"
#define NVM_FILE_CYCLIC_NATIVE_ABI 1u
#define NVM_FILE_CYCLIC_NATIVE_OUTPUT_BYTES (128u*1024u*1024u)
#ifdef NVM_FILE_CYCLIC_NATIVE_PRIVATE
/* Explicit private providers only; no host effects during emission. Immutable
 * input and disjoint output/diagnostics; success publishes malloc-owned C11. */
NvmFileRuntimeStatus nvm2c_file_cyclic_private_emit(const uint8_t *,size_t,char **,char *,size_t);
NvmFileCyclicExecutionReport nvm_file_native_cyclic_execute(const NvmFileCyclicOptions *,NvmFileRuntimeView *);
#endif
#endif
