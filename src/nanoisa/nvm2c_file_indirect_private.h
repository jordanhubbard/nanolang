#ifndef NANOISA_NVM2C_FILE_INDIRECT_PRIVATE_H
#define NANOISA_NVM2C_FILE_INDIRECT_PRIVATE_H
#include "file_indirect_native_abi.h"
#ifdef NVM_FILE_INDIRECT_NATIVE_PRIVATE
/* Explicit private providers only; no host effects during emission. Immutable
 * input and disjoint output/diagnostics; success publishes malloc-owned C11. */
NvmFileRuntimeStatus nvm2c_file_indirect_private_emit(const uint8_t *,size_t,char **,char *,size_t);
NvmFileIndirectExecutionReport nvm_file_native_indirect_execute(const NvmFileIndirectOptions *,NvmFileRuntimeView *);
#endif
#endif
