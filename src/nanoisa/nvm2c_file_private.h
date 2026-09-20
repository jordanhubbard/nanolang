#ifndef NANOISA_NVM2C_FILE_PRIVATE_H
#define NANOISA_NVM2C_FILE_PRIVATE_H
#include "file_runtime_frames.h"
#ifdef NVM_FILE_NATIVE_PRIVATE
#include "file_native_abi.h"
/* No host effects. Input remains immutable; output/diagnostic storage is
 * disjoint. Failure preserves *out; success publishes malloc-owned C11 source.
 * The generated nvm_file_native_execute entry needs the explicitly linked
 * qualified private providers. No public or standalone-C admission. */
NvmFileRuntimeStatus nvm2c_file_private_emit(const uint8_t *,size_t,char **out,char *err,size_t);
#endif
#endif
