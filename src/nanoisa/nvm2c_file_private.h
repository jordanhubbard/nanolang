#ifndef NANOISA_NVM2C_FILE_PRIVATE_H
#define NANOISA_NVM2C_FILE_PRIVATE_H
#include "file_runtime_frames.h"
#ifdef NVM_FILE_NATIVE_PRIVATE
/* Private semantic revision: exact v1 integer/BOOL/passive/File operations,
 * whole-frame NATIVE staging and first-error cleanup. Any incompatible helper,
 * catalog or frame semantic change must revise this value and qualification. */
#define NVM_FILE_NATIVE_ABI 1u
#define NVM_FILE_NATIVE_OUTPUT_BYTES (128u*1024u*1024u)
/* Implemented by the carrier TU, not inferred from an emitter header alone. */
bool nvm_file_runtime_native_abi(uint32_t,size_t,size_t,size_t);
/* No host effects. Input remains immutable; output/diagnostic storage is
 * disjoint. Failure preserves *out; success publishes malloc-owned C11 source.
 * The generated nvm_file_native_execute entry needs the explicitly linked
 * qualified private providers. No public or standalone-C admission. */
NvmFileRuntimeStatus nvm2c_file_private_emit(const uint8_t *,size_t,char **out,char *err,size_t);
#endif
#endif
