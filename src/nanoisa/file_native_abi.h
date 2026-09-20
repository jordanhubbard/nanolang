#ifndef NANOISA_FILE_NATIVE_ABI_H
#define NANOISA_FILE_NATIVE_ABI_H
#include "file_runtime_frames.h"
/* Exact semantic revision shared with the qualified private native engine.
 * Incompatible carrier/frame/catalog/operation changes require a new revision. */
#define NVM_FILE_NATIVE_ABI 1u
#define NVM_FILE_NATIVE_OUTPUT_BYTES (128u*1024u*1024u)
#ifdef __cplusplus
extern "C" {
#endif
bool nvm_file_runtime_native_abi(uint32_t,size_t,size_t,size_t);
#ifdef __cplusplus
}
#endif
#endif
