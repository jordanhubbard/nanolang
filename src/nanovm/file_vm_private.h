#ifndef NANOVM_FILE_VM_PRIVATE_H
#define NANOVM_FILE_VM_PRIVATE_H
/* Explicit private qualification only. Normal builds expose no execution API. */
#ifdef NVM_FILE_VM_PRIVATE
#include "../nanoisa/file_runtime.h"
/* Fresh serialized-v2 invocation, externally serialized like its private cores.
 * scalar_output is non-NULL, disjoint from input, and unchanged on any failure.
 * Success publishes an initialized, nonowning exact INT/BOOL RuntimeView after
 * entry and cleanup. No File/Result or context handle escapes. */
NvmFileRuntimeReport nvm_file_vm_execute(const uint8_t *bytes,size_t size,
                                       NvmFileRuntimeView *scalar_output);
#endif
#endif
