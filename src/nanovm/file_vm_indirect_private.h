#ifndef NANOVM_FILE_VM_INDIRECT_PRIVATE_H
#define NANOVM_FILE_VM_INDIRECT_PRIVATE_H
#include "../nanoisa/file_indirect_runtime.h"
#ifdef NVM_FILE_INDIRECT_VM_PRIVATE
/* Source-private, serialized calls; immutable input and disjoint output.
 * Failure preserves out; only a clean scalar terminal publishes it. */
NvmFileIndirectExecutionReport nvm_file_vm_indirect_execute(const uint8_t *,size_t,
    const NvmFileIndirectOptions *,NvmFileRuntimeView *);
#endif
#endif
