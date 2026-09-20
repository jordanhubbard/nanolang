#ifndef NANOVM_FILE_VM_CYCLIC_PRIVATE_H
#define NANOVM_FILE_VM_CYCLIC_PRIVATE_H
#include "../nanoisa/file_cyclic_runtime.h"
#ifdef NVM_FILE_CYCLIC_VM_PRIVATE
/* Source-private, serialized calls; immutable input and disjoint output.
 * Failure preserves out; only a clean scalar terminal publishes it. */
NvmFileCyclicExecutionReport nvm_file_vm_cyclic_execute(const uint8_t *,size_t,
    const NvmFileCyclicOptions *,NvmFileRuntimeView *);
#endif
#endif
