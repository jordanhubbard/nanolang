#ifndef NANOVM_SOCKET_VM_INDIRECT_PRIVATE_H
#define NANOVM_SOCKET_VM_INDIRECT_PRIVATE_H
#include "../nanoisa/socket_indirect_runtime.h"
#ifdef NVM_SOCKET_INDIRECT_VM_PRIVATE
/* Source-private, serialized calls; immutable input and disjoint output.
 * Failure preserves out; only a clean scalar terminal publishes it. */
NvmSocketIndirectExecutionReport nvm_socket_vm_indirect_execute(const uint8_t *,size_t,
    const NvmSocketIndirectOptions *,NvmSocketRuntimeView *);
#endif
#endif
