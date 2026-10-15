#ifndef NANOVM_SERVICES_VM_INDIRECT_PRIVATE_H
#define NANOVM_SERVICES_VM_INDIRECT_PRIVATE_H
#include "../nanoisa/services_indirect_runtime.h"
#ifdef NVM_SERVICES_INDIRECT_VM_PRIVATE
/* Source-private, serialized calls; immutable input and disjoint output.
 * Failure preserves out; only a clean scalar terminal publishes it. */
NvmServicesIndirectExecutionReport nvm_services_vm_indirect_execute(const uint8_t *,size_t,
    const NvmServicesIndirectOptions *,NvmServicesRuntimeView *);
#endif
#endif
