#ifndef NANOVM_WEBSOCKET_VM_INDIRECT_PRIVATE_H
#define NANOVM_WEBSOCKET_VM_INDIRECT_PRIVATE_H
#include "../nanoisa/websocket_indirect_runtime.h"
#ifdef NVM_WEBSOCKET_INDIRECT_VM_PRIVATE
/* Source-private, serialized calls; immutable input and disjoint output.
 * Failure preserves out; only a clean scalar terminal publishes it. */
NvmWebSocketIndirectExecutionReport nvm_websocket_vm_indirect_execute(const uint8_t *,size_t,
    const NvmWebSocketIndirectOptions *,NvmWebSocketRuntimeView *,const NlWsTransportPolicy *);
#endif
#endif
