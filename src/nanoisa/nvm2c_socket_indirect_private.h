#ifndef NANOISA_NVM2C_SOCKET_INDIRECT_PRIVATE_H
#define NANOISA_NVM2C_SOCKET_INDIRECT_PRIVATE_H
#include "socket_indirect_native_abi.h"
#ifdef NVM_SOCKET_INDIRECT_NATIVE_PRIVATE
/* Explicit private providers only; no host effects during emission. Immutable
 * input and disjoint output/diagnostics; success publishes malloc-owned C11. */
NvmSocketRuntimeStatus nvm2c_socket_indirect_private_emit(const uint8_t *,size_t,char **,char *,size_t);
NvmSocketIndirectExecutionReport nvm_socket_native_indirect_execute(const NvmSocketIndirectOptions *,NvmSocketRuntimeView *);
#endif
#endif
