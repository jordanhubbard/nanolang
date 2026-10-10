#ifndef NANOISA_SOCKET_INDIRECT_NATIVE_PUBLIC_H
#define NANOISA_SOCKET_INDIRECT_NATIVE_PUBLIC_H
/* Installed trusted generated-C details; direct carrier calls are not grants. */
#include "socket_indirect_native_abi.h"
#include "socket_indirect_public_internal.h"
#define NVM_SOCKET_INDIRECT_PUBLIC_ABI 1u
#ifdef __cplusplus
extern "C" {
#endif
bool nvm_socket_runtime_indirect_public_abi(uint32_t, size_t, size_t, size_t);
#ifdef __cplusplus
}
#endif
#endif
