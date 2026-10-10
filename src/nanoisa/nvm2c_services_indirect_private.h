#ifndef NANOISA_NVM2C_SERVICES_INDIRECT_PRIVATE_H
#define NANOISA_NVM2C_SERVICES_INDIRECT_PRIVATE_H
#include "services_indirect_native_abi.h"
#ifdef NVM_SERVICES_INDIRECT_NATIVE_PRIVATE
/* Explicit private providers only; no host effects during emission. Immutable
 * input and disjoint output/diagnostics; success publishes malloc-owned C11. */
NvmServicesRuntimeStatus nvm2c_services_indirect_private_emit(const uint8_t *,size_t,char **,char *,size_t);
NvmServicesIndirectExecutionReport nvm_services_native_indirect_execute(const NvmServicesIndirectOptions *,NvmServicesRuntimeView *);
#endif
#endif
