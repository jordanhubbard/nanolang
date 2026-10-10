#include "services_vm_indirect_private.h"
#ifdef NVM_SERVICES_INDIRECT_VM_PRIVATE
#include "../nanoisa/services_dispatch_config.h"
#include "service_vm_indirect_engine.inc"
NvmServicesIndirectExecutionReport nvm_services_vm_indirect_execute(const uint8_t *bytes,size_t size,
    const NvmServicesIndirectOptions *options,NvmServicesRuntimeView *out) {
    return file_vm_indirect_execute_serialized(bytes,size,options,out);
}
#endif
