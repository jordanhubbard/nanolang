#include "file_vm_cyclic_private.h"
#ifdef NVM_FILE_CYCLIC_VM_PRIVATE
#include "file_vm_cyclic_engine.inc"
NvmFileCyclicExecutionReport nvm_file_vm_cyclic_execute(const uint8_t *bytes,size_t size,
    const NvmFileCyclicOptions *options,NvmFileRuntimeView *out) {
    return file_vm_cyclic_execute_serialized(bytes,size,options,out);
}
#endif
