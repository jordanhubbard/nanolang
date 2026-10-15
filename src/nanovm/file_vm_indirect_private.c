#include "file_vm_indirect_private.h"
#ifdef NVM_FILE_INDIRECT_VM_PRIVATE
#include "file_vm_indirect_engine.inc"
NvmFileIndirectExecutionReport nvm_file_vm_indirect_execute(const uint8_t *bytes,size_t size,
    const NvmFileIndirectOptions *options,NvmFileRuntimeView *out) {
    return file_vm_indirect_execute_serialized(bytes,size,options,out);
}
#endif
