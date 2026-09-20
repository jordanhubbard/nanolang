#include "file_vm_private.h"
#ifdef NVM_FILE_VM_PRIVATE
#include "file_vm_engine.inc"
NvmFileRuntimeReport nvm_file_vm_execute(const uint8_t *bytes, size_t size,
                                       NvmFileRuntimeView *out) {
    return file_vm_execute_serialized(bytes, size, out);
}
#endif
