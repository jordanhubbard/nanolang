#include "nvm2c_file_indirect_private.h"
#ifdef NVM_FILE_INDIRECT_NATIVE_PRIVATE
#include "file_indirect_native_emit.inc"
NvmFileRuntimeStatus nvm2c_file_indirect_private_emit(const uint8_t *bytes,size_t size,
    char **out,char *err,size_t err_size) {
    return file_indirect_native_emit_serialized(bytes,size,FNE_PRIVATE,NULL,out,err,err_size);
}
#endif
