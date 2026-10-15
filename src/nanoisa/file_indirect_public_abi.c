#include "file_indirect_native_public.h"
bool nvm_file_runtime_indirect_public_abi(uint32_t revision, size_t options,
    size_t report, size_t scalar) {
    return revision == NVM_FILE_INDIRECT_PUBLIC_ABI &&
        options == sizeof(NvmFileIndirectOptions) &&
        report == sizeof(NvmFileIndirectExecutionReport) && scalar == sizeof(NvmFileScalar);
}
