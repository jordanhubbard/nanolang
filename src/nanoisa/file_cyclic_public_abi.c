#include "file_cyclic_native_public.h"
bool nvm_file_runtime_cyclic_public_abi(uint32_t revision, size_t options,
    size_t report, size_t scalar) {
    return revision == NVM_FILE_CYCLIC_PUBLIC_ABI &&
        options == sizeof(NvmFileCyclicOptions) &&
        report == sizeof(NvmFileCyclicExecutionReport) && scalar == sizeof(NvmFileScalar);
}
