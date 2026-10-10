#include "socket_indirect_native_public.h"
bool nvm_socket_runtime_indirect_public_abi(uint32_t revision, size_t options,
    size_t report, size_t scalar) {
    return revision == NVM_SOCKET_INDIRECT_PUBLIC_ABI &&
        options == sizeof(NvmSocketIndirectOptions) &&
        report == sizeof(NvmSocketIndirectExecutionReport) && scalar == sizeof(NvmSocketScalar);
}
