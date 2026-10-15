#include "websocket_indirect_native_public.h"
bool nvm_websocket_runtime_indirect_public_abi(uint32_t revision, size_t options,
    size_t report, size_t scalar) {
    return revision == NVM_WEBSOCKET_INDIRECT_PUBLIC_ABI &&
        options == sizeof(NvmWebSocketIndirectOptions) &&
        report == sizeof(NvmWebSocketIndirectExecutionReport) && scalar == sizeof(NvmWebSocketScalar);
}
