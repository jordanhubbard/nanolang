#include "services_indirect_native_public.h"
bool nvm_services_runtime_indirect_public_abi(uint32_t revision, size_t options,
    size_t report, size_t scalar) {
    return revision == NVM_SERVICES_INDIRECT_PUBLIC_ABI &&
        options == sizeof(NvmServicesIndirectOptions) &&
        report == sizeof(NvmServicesIndirectExecutionReport) && scalar == sizeof(NvmServicesScalar);
}
