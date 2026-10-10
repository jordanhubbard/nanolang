#include "../nanoisa/services_indirect_public_internal.h"
#include "../nanoisa/services_dispatch_config.h"
#define SERVICE_EXECUTION_AUTH_ARGUMENT ,const NvmServicesHostGrant *grant
#define SERVICE_EXECUTION_AUTH_CHECK(c) nvm_services_public_grant_status(nvm_services_host_authorize(grant,nvm_services_runtime_indirect_plan(c)))
#include "service_vm_indirect_engine.inc"

NvmServicesIndirectExecutionReport nvm_services_execute_indirect_bytes(NvmServicesHostGrant *grant,
    const uint8_t *bytes, size_t size, const NvmServicesIndirectOptions *options,
    NvmServicesScalar *out) {
    NvmServicesHostStatus entered = nvm_services_host_enter(grant, NVM_SERVICES_HOST_ABI,
                                                  NVM_SERVICES_HOST_CATALOG);
    if (entered != NVM_SERVICES_HOST_OK)
        return nvm_services_indirect_public_refused(nvm_services_public_grant_status(entered),
                                             entered == NVM_SERVICES_HOST_BUSY ? NULL : options);
    NvmServicesIndirectExecutionReport report;
    if (!nvm_services_indirect_public_options(options) || !out)
        report = nvm_services_indirect_public_refused(NVM_SERVICES_RUNTIME_INVALID, options);
    else {
        NvmServicesIndirectOptions copied = *options;
        NvmServicesRuntimeView view = {0};
        report = file_vm_indirect_execute_serialized(bytes, size, &copied, &view, grant);
        report.runtime = nvm_services_public_publish(report.runtime, &view, out);
    }
    nvm_services_host_leave();
    return report;
}
