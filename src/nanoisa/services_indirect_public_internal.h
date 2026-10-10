#ifndef NANOISA_SERVICES_INDIRECT_PUBLIC_INTERNAL_H
#define NANOISA_SERVICES_INDIRECT_PUBLIC_INTERNAL_H
#include "services_indirect_public.h"
#include "services_indirect_runtime.h"
#include "services_public_internal.h"
/* I authorize the complete checked table before copying any runtime policy.
 * The public caller retains the grant gate until invocation cleanup. */
static inline NvmServicesRuntimeStatus nvm_services_public_configure(NvmServicesRuntime *c,const NvmServicesHostGrant *grant) {
    const NvmServicesIndirectHostedPlan *plan=nvm_services_runtime_indirect_plan(c);
    NvmServicesRuntimeStatus status=nvm_services_public_grant_status(nvm_services_host_authorize(grant,plan));
    if(status!=NVM_SERVICES_RUNTIME_OK)return status;
    for(uint32_t i=0;i<NVM_SERVICES_HOST_INSTANCES;i++) {
        NvmServicesNominalLayout type;
        if(!nvm_services_indirect_hosted_type(plan,i*9,&type))break;
        if(nvm_services_indirect_hosted_type(plan,i*9+7,&type))continue;
        NlWsTransportPolicy policy;
        status=nvm_services_public_grant_status(nvm_services_host_websocket_policy(grant,i,&policy));
        if(status==NVM_SERVICES_RUNTIME_OK)status=nvm_services_runtime_websocket_policy(c,i,&policy);
        if(status!=NVM_SERVICES_RUNTIME_OK)return status;
    }
    return NVM_SERVICES_RUNTIME_OK;
}
/* I own no context or gate here. BUSY callers pass NULL without reading options. */
static inline NvmServicesIndirectExecutionReport nvm_services_indirect_public_refused(
    NvmServicesRuntimeStatus status, const NvmServicesIndirectOptions *options) {
    NvmServicesIndirectExecutionReport report = {0};
    report.revision = NVM_SERVICES_INDIRECT_RUNTIME_REVISION;
    report.instruction_limit = options ? options->instruction_limit : 0;
    report.runtime = nvm_services_public_refused(status);
    return report;
}
static inline bool nvm_services_indirect_public_options(const NvmServicesIndirectOptions *options) {
    return options && options->revision == NVM_SERVICES_INDIRECT_RUNTIME_REVISION &&
        options->instruction_limit <= NVM_SERVICES_INDIRECT_FUEL_MAX;
}
#endif
