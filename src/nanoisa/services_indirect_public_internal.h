#ifndef NANOISA_SERVICES_INDIRECT_PUBLIC_INTERNAL_H
#define NANOISA_SERVICES_INDIRECT_PUBLIC_INTERNAL_H
#include "services_indirect_public.h"
#include "services_public_internal.h"
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
