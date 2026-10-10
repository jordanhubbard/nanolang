#ifndef NANOISA_SOCKET_INDIRECT_PUBLIC_INTERNAL_H
#define NANOISA_SOCKET_INDIRECT_PUBLIC_INTERNAL_H
#include "socket_indirect_public.h"
#include "socket_public_internal.h"
/* I own no context or gate here. BUSY callers pass NULL without reading options. */
static inline NvmSocketIndirectExecutionReport nvm_socket_indirect_public_refused(
    NvmSocketRuntimeStatus status, const NvmSocketIndirectOptions *options) {
    NvmSocketIndirectExecutionReport report = {0};
    report.revision = NVM_SOCKET_INDIRECT_RUNTIME_REVISION;
    report.instruction_limit = options ? options->instruction_limit : 0;
    report.runtime = nvm_socket_public_refused(status);
    return report;
}
static inline bool nvm_socket_indirect_public_options(const NvmSocketIndirectOptions *options) {
    return options && options->revision == NVM_SOCKET_INDIRECT_RUNTIME_REVISION &&
        options->instruction_limit <= NVM_SOCKET_INDIRECT_FUEL_MAX;
}
#endif
