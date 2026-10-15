#ifndef NANOISA_WEBSOCKET_INDIRECT_PUBLIC_INTERNAL_H
#define NANOISA_WEBSOCKET_INDIRECT_PUBLIC_INTERNAL_H
#include "websocket_indirect_public.h"
#include "websocket_public_internal.h"
/* I own no context or gate here. BUSY callers pass NULL without reading options. */
static inline NvmWebSocketIndirectExecutionReport nvm_websocket_indirect_public_refused(
    NvmWebSocketRuntimeStatus status, const NvmWebSocketIndirectOptions *options) {
    NvmWebSocketIndirectExecutionReport report = {0};
    report.revision = NVM_WEBSOCKET_INDIRECT_RUNTIME_REVISION;
    report.instruction_limit = options ? options->instruction_limit : 0;
    report.runtime = nvm_websocket_public_refused(status);
    return report;
}
static inline bool nvm_websocket_indirect_public_options(const NvmWebSocketIndirectOptions *options) {
    return options && options->revision == NVM_WEBSOCKET_INDIRECT_RUNTIME_REVISION &&
        options->instruction_limit <= NVM_WEBSOCKET_INDIRECT_FUEL_MAX;
}
#endif
