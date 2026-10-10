#ifndef NANOISA_WEBSOCKET_PUBLIC_INTERNAL_H
#define NANOISA_WEBSOCKET_PUBLIC_INTERNAL_H
#include "websocket_public.h"
#include "websocket_host_grant_internal.h"
#include "nvm_v2_sections.h"

/* Trusted generated/adapter helpers. I own no gate or invocation here. */
static inline NvmWebSocketRuntimeStatus nvm_websocket_public_grant_status(NvmWebSocketHostStatus s) {
    switch (s) {
    case NVM_WEBSOCKET_HOST_OK: return NVM_WEBSOCKET_RUNTIME_OK;
    case NVM_WEBSOCKET_HOST_INVALID: return NVM_WEBSOCKET_RUNTIME_INVALID;
    case NVM_WEBSOCKET_HOST_MEMORY: return NVM_WEBSOCKET_RUNTIME_MEMORY;
    case NVM_WEBSOCKET_HOST_STATE: return NVM_WEBSOCKET_RUNTIME_STATE;
    case NVM_WEBSOCKET_HOST_BUSY: return NVM_WEBSOCKET_RUNTIME_BUSY;
    default: return NVM_WEBSOCKET_RUNTIME_UNRESOLVED;
    }
}
static inline NvmWebSocketRuntimeReport nvm_websocket_public_refused(NvmWebSocketRuntimeStatus s) {
    NvmWebSocketRuntimeReport r = {0};
    r.status = s;
    r.function = r.instruction = NVM_V2_NO_INDEX;
    return r;
}
/* The engine has already destroyed its invocation; the caller still owns the
 * gate. A malformed successful view cannot overwrite the public sentinel. */
static inline NvmWebSocketRuntimeReport nvm_websocket_public_publish(NvmWebSocketRuntimeReport r,
    const NvmWebSocketRuntimeView *view, NvmWebSocketScalar *out) {
    if (r.status != NVM_WEBSOCKET_RUNTIME_OK) return r;
    if (!r.acquired) { r.status = NVM_WEBSOCKET_RUNTIME_STATE; return r; }
    if (r.core_status != NL_WS_VALUE_OK || r.cleanup.execution != NL_WS_VALUE_OK ||
        r.cleanup.cleanup_failures) { r.status = NVM_WEBSOCKET_RUNTIME_CLEANUP; return r; }
    if (!out || !view->initialized || view->owning || view->formal ||
        view->type.mode || view->type.category != NVM_WEBSOCKET_CATEGORY_UNKNOWN ||
        view->type.global_index != NVM_V2_NO_INDEX ||
        view->type.catalog_ordinal != NVM_V2_NO_INDEX || view->fields != 1 ||
        (view->type.tag != TAG_INT && view->type.tag != TAG_BOOL) ||
        (view->type.tag == TAG_BOOL && view->values[0] != 0 && view->values[0] != 1)) {
        r.status = NVM_WEBSOCKET_RUNTIME_TYPE;
        return r;
    }
    NvmWebSocketScalar scalar = {view->type.tag, view->values[0]};
    *out = scalar;
    return r;
}
#endif
