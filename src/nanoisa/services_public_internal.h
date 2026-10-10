#ifndef NANOISA_SERVICES_PUBLIC_INTERNAL_H
#define NANOISA_SERVICES_PUBLIC_INTERNAL_H
#include "services_public.h"
#include "services_host_grant_internal.h"
#include "nvm_v2_sections.h"

/* Trusted generated/adapter helpers. I own no gate or invocation here. */
static inline NvmServicesRuntimeStatus nvm_services_public_grant_status(NvmServicesHostStatus s) {
    switch (s) {
    case NVM_SERVICES_HOST_OK: return NVM_SERVICES_RUNTIME_OK;
    case NVM_SERVICES_HOST_INVALID: return NVM_SERVICES_RUNTIME_INVALID;
    case NVM_SERVICES_HOST_MEMORY: return NVM_SERVICES_RUNTIME_MEMORY;
    case NVM_SERVICES_HOST_STATE: return NVM_SERVICES_RUNTIME_STATE;
    case NVM_SERVICES_HOST_BUSY: return NVM_SERVICES_RUNTIME_BUSY;
    default: return NVM_SERVICES_RUNTIME_UNRESOLVED;
    }
}
static inline NvmServicesRuntimeReport nvm_services_public_refused(NvmServicesRuntimeStatus s) {
    NvmServicesRuntimeReport r = {0};
    r.status = s;
    r.function = r.instruction = NVM_V2_NO_INDEX;
    return r;
}
/* The engine has already destroyed its invocation; the caller still owns the
 * gate. A malformed successful view cannot overwrite the public sentinel. */
static inline NvmServicesRuntimeReport nvm_services_public_publish(NvmServicesRuntimeReport r,
    const NvmServicesRuntimeView *view, NvmServicesScalar *out) {
    if (r.status != NVM_SERVICES_RUNTIME_OK) return r;
    if (!r.acquired) { r.status = NVM_SERVICES_RUNTIME_STATE; return r; }
    if (r.core_status != NL_SERVICES_VALUE_OK || r.cleanup.execution != NL_SERVICES_VALUE_OK ||
        r.cleanup.cleanup_failures) { r.status = NVM_SERVICES_RUNTIME_CLEANUP; return r; }
    if (!out || !view->initialized || view->owning || view->formal ||
        view->type.mode || view->type.category != NVM_SERVICES_CATEGORY_UNKNOWN ||
        view->type.global_index != NVM_V2_NO_INDEX ||
        view->type.catalog_ordinal != NVM_V2_NO_INDEX || view->fields != 1 ||
        (view->type.tag != TAG_INT && view->type.tag != TAG_BOOL) ||
        (view->type.tag == TAG_BOOL && view->values[0] != 0 && view->values[0] != 1)) {
        r.status = NVM_SERVICES_RUNTIME_TYPE;
        return r;
    }
    NvmServicesScalar scalar = {view->type.tag, view->values[0]};
    *out = scalar;
    return r;
}
#endif
