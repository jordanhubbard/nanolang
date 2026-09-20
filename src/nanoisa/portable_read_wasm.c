#include "portable_read_wasm.h"
#ifndef __wasm32__
#error "I require wasm32 for my private offset-based read import"
#endif

__attribute__((import_module("nanolang_host_v1"), import_name("read_text")))
extern int32_t npr_wasm_host_read_text(uint32_t, uint32_t, uint32_t,
                                     uint32_t, uint32_t);

typedef struct {
    uint8_t path[NPR_PATH_LIMIT + 1u];
    uint8_t destination[NPR_TEXT_LIMIT];
    uint32_t length;
    uint32_t busy;
} NprWasmScratch;
_Static_assert(sizeof(void *) == 4 && sizeof(uintptr_t) == 4,
               "I require exact wasm32 offsets");
_Static_assert(sizeof(NprWasmScratch) == 1052684u,
               "I account for all static workspace padding");
static NprWasmScratch scratch;

NprManagedResult npr_wasm_read_managed(NmsRuntime *runtime, NmsHandle argument) {
    NprManagedResult result = {NPR_OK, NMS_OK, 0};
    if (scratch.busy) { result.host_status = NPR_INVALID; return result; }
    if (!runtime) { result.managed_status = NMS_STATE; return result; }
    if (runtime->disposed) { result.managed_status = NMS_DISPOSED; return result; }
    if (!runtime->active) { result.managed_status = NMS_STATE; return result; }
    NmsView view;
    result.managed_status = nms_view(runtime, argument, &view);
    if (result.managed_status != NMS_OK) return result;
    uint32_t length = 0;
    while (length < view.length && length <= NPR_PATH_LIMIT && view.data[length])
        ++length;
    if (length > NPR_PATH_LIMIT) { result.host_status = NPR_LIMIT; return result; }
    scratch.busy = 1;
    /* Volatile stores preserve a freestanding copy without libc imports. */
    volatile uint8_t *path = scratch.path;
    for (uint32_t i = 0; i < length; ++i) path[i] = view.data[i];
    path[length] = 0;
    scratch.length = UINT32_MAX;
    int32_t status = npr_wasm_host_read_text((uint32_t)(uintptr_t)scratch.path,
        length, (uint32_t)(uintptr_t)scratch.destination, NPR_TEXT_LIMIT,
        (uint32_t)(uintptr_t)&scratch.length);
    if (status < NPR_OK || status > NPR_INVALID) result.host_status = NPR_INVALID;
    else result.host_status = (NprStatus)status;
    if (result.host_status == NPR_OK) {
        if (scratch.length > NPR_TEXT_LIMIT) result.host_status = NPR_INVALID;
        else {
            for (uint32_t i = 0; i < scratch.length; ++i) {
                if (!scratch.destination[i]) { result.host_status = NPR_INVALID; break; }
            }
            if (result.host_status == NPR_OK)
                result.managed_status = nms_create(runtime, scratch.destination,
                                                   scratch.length, &result.value);
        }
    }
    scratch.busy = 0;
    return result;
}
