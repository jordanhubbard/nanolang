#include "portable_read_managed.h"
#include <stdlib.h>
#include <string.h>

/* One bounded project allocation before host effects; sizeof includes all
 * padding and the aligned length cell. nms_create has its own runtime budget. */
typedef struct {
    uint8_t path[NPR_PATH_LIMIT + 1u];
    uint8_t destination[NPR_TEXT_LIMIT];
    uint32_t length;
} NprScratch;

NprManagedResult npr_read_managed(NmsRuntime *runtime, NmsHandle argument,
                                 const NprHostBinding *binding) {
    NprManagedResult result = { NPR_OK, NMS_OK, 0 };
    if (!runtime) { result.managed_status = NMS_STATE; return result; }
    if (runtime->disposed) { result.managed_status = NMS_DISPOSED; return result; }
    if (!runtime->active) { result.managed_status = NMS_STATE; return result; }
    NmsView view;
    result.managed_status = nms_view(runtime, argument, &view);
    if (result.managed_status != NMS_OK) return result;
    if (!binding || !binding->read || !binding->context) {
        result.host_status = NPR_DENIED;
        return result;
    }
    NprHostBinding call = *binding;
    uint32_t length = 0;
    while (length < view.length && length <= NPR_PATH_LIMIT && view.data[length])
        ++length;
    if (length > NPR_PATH_LIMIT) { result.host_status = NPR_LIMIT; return result; }
    NprScratch *scratch = malloc(sizeof(*scratch));
    if (!scratch) { result.host_status = NPR_MEMORY; return result; }
    if (length) memcpy(scratch->path, view.data, length);
    scratch->path[length] = 0;
    scratch->length = UINT32_MAX;
    int32_t status = call.read(call.context, scratch->path, length,
                              scratch->destination, NPR_TEXT_LIMIT,
                              &scratch->length);
    if (status < NPR_OK || status > NPR_INVALID) result.host_status = NPR_INVALID;
    else result.host_status = (NprStatus)status;
    if (result.host_status == NPR_OK) {
        if (scratch->length > NPR_TEXT_LIMIT ||
            (scratch->length && memchr(scratch->destination, 0, scratch->length))) {
            result.host_status = NPR_INVALID;
        } else {
            result.managed_status = nms_create(runtime, scratch->destination,
                                               scratch->length, &result.value);
        }
    }
    free(scratch);
    return result;
}
