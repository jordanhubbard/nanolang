#include "file_source_inputs.h"
#include <stdlib.h>
#include <string.h>

/* I expose data-only context operations. No AST, service grant, name resolver,
 * or bytecode producer crosses this boundary. My caller owns each context. */
NlFileSourceSnapshots *nl_source_inputs_new(void) {
    NlFileSourceSnapshots *context = NULL;
    (void)nl_file_source_snapshots_new(&context);
    return context;
}
int64_t nl_source_inputs_valid(NlFileSourceSnapshots *context) {
    return context != NULL;
}
int64_t nl_source_inputs_open(NlFileSourceSnapshots *context,
                            const char *origin, int64_t origin_size,
                            const char *relative, int64_t relative_size) {
    /* My string ABI supplies terminated strings; counts must describe all
     * bytes, never a repaired/truncated declaration or an out-of-bounds span. */
    if (!origin || !relative || origin_size <= 0 || relative_size <= 0 ||
        origin_size > 4096 || relative_size > NL_FILE_BINDING_MAX_BYTES ||
        strlen(origin) != (size_t)origin_size ||
        strlen(relative) != (size_t)relative_size)
        return -(int64_t)NL_FILE_BINDING_INVALID;
    size_t index = 0;
    NlFileBindingStatus status = nl_file_source_snapshot_open(context,
        origin, (size_t)origin_size, relative, (size_t)relative_size, &index);
    return status == NL_FILE_BINDING_OK ? (int64_t)index : -(int64_t)status;
}
int64_t nl_source_inputs_count(NlFileSourceSnapshots *context) {
    return (int64_t)nl_file_source_snapshot_count(context);
}
char *nl_source_inputs_text(NlFileSourceSnapshots *context, int64_t index, int64_t kind) {
    if (index < 0 || index >= NL_FILE_SOURCE_SNAPSHOT_LIMIT || kind < 0 || kind > 3)
        return NULL;
    size_t size = 0;
    const unsigned char *view = nl_file_source_snapshot_bytes(context,
        (size_t)index, (unsigned)kind, &size);
    if (!view || memchr(view, 0, size)) return NULL;
    char *copy = malloc(size + 1);
    if (!copy) return NULL;
    memcpy(copy, view, size);
    copy[size] = 0;
    return copy;
}
void nl_source_inputs_free(NlFileSourceSnapshots *context) {
    nl_file_source_snapshots_free(context);
}
