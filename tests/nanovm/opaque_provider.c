#include <stdint.h>
#include <stddef.h>
#include <string.h>
static int64_t cells[] = {41, 42, 43};
static int64_t calls;
void *opaque_make(int64_t index) {
    ++calls;
    return index >= 0 && index < 3 ? &cells[index] : NULL;
}
void *opaque_same(void *pointer) { ++calls; return pointer; }
int64_t opaque_read(void *pointer) { ++calls; return pointer ? *(int64_t *)pointer : -1; }
void *opaque_large(const char *text) {
    ++calls;
    return text && strlen(text) > 8192 ? &cells[2] : NULL;
}
int64_t opaque_calls(void) { return calls; }

#include "runtime/dyn_array.h"
void *opaque_invalid_array(DynArray *array) {
    ++calls;
    array->length = -1;
    return &cells[1];
}
NANO_EXPORT_ARRAY_ABI(opaque_invalid_array);
