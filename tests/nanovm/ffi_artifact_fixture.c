/* I export the same entry name from two distinct library generations. */
#include <stdint.h>
#include <stdlib.h>
#include "runtime/dyn_array.h"
int64_t nano_artifact_answer(void) { return ARTIFACT_ANSWER; }

static DynArray empty = {.elem_type = ELEM_INT, .elem_size = sizeof(int64_t)};
DynArray *array_matching(void) { return &empty; }
NANO_EXPORT_ARRAY_ABI(array_matching);
DynArray *array_legacy(void) { return &empty; }
/* I abort if the VM enters an incompatible foreign function. */
DynArray *array_mismatch(void) { abort(); }
const uint32_t array_mismatch__nano_array_abi = 99;

DynArray *array_mutate_alias(DynArray *a, DynArray *b) {
    if (a != b || a->elem_type != ELEM_INT || a->length < 1) abort();
    ((int64_t *)a->data)[0] += 1;
    return a;
}
NANO_EXPORT_ARRAY_ABI(array_mutate_alias);
static int64_t cleared_handles;
void array_clear_handles(DynArray *a) {
    for (int64_t i = 0; i < a->length; ++i) {
        if (((int64_t *)a->data)[i]) ++cleared_handles;
        ((int64_t *)a->data)[i] = 0;
    }
}
NANO_EXPORT_ARRAY_ABI(array_clear_handles);
int64_t array_cleared_count(void) { return cleared_handles; }
double array_scale(DynArray *a, double scale) {
    ((double *)a->data)[0] *= scale;
    return ((double *)a->data)[0];
}
NANO_EXPORT_ARRAY_ABI(array_scale);
void array_invalid(DynArray *a) { a->length = -1; }
NANO_EXPORT_ARRAY_ABI(array_invalid);
void array_forbidden(DynArray *a) { (void)a; abort(); }
NANO_EXPORT_ARRAY_ABI(array_forbidden);
DynArray *array_bad_result(DynArray *a) {
    static DynArray invalid = {.length = -1};
    ((int64_t *)a->data)[0] = 999;
    return &invalid;
}
NANO_EXPORT_ARRAY_ABI(array_bad_result);
DynArray *array_grow_once(DynArray *a) {
    static int64_t executions;
    if (!a || a->elem_type != ELEM_INT) abort();
    void *storage = realloc(a->data, 2000 * sizeof(int64_t));
    if (!storage) abort();
    a->data = storage;
    a->length = a->capacity = 2000;
    for (int i = 0; i < 2000; ++i) ((int64_t *)storage)[i] = i;
    ((int64_t *)storage)[0] = ++executions;
    return a;
}
NANO_EXPORT_ARRAY_ABI(array_grow_once);
