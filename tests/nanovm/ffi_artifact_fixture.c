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
