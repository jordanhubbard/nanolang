#ifndef NANOISA_UNION_FIELD_COST_PRIVATE_H
#define NANOISA_UNION_FIELD_COST_PRIVATE_H
#include "record_map_cost_private.h"
#include "ownership_union_fields_private.h"
/* Immutable constructor-owned allocation metadata only; never execution
 * authority or a check of arbitrary allocation extents. Byte output and work
 * must be disjoint from all owner storage. The two writable envelopes are
 * checked explicitly. Work is consumed on failure; byte output is preserved. */
static inline bool ufc_outputs_disjoint(const NvmMapCopyWork *w,const uint64_t *out) {
    if(!w || !out)return false;
    uintptr_t a=(uintptr_t)w,b=(uintptr_t)out;
    return a>b ? a-b>=sizeof *out : b-a>=sizeof *w;
}
NvmMapCostStatus nvm_union_declarations_retained_cost(
    const NvmUnionFieldDeclarations *,NvmMapCopyWork *,uint64_t *);
#endif
