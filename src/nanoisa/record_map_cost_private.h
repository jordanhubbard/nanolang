#ifndef NANOISA_RECORD_MAP_COST_PRIVATE_H
#define NANOISA_RECORD_MAP_COST_PRIVATE_H
#include "ownership_declaration_projection.h"
#include "ownership_map_declarations.h"
/* Work and byte-output objects must not overlap each other or the immutable
 * owner. Work is consumed on failure; byte output is committed only on success.
 * Counts describe owned constructor payloads, never allocator bookkeeping.
 * Input is an immutable internal constructor-owned object; redundant checks
 * cannot establish arbitrary pointer allocation extents or make forged pointers safe. */
typedef struct { uint64_t used,limit; bool limited; } NvmMapCopyWork;
typedef enum { NVM_MAP_COST_OK, NVM_MAP_COST_INVALID, NVM_MAP_COST_LIMIT } NvmMapCostStatus;
static inline bool nvm_map_copy_charge(NvmMapCopyWork *w,uint64_t amount) {
    if(!w)return false;
    if(w->limited || w->used>w->limit || amount>w->limit-w->used){w->limited=true;return false;}
    w->used+=amount;return true;
}
static inline NvmMapCostStatus nvm_map_copy_term(NvmMapCopyWork *w,const void *ptr,
                                                uint64_t count,uint64_t width,uint64_t *sum) {
    if(!nvm_map_copy_charge(w,4))return NVM_MAP_COST_LIMIT;
    if(!!ptr!=!!count || !width)return NVM_MAP_COST_INVALID;
    if(count>UINT64_MAX/width || count*width>UINT64_MAX-*sum)return NVM_MAP_COST_LIMIT;
    *sum+=count*width;return NVM_MAP_COST_OK;
}
static inline NvmMapCostStatus nvm_map_copy_sum(uint64_t *sum,uint64_t add) {
    if(add>UINT64_MAX-*sum)return NVM_MAP_COST_LIMIT;
    *sum+=add;return NVM_MAP_COST_OK;
}
NvmMapCostStatus nvm_ownership_declarations_retained_cost(const NvmOwnershipDeclarationPlan *,NvmMapCopyWork *,uint64_t *);
NvmMapCostStatus nvm_map_declarations_retained_cost(const NvmMapDeclarationPlan *,NvmMapCopyWork *,uint64_t *);
#endif
