#ifndef NANOISA_SERVICE_MULTI_NOMINAL_H
#define NANOISA_SERVICE_MULTI_NOMINAL_H
#include "service_bindings.h"
#include "nvm_format.h"

/* I transport distinct catalog instances, without execution or host authority.
 * I bound tables before allocation and retain the existing v1/v2 codecs. */
#define NVM_MULTI_NOMINAL_VERSION 3u
#define NVM_MULTI_NOMINAL_MAX_INSTANCES 64u
#define NVM_MULTI_NOMINAL_ENTRY_BYTES 64u
#define NVM_MULTI_NOMINAL_MAX_BYTES (16u+64u*64u)
#define NVM_MULTI_NOMINAL_MAX_LAYOUTS 65536u
#define NVM_MULTI_NOMINAL_MAX_IMPORTS (64u*5u)
typedef struct {
    uint32_t catalog;
    uint32_t imports[5], layouts[9];
} NvmServiceInstance;
typedef struct {
    uint32_t count;
    NvmServiceInstance instances[NVM_MULTI_NOMINAL_MAX_INSTANCES];
} NvmMultiNominalBindings;
/* File uses eight layouts and requires layouts[8]==UINT32_MAX. I require all
 * active import and layout indices to be distinct across the entire table.
 * Failure preserves outputs. Value/byte buffers may overlap; size is disjoint.
 * Null byte output selects checked size-only mode. No pointer is retained. */
NvmServiceResult nvm_multi_nominal_check(const NvmMultiNominalBindings *);
NvmServiceResult nvm_multi_nominal_decode(const uint8_t *,size_t,NvmMultiNominalBindings *);
NvmServiceResult nvm_multi_nominal_encode(const NvmMultiNominalBindings *,uint8_t *,size_t,size_t *);

typedef enum { NVM_MULTI_NOMINAL_DESCRIBED, NVM_MULTI_NOMINAL_INVALID,
    NVM_MULTI_NOMINAL_LIMIT, NVM_MULTI_NOMINAL_MEMORY } NvmMultiNominalStatus;
typedef struct {
    uint32_t global_index, instance, catalog, catalog_ordinal, source_ordinal;
    uint8_t layout_kind, ownership_flags;
} NvmMultiNominalLayout;
typedef struct NvmMultiNominalPlan NvmMultiNominalPlan;
/* I validate complete imports/layouts/ownership metadata, not executable code.
 * Caller supplies valid immutable module storage. Failure preserves *out. */
NvmMultiNominalStatus nvm_multi_nominal_plan(const NvmModule *,NvmMultiNominalPlan **);
bool nvm_multi_nominal_storage_bound(uint32_t,size_t *);
uint32_t nvm_multi_nominal_layout_count(const NvmMultiNominalPlan *);
void nvm_multi_nominal_plan_free(NvmMultiNominalPlan *);
bool nvm_multi_nominal_layout(const NvmMultiNominalPlan *,uint32_t,NvmMultiNominalLayout *);
bool nvm_multi_nominal_type(const NvmMultiNominalPlan *,uint32_t,uint32_t,NvmMultiNominalLayout *);
bool nvm_multi_nominal_import(const NvmMultiNominalPlan *,uint32_t,uint32_t,uint32_t *);
uint32_t nvm_multi_nominal_instance_count(const NvmMultiNominalPlan *);
#endif
