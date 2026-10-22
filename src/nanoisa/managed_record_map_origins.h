#ifndef NANOISA_MANAGED_RECORD_MAP_ORIGINS_H
#define NANOISA_MANAGED_RECORD_MAP_ORIGINS_H
#include "managed_record_array_origins.h"
/* Private copied proof facts only. No VM/backend or publication admission. */
#define NVM_RECORD_MAP_ORIGIN 2u
/* Callers precharge bounded logical probes, including misses/checks. Two tag
 * outputs must be distinct; aliased outputs refuse without modification. */
#define NVM_RECORD_MAP_FIELD_QUERY_WORK 24u
#define NVM_RECORD_MAP_ORIGIN_QUERY_WORK 72u
typedef struct NvmRecordMapFrozenContext NvmRecordMapFrozenContext;
typedef struct {
    uint32_t origins,map_origins,fields,checked_map_operations;
    uint64_t peak_bytes_reserved,work_reserved;
} NvmRecordMapCounts;
NvmArrayEligibilityResult nvm_analyze_record_map_origins(const NvmModule *,NvmRecordMapFrozenContext **);
void nvm_record_map_frozen_free(NvmRecordMapFrozenContext *);
bool nvm_record_map_frozen_counts(const NvmRecordMapFrozenContext *,NvmRecordMapCounts *);
bool nvm_record_map_frozen_origin(const NvmRecordMapFrozenContext *,uint32_t,uint8_t *,uint8_t *);
bool nvm_record_map_frozen_field(const NvmRecordMapFrozenContext *,uint32_t,uint16_t,uint8_t *,uint8_t *);
bool nvm_record_map_heap_origin(const NvmRecordMapFrozenContext *,uint32_t,NvmRecordHeapOrigin *);
bool nvm_record_map_field_value(const NvmRecordMapFrozenContext *,uint32_t,NvmRecordValueOrigins *);
#endif
