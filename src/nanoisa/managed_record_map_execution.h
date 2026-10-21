#ifndef NANOISA_MANAGED_RECORD_MAP_EXECUTION_H
#define NANOISA_MANAGED_RECORD_MAP_EXECUTION_H
#include "managed_record_array_execution.h"
#include "managed_record_map_origins.h"
/* Immutable private recipes only, never executable admission. */
#define NVM_RM_INVALID_BASE_RECIPE UINT8_MAX
typedef struct NvmRecordMapExecutionPlan NvmRecordMapExecutionPlan;
typedef enum { NVM_RM_NONE, NVM_RM_NEW, NVM_RM_SET, NVM_RM_GET,
               NVM_RM_HAS, NVM_RM_LEN } NvmRecordMapOperation;
typedef struct {
    NvmRecordArrayExecutionCounts carrier;
    NvmRecordMapCounts maps;
} NvmRecordMapExecutionCounts;
typedef struct {
    NvmRecordArrayExecutionInstruction base;
    uint16_t key_tags,value_tags,result_tags;
    uint8_t map_operation;
} NvmRecordMapExecutionInstruction;
NvmArrayEligibilityResult nvm_prepare_record_map_execution(const NvmModule *,NvmRecordMapExecutionPlan **);
void nvm_record_map_execution_free(NvmRecordMapExecutionPlan *);
bool nvm_record_map_execution_counts(const NvmRecordMapExecutionPlan *,NvmRecordMapExecutionCounts *);
bool nvm_record_map_execution_instruction(const NvmRecordMapExecutionPlan *,uint32_t,NvmRecordMapExecutionInstruction *);
bool nvm_record_map_execution_map_field(const NvmRecordMapExecutionPlan *,uint32_t,uint16_t,uint8_t *,uint8_t *);
bool nvm_record_map_execution_map_origin(const NvmRecordMapExecutionPlan *,uint32_t,uint8_t *,uint8_t *);
bool nvm_record_map_execution_function(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmRecordArrayExecutionFunction *out);
bool nvm_record_map_execution_parameter(const NvmRecordMapExecutionPlan *p,uint32_t fi,uint16_t i,uint8_t *out);
bool nvm_record_map_execution_bytes(const NvmRecordMapExecutionPlan *p,NvmRecordArraySnapshotKind kind,uint32_t index,uint32_t offset,uint32_t count,void *out);
bool nvm_record_map_execution_size(const NvmRecordMapExecutionPlan *p,NvmRecordArraySnapshotKind kind,uint32_t index,uint32_t *out);
bool nvm_record_map_execution_descriptor(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmRecordArrayExecutionDescriptor *out);
bool nvm_record_map_execution_header(const NvmRecordMapExecutionPlan *p,NvmHeader *out);
bool nvm_record_map_execution_section(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmSectionEntry *out);
bool nvm_record_map_execution_metadata(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmMetadataEntry *out);
bool nvm_record_map_execution_debug(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmDebugEntry *out);
bool nvm_record_map_execution_origin_counts(const NvmRecordMapExecutionPlan *p,NvmRecordArrayOriginCounts *out);
bool nvm_record_map_execution_origin(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmRecordHeapOrigin *out);
bool nvm_record_map_execution_field_value(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmRecordValueOrigins *out);
bool nvm_record_map_execution_required_elements(const NvmRecordMapExecutionPlan *p,uint32_t i,uint16_t *out);
bool nvm_record_map_execution_declaration_counts(const NvmRecordMapExecutionPlan *p,NvmDeclarationCounts *out);
bool nvm_record_map_execution_layout(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmDeclarationLayout *out);
bool nvm_record_map_execution_field(const NvmRecordMapExecutionPlan *p,uint32_t i,uint16_t field,NvmV2LayoutField *out);
bool nvm_record_map_execution_type(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmOrdinaryArrayType *out);
bool nvm_record_map_execution_binding(const NvmRecordMapExecutionPlan *p,uint32_t i,NvmOrdinaryArrayBinding *out);
bool nvm_record_map_execution_variant(const NvmRecordMapExecutionPlan *p,uint32_t i,uint16_t variant,NvmUnionVariantFact *out);
#endif
