#ifndef NANOISA_MANAGED_RECORD_ARRAY_EXECUTION_H
#define NANOISA_MANAGED_RECORD_ARRAY_EXECUTION_H
#include "managed_record_array_origins.h"
#include "isa.h"
/* Copied, non-admitting facts. I reserve no executable/public route. */
#define NVM_RECORD_ARRAY_EXECUTION_FRAMES 1024u
#define NVM_RECORD_ARRAY_EXECUTION_BYTES (UINT64_C(128)*1024*1024)
#define NVM_RECORD_ARRAY_EXECUTION_STEPS UINT64_C(33554432)
typedef struct NvmRecordArrayExecutionPlan NvmRecordArrayExecutionPlan;
typedef enum {
    NVM_RA_ROOT_SCALAR, NVM_RA_ROOT_CONSTANT, NVM_RA_ROOT_LOAD,
    NVM_RA_ROOT_STORE, NVM_RA_ROOT_DUP, NVM_RA_ROOT_SWAP, NVM_RA_ROOT_DROP,
    NVM_RA_ROOT_BRANCH, NVM_RA_ROOT_CALL, NVM_RA_ROOT_RETURN,
    NVM_RA_ROOT_STRING, NVM_RA_ROOT_CONSTRUCT, NVM_RA_ROOT_GET,
    NVM_RA_ROOT_SET, NVM_RA_ROOT_ARRAY_PUSH, NVM_RA_ROOT_ARRAY_POP,
    NVM_RA_ROOT_ARRAY_COPY
} NvmRecordArrayRootRecipe;
enum {
    /* I require these runtime checks; I do not claim them statically discharged. */
    NVM_RA_CHECK_TAGS=1u, NVM_RA_CHECK_BOUNDS=2u, NVM_RA_CHECK_NOMINAL=4u,
    NVM_RA_SAFEPOINT_BEFORE=8u, NVM_RA_STAGE_OPERANDS=16u,
    NVM_RA_RETAIN_BEFORE_RELEASE=32u, NVM_RA_FIRST_ERROR_CLEANUP=64u
};
typedef struct {
    uint32_t functions, instructions, globals, initializer, entry, frame_limit;
    uint32_t strings, sections, metadata, debug, records;
    uint64_t peak_bytes_reserved, work_reserved;
} NvmRecordArrayExecutionCounts;
typedef struct {
    NvmFunctionEntry signature;
    uint32_t instruction_start, instruction_count;
    uint16_t maximum_stack;
    uint8_t parameter_tags_present;
} NvmRecordArrayExecutionFunction;
/* Successors are relative byte offsets; code_length denotes implicit return.
 * CALL has its ordinary continuation plus an explicit callee, not a CFG jump.
 * Operand bits copy the encoded immediate bytes into low bits, including F64. */
typedef struct {
    uint32_t function, pc, next_pc, successors[2], callee;
    uint16_t pops, pushes, obligations;
    uint8_t opcode, recipe, successor_count, operand_count;
    uint8_t operand_types[MAX_OPERANDS];
    uint64_t operand_bits[MAX_OPERANDS];
} NvmRecordArrayExecutionInstruction;
typedef struct {
    uint32_t ordinal, layout;
    uint16_t fields;
} NvmRecordArrayExecutionDescriptor;
typedef enum {
    NVM_RA_SNAPSHOT_CODE, NVM_RA_SNAPSHOT_LAYOUTS, NVM_RA_SNAPSHOT_OWNERSHIP,
    NVM_RA_SNAPSHOT_STRING
} NvmRecordArraySnapshotKind;
NvmArrayEligibilityResult nvm_prepare_record_array_execution(
    const NvmModule *, NvmRecordArrayExecutionPlan **);
void nvm_record_array_execution_free(NvmRecordArrayExecutionPlan *);
bool nvm_record_array_execution_counts(const NvmRecordArrayExecutionPlan *,NvmRecordArrayExecutionCounts *);
bool nvm_record_array_execution_function(const NvmRecordArrayExecutionPlan *,uint32_t,NvmRecordArrayExecutionFunction *);
bool nvm_record_array_execution_instruction(const NvmRecordArrayExecutionPlan *,uint32_t,NvmRecordArrayExecutionInstruction *);
bool nvm_record_array_execution_parameter(const NvmRecordArrayExecutionPlan *,uint32_t,uint16_t,uint8_t *);
bool nvm_record_array_execution_bytes(const NvmRecordArrayExecutionPlan *,NvmRecordArraySnapshotKind,uint32_t,uint32_t,uint32_t,void *);
bool nvm_record_array_execution_descriptor(const NvmRecordArrayExecutionPlan *,uint32_t,NvmRecordArrayExecutionDescriptor *);
bool nvm_record_array_execution_header(const NvmRecordArrayExecutionPlan *,NvmHeader *);
bool nvm_record_array_execution_section(const NvmRecordArrayExecutionPlan *,uint32_t,NvmSectionEntry *);
bool nvm_record_array_execution_metadata(const NvmRecordArrayExecutionPlan *,uint32_t,NvmMetadataEntry *);
bool nvm_record_array_execution_debug(const NvmRecordArrayExecutionPlan *,uint32_t,NvmDebugEntry *);
bool nvm_record_array_execution_size(const NvmRecordArrayExecutionPlan *,NvmRecordArraySnapshotKind,uint32_t,uint32_t *);
bool nvm_record_array_execution_origin_counts(const NvmRecordArrayExecutionPlan *,NvmRecordArrayOriginCounts *);
bool nvm_record_array_execution_origin(const NvmRecordArrayExecutionPlan *,uint32_t,NvmRecordHeapOrigin *);
bool nvm_record_array_execution_field_value(const NvmRecordArrayExecutionPlan *,uint32_t,NvmRecordValueOrigins *);
bool nvm_record_array_execution_required_elements(const NvmRecordArrayExecutionPlan *,uint32_t,uint16_t *);
bool nvm_record_array_execution_declaration_counts(const NvmRecordArrayExecutionPlan *,NvmDeclarationCounts *);
bool nvm_record_array_execution_layout(const NvmRecordArrayExecutionPlan *,uint32_t,NvmDeclarationLayout *);
bool nvm_record_array_execution_field(const NvmRecordArrayExecutionPlan *,uint32_t,uint16_t,NvmV2LayoutField *);
bool nvm_record_array_execution_type(const NvmRecordArrayExecutionPlan *,uint32_t,NvmOrdinaryArrayType *);
bool nvm_record_array_execution_binding(const NvmRecordArrayExecutionPlan *,uint32_t,NvmOrdinaryArrayBinding *);
bool nvm_record_array_execution_variant(const NvmRecordArrayExecutionPlan *,uint32_t,uint16_t,NvmUnionVariantFact *);
#endif
