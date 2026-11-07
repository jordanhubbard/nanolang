#ifndef NANOISA_MANAGED_RECORD_ARRAY_ORIGINS_H
#define NANOISA_MANAGED_RECORD_ARRAY_ORIGINS_H
#include "isa.h"
#include "managed_record_shapes.h"
#include "ownership_declaration_projection.h"
/* I expose copied flow facts only, never execution admission. Inputs remain
 * immutable during preparation; every failed operation leaves output intact. */
typedef struct NvmRecordArrayOrigins NvmRecordArrayOrigins;
typedef struct {
    uint32_t origins, fields, checked_field_writes, checked_array_writes, runtime_tag_checks;
    uint64_t work_reserved, peak_bytes_reserved;
} NvmRecordArrayOriginCounts;
NvmArrayEligibilityResult nvm_analyze_record_array_origins(const NvmModule *,NvmRecordArrayOrigins **);
void nvm_record_array_origins_free(NvmRecordArrayOrigins *);
bool nvm_record_array_origins_counts(const NvmRecordArrayOrigins *,NvmRecordArrayOriginCounts *);
bool nvm_record_array_origin(const NvmRecordArrayOrigins *,uint32_t,NvmRecordHeapOrigin *);
bool nvm_record_array_field_value(const NvmRecordArrayOrigins *,uint32_t,NvmRecordValueOrigins *);
/* Zero means unconstrained by a record field; otherwise one exact leaf tag
 * bit or one normalized depth-two primitive obligation. These named bits
 * are not wildcard ARRAY or TAG_STRUCT facts: original referent rows remain
 * available through declaration APIs. */
#define NVM_RECORD_ARRAY_OF_INT ((uint16_t)(1u << TAG_ARRAY))
#define NVM_RECORD_ARRAY_OF_STRING ((uint16_t)(1u << 8))
#define NVM_RECORD_ARRAY_OF_U8 ((uint16_t)(1u << 9))
#define NVM_RECORD_ARRAY_OF_FLOAT ((uint16_t)(1u << 10))
#define NVM_RECORD_ARRAY_OF_BOOL ((uint16_t)(1u << 11))
#define NVM_RECORD_ARRAY_DEPTH3_INT ((uint16_t)(1u << 12))
#define NVM_RECORD_ARRAY_DEPTH3_U8 ((uint16_t)(1u << 13))
#define NVM_RECORD_ARRAY_DEPTH3_FLOAT ((uint16_t)(1u << 14))
#define NVM_RECORD_ARRAY_DEPTH3_BOOL ((uint16_t)(1u << 15))
#define NVM_RECORD_ARRAY_DEPTH3_STRING ((uint16_t)(1u << 6))
/* My normalized namespace is distinct from physical Value.tags. */
static inline uint16_t nvm_record_array_requirement(uint8_t depth,uint8_t leaf) {
    if(leaf<TAG_INT || leaf>TAG_STRING)return 0;
    if(depth==1)return (uint16_t)(1u<<leaf);
    if(depth==2) {
        switch(leaf) {
        case TAG_INT:return NVM_RECORD_ARRAY_OF_INT;
        case TAG_U8:return NVM_RECORD_ARRAY_OF_U8;
        case TAG_FLOAT:return NVM_RECORD_ARRAY_OF_FLOAT;
        case TAG_BOOL:return NVM_RECORD_ARRAY_OF_BOOL;
        case TAG_STRING:return NVM_RECORD_ARRAY_OF_STRING;
        }
    }
    if(depth==3)return leaf==TAG_STRING?NVM_RECORD_ARRAY_DEPTH3_STRING:(uint16_t)(1u<<(11u+leaf));
    return 0;
}
static inline uint8_t nvm_record_array_requirement_depth(uint16_t requirement) {
    for(uint8_t depth=1;depth<=3;depth++)for(uint8_t leaf=TAG_INT;leaf<=TAG_STRING;leaf++)
        if(requirement==nvm_record_array_requirement(depth,leaf))return depth;
    return 0;
}
/* I decode exact obligations only; an OR collision never names a leaf. */
static inline uint8_t nvm_record_array_requirement_leaf(uint16_t requirement) {
    switch(requirement) {
    case NVM_RECORD_ARRAY_DEPTH3_INT:case NVM_RECORD_ARRAY_OF_INT:return TAG_INT;
    case NVM_RECORD_ARRAY_DEPTH3_U8:case NVM_RECORD_ARRAY_OF_U8:return TAG_U8;
    case NVM_RECORD_ARRAY_DEPTH3_FLOAT:case NVM_RECORD_ARRAY_OF_FLOAT:return TAG_FLOAT;
    case NVM_RECORD_ARRAY_DEPTH3_BOOL:case NVM_RECORD_ARRAY_OF_BOOL:return TAG_BOOL;
    case NVM_RECORD_ARRAY_DEPTH3_STRING:case NVM_RECORD_ARRAY_OF_STRING:return TAG_STRING;
    default:return TAG_VOID;
    }
}
bool nvm_record_array_required_elements(const NvmRecordArrayOrigins *,uint32_t,uint16_t *);
bool nvm_record_array_declaration_counts(const NvmRecordArrayOrigins *,NvmDeclarationCounts *);
bool nvm_record_array_declaration_layout(const NvmRecordArrayOrigins *,uint32_t,NvmDeclarationLayout *);
bool nvm_record_array_declaration_field(const NvmRecordArrayOrigins *,uint32_t,uint16_t,NvmV2LayoutField *);
bool nvm_record_array_declaration_type(const NvmRecordArrayOrigins *,uint32_t,NvmOrdinaryArrayType *);
bool nvm_record_array_declaration_binding(const NvmRecordArrayOrigins *,uint32_t,NvmOrdinaryArrayBinding *);
bool nvm_record_array_declaration_variant(const NvmRecordArrayOrigins *,uint32_t,uint16_t,NvmUnionVariantFact *);
#endif
