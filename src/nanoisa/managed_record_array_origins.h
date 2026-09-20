#ifndef NANOISA_MANAGED_RECORD_ARRAY_ORIGINS_H
#define NANOISA_MANAGED_RECORD_ARRAY_ORIGINS_H
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
/* Zero means unconstrained by a record field; otherwise one exact tag bit. */
bool nvm_record_array_required_elements(const NvmRecordArrayOrigins *,uint32_t,uint16_t *);
bool nvm_record_array_declaration_counts(const NvmRecordArrayOrigins *,NvmDeclarationCounts *);
bool nvm_record_array_declaration_layout(const NvmRecordArrayOrigins *,uint32_t,NvmDeclarationLayout *);
bool nvm_record_array_declaration_field(const NvmRecordArrayOrigins *,uint32_t,uint16_t,NvmV2LayoutField *);
bool nvm_record_array_declaration_type(const NvmRecordArrayOrigins *,uint32_t,NvmOrdinaryArrayType *);
bool nvm_record_array_declaration_binding(const NvmRecordArrayOrigins *,uint32_t,NvmOrdinaryArrayBinding *);
bool nvm_record_array_declaration_variant(const NvmRecordArrayOrigins *,uint32_t,uint16_t,NvmUnionVariantFact *);
#endif
