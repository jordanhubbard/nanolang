#ifndef NANOISA_MANAGED_RECORD_PLAN_H
#define NANOISA_MANAGED_RECORD_PLAN_H
#include "nvm_format.h"
#include "nvm_v2_sections.h"

#define NVM_RECORD_PLAN_MAX_LAYOUTS 256u
#define NVM_RECORD_PLAN_MAX_FIELDS 65536u
/* DESCRIBED supplies identity/shape facts, never shared-storage authority. */
typedef enum {
    NVM_RECORD_DESCRIBED, NVM_RECORD_UNRESOLVED, NVM_RECORD_INVALID,
    NVM_RECORD_LIMIT, NVM_RECORD_MEMORY
} NvmRecordPlanStatus;
typedef enum { NVM_RECORD_AUTHORITY_UNKNOWN = 0 } NvmRecordAuthority;
typedef struct {
    NvmRecordPlanStatus status;
    uint32_t layout, field;
    const char *message; /* Static storage. */
} NvmRecordPlanResult;
typedef struct {
    NvmV2Layouts layouts; /* Owned fields preserve numeric name/tag/nested facts. */
    uint32_t record_count;
    uint32_t *record_to_layout;
    uint32_t *layout_to_record; /* NO_INDEX for non-record layouts. */
    NvmRecordAuthority authority;
} NvmRecordPlan;
/* I borrow an immutable module only for this call. The resulting numeric plan
 * owns its fields/maps; name indices are facts, not borrowed string pointers.
 * Every failure leaves *out unchanged. Ownership metadata remains unresolved. */
NvmRecordPlanResult nvm_describe_managed_records(const NvmModule *, NvmRecordPlan **);
void nvm_record_plan_free(NvmRecordPlan *);
#endif
