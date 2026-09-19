#ifndef NL_NSI_FILE_PLAN_H
#define NL_NSI_FILE_PLAN_H
#include "nsi.h"
#include <stdint.h>

/* Private descriptive plan only. I grant no callable/import authority. */
typedef struct NlFilePlan NlFilePlan;
typedef enum { NL_FILE_PLAN_OK, NL_FILE_PLAN_INVALID, NL_FILE_PLAN_MEMORY } NlFilePlanStatus;
typedef enum { NL_FILE_INPUT_NONE, NL_FILE_INPUT_EXCLUSIVE, NL_FILE_INPUT_CONSUME } NlFileInputMode;
typedef enum { NL_FILE_OWNER_NONE, NL_FILE_OWNER_PRESERVED, NL_FILE_OWNER_CONSUMED } NlFileOwnerState;
typedef enum { NL_FILE_DOMAIN_NONE, NL_FILE_DOMAIN_BYTE_INT } NlFileDomain;
typedef struct {
    const char *id, *name, *type_id;
    NlNsiDirection direction;
    NlNsiOwnership ownership;
    NlNsiLifetime lifetime;
    NlNsiMutability mutability;
    NlFileDomain domain;
} NlFilePlanParam;
typedef struct { const char *id, *name, *type_id; NlFileDomain domain; } NlFilePlanMember;
typedef struct {
    const char *id, *name;
    NlNsiTypeKind kind;
    const NlFilePlanMember *members;
    size_t member_count;
} NlFilePlanType;
typedef struct {
    NlFileOwnerState input_state;
    /* NULL means no owned payload; otherwise the sole payload itself is File. */
    const char *owned_payload_type;
} NlFilePlanOutcome;
typedef struct {
    const char *id, *name, *generated_name, *binding_id;
    uint32_t abi_version, required_rights, acquired_rights;
    NlFileInputMode input_mode;
    const NlFilePlanParam *params;
    size_t param_count;
    NlFilePlanOutcome outcomes[2]; /* Exact declared Ok, Error order. */
} NlFilePlanMethod;

/* Caller supplies a valid in-memory NlNsi (including its arrays/strings).
 * I validate its exact catalog shape before allocating. Failure leaves *out
 * unchanged. Success publishes one owned plan, independent of input lifetime.
 * All queried views borrow immutable process-lifetime catalog data; no caller
 * can provide an alternative catalog. No service or generator is invoked. */
NlFilePlanStatus nl_file_plan_build(const NlNsi *, NlFilePlan **out);
void nl_file_plan_free(NlFilePlan *);
const char *nl_file_plan_interface(const NlFilePlan *);
size_t nl_file_plan_method_count(const NlFilePlan *);
size_t nl_file_plan_type_count(const NlFilePlan *);
const NlFilePlanMethod *nl_file_plan_method(const NlFilePlan *, size_t);
const NlFilePlanType *nl_file_plan_type(const NlFilePlan *, size_t);
#endif
