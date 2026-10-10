#ifndef NL_NSI_FILE_PLAN_H
#define NL_NSI_FILE_PLAN_H
#include "nsi_service_catalog.h"
#include <stdint.h>

/* Private descriptive plan only. I grant no callable/import authority. */
typedef struct NlFilePlan NlFilePlan;
typedef enum { NL_FILE_PLAN_OK, NL_FILE_PLAN_INVALID, NL_FILE_PLAN_MEMORY } NlFilePlanStatus;
/* I preserve catalog1's names, numeric enums and layout through shared facts. */
typedef NlServiceInputMode NlFileInputMode;
typedef NlServiceOwnerState NlFileOwnerState;
typedef NlServiceDomain NlFileDomain;
#define NL_FILE_INPUT_NONE NL_SERVICE_INPUT_NONE
#define NL_FILE_INPUT_EXCLUSIVE NL_SERVICE_INPUT_EXCLUSIVE
#define NL_FILE_INPUT_CONSUME NL_SERVICE_INPUT_CONSUME
#define NL_FILE_OWNER_NONE NL_SERVICE_OWNER_NONE
#define NL_FILE_OWNER_PRESERVED NL_SERVICE_OWNER_PRESERVED
#define NL_FILE_OWNER_CONSUMED NL_SERVICE_OWNER_CONSUMED
#define NL_FILE_DOMAIN_NONE NL_SERVICE_DOMAIN_NONE
#define NL_FILE_DOMAIN_BYTE_INT NL_SERVICE_DOMAIN_BYTE_INT
typedef NlServicePlanParam NlFilePlanParam;
typedef NlServicePlanMember NlFilePlanMember;
typedef NlServicePlanType NlFilePlanType;
typedef NlServicePlanOutcome NlFilePlanOutcome;
typedef NlServicePlanMethod NlFilePlanMethod;

/* Caller supplies a valid in-memory NlNsi (including its arrays/strings).
 * I validate its exact catalog shape before allocating. Failure leaves *out
 * unchanged. Success publishes one owned plan, independent of input lifetime.
 * All queried views borrow immutable process-lifetime catalog data; no caller
 * can provide an alternative catalog. No service or generator is invoked. */
NlFilePlanStatus nl_file_plan_build(const NlNsi *, NlFilePlan **out);
void nl_file_plan_free(NlFilePlan *);
/* I report the exact owning allocation without allocating. */
size_t nl_file_plan_storage_size(void);
const char *nl_file_plan_interface(const NlFilePlan *);
size_t nl_file_plan_method_count(const NlFilePlan *);
size_t nl_file_plan_type_count(const NlFilePlan *);
const NlFilePlanMethod *nl_file_plan_method(const NlFilePlan *, size_t);
const NlFilePlanType *nl_file_plan_type(const NlFilePlan *, size_t);
#endif
