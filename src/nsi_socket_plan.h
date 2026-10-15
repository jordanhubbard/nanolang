#ifndef NL_NSI_SOCKET_PLAN_H
#define NL_NSI_SOCKET_PLAN_H
#include "nsi_service_catalog.h"

/* I describe one immutable TCP contract; success admits no source or execution. */
#define NL_SOCKET_PLAN_METHODS 5u
#define NL_SOCKET_PLAN_TYPES 9u
typedef struct NlSocketPlan NlSocketPlan;
typedef enum { NL_SOCKET_PLAN_OK, NL_SOCKET_PLAN_INVALID, NL_SOCKET_PLAN_MEMORY } NlSocketPlanStatus;
/* Valid immutable NSI arrays/strings are required for the call. I validate the
 * exact contract before allocation; failure preserves *out. Query views borrow
 * process-lifetime catalog data, independent of the input document's lifetime. */
NlSocketPlanStatus nl_socket_plan_build(const NlNsi *, NlSocketPlan **out);
void nl_socket_plan_free(NlSocketPlan *);
size_t nl_socket_plan_storage_size(void);
const char *nl_socket_plan_interface(const NlSocketPlan *);
size_t nl_socket_plan_method_count(const NlSocketPlan *);
size_t nl_socket_plan_type_count(const NlSocketPlan *);
const NlServicePlanMethod *nl_socket_plan_method(const NlSocketPlan *, size_t);
const NlServicePlanType *nl_socket_plan_type(const NlSocketPlan *, size_t);
/* I expose immutable facts for future source producers, not an admission token. */
const char *nl_socket_catalog_interface(void);
const NlServicePlanMethod *nl_socket_catalog_method(size_t);
const NlServicePlanType *nl_socket_catalog_type(size_t);
#endif
