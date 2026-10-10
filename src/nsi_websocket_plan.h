#ifndef NL_NSI_WEBSOCKET_PLAN_H
#define NL_NSI_WEBSOCKET_PLAN_H
#include "nsi_service_catalog.h"
#define NL_WEBSOCKET_PLAN_METHODS 4u
#define NL_WEBSOCKET_PLAN_TYPES 7u
#define NL_WEBSOCKET_PLAN_CAPABILITIES 2u
typedef struct NlWebSocketPlan NlWebSocketPlan;
typedef enum { NL_WEBSOCKET_PLAN_OK, NL_WEBSOCKET_PLAN_INVALID, NL_WEBSOCKET_PLAN_MEMORY } NlWebSocketPlanStatus;
/* I validate an exact immutable contract before allocation. Input arrays and
 * strings must be valid for this call; failure preserves *out. Queries borrow
 * process-lifetime catalog facts, never execution authority or source storage. */
NlWebSocketPlanStatus nl_websocket_plan_build(const NlNsi *,NlWebSocketPlan **out);
void nl_websocket_plan_free(NlWebSocketPlan *);
size_t nl_websocket_plan_storage_size(void);
const char *nl_websocket_plan_interface(const NlWebSocketPlan *);
size_t nl_websocket_plan_method_count(const NlWebSocketPlan *);
size_t nl_websocket_plan_type_count(const NlWebSocketPlan *);
size_t nl_websocket_plan_capability_count(const NlWebSocketPlan *);
const NlServicePlanMethod *nl_websocket_plan_method(const NlWebSocketPlan *,size_t);
const NlServicePlanType *nl_websocket_plan_type(const NlWebSocketPlan *,size_t);
const NlServicePlanCapability *nl_websocket_plan_capability(const NlWebSocketPlan *,size_t);
const char *nl_websocket_catalog_interface(void);
const NlServicePlanMethod *nl_websocket_catalog_method(size_t);
const NlServicePlanType *nl_websocket_catalog_type(size_t);
const NlServicePlanCapability *nl_websocket_catalog_capability(size_t);
#endif
