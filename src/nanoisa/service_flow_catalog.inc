/* I preserve the exact legacy single-catalog configuration. Mixed engines
 * supply instance-aware lookup functions before including shared transfers. */
#ifndef FLOW_CATALOG_TYPE
#define FLOW_CATALOG_TYPE(p,i) NOMINAL_CATALOG_TYPE(i)
#define FLOW_CATALOG_METHOD(p,i) NOMINAL_CATALOG_METHOD(i)
#define FLOW_TYPE_BASE(i) ((void)(i),0u)
#define FLOW_TYPE_COUNT(p,i) ((void)(i),NOMINAL_TYPES)
#define FLOW_IMPORT_COUNT(p) NVM_SERVICE_BINDING_COUNT
#define FLOW_METHOD_OWNER(i) ((void)(i),0u)
#define FLOW_METHOD_LOCAL(i) (i)
#define FLOW_IMPORT_COUNT_VALID(n) ((n)==NVM_SERVICE_BINDING_COUNT)
#define FLOW_IS_ENDPOINT(p,i) ((i)==FLOW_ENDPOINT_TYPE)
#endif
