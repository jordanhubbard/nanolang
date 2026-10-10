#ifndef NANOISA_SERVICES_NOMINAL_H
#define NANOISA_SERVICES_NOMINAL_H
#include "service_multi_nominal.h"
#include "../nsi_file_catalog.h"
#include "../nsi_socket_plan.h"

/* I use instance*9+type and instance*5+method as private flow identifiers.
 * They are not wire catalog ordinals or host handles. Missing slots fail. */
typedef enum {
 NVM_SERVICES_CATEGORY_UNKNOWN, NVM_SERVICES_CATEGORY_OWNER,
 NVM_SERVICES_CATEGORY_OWNER_RESULT, NVM_SERVICES_CATEGORY_SCALAR_RESULT,
 NVM_SERVICES_CATEGORY_RECORD
} NvmServicesCategory;
typedef struct {
 uint32_t global_index,catalog_ordinal,source_ordinal;
 uint8_t layout_kind,ownership_flags;
 NvmServicesCategory category;
} NvmServicesNominalLayout;
typedef NvmMultiNominalPlan NvmServicesNominalPlan;
typedef NvmMultiNominalBindings NvmServicesNominalBindings;
#define NVM_SERVICES_NOMINAL_TYPES 8u
#define NVM_SERVICES_NOMINAL_MAX_LAYOUTS NVM_MULTI_NOMINAL_MAX_LAYOUTS
#define NVM_SERVICES_NOMINAL_BYTES NVM_MULTI_NOMINAL_MAX_BYTES
/* I admit only catalogs supported by the mixed executable flow engine. */
NvmMultiNominalStatus nvm_services_nominal_plan(const NvmModule *,NvmServicesNominalPlan **);
#define nvm_services_nominal_plan_free nvm_multi_nominal_plan_free
#define nvm_services_nominal_storage_bound nvm_multi_nominal_storage_bound
#define nvm_services_nominal_layout_count nvm_multi_nominal_layout_count
#define nvm_services_nominal_decode nvm_multi_nominal_decode
bool nvm_services_nominal_layout(const NvmServicesNominalPlan *,uint32_t,NvmServicesNominalLayout *);
bool nvm_services_nominal_type(const NvmServicesNominalPlan *,uint32_t,NvmServicesNominalLayout *);
bool nvm_services_nominal_source(const NvmServicesNominalPlan *,uint8_t,uint32_t,NvmServicesNominalLayout *);
bool nvm_services_nominal_import(const NvmServicesNominalPlan *,uint32_t,uint32_t *);
const NlServicePlanType *nvm_services_catalog_type(const NvmServicesNominalPlan *,uint32_t);
const NlServicePlanMethod *nvm_services_catalog_method(const NvmServicesNominalPlan *,uint32_t);
uint32_t nvm_services_catalog_types(const NvmServicesNominalPlan *,uint32_t);
bool nvm_services_catalog_endpoint(const NvmServicesNominalPlan *,uint32_t);
#endif
