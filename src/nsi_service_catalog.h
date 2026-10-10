#ifndef NL_NSI_SERVICE_CATALOG_H
#define NL_NSI_SERVICE_CATALOG_H
#include "nsi.h"
#include <stdint.h>

/* I share descriptive shapes, never caller-selected execution authority. */
typedef enum { NL_SERVICE_INPUT_NONE, NL_SERVICE_INPUT_EXCLUSIVE, NL_SERVICE_INPUT_CONSUME } NlServiceInputMode;
typedef enum { NL_SERVICE_OWNER_NONE, NL_SERVICE_OWNER_PRESERVED, NL_SERVICE_OWNER_CONSUMED } NlServiceOwnerState;
typedef enum {
    NL_SERVICE_DOMAIN_NONE, NL_SERVICE_DOMAIN_BYTE_INT,
    NL_SERVICE_DOMAIN_U32_INT, NL_SERVICE_DOMAIN_PORT_INT, NL_SERVICE_DOMAIN_IP_FAMILY
} NlServiceDomain;
typedef struct {
    const char *id, *name, *type_id;
    NlNsiDirection direction;
    NlNsiOwnership ownership;
    NlNsiLifetime lifetime;
    NlNsiMutability mutability;
    NlServiceDomain domain;
} NlServicePlanParam;
typedef struct { const char *id, *name, *type_id; NlServiceDomain domain; } NlServicePlanMember;
typedef struct {
    const char *id, *name;
    NlNsiTypeKind kind;
    const NlServicePlanMember *members;
    size_t member_count;
} NlServicePlanType;
typedef struct {
    NlServiceOwnerState input_state;
    const char *owned_payload_type; /* NULL means no owned payload. */
} NlServicePlanOutcome;
typedef struct {
    const char *id, *name, *generated_name, *binding_id;
    uint32_t abi_version, required_rights, acquired_rights;
    NlServiceInputMode input_mode;
    const NlServicePlanParam *params;
    size_t param_count;
    NlServicePlanOutcome outcomes[2]; /* I retain exact Ok, Error order. */
} NlServicePlanMethod;
#endif
