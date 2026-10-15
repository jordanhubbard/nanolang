#ifndef NANOISA_SERVICE_SOCKET_NOMINAL_H
#define NANOISA_SERVICE_SOCKET_NOMINAL_H
#include "service_bindings.h"
#include "nvm_format.h"

/* I describe catalog2 TCP in a private v2 map. I grant no execution authority. */
#define NVM_SOCKET_NOMINAL_VERSION 2u
#define NVM_SOCKET_NOMINAL_TYPES 9u
#define NVM_SOCKET_NOMINAL_BYTES 128u
#define NVM_SOCKET_NOMINAL_MAX_LAYOUTS 65536u

typedef struct {
    uint32_t imports[NVM_SERVICE_BINDING_COUNT];
    uint32_t layouts[NVM_SOCKET_NOMINAL_TYPES];
} NvmSocketNominalBindings;
NvmServiceResult nvm_socket_nominal_check(const NvmSocketNominalBindings *);
/* Failure preserves output; buffers may overlap; no input is retained. */
NvmServiceResult nvm_socket_nominal_decode(const uint8_t *,size_t,NvmSocketNominalBindings *);
/* Null bytes selects checked size-only mode. Size storage must be disjoint
 * from input and bytes; input/bytes may overlap. Failure preserves both outputs. */
NvmServiceResult nvm_socket_nominal_encode(const NvmSocketNominalBindings *,uint8_t *,size_t,size_t *);

typedef enum {
    NVM_SOCKET_NOMINAL_DESCRIBED, NVM_SOCKET_NOMINAL_INVALID,
    NVM_SOCKET_NOMINAL_LIMIT, NVM_SOCKET_NOMINAL_MEMORY
} NvmSocketNominalStatus;
typedef enum {
    NVM_SOCKET_CATEGORY_UNKNOWN, NVM_SOCKET_CATEGORY_CONN,
    NVM_SOCKET_CATEGORY_CONNECT_RESULT, NVM_SOCKET_CATEGORY_SCALAR_RESULT,
    NVM_SOCKET_CATEGORY_RECORD
} NvmSocketCategory;
typedef struct {
    uint32_t global_index, catalog_ordinal, source_ordinal;
    uint8_t layout_kind, ownership_flags;
    NvmSocketCategory category;
} NvmSocketNominalLayout;
typedef struct NvmSocketNominalPlan NvmSocketNominalPlan;
/* Caller supplies valid immutable in-memory module arrays for this call.
 * I check complete v2 service, layout, import and ownership-v1 declarations,
 * not code or host authority. Success publishes an independently owned map;
 * failure preserves *out. Shared ownership validation is not relaxed. */
NvmSocketNominalStatus nvm_socket_nominal_plan(const NvmModule *,NvmSocketNominalPlan **);
void nvm_socket_nominal_plan_free(NvmSocketNominalPlan *);
/* Exact sole-allocation size for bounded layout count; no metadata authority.
 * Failure preserves output. Composed private analyses use this before allocation. */
bool nvm_socket_nominal_storage_bound(uint32_t layouts, size_t *out);
uint32_t nvm_socket_nominal_layout_count(const NvmSocketNominalPlan *);
/* Failure preserves output. UNKNOWN rows have no catalog/runtime authority. */
bool nvm_socket_nominal_layout(const NvmSocketNominalPlan *,uint32_t,NvmSocketNominalLayout *);
bool nvm_socket_nominal_type(const NvmSocketNominalPlan *,uint32_t,NvmSocketNominalLayout *);
bool nvm_socket_nominal_source(const NvmSocketNominalPlan *,uint8_t,uint32_t,NvmSocketNominalLayout *);
bool nvm_socket_nominal_import(const NvmSocketNominalPlan *,uint32_t,uint32_t *);
#endif
