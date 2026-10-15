#ifndef NANOISA_SERVICE_WEBSOCKET_NOMINAL_H
#define NANOISA_SERVICE_WEBSOCKET_NOMINAL_H
#include "service_bindings.h"
#include "nvm_format.h"

/* I describe catalog3 WebSocket in a private v2 map. I grant no execution authority. */
#define NVM_WEBSOCKET_NOMINAL_METHODS 4u
#define NVM_WEBSOCKET_NOMINAL_VERSION 2u
#define NVM_WEBSOCKET_NOMINAL_TYPES 7u
#define NVM_WEBSOCKET_NOMINAL_BYTES 104u
#define NVM_WEBSOCKET_NOMINAL_MAX_LAYOUTS 65536u

typedef struct {
    uint32_t imports[NVM_WEBSOCKET_NOMINAL_METHODS];
    uint32_t layouts[NVM_WEBSOCKET_NOMINAL_TYPES];
} NvmWebSocketNominalBindings;
NvmServiceResult nvm_websocket_nominal_check(const NvmWebSocketNominalBindings *);
/* Failure preserves output; buffers may overlap; no input is retained. */
NvmServiceResult nvm_websocket_nominal_decode(const uint8_t *,size_t,NvmWebSocketNominalBindings *);
/* Null bytes selects checked size-only mode. Size storage must be disjoint
 * from input and bytes; input/bytes may overlap. Failure preserves both outputs. */
NvmServiceResult nvm_websocket_nominal_encode(const NvmWebSocketNominalBindings *,uint8_t *,size_t,size_t *);

typedef enum {
    NVM_WEBSOCKET_NOMINAL_DESCRIBED, NVM_WEBSOCKET_NOMINAL_INVALID,
    NVM_WEBSOCKET_NOMINAL_LIMIT, NVM_WEBSOCKET_NOMINAL_MEMORY
} NvmWebSocketNominalStatus;
typedef enum {
    NVM_WEBSOCKET_CATEGORY_UNKNOWN, NVM_WEBSOCKET_CATEGORY_CONN,
    NVM_WEBSOCKET_CATEGORY_CONNECT_RESULT, NVM_WEBSOCKET_CATEGORY_SCALAR_RESULT,
    NVM_WEBSOCKET_CATEGORY_RECORD
} NvmWebSocketCategory;
typedef struct {
    uint32_t global_index, catalog_ordinal, source_ordinal;
    uint8_t layout_kind, ownership_flags;
    NvmWebSocketCategory category;
} NvmWebSocketNominalLayout;
typedef struct NvmWebSocketNominalPlan NvmWebSocketNominalPlan;
/* Caller supplies valid immutable in-memory module arrays for this call.
 * I check complete v2 service, layout, import and ownership-v1 declarations,
 * not code or host authority. Success publishes an independently owned map;
 * failure preserves *out. Shared ownership validation is not relaxed. */
NvmWebSocketNominalStatus nvm_websocket_nominal_plan(const NvmModule *,NvmWebSocketNominalPlan **);
void nvm_websocket_nominal_plan_free(NvmWebSocketNominalPlan *);
/* Exact sole-allocation size for bounded layout count; no metadata authority.
 * Failure preserves output. Composed private analyses use this before allocation. */
bool nvm_websocket_nominal_storage_bound(uint32_t layouts, size_t *out);
uint32_t nvm_websocket_nominal_layout_count(const NvmWebSocketNominalPlan *);
/* Failure preserves output. UNKNOWN rows have no catalog/runtime authority. */
bool nvm_websocket_nominal_layout(const NvmWebSocketNominalPlan *,uint32_t,NvmWebSocketNominalLayout *);
bool nvm_websocket_nominal_type(const NvmWebSocketNominalPlan *,uint32_t,NvmWebSocketNominalLayout *);
bool nvm_websocket_nominal_source(const NvmWebSocketNominalPlan *,uint8_t,uint32_t,NvmWebSocketNominalLayout *);
bool nvm_websocket_nominal_import(const NvmWebSocketNominalPlan *,uint32_t,uint32_t *);
#endif
