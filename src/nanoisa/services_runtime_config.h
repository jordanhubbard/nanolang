#include "../nsi_websocket_plan.h"
#include "services_flow_config.h"
#define CoreValues NlServicesValues
#define CoreValue NlServicesValue
#define CoreBorrow NlServicesBorrow
#define CoreStatus NlServicesValueStatus
#define CoreFinish NlServicesFinish
#define CoreResult NlServicesOpenView
#define CoreScalarResult NlServicesScalarResult
#define CoreOpenView NlServicesOpenView
#define CORE_STATUS(name) NL_SERVICES_VALUE_##name
#define CORE_FN(name) RUNTIME_CORE_##name
#define RUNTIME_CORE_open_take_error nl_services_value_take_error
#define RUNTIME_CORE_open_take_ok nl_services_value_take_ok
#define RUNTIME_CORE_open_view nl_services_value_view
#define RUNTIME_CORE_value_borrow nl_services_value_borrow
#define RUNTIME_CORE_value_borrow_validate nl_services_borrow_validate
#define RUNTIME_CORE_value_drop nl_services_value_drop
#define RUNTIME_CORE_value_end_borrow nl_services_value_end_borrow
#define RUNTIME_CORE_value_move nl_services_value_move
#define RUNTIME_CORE_value_validate nl_services_value_validate
#define RUNTIME_CORE_values_report nl_services_values_report
#define RUNTIME_CORE_values_destroy fr_mixed_destroy
#define RUNTIME_VALUE_ADAPTER "services_runtime_values.inc"
#define RUNTIME_MIXED 1
#define RUNTIME_TYPE_CAPACITY (64u*9u)
#define RUNTIME_CONTEXT_MEMBERS NlServicesCatalog catalogs[64]; uint32_t instances; NlServicesValueConfig configs[64]; bool policy_set[64]; char resolver_helpers[64][4096];
#define RUNTIME_TYPE_BASE(id) ((id)/9*9)
#define RUNTIME_TYPE_COUNT(c,id) ((c)->catalogs[(id)/9]==NL_SERVICES_TCP?9u:(c)->catalogs[(id)/9]==NL_SERVICES_WEBSOCKET?7u:8u)
#define RUNTIME_TYPE_VALID(c,id) ((id)<(c)->instances*9 && (id)%9<RUNTIME_TYPE_COUNT(c,id))
#define RUNTIME_CATALOG_TYPE(c,id) (!RUNTIME_TYPE_VALID(c,id)?NULL:(c)->catalogs[(id)/9]==NL_SERVICES_TCP?nl_socket_catalog_type((id)%9):(c)->catalogs[(id)/9]==NL_SERVICES_WEBSOCKET?nl_websocket_catalog_type((id)%9):nl_file_catalog_type((id)%9))
#define RUNTIME_METHOD_LOCAL(id) ((id)%5)
#define RUNTIME_METHOD_OWNER(id) ((id)/5*9)
#define RUNTIME_IMPORT_OPTIONAL(c,id) ((c)->catalogs[(id)/5]==NL_SERVICES_WEBSOCKET && (id)%5==4)
#define RUNTIME_IMPORTS(c) ((c)->instances*5)
#define RUNTIME_ENDPOINT(c,id) ((c)->catalogs[(id)/5]==NL_SERVICES_TCP)
#define RUNTIME_ENDPOINT_TYPE(id) ((id)/5*9+8)
#define RUNTIME_ERROR_FIELDS(c,id) ((c)->catalogs[(id)/9]==NL_SERVICES_TCP?11u:(c)->catalogs[(id)/9]==NL_SERVICES_WEBSOCKET?9u:7u)
#define RUNTIME_ERROR(view) (view)
#define RUNTIME_SCALAR_PAYLOAD(c,p) fr_mixed_payload(c,p)
#define RUNTIME_BOUND_HOSTED(p,out) fr_mixed_hosted_bound(p,out)
#define RUNTIME_BOUND_CYCLIC(p,out) fr_mixed_cyclic_hosted_bound(p,out)
#define RUNTIME_BOUND_INDIRECT(p,out) fr_mixed_indirect_hosted_bound(p,out)
#define RUNTIME_WEBSOCKET_BUDGET (32u*1024u*1024u)
#define RUNTIME_BEGIN(c) fr_mixed_begin(c)
#define RUNTIME_MAX_INSTANCES 64u
#define RUNTIME_INSTANCES(c) ((c)->instances)
#define RUNTIME_CORE_INDEX(v) ((v).instance-1)
#define RUNTIME_CORE_SLOT(v) ((v).catalog==NL_SERVICES_TCP?(v).value.tcp.slot:(v).catalog==NL_SERVICES_WEBSOCKET?(v).value.websocket.slot:(v).value.file.slot)
#define RUNTIME_VALUE_MATCH(c,v,t) ((v).instance==(t).catalog_ordinal/9+1 && (v).instance<=(c)->instances && (v).catalog==(c)->catalogs[(v).instance-1])
#define RUNTIME_BORROW_VALUE(b) fr_mixed_borrow_value(b)
#define RUNTIME_BORROW_EPOCH(b) ((b).catalog==NL_SERVICES_TCP?(b).borrow.tcp.epoch:(b).catalog==NL_SERVICES_WEBSOCKET?(b).borrow.websocket.epoch:(b).borrow.file.epoch)
#define RUNTIME_IDENTITY(a,b) fr_mixed_identity(a,b)
#define RUNTIME_ZERO(a) (!(a).instance && !(a).catalog && !(a).value.file.invocation && !(a).value.file.generation && !(a).value.file.slot)
#define RUNTIME_LIVE_SLOTS(c,i,o,b) nl_services_values_live_slots((c)->files,i,o,b)
/* I derive catalogs only from the complete checked hosted nominal table. The
 * catalogs have eight File, nine TCP or seven WebSocket types; no caller table
 * or global mutable catalog selection is used by executable frames. */
#define RUNTIME_LOAD_TYPES(c,get,p) do { \
    for(uint32_t instance=0;instance<64;instance++) { \
        NominalLayout type;uint32_t base=instance*9; \
        if(!get(p,base,&type))break; \
        if(!get(p,base+6,&type))goto fail; \
        (c)->catalogs[instance]=get(p,base+8,&type)?NL_SERVICES_TCP:get(p,base+7,&type)?NL_SERVICES_FILE:NL_SERVICES_WEBSOCKET; \
        (c)->instances++; \
        for(uint32_t i=base;i<base+RUNTIME_TYPE_COUNT(c,base);i++) { \
            if(!get(p,i,&type) || type.catalog_ordinal!=i)goto fail; \
            (c)->types[i]=(SERVICE_TYPE(FlowDeclaration)){type.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION, \
                0,type.global_index,type.catalog_ordinal,type.category}; \
        } \
    } \
    if(!(c)->instances)goto fail; \
} while(0)
