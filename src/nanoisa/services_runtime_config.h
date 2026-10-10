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
#define RUNTIME_CONTEXT_MEMBERS NlServicesCatalog catalogs[64]; uint32_t instances;
#define RUNTIME_TYPE_BASE(id) ((id)/9*9)
#define RUNTIME_TYPE_COUNT(c,id) ((c)->catalogs[(id)/9]==NL_SERVICES_TCP?9u:8u)
#define RUNTIME_TYPE_VALID(c,id) ((id)<(c)->instances*9 && (id)%9<RUNTIME_TYPE_COUNT(c,id))
#define RUNTIME_CATALOG_TYPE(c,id) (!RUNTIME_TYPE_VALID(c,id)?NULL:(c)->catalogs[(id)/9]==NL_SERVICES_TCP?nl_socket_catalog_type((id)%9):nl_file_catalog_type((id)%9))
#define RUNTIME_METHOD_LOCAL(id) ((id)%5)
#define RUNTIME_METHOD_OWNER(id) ((id)/5*9)
#define RUNTIME_IMPORTS(c) ((c)->instances*5)
#define RUNTIME_ENDPOINT(c,id) ((c)->catalogs[(id)/5]==NL_SERVICES_TCP)
#define RUNTIME_ENDPOINT_TYPE(id) ((id)/5*9+8)
#define RUNTIME_ERROR_FIELDS(c,id) ((c)->catalogs[(id)/9]==NL_SERVICES_TCP?11u:7u)
#define RUNTIME_ERROR(view) (view)
#define RUNTIME_SCALAR_PAYLOAD(c,p) fr_mixed_payload(c,p)
#define RUNTIME_BOUND(out) fr_mixed_bound(out)
#define RUNTIME_BEGIN(c) nl_services_values_create((c)->catalogs,(c)->instances,&(c)->files)
#define RUNTIME_MAX_INSTANCES 64u
#define RUNTIME_INSTANCES(c) ((c)->instances)
#define RUNTIME_CORE_INDEX(v) ((v).instance-1)
#define RUNTIME_CORE_SLOT(v) ((v).catalog==NL_SERVICES_TCP?(v).value.tcp.slot:(v).value.file.slot)
#define RUNTIME_VALUE_MATCH(c,v,t) ((v).instance==(t).catalog_ordinal/9+1 && (v).instance<=(c)->instances && (v).catalog==(c)->catalogs[(v).instance-1])
#define RUNTIME_BORROW_VALUE(b) fr_mixed_borrow_value(b)
#define RUNTIME_BORROW_EPOCH(b) ((b).catalog==NL_SERVICES_TCP?(b).borrow.tcp.epoch:(b).borrow.file.epoch)
#define RUNTIME_IDENTITY(a,b) fr_mixed_identity(a,b)
#define RUNTIME_ZERO(a) (!(a).instance && !(a).catalog && !(a).value.file.invocation && !(a).value.file.generation && !(a).value.file.slot)
#define RUNTIME_LIVE_SLOTS(c,i,o,b) nl_services_values_live_slots((c)->files,i,o,b)
/* I derive catalogs only from the complete checked hosted nominal table. The
 * only supported catalogs have eight File or nine TCP types; no caller table
 * or global mutable catalog selection is used by executable frames. */
#define RUNTIME_LOAD_TYPES(c,get,p) do { \
    for(uint32_t instance=0;instance<64;instance++) { \
        NominalLayout type;uint32_t base=instance*9; \
        if(!get(p,base,&type))break; \
        (c)->catalogs[instance]=get(p,base+8,&type)?NL_SERVICES_TCP:NL_SERVICES_FILE; \
        (c)->instances++; \
        for(uint32_t i=base;i<base+RUNTIME_TYPE_COUNT(c,base);i++) { \
            if(!get(p,i,&type) || type.catalog_ordinal!=i)goto fail; \
            (c)->types[i]=(SERVICE_TYPE(FlowDeclaration)){type.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION, \
                0,type.global_index,type.catalog_ordinal,type.category}; \
        } \
    } \
    if(!(c)->instances)goto fail; \
} while(0)
