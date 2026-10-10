/* I keep single-catalog defaults; mixed configuration supplies instance hooks. */
#ifndef RUNTIME_TYPE_CAPACITY
#define RUNTIME_TYPE_CAPACITY NOMINAL_TYPES
#ifndef RUNTIME_CONTEXT_MEMBERS
#define RUNTIME_CONTEXT_MEMBERS
#endif
#define RUNTIME_TYPE_BASE(id) (0u)
#define RUNTIME_TYPE_COUNT(c,id) NOMINAL_TYPES
#define RUNTIME_TYPE_VALID(c,id) ((id)<NOMINAL_TYPES)
#define RUNTIME_CATALOG_TYPE(c,id) NOMINAL_CATALOG_TYPE(id)
#define RUNTIME_METHOD_LOCAL(id) (id)
#define RUNTIME_METHOD_OWNER(id) (0u)
#ifndef RUNTIME_IMPORTS
#define RUNTIME_IMPORTS(c) NVM_SERVICE_BINDING_COUNT
#endif
#define RUNTIME_ENDPOINT(c,id) RUNTIME_HAS_ENDPOINT
#define RUNTIME_ENDPOINT_TYPE(id) RUNTIME_ENDPOINT_ORDINAL
#define RUNTIME_ERROR_FIELDS(c,id) SERVICE_CONST(RUNTIME_FIELDS)
#define RUNTIME_ERROR(view) ((view).error)
#ifndef RUNTIME_SCALAR_PAYLOAD
#define RUNTIME_SCALAR_PAYLOAD(c,p) fr_scalar_payload(p)
#endif
#ifndef RUNTIME_BOUND
#define RUNTIME_BOUND(out) CORE_FN(values_storage_bound)(out)
#endif
#ifndef RUNTIME_BEGIN
#define RUNTIME_BEGIN(c) CORE_FN(values_create)(&(c)->files)
#endif
#define RUNTIME_MAX_INSTANCES 1u
#define RUNTIME_INSTANCES(c) 1u
#define RUNTIME_CORE_INDEX(v) 0u
#define RUNTIME_CORE_SLOT(v) ((v).slot)
#define RUNTIME_VALUE_MATCH(c,v,t) true
#define RUNTIME_BORROW_VALUE(b) ((b).value)
#define RUNTIME_BORROW_EPOCH(b) ((b).epoch)
#define RUNTIME_IDENTITY(a,b) ((a).invocation==(b).invocation && (a).generation==(b).generation && (a).slot==(b).slot)
#define RUNTIME_ZERO(a) (!(a).invocation && !(a).generation && !(a).slot)
#define RUNTIME_LIVE_SLOTS(c,i,o,b) CORE_FN(values_live_slots)((c)->files,o,b)
#define RUNTIME_LOAD_TYPES(c,get,p) do { \
    for(uint32_t i=0;i<NOMINAL_TYPES;i++) { \
        NominalLayout type;if(!get(p,i,&type))goto fail; \
        (c)->types[i]=(SERVICE_TYPE(FlowDeclaration)){type.layout_kind==NVM_V2_LAYOUT_STRUCT?TAG_STRUCT:TAG_UNION, \
            0,type.global_index,type.catalog_ordinal,type.category}; \
    } \
} while(0)
#endif

#ifndef RUNTIME_BOUND_HOSTED
#define RUNTIME_BOUND_HOSTED(p,out) RUNTIME_BOUND(out)
#define RUNTIME_BOUND_CYCLIC(p,out) RUNTIME_BOUND(out)
#define RUNTIME_BOUND_INDIRECT(p,out) RUNTIME_BOUND(out)
#endif
