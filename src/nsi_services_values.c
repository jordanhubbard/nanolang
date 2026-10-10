#include "nsi_services_values.h"
#include "nsi_file_values_internal.h"
#include <stdlib.h>
#include <string.h>

struct NlServicesValues {
    uint32_t count;
    bool terminal;
    struct {
        NlServicesCatalog catalog;
        union { NlFileValues *file; NlSocketValues *tcp; NlWsValues *websocket; } core;
    } instances[NL_SERVICES_VALUE_INSTANCES];
    NlServicesFinish report;
};
static NlServicesValueStatus from_file(NlFileValueStatus status) {
    switch(status) {
    case NL_FILE_VALUE_OK:return NL_SERVICES_VALUE_OK;
    case NL_FILE_VALUE_ARGUMENT:return NL_SERVICES_VALUE_ARGUMENT;
    case NL_FILE_VALUE_STALE:return NL_SERVICES_VALUE_STALE;
    case NL_FILE_VALUE_TYPE:return NL_SERVICES_VALUE_TYPE;
    case NL_FILE_VALUE_BORROWED:return NL_SERVICES_VALUE_BORROWED;
    case NL_FILE_VALUE_LIMIT:return NL_SERVICES_VALUE_LIMIT;
    case NL_FILE_VALUE_MEMORY:return NL_SERVICES_VALUE_MEMORY;
    case NL_FILE_VALUE_DISPOSED:return NL_SERVICES_VALUE_DISPOSED;
    case NL_FILE_VALUE_STATE:return NL_SERVICES_VALUE_STATE;
    default:return NL_SERVICES_VALUE_STATE;
    }
}
static NlFileValueStatus to_file(NlServicesValueStatus status) {
    switch(status) {
    case NL_SERVICES_VALUE_OK:return NL_FILE_VALUE_OK;
    case NL_SERVICES_VALUE_ARGUMENT:return NL_FILE_VALUE_ARGUMENT;
    case NL_SERVICES_VALUE_STALE:return NL_FILE_VALUE_STALE;
    case NL_SERVICES_VALUE_TYPE:return NL_FILE_VALUE_TYPE;
    case NL_SERVICES_VALUE_BORROWED:return NL_FILE_VALUE_BORROWED;
    case NL_SERVICES_VALUE_LIMIT:return NL_FILE_VALUE_LIMIT;
    case NL_SERVICES_VALUE_MEMORY:return NL_FILE_VALUE_MEMORY;
    case NL_SERVICES_VALUE_DISPOSED:return NL_FILE_VALUE_DISPOSED;
    case NL_SERVICES_VALUE_STATE:return NL_FILE_VALUE_STATE;
    default:return NL_FILE_VALUE_STATE;
    }
}
static NlServicesValueStatus from_socket(NlSocketValueStatus status) {
    switch(status) {
    case NL_SOCKET_VALUE_OK:return NL_SERVICES_VALUE_OK;
    case NL_SOCKET_VALUE_ARGUMENT:return NL_SERVICES_VALUE_ARGUMENT;
    case NL_SOCKET_VALUE_STALE:return NL_SERVICES_VALUE_STALE;
    case NL_SOCKET_VALUE_TYPE:return NL_SERVICES_VALUE_TYPE;
    case NL_SOCKET_VALUE_BORROWED:return NL_SERVICES_VALUE_BORROWED;
    case NL_SOCKET_VALUE_LIMIT:return NL_SERVICES_VALUE_LIMIT;
    case NL_SOCKET_VALUE_MEMORY:return NL_SERVICES_VALUE_MEMORY;
    case NL_SOCKET_VALUE_DISPOSED:return NL_SERVICES_VALUE_DISPOSED;
    case NL_SOCKET_VALUE_STATE:return NL_SERVICES_VALUE_STATE;
    default:return NL_SERVICES_VALUE_STATE;
    }
}
static NlSocketValueStatus to_socket(NlServicesValueStatus status) {
    switch(status) {
    case NL_SERVICES_VALUE_OK:return NL_SOCKET_VALUE_OK;
    case NL_SERVICES_VALUE_ARGUMENT:return NL_SOCKET_VALUE_ARGUMENT;
    case NL_SERVICES_VALUE_STALE:return NL_SOCKET_VALUE_STALE;
    case NL_SERVICES_VALUE_TYPE:return NL_SOCKET_VALUE_TYPE;
    case NL_SERVICES_VALUE_BORROWED:return NL_SOCKET_VALUE_BORROWED;
    case NL_SERVICES_VALUE_LIMIT:return NL_SOCKET_VALUE_LIMIT;
    case NL_SERVICES_VALUE_MEMORY:return NL_SOCKET_VALUE_MEMORY;
    case NL_SERVICES_VALUE_DISPOSED:return NL_SOCKET_VALUE_DISPOSED;
    case NL_SERVICES_VALUE_STATE:return NL_SOCKET_VALUE_STATE;
    default:return NL_SOCKET_VALUE_STATE;
    }
}
static NlServicesValueStatus from_websocket(NlWsValueStatus status) {
    switch(status) {
    case NL_WS_VALUE_OK:return NL_SERVICES_VALUE_OK;
    case NL_WS_VALUE_ARGUMENT:return NL_SERVICES_VALUE_ARGUMENT;
    case NL_WS_VALUE_STALE:return NL_SERVICES_VALUE_STALE;
    case NL_WS_VALUE_TYPE:return NL_SERVICES_VALUE_TYPE;
    case NL_WS_VALUE_BORROWED:return NL_SERVICES_VALUE_BORROWED;
    case NL_WS_VALUE_LIMIT:return NL_SERVICES_VALUE_LIMIT;
    case NL_WS_VALUE_MEMORY:return NL_SERVICES_VALUE_MEMORY;
    case NL_WS_VALUE_DISPOSED:return NL_SERVICES_VALUE_DISPOSED;
    case NL_WS_VALUE_STATE:return NL_SERVICES_VALUE_STATE;
    default:return NL_SERVICES_VALUE_STATE;
    }
}
static NlWsValueStatus to_websocket(NlServicesValueStatus status) {
    switch(status) {
    case NL_SERVICES_VALUE_OK:return NL_WS_VALUE_OK;
    case NL_SERVICES_VALUE_ARGUMENT:return NL_WS_VALUE_ARGUMENT;
    case NL_SERVICES_VALUE_STALE:return NL_WS_VALUE_STALE;
    case NL_SERVICES_VALUE_TYPE:return NL_WS_VALUE_TYPE;
    case NL_SERVICES_VALUE_BORROWED:return NL_WS_VALUE_BORROWED;
    case NL_SERVICES_VALUE_LIMIT:return NL_WS_VALUE_LIMIT;
    case NL_SERVICES_VALUE_MEMORY:return NL_WS_VALUE_MEMORY;
    case NL_SERVICES_VALUE_DISPOSED:return NL_WS_VALUE_DISPOSED;
    case NL_SERVICES_VALUE_STATE:return NL_WS_VALUE_STATE;
    default:return NL_WS_VALUE_STATE;
    }
}
static NlServicesValueStatus ready(const NlServicesValues *c) {
    return !c?NL_SERVICES_VALUE_ARGUMENT:c->terminal?NL_SERVICES_VALUE_DISPOSED:NL_SERVICES_VALUE_OK;
}
static NlServicesValueStatus instance(const NlServicesValues *c,uint32_t id) {
    NlServicesValueStatus status=ready(c);
    return status!=NL_SERVICES_VALUE_OK?status:!id || id>c->count?NL_SERVICES_VALUE_TYPE:NL_SERVICES_VALUE_OK;
}
bool nl_services_values_storage_bound(const NlServicesCatalog *catalogs,uint32_t count,size_t *out) {
    if(!catalogs || !out || !count || count>NL_SERVICES_VALUE_INSTANCES)return false;
    size_t total=sizeof(NlServicesValues);
    for(uint32_t i=0;i<count;i++) {
        size_t core;
        bool ok=catalogs[i]==NL_SERVICES_FILE?nl_file_values_storage_bound(&core):
            catalogs[i]==NL_SERVICES_TCP?nl_socket_values_storage_bound(&core):false;
        if(!ok || core>SIZE_MAX-total)return false;
        total+=core;
    }
    *out=total;return true;
}
static void destroy_core(NlServicesValues *c,uint32_t i,NlServicesValueStatus status) {
    if(c->instances[i].catalog==NL_SERVICES_FILE)
        (void)nl_file_values_destroy(c->instances[i].core.file,to_file(status));
    else if(c->instances[i].catalog==NL_SERVICES_TCP)
        (void)nl_socket_values_destroy(c->instances[i].core.tcp,to_socket(status));
    else (void)nl_ws_values_destroy(c->instances[i].core.websocket,to_websocket(status));
}
bool nl_services_values_config_storage_bound(const NlServicesValueConfig *configs,uint32_t count,size_t *out) {
    if(!configs || !out || !count || count>NL_SERVICES_VALUE_INSTANCES)return false;
    size_t total=sizeof(NlServicesValues);
    for(uint32_t i=0;i<count;i++) {
        const NlServicesValueConfig *config=&configs[i];size_t core=0;
        if(config->catalog==NL_SERVICES_WEBSOCKET) {
            const NlWsTransportPolicy *p=&config->websocket;
            if(p->max_timeout_ms>60000 || (p->resolver_helper &&
               (p->resolver_helper[0]!='/' || strlen(p->resolver_helper)>=4096)) ||
               !nl_ws_values_minimum_storage(&core) || config->websocket_storage_limit<core)return false;
            core=config->websocket_storage_limit;
        } else {
            if(config->websocket_storage_limit || config->websocket.allow_network || config->websocket.allow_lookup ||
               config->websocket.resolver_helper || config->websocket.max_timeout_ms)return false;
            bool ok=config->catalog==NL_SERVICES_FILE?nl_file_values_storage_bound(&core):
                config->catalog==NL_SERVICES_TCP?nl_socket_values_storage_bound(&core):false;
            if(!ok)return false;
        }
        if(core>SIZE_MAX-total)return false;
        total+=core;
    }
    *out=total;return true;
}
NlServicesValueStatus nl_services_values_create_config(const NlServicesValueConfig *configs,uint32_t count,NlServicesValues **out) {
    size_t bound;
    if(!out || *out || !nl_services_values_config_storage_bound(configs,count,&bound))return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValues *c=calloc(1,sizeof *c);
    if(!c)return NL_SERVICES_VALUE_MEMORY;
    for(uint32_t i=0;i<count;i++) {
        c->instances[i].catalog=configs[i].catalog;
        NlServicesValueStatus status=configs[i].catalog==NL_SERVICES_FILE?
            from_file(nl_file_values_create(&c->instances[i].core.file)):
            configs[i].catalog==NL_SERVICES_TCP?from_socket(nl_socket_values_create(&c->instances[i].core.tcp)):
            from_websocket(nl_ws_values_create_bounded(&configs[i].websocket,configs[i].websocket_storage_limit,&c->instances[i].core.websocket));
        if(status!=NL_SERVICES_VALUE_OK) {
            for(uint32_t j=0;j<c->count;j++)destroy_core(c,j,status);
            free(c);return status;
        }
        c->count++;
    }
    *out=c;return NL_SERVICES_VALUE_OK;
}
NlServicesValueStatus nl_services_values_create(const NlServicesCatalog *catalogs,uint32_t count,NlServicesValues **out) {
    size_t bound;
    if(!nl_services_values_storage_bound(catalogs,count,&bound))return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValueConfig configs[NL_SERVICES_VALUE_INSTANCES]={0};
    for(uint32_t i=0;i<count;i++)configs[i].catalog=catalogs[i];
    return nl_services_values_create_config(configs,count,out);
}
NlServicesValueStatus nl_services_values_acquire(NlServicesValues *c,uint32_t index,const NlSocketEndpoint *endpoint,NlServicesValue *out) {
    NlServicesValueStatus status=ready(c);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    if(index>=c->count || !out || out->instance)return NL_SERVICES_VALUE_ARGUMENT;
    if(c->instances[index].catalog==NL_SERVICES_WEBSOCKET)return NL_SERVICES_VALUE_TYPE;
    bool file=c->instances[index].catalog==NL_SERVICES_FILE;
    if(file?endpoint!=NULL:endpoint==NULL)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValue value={0};value.instance=index+1;value.catalog=c->instances[index].catalog;
    status=file?from_file(nl_file_values_temp(c->instances[index].core.file,&value.value.file)):
        from_socket(nl_socket_values_begin_connect(c->instances[index].core.tcp,endpoint,&value.value.tcp));
    if(status==NL_SERVICES_VALUE_OK)*out=value;
    return status;
}
NlServicesValueStatus nl_services_value_move(NlServicesValues *c,NlServicesValue *source,NlServicesValue *out) {
    if(!source || !out || source==out || out->instance)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValueStatus status=instance(c,source->instance);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    uint32_t i=source->instance-1;
    if(source->catalog!=c->instances[i].catalog)return NL_SERVICES_VALUE_TYPE;
    NlServicesValue value={0};value.instance=source->instance;value.catalog=source->catalog;
    status=c->instances[i].catalog==NL_SERVICES_FILE?
        from_file(nl_file_value_move(c->instances[i].core.file,&source->value.file,&value.value.file)):
        c->instances[i].catalog==NL_SERVICES_TCP?from_socket(nl_socket_value_move(c->instances[i].core.tcp,&source->value.tcp,&value.value.tcp)):
        from_websocket(nl_ws_value_move(c->instances[i].core.websocket,&source->value.websocket,&value.value.websocket));
    if(status==NL_SERVICES_VALUE_OK){*source=(NlServicesValue){0};*out=value;}
    return status;
}
NlServicesValueStatus nl_services_value_take_ok(NlServicesValues *c,NlServicesValue *source,NlServicesValue *out) {
    if(!source || !out || source==out || out->instance)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValueStatus status=instance(c,source->instance);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    uint32_t i=source->instance-1;
    if(source->catalog!=c->instances[i].catalog)return NL_SERVICES_VALUE_TYPE;
    NlServicesValue value={0};value.instance=source->instance;value.catalog=source->catalog;
    status=c->instances[i].catalog==NL_SERVICES_FILE?
        from_file(nl_file_open_take_ok(c->instances[i].core.file,&source->value.file,&value.value.file)):
        c->instances[i].catalog==NL_SERVICES_TCP?from_socket(nl_socket_connect_take_ok(c->instances[i].core.tcp,&source->value.tcp,&value.value.tcp)):
        from_websocket(nl_ws_connect_take_ok(c->instances[i].core.websocket,&source->value.websocket,&value.value.websocket));
    if(status==NL_SERVICES_VALUE_OK){*source=(NlServicesValue){0};*out=value;}
    return status;
}
NlServicesValueStatus nl_services_value_view(NlServicesValues *c,const NlServicesValue *source,NlServicesOpenView *out) {
    if(!source || !out)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValueStatus status=instance(c,source->instance);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    uint32_t i=source->instance-1;
    if(source->catalog!=c->instances[i].catalog)return NL_SERVICES_VALUE_TYPE;
    NlServicesOpenView view={0};view.catalog=c->instances[i].catalog;
    if(view.catalog==NL_SERVICES_FILE) {
        NlFileOpenView actual;
        status=from_file(nl_file_open_view(c->instances[i].core.file,&source->value.file,&actual));
        if(status==NL_SERVICES_VALUE_OK){view.ok=actual.ok;view.error.file=actual.error;}
    } else if(view.catalog==NL_SERVICES_TCP) {
        NlSocketConnectView actual;
        status=from_socket(nl_socket_connect_view(c->instances[i].core.tcp,&source->value.tcp,&actual));
        if(status==NL_SERVICES_VALUE_OK){view.ok=actual.ok;view.pending=actual.pending;view.error.tcp=actual.error;}
    }
    if(view.catalog==NL_SERVICES_WEBSOCKET) {
        NlWsConnectView actual;
        status=from_websocket(nl_ws_connect_view(c->instances[i].core.websocket,&source->value.websocket,&actual));
        if(status==NL_SERVICES_VALUE_OK){view.ok=actual.ok;view.error.websocket=actual.error;}
    }
    if(status==NL_SERVICES_VALUE_OK)*out=view;
    return status;
}
NlServicesValueStatus nl_services_value_take_error(NlServicesValues *c,NlServicesValue *source,NlServicesOpenView *out) {
    if(!source || !out)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValueStatus status=instance(c,source->instance);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    uint32_t i=source->instance-1;
    if(source->catalog!=c->instances[i].catalog)return NL_SERVICES_VALUE_TYPE;
    NlServicesOpenView view={0};view.catalog=c->instances[i].catalog;
    status=view.catalog==NL_SERVICES_FILE?
        from_file(nl_file_open_take_error(c->instances[i].core.file,&source->value.file,&view.error.file)):
        c->instances[i].catalog==NL_SERVICES_TCP?from_socket(nl_socket_connect_take_error(c->instances[i].core.tcp,&source->value.tcp,&view.error.tcp)):
        from_websocket(nl_ws_connect_take_error(c->instances[i].core.websocket,&source->value.websocket,&view.error.websocket));
    if(status==NL_SERVICES_VALUE_OK){*source=(NlServicesValue){0};*out=view;}
    return status;
}
NlServicesValueStatus nl_services_value_borrow(NlServicesValues *c,const NlServicesValue *source,NlServicesBorrow *out) {
    if(!source || !out || out->instance)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValueStatus status=instance(c,source->instance);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    uint32_t i=source->instance-1;
    if(source->catalog!=c->instances[i].catalog)return NL_SERVICES_VALUE_TYPE;
    NlServicesBorrow borrow={0};borrow.instance=source->instance;borrow.catalog=source->catalog;
    status=c->instances[i].catalog==NL_SERVICES_FILE?
        from_file(nl_file_value_borrow(c->instances[i].core.file,&source->value.file,&borrow.borrow.file)):
        c->instances[i].catalog==NL_SERVICES_TCP?from_socket(nl_socket_value_borrow(c->instances[i].core.tcp,&source->value.tcp,&borrow.borrow.tcp)):
        from_websocket(nl_ws_value_borrow(c->instances[i].core.websocket,&source->value.websocket,&borrow.borrow.websocket));
    if(status==NL_SERVICES_VALUE_OK)*out=borrow;
    return status;
}
NlServicesValueStatus nl_services_value_end_borrow(NlServicesValues *c,NlServicesBorrow *borrow) {
    if(!borrow)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValueStatus status=instance(c,borrow->instance);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    uint32_t i=borrow->instance-1;
    if(borrow->catalog!=c->instances[i].catalog)return NL_SERVICES_VALUE_TYPE;
    status=c->instances[i].catalog==NL_SERVICES_FILE?
        from_file(nl_file_value_end_borrow(c->instances[i].core.file,&borrow->borrow.file)):
        c->instances[i].catalog==NL_SERVICES_TCP?from_socket(nl_socket_value_end_borrow(c->instances[i].core.tcp,&borrow->borrow.tcp)):
        from_websocket(nl_ws_value_end_borrow(c->instances[i].core.websocket,&borrow->borrow.websocket));
    if(status==NL_SERVICES_VALUE_OK)*borrow=(NlServicesBorrow){0};
    return status;
}
NlServicesValueStatus nl_services_value_call(NlServicesValues *c,uint32_t index,uint32_t method,const NlServicesBorrow *borrow,int64_t byte,NlServicesScalarResult *out) {
    NlServicesValueStatus status=ready(c);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    if(index>=c->count || !borrow || !out || method<1 || method>3 || (method!=1 && byte))return NL_SERVICES_VALUE_ARGUMENT;
    if(borrow->instance!=index+1 || borrow->catalog!=c->instances[index].catalog)return NL_SERVICES_VALUE_TYPE;
    if(c->instances[index].catalog==NL_SERVICES_WEBSOCKET)return NL_SERVICES_VALUE_TYPE;
    NlServicesScalarResult result={0};result.catalog=c->instances[index].catalog;
    if(result.catalog==NL_SERVICES_FILE) {
        NlFileValues *core=c->instances[index].core.file;
        const NlFileValueBorrow *b=&borrow->borrow.file;
        status=from_file(method==1?nl_file_value_write_byte(core,b,byte,&result.result.file):
            method==2?nl_file_value_rewind(core,b,&result.result.file):nl_file_value_read_byte(core,b,&result.result.file));
    } else {
        NlSocketValues *core=c->instances[index].core.tcp;
        const NlSocketValueBorrow *b=&borrow->borrow.tcp;
        status=from_socket(method==1?nl_socket_value_send_byte(core,b,byte,&result.result.tcp):
            method==2?nl_socket_value_finish_connect(core,b,&result.result.tcp):nl_socket_value_receive_byte(core,b,&result.result.tcp));
    }
    if(status==NL_SERVICES_VALUE_OK)*out=result;
    return status;
}
NlServicesValueStatus nl_services_value_close(NlServicesValues *c,uint32_t index,NlServicesValue *source,NlServicesScalarResult *out) {
    NlServicesValueStatus status=ready(c);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    if(index>=c->count || !source || !out)return NL_SERVICES_VALUE_ARGUMENT;
    if(source->instance!=index+1 || source->catalog!=c->instances[index].catalog)return NL_SERVICES_VALUE_TYPE;
    if(c->instances[index].catalog==NL_SERVICES_WEBSOCKET)return NL_SERVICES_VALUE_TYPE;
    NlServicesScalarResult result={0};result.catalog=c->instances[index].catalog;
    status=result.catalog==NL_SERVICES_FILE?
        from_file(nl_file_value_close(c->instances[index].core.file,&source->value.file,&result.result.file)):
        from_socket(nl_socket_value_close(c->instances[index].core.tcp,&source->value.tcp,&result.result.tcp));
    if(status==NL_SERVICES_VALUE_OK){*source=(NlServicesValue){0};*out=result;}
    return status;
}
NlServicesValueStatus nl_services_value_validate(NlServicesValues *c,const NlServicesValue *value,bool result) {
    if(!value)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValueStatus status=instance(c,value->instance);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    uint32_t i=value->instance-1;
    if(value->catalog!=c->instances[i].catalog)return NL_SERVICES_VALUE_TYPE;
    return value->catalog==NL_SERVICES_FILE?
        from_file(nl_file_value_validate(c->instances[i].core.file,&value->value.file,result)):
        c->instances[i].catalog==NL_SERVICES_TCP?from_socket(nl_socket_value_validate(c->instances[i].core.tcp,&value->value.tcp,result)):
        from_websocket(nl_ws_value_validate(c->instances[i].core.websocket,&value->value.websocket,result));
}
NlServicesValueStatus nl_services_borrow_validate(NlServicesValues *c,const NlServicesBorrow *borrow) {
    if(!borrow)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValueStatus status=instance(c,borrow->instance);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    uint32_t i=borrow->instance-1;
    if(borrow->catalog!=c->instances[i].catalog)return NL_SERVICES_VALUE_TYPE;
    return borrow->catalog==NL_SERVICES_FILE?
        from_file(nl_file_value_borrow_validate(c->instances[i].core.file,&borrow->borrow.file)):
        c->instances[i].catalog==NL_SERVICES_TCP?from_socket(nl_socket_value_borrow_validate(c->instances[i].core.tcp,&borrow->borrow.tcp)):
        from_websocket(nl_ws_value_borrow_validate(c->instances[i].core.websocket,&borrow->borrow.websocket));
}
NlServicesValueStatus nl_services_values_live_slots(NlServicesValues *c,uint32_t index,uint64_t *owners,uint64_t *borrowed) {
    NlServicesValueStatus status=ready(c);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    if(index>=c->count || !owners || !borrowed || owners==borrowed)return NL_SERVICES_VALUE_ARGUMENT;
    return c->instances[index].catalog==NL_SERVICES_FILE?
        from_file(nl_file_values_live_slots(c->instances[index].core.file,owners,borrowed)):
        c->instances[index].catalog==NL_SERVICES_TCP?from_socket(nl_socket_values_live_slots(c->instances[index].core.tcp,owners,borrowed)):
        from_websocket(nl_ws_values_live_slots(c->instances[index].core.websocket,owners,borrowed));
}
NlServicesValueStatus nl_services_value_drop(NlServicesValues *c,NlServicesValue *source) {
    NlServicesValueStatus status=ready(c);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    if(!source)return NL_SERVICES_VALUE_ARGUMENT;
    if(!source->instance)return NL_SERVICES_VALUE_OK;
    status=instance(c,source->instance);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    uint32_t i=source->instance-1;
    if(source->catalog!=c->instances[i].catalog)return NL_SERVICES_VALUE_TYPE;
    status=c->instances[i].catalog==NL_SERVICES_FILE?
        from_file(nl_file_value_drop(c->instances[i].core.file,&source->value.file)):
        c->instances[i].catalog==NL_SERVICES_TCP?from_socket(nl_socket_value_drop(c->instances[i].core.tcp,&source->value.tcp)):
        from_websocket(nl_ws_value_drop(c->instances[i].core.websocket,&source->value.websocket));
    if(status==NL_SERVICES_VALUE_OK)*source=(NlServicesValue){0};
    return status;
}
bool nl_services_values_report(const NlServicesValues *c,NlServicesFinish *out) {
    if(!c || !out)return false;
    if(c->terminal){*out=c->report;return true;}
    NlServicesFinish report={0};report.count=c->count;
    for(uint32_t i=0;i<c->count;i++) {
        report.instances[i].catalog=c->instances[i].catalog;
        if(c->instances[i].catalog==NL_SERVICES_FILE) {
            NlFileValuesFinish f;
            if(!nl_file_values_report(c->instances[i].core.file,&f))return false;
            report.instances[i].finish.file=f;report.cleanup_failures+=f.cleanup_failures;
            if(report.execution==NL_SERVICES_VALUE_OK)report.execution=from_file(f.execution);
        } else if(c->instances[i].catalog==NL_SERVICES_TCP) {
            NlSocketValuesFinish f;
            if(!nl_socket_values_report(c->instances[i].core.tcp,&f))return false;
            report.instances[i].finish.tcp=f;report.cleanup_failures+=f.cleanup_failures;
            if(report.execution==NL_SERVICES_VALUE_OK)report.execution=from_socket(f.execution);
        } else {
            NlWsValuesFinish f;
            if(!nl_ws_values_report(c->instances[i].core.websocket,&f))return false;
            report.instances[i].finish.websocket=f;report.cleanup_failures+=f.cleanup_failures;
            if(report.execution==NL_SERVICES_VALUE_OK)report.execution=from_websocket(f.execution);
        }
    }
    *out=report;return true;
}
NlServicesValueStatus nl_services_values_finish(NlServicesValues *c,NlServicesValueStatus execution,NlServicesFinish *out) {
    if(!c || !out || execution<NL_SERVICES_VALUE_OK || execution>NL_SERVICES_VALUE_STATE)return NL_SERVICES_VALUE_ARGUMENT;
    if(!c->terminal) {
        c->terminal=true;c->report.execution=execution;c->report.count=c->count;
        for(uint32_t i=0;i<c->count;i++) {
            c->report.instances[i].catalog=c->instances[i].catalog;
            if(c->instances[i].catalog==NL_SERVICES_FILE) {
                NlFileValuesFinish f=nl_file_values_finish(c->instances[i].core.file,to_file(execution));
                c->report.instances[i].finish.file=f;c->report.cleanup_failures+=f.cleanup_failures;
                if(c->report.execution==NL_SERVICES_VALUE_OK)c->report.execution=from_file(f.execution);
            } else if(c->instances[i].catalog==NL_SERVICES_TCP) {
                NlSocketValuesFinish f=nl_socket_values_finish(c->instances[i].core.tcp,to_socket(execution));
                c->report.instances[i].finish.tcp=f;c->report.cleanup_failures+=f.cleanup_failures;
                if(c->report.execution==NL_SERVICES_VALUE_OK)c->report.execution=from_socket(f.execution);
            } else {
                NlWsValuesFinish f=nl_ws_values_finish(c->instances[i].core.websocket,to_websocket(execution));
                c->report.instances[i].finish.websocket=f;c->report.cleanup_failures+=f.cleanup_failures;
                if(c->report.execution==NL_SERVICES_VALUE_OK)c->report.execution=from_websocket(f.execution);
            }
        }
    }
    *out=c->report;return NL_SERVICES_VALUE_OK;
}
NlServicesValueStatus nl_services_values_destroy(NlServicesValues **context,NlServicesValueStatus execution,NlServicesFinish *out) {
    if(!context)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValues *c=*context;
    NlServicesValueStatus status=nl_services_values_finish(c,execution,out);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    for(uint32_t i=0;i<c->count;i++)destroy_core(c,i,c->report.execution);
    free(c);*context=NULL;return NL_SERVICES_VALUE_OK;
}

static NlServicesValueStatus websocket_instance(NlServicesValues *c,uint32_t index) {
    NlServicesValueStatus status=ready(c);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    if(index>=c->count)return NL_SERVICES_VALUE_ARGUMENT;
    return c->instances[index].catalog==NL_SERVICES_WEBSOCKET?NL_SERVICES_VALUE_OK:NL_SERVICES_VALUE_TYPE;
}
NlServicesValueStatus nl_services_values_websocket_connect(NlServicesValues *c,uint32_t index,const void *url,size_t length,int64_t timeout,NlServicesValue *out) {
    NlServicesValueStatus status=websocket_instance(c,index);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    if(!out || out->instance)return NL_SERVICES_VALUE_ARGUMENT;
    NlServicesValue value={0};value.instance=index+1;value.catalog=NL_SERVICES_WEBSOCKET;
    status=from_websocket(nl_ws_values_connect(c->instances[index].core.websocket,url,length,timeout,&value.value.websocket));
    if(status==NL_SERVICES_VALUE_OK)*out=value;
    return status;
}
static NlServicesValueStatus websocket_borrow(NlServicesValues *c,uint32_t index,const NlServicesBorrow *borrow) {
    NlServicesValueStatus status=websocket_instance(c,index);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    if(!borrow)return NL_SERVICES_VALUE_ARGUMENT;
    return borrow->instance==index+1 && borrow->catalog==NL_SERVICES_WEBSOCKET?NL_SERVICES_VALUE_OK:NL_SERVICES_VALUE_TYPE;
}
NlServicesValueStatus nl_services_value_websocket_send(NlServicesValues *c,uint32_t index,const NlServicesBorrow *borrow,bool binary,const void *bytes,size_t length,int64_t timeout,NlWsTransportResult *out) {
    NlServicesValueStatus status=websocket_borrow(c,index,borrow);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    return from_websocket(nl_ws_value_send(c->instances[index].core.websocket,&borrow->borrow.websocket,binary,bytes,length,timeout,out));
}
NlServicesValueStatus nl_services_value_websocket_receive(NlServicesValues *c,uint32_t index,const NlServicesBorrow *borrow,int64_t timeout,NlWsMessage *message,NlWsTransportResult *out) {
    NlServicesValueStatus status=websocket_borrow(c,index,borrow);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    return from_websocket(nl_ws_value_receive(c->instances[index].core.websocket,&borrow->borrow.websocket,timeout,message,out));
}
NlServicesValueStatus nl_services_value_websocket_close(NlServicesValues *c,uint32_t index,NlServicesValue *value,int64_t timeout,NlWsTransportResult *out) {
    NlServicesValueStatus status=websocket_instance(c,index);
    if(status!=NL_SERVICES_VALUE_OK)return status;
    if(!value || !out)return NL_SERVICES_VALUE_ARGUMENT;
    if(value->instance!=index+1 || value->catalog!=NL_SERVICES_WEBSOCKET)return NL_SERVICES_VALUE_TYPE;
    status=from_websocket(nl_ws_value_close(c->instances[index].core.websocket,&value->value.websocket,timeout,out));
    if(status==NL_SERVICES_VALUE_OK)*value=(NlServicesValue){0};
    return status;
}
