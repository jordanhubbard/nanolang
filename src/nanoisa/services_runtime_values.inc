static bool fr_mixed_bound(size_t *out) {
    NlServicesCatalog catalogs[64];size_t file,tcp;
    for(unsigned i=0;i<64;i++)catalogs[i]=NL_SERVICES_FILE;
    if(!nl_services_values_storage_bound(catalogs,64,&file))return false;
    for(unsigned i=0;i<64;i++)catalogs[i]=NL_SERVICES_TCP;
    if(!nl_services_values_storage_bound(catalogs,64,&tcp))return false;
    *out=file>tcp?file:tcp;return true;
}
static NlServicesValue fr_mixed_borrow_value(NlServicesBorrow borrow) {
    NlServicesValue value={0};value.instance=borrow.instance;value.catalog=borrow.catalog;
    if(borrow.catalog==NL_SERVICES_TCP)value.value.tcp=borrow.borrow.tcp.value;
    else value.value.file=borrow.borrow.file.value;
    return value;
}
static bool fr_mixed_identity(NlServicesValue a,NlServicesValue b) {
    if(a.instance!=b.instance || a.catalog!=b.catalog)return false;
    if(a.catalog==NL_SERVICES_TCP)return a.value.tcp.invocation==b.value.tcp.invocation &&
        a.value.tcp.generation==b.value.tcp.generation && a.value.tcp.slot==b.value.tcp.slot;
    return a.value.file.invocation==b.value.file.invocation &&
        a.value.file.generation==b.value.file.generation && a.value.file.slot==b.value.file.slot;
}
static NlServicesFinish fr_mixed_destroy(NlServicesValues *values,NlServicesValueStatus status) {
    NlServicesFinish report={0};
    if(nl_services_values_destroy(&values,status,&report)!=NL_SERVICES_VALUE_OK)
        report.execution=NL_SERVICES_VALUE_STATE;
    return report;
}
static bool fr_error_payload(NlServicesOpenView view,int64_t out[NVM_SERVICES_RUNTIME_FIELDS]) {
    if(view.catalog==NL_SERVICES_FILE) {
        NlFileResult r=view.error.file;if(r.bytes>(uint64_t)INT64_MAX)return false;
        out[0]=r.status;out[1]=r.host_errno;out[2]=r.cleanup_errno;out[3]=(int64_t)r.bytes;
        out[4]=r.eof;out[5]=r.consumed;out[6]=r.cleanup_failed;return true;
    }
    if(view.catalog!=NL_SERVICES_TCP)return false;
    NlSocketResult r=view.error.tcp;if(r.bytes>(uint64_t)INT64_MAX)return false;
    out[0]=r.status;out[1]=r.host_errno;out[2]=r.cleanup_errno;out[3]=(int64_t)r.bytes;
    out[4]=r.close_attempts;out[5]=r.closed_count;out[6]=r.eof;out[7]=r.consumed;
    out[8]=r.cleanup_failed;out[9]=r.closure_unknown;out[10]=r.connect_pending;return true;
}
static bool fr_scalar_result(NvmServicesRuntime *c,uint32_t id,NlServicesScalarResult actual,NvmServicesRuntimeView *out) {
    uint32_t method=id%5,instance=id/5;
    if(instance>=c->instances || !method || actual.catalog!=c->catalogs[instance])return false;
    bool ok,eof;int64_t value;NlServicesOpenView error={0};error.catalog=actual.catalog;
    if(actual.catalog==NL_SERVICES_FILE) {
        NlFileScalarResult r=actual.result.file;
        if(r.kind!=(NlFileScalarKind)(method-1) || r.ok!=(r.detail.status==NL_FILE_OK))return false;
        ok=r.ok;eof=r.eof;value=r.value;error.error.file=r.detail;
    } else {
        static const NlSocketScalarKind kinds[]={NL_SOCKET_VALUE_CONNECT,NL_SOCKET_VALUE_SEND,
            NL_SOCKET_VALUE_CONNECT,NL_SOCKET_VALUE_RECEIVE,NL_SOCKET_VALUE_CLOSE};
        NlSocketScalarResult r=actual.result.tcp;
        if(r.kind!=kinds[method] || r.ok!=(r.detail.status==NL_SOCKET_OK || (method==3 && r.detail.status==NL_SOCKET_EOF)))return false;
        ok=r.ok;eof=r.eof;value=r.value;error.error.tcp=r.detail;
    }
    NvmServicesRuntimeView result=fr_view(c->types[instance*9+method+3]);
    result.arm=ok?NVM_SERVICES_FLOW_ARM_OK:NVM_SERVICES_FLOW_ARM_ERROR;
    if(!ok) {
        result.fields=RUNTIME_ERROR_FIELDS(c,instance*9);
        if(!fr_error_payload(error,result.values))return false;
    } else if(method==1) {
        if(value<0 || value>1)return false;
        result.fields=1;result.values[0]=value;
    } else if(method==3) {
        if(value<0 || value>255 || (eof && value))return false;
        result.fields=2;result.values[0]=value;result.values[1]=eof;
    } else if(value || eof)return false;
    *out=result;return true;
}
static bool fr_mixed_payload(const NvmServicesRuntime *c,const NvmServicesRuntimeView *p) {
    uint32_t id=p->type.catalog_ordinal;
    if(!RUNTIME_TYPE_VALID(c,id))return false;
    if(p->arm==NVM_SERVICES_FLOW_ARM_ERROR) {
        if(p->fields!=RUNTIME_ERROR_FIELDS(c,id))return false;
        uint8_t start=c->catalogs[id/9]==NL_SERVICES_TCP?6:4;
        for(uint8_t i=start;i<p->fields;i++)if(p->values[i]!=0 && p->values[i]!=1)return false;
        return true;
    }
    uint32_t ordinal=id%9;
    if(ordinal==4)return p->fields==1;
    if(ordinal==6)return p->fields==2 && (p->values[1]==0 || p->values[1]==1);
    return (ordinal==5 || ordinal==7) && !p->fields;
}
static NlServicesValueStatus fr_mixed_service(NvmServicesRuntime *c,uint32_t id,uint32_t reference,
    uint32_t input,uint32_t output,NlServicesScalarResult *scalar) {
    uint32_t instance=id/5,method=id%5;
    if(!method) {
        NlSocketEndpoint endpoint={0};const NlSocketEndpoint *address=NULL;
        if(c->catalogs[instance]==NL_SERVICES_TCP) {
            const int64_t *v=c->values[input].view.values;
            endpoint=(NlSocketEndpoint){v[0],v[1],v[2],v[3],v[4],v[5],v[6]};address=&endpoint;
        }
        return nl_services_values_acquire(c->files,instance,address,&c->values[output].owner);
    }
    if(method==4)return nl_services_value_close(c->files,instance,&c->values[input].owner,scalar);
    return nl_services_value_call(c->files,instance,method,&c->references[reference].borrow,
        method==1?c->values[input].view.values[0]:0,scalar);
}
