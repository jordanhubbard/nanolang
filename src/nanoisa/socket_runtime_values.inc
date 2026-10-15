/* I preserve the TCP catalog's eleven fields and the core's distinct result kinds. */
static bool fr_error_payload(NlSocketResult r,int64_t out[NVM_SOCKET_RUNTIME_FIELDS]) {
    if(r.bytes>(uint64_t)INT64_MAX)return false;
    out[0]=r.status;out[1]=r.host_errno;out[2]=r.cleanup_errno;out[3]=(int64_t)r.bytes;
    out[4]=r.close_attempts;out[5]=r.closed_count;out[6]=r.eof;out[7]=r.consumed;
    out[8]=r.cleanup_failed;out[9]=r.closure_unknown;out[10]=r.connect_pending;return true;
}
static bool fr_scalar_result(NvmSocketRuntime *c,uint32_t ordinal,NlSocketScalarResult actual,NvmSocketRuntimeView *out) {
    static const NlSocketScalarKind kinds[]={NL_SOCKET_VALUE_CONNECT,NL_SOCKET_VALUE_SEND,
        NL_SOCKET_VALUE_CONNECT,NL_SOCKET_VALUE_RECEIVE,NL_SOCKET_VALUE_CLOSE};
    if(!ordinal || ordinal>=NVM_SERVICE_BINDING_COUNT || actual.kind!=kinds[ordinal] ||
       actual.ok!=(actual.detail.status==NL_SOCKET_OK || (ordinal==3 && actual.detail.status==NL_SOCKET_EOF)))return false;
    NvmSocketRuntimeView result=fr_view(c->types[ordinal+3]);
    result.arm=actual.ok?NVM_SOCKET_FLOW_ARM_OK:NVM_SOCKET_FLOW_ARM_ERROR;
    if(!actual.ok) {
        result.fields=NVM_SOCKET_RUNTIME_FIELDS;
        if(!fr_error_payload(actual.detail,result.values))return false;
    } else if(ordinal==1) {
        if(actual.value<0 || actual.value>1)return false;
        result.fields=1;result.values[0]=actual.value;
    } else if(ordinal==3) {
        if(actual.value<0 || actual.value>255 || (actual.eof && actual.value))return false;
        result.fields=2;result.values[0]=actual.value;result.values[1]=actual.eof;
    } else if(actual.value || actual.eof)return false;
    *out=result;return true;
}
static NlSocketValueStatus fr_acquire(NlSocketValues *values,const NvmSocketRuntimeView *input,NlSocketValue *out) {
    if(!input || input->fields!=7)return NL_SOCKET_VALUE_ARGUMENT;
    NlSocketEndpoint endpoint={input->values[0],input->values[1],input->values[2],input->values[3],
        input->values[4],input->values[5],input->values[6]};
    return nl_socket_values_begin_connect(values,&endpoint,out);
}
static bool fr_scalar_payload(const NvmSocketRuntimeView *p) {
    if(p->arm==NVM_SOCKET_FLOW_ARM_ERROR) {
        if(p->fields!=NVM_SOCKET_RUNTIME_FIELDS)return false;
        for(uint8_t i=6;i<11;i++)if(p->values[i]!=0 && p->values[i]!=1)return false;
        return true;
    }
    uint32_t ordinal=p->type.catalog_ordinal;
    if(ordinal==4)return p->fields==1;
    if(ordinal==6)return p->fields==2 && (p->values[1]==0 || p->values[1]==1);
    return (ordinal==5 || ordinal==7) && !p->fields;
}
